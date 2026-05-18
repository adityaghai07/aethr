"""
Slack bot integration for Aethr.

Uses Slack's Bolt framework (slack-bolt) with Socket Mode for development
or HTTP mode (Events API) for production.

Setup:
  1. Create a Slack app at https://api.slack.com/apps
  2. Enable Socket Mode (Settings → Socket Mode) for dev
  3. Add Bot Token Scopes: chat:write, app_mentions:read, im:history, im:read
  4. Subscribe to events: app_mention, message.im
  5. Set env vars: SLACK_BOT_TOKEN, SLACK_APP_TOKEN (for Socket Mode)
                   or SLACK_SIGNING_SECRET (for HTTP mode)

Install:
    pip install slack-bolt

Run (Socket Mode, dev):
    python -m bot.slack_handler

Run (HTTP mode, production):
    Use your WSGI server to serve the Flask/FastAPI app returned by `build_app()`.
"""
from __future__ import annotations
import logging
import os
from typing import Any

from bot.base_handler import BaseAssistantHandler, MessageContext

logger = logging.getLogger(__name__)

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "")     # xapp-... for Socket Mode
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")

# Slack's message length limit
SLACK_MAX_LENGTH = 3000


class SlackMessageHandle:
    """Opaque handle for a Slack message that can be updated (thread ts + channel)."""
    def __init__(self, client, channel: str, ts: str):
        self.client = client
        self.channel = channel
        self.ts = ts


class SlackAssistantHandler(BaseAssistantHandler):
    """
    Concrete handler that talks to Slack.
    Streams responses by continuously updating the posted message.
    """

    def __init__(self, client):
        self._client = client

    async def send_placeholder(self, ctx: MessageContext) -> SlackMessageHandle:
        channel = ctx.extra.get("channel", "")
        resp = await self._client.chat_postMessage(
            channel=channel,
            text="_thinking…_",
        )
        return SlackMessageHandle(self._client, channel, resp["ts"])

    async def update_message(self, handle: SlackMessageHandle, text: str) -> None:
        try:
            await self._client.chat_update(
                channel=handle.channel,
                ts=handle.ts,
                text=text + " ▌",
            )
        except Exception as e:
            logger.debug(f"Slack update_message failed (non-fatal): {e}")

    async def finalize_message(self, handle: SlackMessageHandle, text: str) -> None:
        await self._client.chat_update(
            channel=handle.channel,
            ts=handle.ts,
            text=text,
        )

    async def send_followup(self, ctx: MessageContext, text: str) -> None:
        await self._client.chat_postMessage(
            channel=ctx.extra.get("channel", ""),
            text=text,
        )

    async def on_error(self, ctx: MessageContext, error: Exception) -> None:
        logger.error(f"Slack generation failed for {ctx.user_id}: {error}")
        await self._client.chat_postMessage(
            channel=ctx.extra.get("channel", ""),
            text="The inference server seems offline. Use `/seturl` to update it.",
        )


# ── Slack Bolt app setup ───────────────────────────────────────────────────────

def build_app():
    """
    Build and return the Slack Bolt App (async version).
    Call app.start() for Socket Mode or mount app.server() for HTTP.
    """
    try:
        from slack_bolt.async_app import AsyncApp
        from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
    except ImportError:
        raise ImportError(
            "slack-bolt not installed. Run: pip install slack-bolt"
        )

    app = AsyncApp(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
    handler = SlackAssistantHandler(app.client)
    _user_states: dict[str, dict] = {}   # user_id → {conversation_id, last_msg_id}

    @app.event("app_mention")
    @app.event("message")
    async def handle_message(event, say):
        # Ignore bot messages (including our own)
        if event.get("bot_id"):
            return

        user_id = event.get("user", "unknown")
        text = event.get("text", "").strip()
        channel = event.get("channel", "")

        # Strip bot mention prefix if present
        if text.startswith("<@"):
            text = text.split(">", 1)[-1].strip()

        if not text:
            return

        if user_id not in _user_states:
            _user_states[user_id] = {}

        ctx = MessageContext(
            user_id=user_id,
            text=text,
            platform="slack",
            extra={"channel": channel},
        )
        await handler.process_message(ctx, _user_states[user_id], SLACK_MAX_LENGTH)

    @app.command("/reset")
    async def reset_command(ack, body):
        await ack()
        user_id = body["user_id"]
        _user_states[user_id] = {}
        from db.queries import create_conversation
        conv = await create_conversation(user_id)
        _user_states[user_id]["conversation_id"] = conv.id
        await app.client.chat_postMessage(
            channel=body["channel_id"],
            text="Fresh start! What can I help with?",
        )

    return app


def main():
    """Run Aethr Slack bot in Socket Mode (development)."""
    import asyncio
    try:
        from slack_bolt.async_app import AsyncApp
        from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
    except ImportError:
        raise ImportError("Install: pip install slack-bolt")

    logging.basicConfig(level=logging.INFO)
    app = build_app()

    async def run():
        handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
        await handler.start_async()

    asyncio.run(run())


if __name__ == "__main__":
    main()
