"""
BaseAssistantHandler — platform-agnostic conversation handling.

Contains all the business logic for message processing, streaming, safety checks,
and reward scoring. Platform-specific code (Telegram send/edit, Slack post/update)
is implemented in concrete subclasses.

Subclassing:
    class SlackHandler(BaseAssistantHandler):
        async def send_placeholder(self, ctx) -> Any:
            return await slack_client.chat_postMessage(...)
        async def update_message(self, handle, text) -> None:
            await slack_client.chat_update(ts=handle.ts, text=text)
        async def finalize_message(self, handle, text) -> None:
            await slack_client.chat_update(ts=handle.ts, text=text)

    class TelegramHandler(BaseAssistantHandler):
        async def send_placeholder(self, ctx) -> Any:
            return await ctx.update.message.reply_text('▌')
        async def update_message(self, handle, text) -> None:
            await handle.edit_text(text + '▌')
        async def finalize_message(self, handle, text) -> None:
            await handle.edit_text(text)
"""
from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from config import SYSTEM_PROMPT
from inference.client import llm
from bot.conversation_manager import build_context
from bot.safety import safe_generate, is_safe
from db.queries import (
    create_conversation,
    save_message,
    get_conversation_history,
    save_reward_score,
)
from rewards.composite import compute_composite_reward
from rewards.rule_based import score_rules

logger = logging.getLogger(__name__)

STREAM_EDIT_INTERVAL = 0.6   # seconds between live message edits (rate-limit safe)


class MessageContext:
    """
    Thin wrapper so BaseAssistantHandler doesn't depend on Telegram or Slack types.
    Subclasses construct this from the platform's native event object.
    """
    def __init__(
        self,
        user_id: str,
        text: str,
        platform: str = "unknown",
        extra: dict | None = None,
    ):
        self.user_id = user_id
        self.text = text
        self.platform = platform
        self.extra = extra or {}


class BaseAssistantHandler(ABC):
    """
    Platform-agnostic message processing for Aethr.

    Subclasses must implement three methods:
      send_placeholder  — post "typing…" indicator, return a handle
      update_message    — update the in-progress message with a new partial text
      finalize_message  — set the final text (remove spinner, etc.)

    All conversation state is stored per-user in a dict passed to process_message.
    """

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    async def send_placeholder(self, ctx: MessageContext) -> Any:
        """Post a typing indicator. Return an opaque handle for update_message."""
        ...

    @abstractmethod
    async def update_message(self, handle: Any, text: str) -> None:
        """Live-update the message with partial text. Called every STREAM_EDIT_INTERVAL."""
        ...

    @abstractmethod
    async def finalize_message(self, handle: Any, text: str) -> None:
        """Replace placeholder with the final complete response."""
        ...

    async def send_followup(self, ctx: MessageContext, text: str) -> None:
        """
        Send a new separate message (used when response exceeds platform limits).
        Default: no-op. Override for platforms that support it.
        """

    async def on_error(self, ctx: MessageContext, error: Exception) -> None:
        """
        Called when generation fails. Default: log and send error message.
        Override to customize error handling.
        """
        logger.error(f"Generation failed for user {ctx.user_id}: {error}")

    # ── Core processing logic ─────────────────────────────────────────────────

    async def process_message(
        self,
        ctx: MessageContext,
        user_state: dict,
        max_message_length: int = 4096,
    ) -> str | None:
        """
        Process one user message end-to-end:
          1. Get/create conversation
          2. Stream response with live updates
          3. Safety check (regenerate if needed)
          4. Save to DB and trigger background reward scoring
          5. Return the full response text

        user_state: mutable dict for per-user state (conversation_id, last_msg_id, etc.)
                    caller is responsible for persisting this between calls.
        """
        conv_id = await self._get_or_create_conv(user_state, ctx.user_id)
        await save_message(conv_id, role="user", content=ctx.text)

        history = await get_conversation_history(conv_id)
        messages = build_context(history)

        # Send typing indicator
        handle = await self.send_placeholder(ctx)

        # Stream generation
        accumulated = ""
        last_edit_at = 0.0

        try:
            async for chunk in llm.chat_stream(messages):
                accumulated += chunk
                now = time.monotonic()
                if now - last_edit_at >= STREAM_EDIT_INTERVAL:
                    try:
                        await self.update_message(handle, accumulated[-max_message_length:])
                        last_edit_at = now
                    except Exception:
                        pass
        except Exception as e:
            await self.on_error(ctx, e)
            return None

        full_response = accumulated

        # Safety check — regenerate with stricter prompt if needed
        safe, violations = is_safe(full_response)
        if not safe:
            logger.warning(f"Unsafe streaming response, retrying: {violations}")

            async def _gen(msgs):
                out = ""
                async for chunk in llm.chat_stream(msgs):
                    out += chunk
                return out

            full_response = await safe_generate(_gen, messages, SYSTEM_PROMPT)

        # Finalize and send (handle platform message length limits)
        await self.finalize_message(handle, full_response[:max_message_length])
        for i in range(max_message_length, len(full_response), max_message_length):
            await self.send_followup(ctx, full_response[i: i + max_message_length])

        # Save to DB and kick off background reward scoring
        msg_record = await save_message(
            conv_id,
            role="assistant",
            content=full_response,
            generation_params={"temperature": 0.7, "max_tokens": 2048},
        )
        user_state["last_assistant_msg_id"] = msg_record.id

        return full_response

    async def score_in_background(self, msg_id, messages: list[dict], response: str) -> None:
        """Fire-and-forget reward scoring. Call via create_task()."""
        try:
            rule_scores = score_rules(
                prompt=messages[-1]["content"] if messages else "",
                response=response,
                conversation_history=messages,
            )
            composite = compute_composite_reward(
                rule_scores=vars(rule_scores),
                judge_scores=None,
                user_feedback=None,
            )
            await save_reward_score(
                message_id=msg_id,
                composite_score=composite,
                rule_based_scores=vars(rule_scores),
                llm_judge_scores={},
                user_feedback_scores={},
            )
        except Exception as e:
            logger.warning(f"Inline scoring failed for {msg_id}: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _get_or_create_conv(self, user_state: dict, user_id: str):
        conv_id = user_state.get("conversation_id")
        if not conv_id:
            conv = await create_conversation(user_id)
            user_state["conversation_id"] = conv.id
        return user_state["conversation_id"]
