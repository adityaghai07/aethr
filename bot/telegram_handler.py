"""
Telegram bot — the user-facing entry point of Aethr.

Commands:
  /start   — greet and create a new conversation
  /reset   — clear conversation history (new conversation)
  /rate    — prompt the user to rate the last response
  /stats   — basic usage stats
  /seturl  — (admin only) update the inference server URL after Kaggle restart
  /health  — check inference server reachability
"""
import logging
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from config import TELEGRAM_TOKEN, ADMIN_USER_ID
from inference.client import llm
from bot.base_handler import BaseAssistantHandler, MessageContext
from bot.feedback_collector import classify_user_message, feedback_to_score
from db.queries import create_conversation, get_conversation_history

logger = logging.getLogger(__name__)

# Telegram's hard message length limit
TELEGRAM_MAX_LENGTH = 4096


class TelegramAssistantHandler(BaseAssistantHandler):
    """Concrete handler that talks to Telegram."""

    async def send_placeholder(self, ctx: MessageContext):
        update: Update = ctx.extra["update"]
        return await update.message.reply_text("▌")

    async def update_message(self, handle, text: str) -> None:
        try:
            await handle.edit_text(text + "▌")
        except Exception:
            pass  # "message not modified" or transient — ignore

    async def finalize_message(self, handle, text: str) -> None:
        await handle.edit_text(text)

    async def send_followup(self, ctx: MessageContext, text: str) -> None:
        update: Update = ctx.extra["update"]
        await update.message.reply_text(text)

    async def on_error(self, ctx: MessageContext, error: Exception) -> None:
        logger.error(f"Generation failed for user {ctx.user_id}: {error}")
        handle = ctx.extra.get("placeholder_handle")
        if handle:
            await handle.edit_text(
                "The inference server seems offline. Use /seturl to update the Kaggle URL."
            )


_handler = TelegramAssistantHandler()


# ── Handlers ──────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    conv = await create_conversation(user_id)
    context.user_data["conversation_id"] = conv.id
    await update.message.reply_text(
        "Hey, I'm Aethr — your personal AI assistant. Just message me anything.\n\n"
        "Commands: /reset (new conversation) · /rate (rate last response) · /stats"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    user_text = update.message.text

    feedback_type = classify_user_message(user_text)
    if feedback_type and (prev_msg_id := context.user_data.get("last_assistant_msg_id")):
        fb = feedback_to_score(feedback_type)
        context.user_data["pending_feedback"] = {"msg_id": prev_msg_id, "feedback": fb}

    ctx = MessageContext(
        user_id=user_id,
        text=user_text,
        platform="telegram",
        extra={"update": update},
    )

    response = await _handler.process_message(
        ctx,
        context.user_data,
        max_message_length=TELEGRAM_MAX_LENGTH,
    )

    if response and (last_id := context.user_data.get("last_assistant_msg_id")):
        history = await get_conversation_history(context.user_data.get("conversation_id"), limit=10)
        context.application.create_task(
            _handler.score_in_background(last_id, history, response)
        )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    conv = await create_conversation(user_id)
    context.user_data.clear()
    context.user_data["conversation_id"] = conv.id
    await update.message.reply_text("Fresh start! What can I help with?")


async def rate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "How was my last response?\n"
        "Reply with: 👍 helpful · 👎 wrong · 💡 could be better\n"
        "Or just correct me directly — I'll pick it up."
    )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conv_id = context.user_data.get("conversation_id")
    if not conv_id:
        await update.message.reply_text("No conversation yet — send me a message!")
        return
    history = await get_conversation_history(conv_id)
    turns = sum(1 for m in history if m["role"] == "user")
    await update.message.reply_text(f"This conversation: {turns} messages so far.")


async def seturl_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin-only: update inference URL after Kaggle session restarts."""
    if update.effective_user.id != ADMIN_USER_ID:
        return
    if not context.args:
        await update.message.reply_text("Usage: /seturl <ngrok-url>")
        return
    new_url = context.args[0]
    llm.set_url(new_url)
    ok = await llm.health()
    status = "reachable" if ok else "not responding yet"
    await update.message.reply_text(f"Inference URL set to:\n{new_url}\nStatus: {status}")


async def health_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok = await llm.health()
    await update.message.reply_text(
        "Inference server: online" if ok else "Inference server: offline"
    )


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO)
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",   start))
    app.add_handler(CommandHandler("reset",   reset_command))
    app.add_handler(CommandHandler("rate",    rate_command))
    app.add_handler(CommandHandler("stats",   stats_command))
    app.add_handler(CommandHandler("seturl",  seturl_command))
    app.add_handler(CommandHandler("health",  health_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Aethr bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
