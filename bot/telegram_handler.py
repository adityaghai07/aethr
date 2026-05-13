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
from config import TELEGRAM_TOKEN, ADMIN_USER_ID, SYSTEM_PROMPT
from inference.client import llm
from bot.conversation_manager import build_context
from bot.feedback_collector import classify_user_message, feedback_to_score
from db.queries import (
    create_conversation,
    save_message,
    get_conversation_history,
    save_reward_score,
)
from rewards.composite import compute_composite_reward
from rewards.rule_based import score_rules

logger = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_or_create_conv(user_data: dict, user_id: str):
    conv_id = user_data.get("conversation_id")
    if not conv_id:
        conv = await create_conversation(user_id)
        user_data["conversation_id"] = conv.id
    return user_data["conversation_id"]


async def _score_async(msg_id, messages: list[dict], response: str):
    """Fire-and-forget reward scoring — runs after the response is sent."""
    try:
        rule_scores = score_rules(
            prompt=messages[-1]["content"] if messages else "",
            response=response,
            conversation_history=messages,
        )
        composite = compute_composite_reward(
            rule_scores=vars(rule_scores),
            judge_scores=None,   # LLM judge runs in the background reward worker
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
    conv_id = await _get_or_create_conv(context.user_data, user_id)

    # Detect implicit feedback about the previous response
    feedback_type = classify_user_message(user_text)
    if feedback_type and (prev_msg_id := context.user_data.get("last_assistant_msg_id")):
        fb = feedback_to_score(feedback_type)
        # Re-score the previous message with user feedback included
        # (the reward worker will pick this up and update the composite)
        context.user_data["pending_feedback"] = {"msg_id": prev_msg_id, "feedback": fb}

    await save_message(conv_id, role="user", content=user_text)

    history = await get_conversation_history(conv_id)
    messages = build_context(history)

    await update.message.chat.send_action("typing")

    # ── Generate ──────────────────────────────────────────────────────────────
    full_response = ""
    try:
        async for chunk in llm.chat_stream(messages):
            full_response += chunk
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        await update.message.reply_text(
            "The inference server seems offline. Use /seturl to update the Kaggle URL."
        )
        return

    # ── Save & send ───────────────────────────────────────────────────────────
    msg_record = await save_message(
        conv_id,
        role="assistant",
        content=full_response,
        generation_params={"temperature": 0.7, "max_tokens": 2048},
    )
    context.user_data["last_assistant_msg_id"] = msg_record.id

    for i in range(0, len(full_response), 4096):  # Telegram's 4096-char limit
        await update.message.reply_text(full_response[i : i + 4096])

    # Inline rule-based scoring (fast), LLM judge runs via reward worker
    context.application.create_task(
        _score_async(msg_record.id, messages, full_response)
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
    status = "✓ reachable" if ok else "✗ not responding yet"
    await update.message.reply_text(f"Inference URL set to:\n{new_url}\nStatus: {status}")


async def health_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok = await llm.health()
    await update.message.reply_text(
        "Inference server: online ✓" if ok else "Inference server: offline ✗"
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
