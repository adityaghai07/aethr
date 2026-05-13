"""
Background reward worker — runs on your local machine alongside the bot.
Polls Supabase for unscored assistant messages, runs the full reward stack
(all active plugins), and writes scores back.

Start with: python -m rewards.worker
"""
import asyncio
import logging
from config import get_active_plugins
from db.queries import get_unscored_messages, save_reward_score, get_message_context
from rewards.composite import compute_composite

logger = logging.getLogger(__name__)
POLL_INTERVAL = 10  # seconds between DB polls


async def _score_message(msg, plugins) -> None:
    try:
        ctx = await get_message_context(msg.id)

        composite, details, any_violation = await compute_composite(
            prompt=ctx["last_user_message"],
            response=msg.content,
            history=ctx["history"],
            plugins=plugins,
        )

        if any_violation:
            logger.warning(f"Message {msg.id} has guardrail violations: {details}")

        # Extract per-dimension scores for DB columns (from llm_judge plugin if available)
        judge_detail = details.get("llm_judge", {})
        rule_detail = details.get("rule_based", {})

        await save_reward_score(
            message_id=msg.id,
            composite_score=composite,
            rule_based_scores=rule_detail,
            llm_judge_scores=judge_detail,
            user_feedback_scores={},
            judge_cost_usd=judge_detail.get("cost_usd", 0.0),
            helpfulness=judge_detail.get("helpfulness"),
            tone_match=judge_detail.get("tone_match"),
            conciseness=judge_detail.get("conciseness"),
            factuality=judge_detail.get("factuality"),
            instruction_following=judge_detail.get("instruction_following"),
        )

        logger.info(f"Scored {msg.id}: composite={composite:.3f} violated={any_violation}")

    except Exception as e:
        logger.error(f"Scoring failed for {msg.id}: {e}", exc_info=True)


async def run() -> None:
    logging.basicConfig(level=logging.INFO)
    plugins = get_active_plugins()
    logger.info(f"Reward worker started. Active plugins: {[p.name for p in plugins]}")

    while True:
        unscored = await get_unscored_messages(limit=20)
        if unscored:
            logger.info(f"Scoring {len(unscored)} messages...")
            for msg in unscored:
                await _score_message(msg, plugins)
        await asyncio.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    asyncio.run(run())
