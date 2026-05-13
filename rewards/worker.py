"""
Background reward worker — runs on your local machine alongside the bot.
Polls Supabase for unscored assistant messages, runs the full reward stack,
and writes scores back. The LLM judge (Tier 2) runs here, not inline.

Phase 3 activation. Runs as a separate process: python -m rewards.worker
"""
import asyncio
import logging
from db.queries import get_unscored_messages, save_reward_score, get_message_context
from rewards.rule_based import score_rules
from rewards.llm_judge import judge_response
from rewards.composite import compute_composite_reward

logger = logging.getLogger(__name__)
POLL_INTERVAL = 10  # seconds


async def _score_message(msg) -> None:
    try:
        ctx = await get_message_context(msg.id)

        rule_scores = score_rules(
            prompt=ctx["last_user_message"],
            response=msg.content,
            conversation_history=ctx["history"],
        )

        judge_scores = await judge_response(
            conversation_history=ctx["history"],
            assistant_response=msg.content,
        )

        composite = compute_composite_reward(
            rule_scores=vars(rule_scores),
            judge_scores=judge_scores,
            user_feedback=None,   # picked up later if user reacts
        )

        judge_cost = judge_scores.pop("judge_cost_usd", 0.0) if judge_scores else 0.0

        await save_reward_score(
            message_id=msg.id,
            composite_score=composite,
            rule_based_scores=vars(rule_scores),
            llm_judge_scores=judge_scores or {},
            user_feedback_scores={},
            judge_model=None,
            judge_cost_usd=judge_cost,
            helpfulness=judge_scores.get("helpfulness") if judge_scores else None,
            tone_match=judge_scores.get("tone_match") if judge_scores else None,
            conciseness=judge_scores.get("conciseness") if judge_scores else None,
            factuality=judge_scores.get("factuality") if judge_scores else None,
            instruction_following=judge_scores.get("instruction_following") if judge_scores else None,
        )
        logger.info(f"Scored {msg.id}: composite={composite:.3f}")

    except Exception as e:
        logger.error(f"Scoring failed for {msg.id}: {e}", exc_info=True)


async def run():
    logging.basicConfig(level=logging.INFO)
    logger.info("Reward worker started")
    while True:
        unscored = await get_unscored_messages(limit=20)
        if unscored:
            logger.info(f"Scoring {len(unscored)} messages...")
            for msg in unscored:
                await _score_message(msg)
        await asyncio.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    asyncio.run(run())
