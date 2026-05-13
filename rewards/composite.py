"""
Composite reward computation using the plugin registry.

The active plugins and their weights are configured in config.ACTIVE_PLUGINS.
Add, remove, or reorder plugins there — no code changes needed here.

Final score is a weighted average of all enabled plugins, mapped to [-1, 1].
If any plugin sets violated=True, the response is flagged for hard-block.
"""
import asyncio
import logging
from rewards.registry import RewardResult

logger = logging.getLogger(__name__)


async def compute_composite(
    prompt: str,
    response: str,
    history: list[dict],
    plugins,               # list of RewardPlugin instances from config.ACTIVE_PLUGINS
) -> tuple[float, dict, bool]:
    """
    Run all enabled plugins and return:
      (composite_score, details_per_plugin, any_violation)

    composite_score is in [-1.0, 1.0], centered at 0 for GRPO advantage calculation.
    """
    if not plugins:
        logger.warning("No reward plugins configured — returning neutral score")
        return 0.0, {}, False

    # Run all plugins concurrently
    tasks = {
        plugin.name: plugin.score(prompt, response, history)
        for plugin in plugins
        if plugin.enabled
    }
    results: dict[str, RewardResult] = dict(
        zip(tasks.keys(), await asyncio.gather(*tasks.values(), return_exceptions=True))
    )

    total_weight = 0.0
    weighted_sum = 0.0
    details = {}
    any_violation = False

    for plugin in plugins:
        if not plugin.enabled:
            continue
        result = results.get(plugin.name)

        if isinstance(result, Exception):
            logger.error(f"Plugin {plugin.name} raised: {result}")
            continue

        weighted_sum += plugin.weight * result.score
        total_weight += plugin.weight
        details[plugin.name] = {"score": result.score, **result.details}

        if result.violated:
            any_violation = True
            logger.warning(f"Hard violation from plugin '{plugin.name}'")

    if total_weight == 0:
        return 0.0, details, any_violation

    # Weighted average in [0, 1] domain, then center at 0 for GRPO
    avg = weighted_sum / total_weight
    centered = (avg - 0.5) * 2.0
    return max(-1.0, min(1.0, centered)), details, any_violation


# ── Backward-compatible helper used by the inline bot scorer ─────────────────

def compute_composite_reward(
    rule_scores: dict,
    judge_scores: dict | None,
    user_feedback: dict | None,
) -> float:
    """
    Lightweight synchronous version used inline in the bot (rule-based only).
    The full async version runs in the reward worker.
    """
    from config import REWARD_WEIGHTS

    rule_avg = (
        rule_scores.get("length_appropriate", 0.5) * 0.20 +
        rule_scores.get("no_refusal_leak",    1.0) * 0.30 +
        rule_scores.get("format_quality",     1.0) * 0.20 +
        rule_scores.get("language_match",     1.0) * 0.15 +
        rule_scores.get("repetition_penalty", 1.0) * 0.15
    )

    if judge_scores:
        judge_avg = (
            judge_scores.get("helpfulness",          0.5) * 0.35 +
            judge_scores.get("tone_match",           0.5) * 0.20 +
            judge_scores.get("conciseness",          0.5) * 0.15 +
            judge_scores.get("factuality",           0.5) * 0.20 +
            judge_scores.get("instruction_following",0.5) * 0.10
        )
        raw = (
            REWARD_WEIGHTS["rule_based"] * rule_avg +
            REWARD_WEIGHTS["llm_judge"]  * judge_avg
            + (REWARD_WEIGHTS["user_feedback"] * user_feedback["score"] if user_feedback else 0)
        )
        if not user_feedback:
            raw = 0.30 * rule_avg + 0.70 * judge_avg
    else:
        raw = 0.60 * rule_avg + 0.40 * user_feedback["score"] if user_feedback else rule_avg

    return (raw - 0.5) * 2.0
