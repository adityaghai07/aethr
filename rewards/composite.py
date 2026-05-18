"""
Composite reward computation using the plugin registry.

Ensemble mode is controlled by REWARD_ENSEMBLE_MODE env var:
  "wco"          — Worst-Case Optimization: composite = min(plugin scores).
                   Forces the model to satisfy ALL criteria, not just game one.
  "weighted_avg" — Classic weighted average (backward-compat fallback).

WCO reference: "Improving Alignment via Worst-Case Optimization" (2024).
The intuition: if helpfulness=0.9 but tone_match=0.1, the WCO score is 0.1,
not 0.55. The model must improve the worst dimension rather than exploit the best.

Group-level variance normalization (for GRPO):
  `normalize_group_advantages()` — normalizes a group's reward list to
  zero mean + unit variance. Prevents reward collapse where all rollouts
  score ~0.65±0.02, which makes GRPO advantages ≈ 0 and stalls training.
"""
import asyncio
import logging
import os
import statistics
from rewards.registry import RewardResult

logger = logging.getLogger(__name__)

_ENSEMBLE_MODE = os.getenv("REWARD_ENSEMBLE_MODE", "wco")


async def compute_composite(
    prompt: str,
    response: str,
    history: list[dict],
    plugins,               # list[RewardPlugin] from config.get_active_plugins()
) -> tuple[float, dict, bool]:
    """
    Run all enabled plugins and return:
      (composite_score, details_per_plugin, any_violation)

    composite_score is in [-1.0, 1.0], centered at 0 for GRPO advantage calculation.
    """
    if not plugins:
        logger.warning("No reward plugins configured — returning neutral score")
        return 0.0, {}, False

    active = [p for p in plugins if p.enabled]
    if not active:
        return 0.0, {}, False

    # Run all plugins concurrently
    tasks = {p.name: p.score(prompt, response, history) for p in active}
    raw_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    results: dict[str, RewardResult] = dict(zip(tasks.keys(), raw_results))

    plugin_scores: list[float] = []
    plugin_weights: list[float] = []
    details: dict = {}
    any_violation = False

    plugin_map = {p.name: p for p in active}
    for name, result in results.items():
        plugin = plugin_map[name]
        if isinstance(result, Exception):
            logger.error(f"Plugin {name} raised: {result}")
            continue

        plugin_scores.append(result.score)
        plugin_weights.append(plugin.weight)
        details[name] = {"score": result.score, **result.details}

        if result.violated:
            any_violation = True
            logger.warning(f"Hard violation from plugin '{name}'")

    if not plugin_scores:
        return 0.0, details, any_violation

    # ── Ensemble combination ───────────────────────────────────────────────────
    if _ENSEMBLE_MODE == "wco":
        # Worst-Case Optimization: composite = minimum score across all plugins.
        # Weighted softmin variant: weight each plugin's score, then take min
        # of (score / weight_normalized) to account for different importances.
        # For equal weights this reduces to pure min().
        total_w = sum(plugin_weights)
        normalized_w = [w / total_w for w in plugin_weights]
        # Penalize low scores on high-weight plugins more severely
        adjusted = [s * (1.0 - 0.3 * (1.0 - w)) for s, w in zip(plugin_scores, normalized_w)]
        raw = min(adjusted)
    else:
        # Weighted average (backward-compat)
        total_w = sum(plugin_weights)
        raw = sum(s * w for s, w in zip(plugin_scores, plugin_weights)) / total_w

    # Center at 0 for GRPO (raw is in [0,1], centered becomes [-1,1])
    centered = (raw - 0.5) * 2.0
    return max(-1.0, min(1.0, centered)), details, any_violation


def normalize_group_advantages(rewards: list[float]) -> list[float]:
    """
    Group-level advantage normalization for GRPO.

    Given N rewards from rollouts on the same prompt:
      - Subtract group mean (center advantages around 0)
      - Divide by group std dev (unit variance — prevents collapse when all
        rollouts score similarly, which would make all advantages ≈ 0)

    Returns normalized advantages. If all rewards are identical (std=0),
    returns all-zero advantages (no update — skip this prompt's gradient).

    Usage in training loop:
        for prompt, rollouts in trajectory_groups:
            rewards = [score(r) for r in rollouts]
            advantages = normalize_group_advantages(rewards)
            # use advantages in GRPO policy gradient update
    """
    if len(rewards) < 2:
        return [0.0] * len(rewards)

    mean = sum(rewards) / len(rewards)
    try:
        std = statistics.stdev(rewards)
    except statistics.StatisticsError:
        std = 0.0

    if std < 1e-6:
        # All rollouts scored the same — no learning signal for this prompt
        return [0.0] * len(rewards)

    return [(r - mean) / std for r in rewards]


# ── Backward-compatible helper used by the inline bot scorer ─────────────────

def compute_composite_reward(
    rule_scores: dict,
    judge_scores: dict | None,
    user_feedback: dict | None,
) -> float:
    """
    Lightweight synchronous version used inline in the bot (rule-based only).
    The full async version runs in the reward worker.
    Uses weighted average (WCO requires multiple plugin results, not available inline).
    """
    from config import REWARD_WEIGHTS

    rule_avg = (
        rule_scores.get("length",        0.5) * 0.20 +
        rule_scores.get("no_refusal",    1.0) * 0.30 +
        rule_scores.get("format",        1.0) * 0.20 +
        rule_scores.get("language_match",1.0) * 0.15 +
        rule_scores.get("repetition",    1.0) * 0.15
    )

    if judge_scores:
        judge_avg = (
            judge_scores.get("helpfulness",           0.5) * 0.35 +
            judge_scores.get("tone_match",            0.5) * 0.20 +
            judge_scores.get("conciseness",           0.5) * 0.15 +
            judge_scores.get("factuality",            0.5) * 0.20 +
            judge_scores.get("instruction_following", 0.5) * 0.10
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
