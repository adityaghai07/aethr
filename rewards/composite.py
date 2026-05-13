"""
Combines rule-based, LLM judge, and user feedback into a single scalar reward.
Output is in [-1.0, 1.0], centered at 0.0 — required by GRPO's advantage computation.
"""
from config import REWARD_WEIGHTS


def compute_composite_reward(
    rule_scores: dict,
    judge_scores: dict | None,
    user_feedback: dict | None,
) -> float:
    """
    Returns a reward in [-1.0, 1.0].
    0.5 = average response. Mapped to 0.0 for GRPO centering.
    """
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
        if user_feedback:
            raw = (
                REWARD_WEIGHTS["rule_based"]    * rule_avg +
                REWARD_WEIGHTS["llm_judge"]     * judge_avg +
                REWARD_WEIGHTS["user_feedback"] * user_feedback.get("score", 0.5)
            )
        else:
            raw = 0.30 * rule_avg + 0.70 * judge_avg
    else:
        if user_feedback:
            raw = 0.60 * rule_avg + 0.40 * user_feedback.get("score", 0.5)
        else:
            raw = rule_avg

    # Map [0, 1] → [-1, 1] for GRPO advantage centering
    return (raw - 0.5) * 2.0
