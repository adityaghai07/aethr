"""
General-purpose reward plugins — applicable to any conversational assistant.
These are domain-agnostic and should always be in your ACTIVE_PLUGINS.

Plugins defined here:
  - RuleBasedPlugin   (instant, free, synchronous)
  - LLMJudgePlugin    (async, ~$0.002/call, most informative)
"""
import re
import logging
from rewards.registry import RewardPlugin, RewardResult, register
from rewards.llm_judge import judge_response

logger = logging.getLogger(__name__)


class RuleBasedPlugin(RewardPlugin):
    """
    Deterministic signals that fire in milliseconds with no API cost.
    Checks: length appropriateness, no spurious refusals, format quality,
    language match, repetition.
    """
    name = "rule_based"
    weight = 0.25

    async def score(self, prompt: str, response: str, history: list[dict]) -> RewardResult:
        prompt_words = len(prompt.split())
        response_words = len(response.split())

        # Length appropriateness
        if prompt_words < 10:
            ideal = (10, 150)
        elif prompt_words < 50:
            ideal = (30, 500)
        else:
            ideal = (50, 1000)

        if ideal[0] <= response_words <= ideal[1]:
            length_score = 1.0
        elif response_words < ideal[0]:
            length_score = max(0.0, response_words / ideal[0])
        else:
            length_score = max(0.0, 1.0 - (response_words / ideal[1] - 1.0) * 0.5)

        # No spurious refusal
        refusal_phrases = [
            "I cannot", "I'm unable to", "I can't help with",
            "As an AI", "I don't have the ability",
        ]
        benign = not any(w in prompt.lower() for w in ["hack", "exploit", "illegal"])
        has_refusal = any(p.lower() in response.lower() for p in refusal_phrases)
        no_refusal = 0.0 if (has_refusal and benign) else 1.0

        # Format quality
        fmt = 1.0
        if response.count("```") % 2 != 0:
            fmt -= 0.5
        if response.count("**") % 2 != 0:
            fmt -= 0.2
        if prompt_words < 20 and response.count("\n- ") > 5:
            fmt -= 0.3
        fmt = max(0.0, fmt)

        # Language match (ASCII ratio heuristic)
        def _ascii(text: str) -> float:
            return sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
        lang_match = 1.0 if abs(_ascii(prompt) - _ascii(response)) < 0.3 else 0.5

        # Repetition penalty
        sentences = [s.strip() for s in response.split(". ") if len(s.strip()) > 10]
        unique = {s.lower() for s in sentences}
        repetition = len(unique) / len(sentences) if len(sentences) > 2 else 1.0

        raw = (
            length_score * 0.20 +
            no_refusal   * 0.30 +
            fmt          * 0.20 +
            lang_match   * 0.15 +
            repetition   * 0.15
        )

        return RewardResult(
            score=raw,
            details={
                "length": length_score,
                "no_refusal": no_refusal,
                "format": fmt,
                "language_match": lang_match,
                "repetition": repetition,
            },
        )


class LLMJudgePlugin(RewardPlugin):
    """
    External LLM scores the response on 5 rubric dimensions.
    Most informative signal — worth the ~$0.002/call cost.
    Runs async in the background reward worker, not inline.
    """
    name = "llm_judge"
    weight = 0.50

    async def score(self, prompt: str, response: str, history: list[dict]) -> RewardResult:
        scores = await judge_response(
            conversation_history=history,
            assistant_response=response,
        )
        if not scores:
            # Judge unavailable — return neutral, don't block
            return RewardResult(score=0.5, details={"error": "judge_unavailable"})

        raw = (
            scores.get("helpfulness",           0.5) * 0.35 +
            scores.get("tone_match",            0.5) * 0.20 +
            scores.get("conciseness",           0.5) * 0.15 +
            scores.get("factuality",            0.5) * 0.20 +
            scores.get("instruction_following", 0.5) * 0.10
        )

        return RewardResult(
            score=raw,
            details={
                "helpfulness":           scores.get("helpfulness"),
                "tone_match":            scores.get("tone_match"),
                "conciseness":           scores.get("conciseness"),
                "factuality":            scores.get("factuality"),
                "instruction_following": scores.get("instruction_following"),
                "reasoning":             scores.get("reasoning", ""),
                "cost_usd":              scores.get("judge_cost_usd", 0.0),
            },
        )


# Register instances — these names are referenced in config.ACTIVE_PLUGINS
register(RuleBasedPlugin())
register(LLMJudgePlugin())
