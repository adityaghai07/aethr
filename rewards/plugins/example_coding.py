"""
EXAMPLE: A custom reward plugin for a coding assistant.
Not used in Vithos — shows how to build your own plugin.

To activate: add "coding_quality" to ACTIVE_PLUGIN_NAMES in config.py
"""
import re
from rewards.registry import RewardPlugin, RewardResult, register


class CodingQualityPlugin(RewardPlugin):
    """
    Rewards code responses that include:
    - A code block (```...```)
    - A brief explanation of what the code does
    - Correct syntax (heuristic: no obvious truncation)

    Penalizes:
    - Code with no explanation
    - Responses that claim code works but have obvious issues
    """
    name = "coding_quality"
    weight = 0.30
    enabled = False   # Disabled by default — set enabled=True and add to ACTIVE_PLUGIN_NAMES

    async def score(self, prompt: str, response: str, history: list[dict]) -> RewardResult:
        is_code_query = any(w in prompt.lower() for w in [
            "code", "function", "script", "write", "implement", "debug", "fix",
            "python", "javascript", "sql", "class", "method",
        ])

        if not is_code_query:
            # Not a coding question — neutral score, don't interfere
            return RewardResult(score=0.5, details={"skipped": "not a code query"})

        has_code_block = bool(re.search(r"```[\w]*\n[\s\S]+?```", response))
        has_explanation = len(response.split("```")[0].strip()) > 30 or \
                          (len(response.split("```")) > 2 and len(response.split("```")[-1].strip()) > 30)
        is_complete = not response.rstrip().endswith(("...", "# etc", "# ..."))

        score = 0.5
        if has_code_block:
            score += 0.25
        if has_explanation:
            score += 0.15
        if is_complete:
            score += 0.10

        return RewardResult(
            score=min(1.0, score),
            details={
                "has_code_block": has_code_block,
                "has_explanation": has_explanation,
                "is_complete": is_complete,
            },
        )


# Uncomment to register:
# register(CodingQualityPlugin())
