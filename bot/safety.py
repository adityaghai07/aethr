"""
Hard safety filter — runs synchronously before a response is sent to the user.
This is separate from the reward system. Rewards shape the model over time.
This catches anything that slips through right now.

If check_response() returns False, the bot regenerates with a stricter system prompt.
After MAX_RETRIES failed attempts, it sends the fallback message instead.
"""
import asyncio
import logging
from rewards.plugins.medical import (
    _match_any,
    _DIAGNOSIS_PATTERNS,
    _ALARM_PATTERNS,
    _TREATMENT_PATTERNS,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

# Injected into the system prompt on retry to nudge the model away from violations
RETRY_SUFFIX = """

IMPORTANT REMINDER for this response:
- Observe and contextualize only. Never diagnose or name a condition.
- Never suggest starting, stopping, or adjusting any medication or treatment.
- Use calm language. No urgent or alarming words.
- If the question is clinical, gently redirect: "This sounds like something to go over with your doctor."
- Your disclaimer should sound human: "These are observations from your data, not a diagnosis. Your doctor has the final word."
"""

FALLBACK_MESSAGE = (
    "I can see some interesting patterns in your data — "
    "this one's best discussed with your doctor directly. "
    "They'll have the full picture."
)


def is_safe(response: str) -> tuple[bool, list[str]]:
    """
    Returns (safe, list_of_violations).
    Called before every response is sent to the user.
    """
    violations: list[str] = []

    for pattern_list in [_DIAGNOSIS_PATTERNS, _ALARM_PATTERNS, _TREATMENT_PATTERNS]:
        hits = _match_any(response, pattern_list)
        violations.extend(hits)

    return (len(violations) == 0, violations)


async def safe_generate(
    generate_fn,           # async callable: (messages) -> str
    messages: list[dict],
    system_prompt: str,
) -> str:
    """
    Generate a response, retrying with a stricter prompt if safety checks fail.
    Returns the safe response text, or FALLBACK_MESSAGE if all retries fail.
    """
    current_messages = messages

    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            # Strengthen system prompt on retry
            stricter_system = system_prompt + RETRY_SUFFIX
            current_messages = [
                m if m["role"] != "system" else {"role": "system", "content": stricter_system}
                for m in messages
            ]
            logger.warning(f"Safety retry {attempt}/{MAX_RETRIES}")

        response = await generate_fn(current_messages)
        safe, violations = is_safe(response)

        if safe:
            return response

        logger.warning(f"Unsafe response blocked (attempt {attempt + 1}): {violations}")

    logger.error("All retries failed — sending fallback message")
    return FALLBACK_MESSAGE
