"""
Tier 1 rewards — deterministic, instant, free.
These fire synchronously on every response, giving an immediate signal.
"""
from dataclasses import dataclass


@dataclass
class RuleScores:
    length_appropriate: float   # Is response length proportional to prompt complexity?
    no_refusal_leak: float      # Did the model refuse a benign request?
    format_quality: float       # Unclosed code blocks, excessive bullets in casual chat
    language_match: float       # Response language matches prompt language
    repetition_penalty: float   # Model repeating itself


def score_rules(
    prompt: str,
    response: str,
    conversation_history: list[dict],
) -> RuleScores:
    # 1. Length appropriateness
    prompt_words = len(prompt.split())
    response_words = len(response.split())

    if prompt_words < 10:
        ideal_range = (10, 150)
    elif prompt_words < 50:
        ideal_range = (30, 500)
    else:
        ideal_range = (50, 1000)

    if ideal_range[0] <= response_words <= ideal_range[1]:
        length_score = 1.0
    elif response_words < ideal_range[0]:
        length_score = max(0.0, response_words / ideal_range[0])
    else:
        overshoot = response_words / ideal_range[1]
        length_score = max(0.0, 1.0 - (overshoot - 1.0) * 0.5)

    # 2. No spurious refusal
    refusal_phrases = [
        "I cannot", "I'm unable to", "I can't help with",
        "As an AI", "I don't have the ability",
    ]
    benign_prompt = not any(w in prompt.lower() for w in ["hack", "exploit", "illegal"])
    has_refusal = any(p.lower() in response.lower() for p in refusal_phrases)
    no_refusal = 0.0 if (has_refusal and benign_prompt) else 1.0

    # 3. Format quality
    format_score = 1.0
    if response.count("```") % 2 != 0:
        format_score -= 0.5
    if response.count("**") % 2 != 0:
        format_score -= 0.2
    if prompt_words < 20 and response.count("\n- ") > 5:
        format_score -= 0.3
    format_score = max(0.0, format_score)

    # 4. Language match (ASCII ratio heuristic)
    def ascii_ratio(text: str) -> float:
        return sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    language_match = 1.0 if abs(ascii_ratio(prompt) - ascii_ratio(response)) < 0.3 else 0.5

    # 5. Repetition
    sentences = [s.strip() for s in response.split(". ") if len(s.strip()) > 10]
    unique = set(s.lower() for s in sentences)
    repetition_score = len(unique) / len(sentences) if len(sentences) > 2 else 1.0

    return RuleScores(
        length_appropriate=length_score,
        no_refusal_leak=no_refusal,
        format_quality=format_score,
        language_match=language_match,
        repetition_penalty=repetition_score,
    )
