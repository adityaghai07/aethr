"""
Implicit user feedback signals — detected from the text of user messages.
No extra UI burden; the model just notices how the user responds.
"""
import re

_CORRECTION = [
    r"^no[,.]?\s",
    r"^wrong",
    r"^actually[,.]?\s",
    r"^that'?s (not|in)correct",
    r"^I meant",
    r"^not (quite|really)",
]

_APPRECIATION = [
    r"^thanks",
    r"^thank you",
    r"^perfect",
    r"^great",
    r"^awesome",
    r"^exactly",
    r"^that'?s (right|correct|helpful|perfect)",
    r"^nice[!.]?$",
]

# Reward deltas applied to the previous assistant message
FEEDBACK_REWARD = {
    "correction":   -0.50,
    "appreciation": +0.30,
    "abandonment":  -0.30,   # user rephrases same question (detected externally)
    "follow_up":    +0.20,   # user builds on the answer (detected externally)
}


def classify_user_message(text: str) -> str | None:
    """
    Returns 'correction', 'appreciation', or None (neutral / new topic).
    Called on every incoming user message before generating a response.
    """
    lower = text.lower().strip()
    for pattern in _CORRECTION:
        if re.match(pattern, lower):
            return "correction"
    for pattern in _APPRECIATION:
        if re.match(pattern, lower):
            return "appreciation"
    return None


def feedback_to_score(feedback_type: str) -> dict:
    """Convert a feedback classification to a score dict for the reward system."""
    if feedback_type not in FEEDBACK_REWARD:
        return {}
    return {
        "type": feedback_type,
        "score": FEEDBACK_REWARD[feedback_type],
    }
