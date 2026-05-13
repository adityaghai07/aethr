"""
Context window management: builds the LLM message list from DB history,
respecting the token budget by keeping recent messages and trimming older ones.
"""
import tiktoken
from config import MAX_CONTEXT_MESSAGES, MAX_CONTEXT_TOKENS, SYSTEM_PROMPT

# gpt-4 encoder is a close approximation for Qwen3's tokenizer
_enc = tiktoken.encoding_for_model("gpt-4")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def build_context(
    message_history: list[dict],
    system_prompt: str = SYSTEM_PROMPT,
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> list[dict]:
    """
    Build a context window from conversation history.

    Strategy:
    - Always include system prompt
    - Always keep the last 4 messages (current turn)
    - Fill remaining token budget with older messages, newest-first
    """
    system_tokens = _count_tokens(system_prompt)
    remaining = max_tokens - system_tokens

    recent = message_history[-4:]
    older = message_history[:-4]

    remaining -= sum(_count_tokens(m["content"]) for m in recent)

    included_older: list[dict] = []
    for msg in reversed(older):
        cost = _count_tokens(msg["content"])
        if remaining - cost < 0:
            break
        included_older.insert(0, msg)
        remaining -= cost

    return [{"role": "system", "content": system_prompt}] + included_older + recent
