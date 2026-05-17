"""
Tier 2 reward — LLM-as-judge scoring.
~$0.002/call with Claude Sonnet. Runs async after every response.
Rubric-based scoring is 30-40% more consistent than generic "rate 1-10" prompts.

A module-level shared httpx.AsyncClient reuses TLS connections across calls,
so parallel judge requests during GRPO training don't pay the ~150ms TLS
handshake cost on every call.
"""
import json
import logging
import httpx
from config import JUDGE_API_KEY, JUDGE_MODEL

logger = logging.getLogger(__name__)

# Shared client — keeps the HTTPS/HTTP-2 connection pool warm across calls.
# HTTP/2 multiplexes multiple parallel requests over a single connection,
# so concurrent judge calls from GRPO rollouts don't queue at the transport layer.
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=30.0,
            http2=True,
            limits=httpx.Limits(
                max_connections=20,           # plenty for parallel rollouts
                max_keepalive_connections=20,
                keepalive_expiry=60.0,
            ),
        )
    return _client


RUBRIC = """You are evaluating an AI assistant's response in a personal conversation.

Score EACH dimension from 0.0 to 1.0:

1. helpfulness: Does it address what the user asked? (0=off-topic, 1=fully answers)
2. tone_match: Is the tone appropriate for the context? (0=robotic/formal, 1=natural/warm)
3. conciseness: Is it the right length? (0=verbose/repetitive, 1=no fluff, no gaps)
4. factuality: Is the information accurate? (0=clear errors, 1=accurate or hedged)
5. instruction_following: Were explicit instructions followed? (0=ignored, 1=precise)

Respond with ONLY valid JSON — no markdown fences, no explanation:
{"helpfulness": X, "tone_match": X, "conciseness": X, "factuality": X, "instruction_following": X, "reasoning": "one sentence"}"""


async def judge_response(
    conversation_history: list[dict],
    assistant_response: str,
) -> dict | None:
    """
    Score an assistant response.
    Returns score dict or None if the judge call fails.
    Cost is tracked in the return dict for DB logging.
    """
    conv_text = ""
    for msg in conversation_history[-6:]:
        conv_text += f"{msg['role'].upper()}: {msg['content']}\n\n"

    prompt = f"## Conversation\n{conv_text}\n## Response to evaluate\nASSISTANT: {assistant_response}\n\n## Rubric\n{RUBRIC}"

    client = _get_client()
    try:
        if "claude" in JUDGE_MODEL:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": JUDGE_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": JUDGE_MODEL,
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            result_text = resp.json()["content"][0]["text"]
            cost = (500 * 3 + 50 * 15) / 1_000_000
        else:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {JUDGE_API_KEY}"},
                json={
                    "model": JUDGE_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.0,
                },
            )
            result_text = resp.json()["choices"][0]["message"]["content"]
            cost = 0.001

        result_text = result_text.strip().lstrip("```json").rstrip("```").strip()
        scores = json.loads(result_text)
        scores["judge_cost_usd"] = cost
        return scores

    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return None
