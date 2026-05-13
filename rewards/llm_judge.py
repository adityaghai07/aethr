"""
Tier 2 reward — LLM-as-judge scoring.
~$0.002/call with Claude Sonnet. Runs async after every response.
Rubric-based scoring is 30-40% more consistent than generic "rate 1-10" prompts.

Phase 3 implementation. Stub until then.
"""
import json
import logging
import httpx
from config import JUDGE_API_KEY, JUDGE_MODEL

logger = logging.getLogger(__name__)

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

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
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
                # Rough cost: ~500 input + 50 output tokens at Sonnet rates
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
