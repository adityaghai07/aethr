"""
REWARDANYTHING-8B reward model wrapper.

Ranked #3 on RM-Bench (86.4 overall, 84.4 Hard). A reasoning GenRM — it reads
a natural-language rubric and scores against it, making it ideal for custom
user-defined criteria (FSPO-style personalization).

Two operation modes (same as skywork_v2.py):
  1. Remote server (REWARDANYTHING_URL set): calls the reward server.
  2. Local inference: use RewardAnythingModel.load_local() in training notebooks.

The key differentiator over Skywork: REWARDANYTHING accepts a custom rubric,
enabling per-user reward customization via RewardRubric.
"""
from __future__ import annotations
import logging
import os
import httpx

from rewards.registry import RewardPlugin, RewardResult, register
from rewards.rubric import RewardRubric

logger = logging.getLogger(__name__)

MODEL_ID = "REWARDANYTHING/REWARDANYTHING-8B"
_REWARD_URL = os.getenv("REWARDANYTHING_URL", "")


class RewardAnythingModel:
    """
    Direct wrapper for local REWARDANYTHING-8B inference.
    Use in the Kaggle training notebook, not the bot process.

    Example:
        from rewards.models.rewardanything import RewardAnythingModel
        from rewards.rubric import RewardRubric
        rm = RewardAnythingModel.load_local()
        rubric = RewardRubric(name="concise", criteria=["Be concise", "Be friendly"])
        score = rm.score_with_rubric(history, response, rubric)
    """

    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer

    @classmethod
    def load_local(cls) -> "RewardAnythingModel":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
        ).eval()

        if torch.cuda.is_available():
            model = model.cuda()

        logger.info(f"Loaded {MODEL_ID}")
        return cls(model, tokenizer)

    def score_with_rubric(
        self,
        history: list[dict],
        response: str,
        rubric: RewardRubric | None = None,
    ) -> float:
        """
        Score response using a custom rubric or the model's default criteria.
        Returns a score in [0, 1].
        """
        import torch

        rubric_text = rubric.to_prompt() if rubric else (
            "Score the response 0.0–1.0 on helpfulness, accuracy, and quality.\n"
            'Return ONLY: {"score": <0.0–1.0>, "reasoning": "<one sentence>"}'
        )

        conv_text = ""
        for msg in history[-6:]:
            conv_text += f"{msg['role'].upper()}: {msg['content']}\n\n"
        conv_text += f"ASSISTANT: {response}"

        prompt = f"## Conversation\n{conv_text}\n\n## Scoring Rubric\n{rubric_text}"

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                temperature=1.0,
            )

        generated = self._tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        try:
            import json
            data = json.loads(generated.split("{", 1)[1].rsplit("}", 1)[0].join(["{", "}"]))
            return float(data.get("score", 0.5))
        except Exception:
            logger.warning(f"REWARDANYTHING parse failed: {generated!r}")
            return 0.5

    def score_batch(
        self,
        conversations: list[list[dict]],
        responses: list[str],
        rubric: RewardRubric | None = None,
    ) -> list[float]:
        """Score a batch sequentially (no padding trick — model generates text)."""
        return [
            self.score_with_rubric(conv, resp, rubric)
            for conv, resp in zip(conversations, responses)
        ]


# ── RewardPlugin wrapper ───────────────────────────────────────────────────────

class RewardAnythingPlugin(RewardPlugin):
    """
    Calls a running REWARDANYTHING server, optionally with a custom rubric.
    Disabled when REWARDANYTHING_URL is not set.

    Set REWARDANYTHING_URL to enable. Rubric can be set at runtime via
    plugin.set_rubric(rubric) to enable per-user customization.
    """
    name = "rewardanything"
    weight = 0.35

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=20.0)
        self._rubric: RewardRubric | None = None

    @property
    def enabled(self) -> bool:
        return bool(_REWARD_URL)

    def set_rubric(self, rubric: RewardRubric) -> None:
        """Swap in a custom rubric at runtime (FSPO personalization)."""
        self._rubric = rubric
        logger.info(f"RewardAnythingPlugin rubric set to: {rubric.name}")

    async def score(self, prompt: str, response: str, history: list[dict]) -> RewardResult:
        if not _REWARD_URL:
            return RewardResult(score=0.5, details={"status": "no_server"})

        payload: dict = {
            "conversation": history,
            "response": response,
        }
        if self._rubric:
            payload["rubric"] = self._rubric.to_prompt()

        try:
            resp = await self._client.post(
                f"{_REWARD_URL.rstrip('/')}/score",
                json=payload,
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            raw_score = data.get("score", 0.5)
            return RewardResult(
                score=max(0.0, min(1.0, raw_score)),
                details={
                    "reasoning": data.get("reasoning", ""),
                    "model": MODEL_ID,
                    "rubric": self._rubric.name if self._rubric else "default",
                },
            )
        except Exception as e:
            logger.warning(f"REWARDANYTHING server call failed: {e}")
            return RewardResult(score=0.5, details={"error": str(e)})


register(RewardAnythingPlugin())
