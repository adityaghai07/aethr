"""
Skywork-Reward-V2-Llama-3.1-8B-40M reward model wrapper.

Ranked #1 on RM-Bench (96.0 overall, 93.5 Hard). This model correlates 0.55
with downstream RLHF improvement vs 0.21 for RewardBench-ranked models.

Two operation modes:
  1. Remote server (SKYWORK_REWARD_URL set): calls a running reward server.
     Deploy on Kaggle/A100 alongside training — shares compute with the
     training run and scores rollouts in parallel.

  2. Local inference (when running on GPU): loads the model directly.
     Use `SkyworkRewardModel.load_local()` in the training notebook.

The RewardPlugin wraps mode 1 for the background reward worker.
For training-time scoring, use `SkyworkRewardModel` directly.

IMPORTANT: This model MUST use attn_implementation="flash_attention_2" or
"eager" — SDPA has a documented bug with Skywork-V2. flash_attention_2 is
preferred on A100; "eager" for T4/V100.
"""
from __future__ import annotations
import json
import logging
import os
import httpx

from rewards.registry import RewardPlugin, RewardResult, register

logger = logging.getLogger(__name__)

MODEL_ID = "Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M"
_REWARD_URL = os.getenv("SKYWORK_REWARD_URL", "")   # e.g. http://ngrok-url/score


class SkyworkRewardModel:
    """
    Direct wrapper for local Skywork-V2 inference.
    Use this in the Kaggle training notebook, NOT in the bot process.

    Example (training notebook):
        from rewards.models.skywork_v2 import SkyworkRewardModel
        rm = SkyworkRewardModel.load_local(attn_impl="flash_attention_2")
        score = rm.score_single(conversation, response)
    """

    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer

    @classmethod
    def load_local(cls, attn_impl: str = "eager") -> "SkyworkRewardModel":
        """
        Load Skywork-V2 on the current GPU.
        attn_impl: "flash_attention_2" (A100+) or "eager" (T4/V100).
        SDPA is NOT supported — do not pass "sdpa".
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            num_labels=1,
        ).eval()

        if torch.cuda.is_available():
            model = model.cuda()

        logger.info(f"Loaded {MODEL_ID} with attn={attn_impl}")
        return cls(model, tokenizer)

    def score_batch(self, conversations: list[list[dict]]) -> list[float]:
        """
        Score a batch of conversations. Each conversation is a list of
        {"role": ..., "content": ...} dicts including the assistant response.
        Returns raw reward logits (higher = better, not bounded to [0,1]).
        """
        import torch

        inputs = self._tokenizer.apply_chat_template(
            conversations,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=4096,
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits.squeeze(-1)

        return logits.cpu().tolist()

    def score_single(self, history: list[dict], response: str) -> float:
        """Score a single response. Returns a raw logit (not normalized)."""
        conv = history + [{"role": "assistant", "content": response}]
        scores = self.score_batch([conv])
        return scores[0]

    def normalize(self, scores: list[float], lo: float = -5.0, hi: float = 5.0) -> list[float]:
        """Clip raw logits to [lo, hi] and rescale to [0, 1]."""
        return [max(0.0, min(1.0, (s - lo) / (hi - lo))) for s in scores]


# ── RewardPlugin wrapper (used by background reward worker) ───────────────────

class SkyworkRewardPlugin(RewardPlugin):
    """
    Calls a running Skywork-V2 reward server for online scoring.
    Disabled when SKYWORK_REWARD_URL is not set — gracefully returns neutral.

    Enable by setting SKYWORK_REWARD_URL to the ngrok/cloudflare URL of
    a Kaggle notebook running rewards/models/server.py.
    """
    name = "skywork_v2"
    weight = 0.40   # highest weight — best RM-Bench correlation

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=15.0)

    @property
    def enabled(self) -> bool:
        return bool(_REWARD_URL)

    async def score(self, prompt: str, response: str, history: list[dict]) -> RewardResult:
        if not _REWARD_URL:
            return RewardResult(score=0.5, details={"status": "no_server"})

        conv = history + [{"role": "assistant", "content": response}]
        try:
            resp = await self._client.post(
                f"{_REWARD_URL.rstrip('/')}/score",
                json={"conversation": conv},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            raw_score = data.get("score", 0.0)   # server returns [0, 1]
            return RewardResult(
                score=max(0.0, min(1.0, raw_score)),
                details={"raw_logit": data.get("raw_logit"), "model": MODEL_ID},
            )
        except Exception as e:
            logger.warning(f"Skywork-V2 server call failed: {e}")
            return RewardResult(score=0.5, details={"error": str(e)})


register(SkyworkRewardPlugin())
