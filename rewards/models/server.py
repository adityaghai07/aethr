"""
Lightweight FastAPI server that exposes Skywork-V2 or REWARDANYTHING via HTTP.

Run this on the Kaggle/A100 training notebook to enable remote reward scoring
by the local bot reward worker (which can't load 8B models itself).

Usage (in Kaggle cell):
    import subprocess, threading
    subprocess.Popen(["python", "-m", "rewards.models.server", "--model", "skywork"])
    # then tunnel with ngrok and set SKYWORK_REWARD_URL=http://...

CLI:
    python -m rewards.models.server --model skywork --port 7860
    python -m rewards.models.server --model rewardanything --port 7861
"""
from __future__ import annotations
import argparse
import logging
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)
app = FastAPI(title="Aethr Reward Model Server")

_model = None
_model_type = "skywork"


class ScoreRequest(BaseModel):
    conversation: list[dict]        # [{role, content}, ...] including assistant turn
    response: str = ""              # assistant response (if not the last conv turn)
    rubric: str | None = None       # optional rubric text (REWARDANYTHING only)


class ScoreResponse(BaseModel):
    score: float                    # [0, 1]
    raw_logit: float | None = None
    reasoning: str = ""


@app.get("/health")
async def health():
    return {"status": "ok", "model": _model_type, "loaded": _model is not None}


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    if _model is None:
        return ScoreResponse(score=0.5, reasoning="model_not_loaded")

    # Build the conversation to score
    conv = req.conversation
    if req.response and (not conv or conv[-1]["role"] != "assistant"):
        conv = conv + [{"role": "assistant", "content": req.response}]

    if _model_type == "skywork":
        raw_logits = _model.score_batch([conv])
        raw = raw_logits[0]
        normalized = _model.normalize([raw])[0]
        return ScoreResponse(score=normalized, raw_logit=raw)

    else:  # rewardanything
        from rewards.rubric import RewardRubric
        rubric = None
        if req.rubric:
            # Treat as pre-compiled prompt text (already formatted)
            rubric_obj = RewardRubric(name="custom", criteria=[req.rubric])
            rubric = rubric_obj
        history = [m for m in conv if m["role"] != "assistant"]
        response = conv[-1]["content"] if conv and conv[-1]["role"] == "assistant" else req.response
        s = _model.score_with_rubric(history, response, rubric)
        return ScoreResponse(score=s)


def _load_model(model_type: str):
    global _model, _model_type
    _model_type = model_type
    if model_type == "skywork":
        from rewards.models.skywork_v2 import SkyworkRewardModel
        _model = SkyworkRewardModel.load_local()
    elif model_type == "rewardanything":
        from rewards.models.rewardanything import RewardAnythingModel
        _model = RewardAnythingModel.load_local()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'skywork' or 'rewardanything'.")
    logger.info(f"Reward model loaded: {model_type}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="skywork", choices=["skywork", "rewardanything"])
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    _load_model(args.model)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
