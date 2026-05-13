# Aethr — Kaggle Inference Notebook (Notebook A)
# Run this in a Kaggle notebook with GPU accelerator enabled.
# RTX PRO 6000 (96GB) recommended. T4 x2 also works with 4-bit model.
#
# This notebook:
#   1. Installs dependencies
#   2. Loads Qwen3-8B via Unsloth (4-bit quantized — fits any Kaggle GPU)
#   3. Starts a FastAPI server (OpenAI-compatible /v1/chat/completions)
#   4. Exposes it via ngrok — copy the printed URL into your bot's /seturl command
#   5. Loads the latest LoRA adapter from HF Hub (if one exists)
#
# Sessions last up to 12 hours. When the session ends, restart and run all cells again.
# The bot handles the URL rotation via /seturl.

# ── Cell 1: Install ────────────────────────────────────────────────────────────
# In[ ]:

import subprocess
subprocess.run([
    "pip", "install", "-q",
    "unsloth",
    "vllm",
    "fastapi",
    "uvicorn[standard]",
    "pyngrok",
    "huggingface_hub",
    "python-dotenv",
], check=True)
print("✓ Packages installed")

# ── Cell 2: Configuration ─────────────────────────────────────────────────────
# In[ ]:

import os

# Paste your secrets here OR add them as Kaggle secrets (recommended)
HF_TOKEN         = os.environ.get("HF_TOKEN", "hf_YOUR_TOKEN_HERE")
HF_ADAPTER_REPO  = os.environ.get("HF_ADAPTER_REPO", "your-hf-username/aethr-adapters")
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "your-ngrok-token")
BASE_MODEL       = "unsloth/Qwen3-8B-bnb-4bit"   # 4-bit: ~8GB VRAM, works on any Kaggle GPU

os.environ["HF_TOKEN"] = HF_TOKEN

# ── Cell 3: Load model ────────────────────────────────────────────────────────
# In[ ]:
# What's happening here:
#   - Unsloth loads the model in 4-bit (bnb) — cuts VRAM from 16GB to ~8GB
#   - fast_inference=True enables the vLLM backend for throughput
#   - get_peft_model attaches LoRA adapter slots — needed for training hot-swap later
#   - gpu_memory_utilization=0.6 leaves headroom; raise to 0.8 if not training on this GPU

import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"   # share weights between vLLM and training

from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=8192,
    load_in_4bit=True,
    fast_inference=True,        # enables vLLM backend
    gpu_memory_utilization=0.6,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)

print(f"✓ Model loaded: {BASE_MODEL}")
print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB allocated")

# ── Cell 4: Load latest adapter from HF Hub (if available) ───────────────────
# In[ ]:
# On first run there won't be an adapter — that's fine, model runs on base weights.
# After Phase 4 training, this will load the latest improved adapter automatically.

from huggingface_hub import snapshot_download, list_repo_commits
from peft import PeftModel

ADAPTER_DIR = "/kaggle/working/active_adapter"

try:
    commits = list(list_repo_commits(HF_ADAPTER_REPO, repo_type="model", token=HF_TOKEN))
    if commits:
        snapshot_download(
            repo_id=HF_ADAPTER_REPO,
            local_dir=ADAPTER_DIR,
            token=HF_TOKEN,
            repo_type="model",
        )
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
        print(f"✓ Loaded adapter from {HF_ADAPTER_REPO} (commit: {commits[0].commit_id[:8]})")
    else:
        print("ℹ No adapter found — running on base model weights")
except Exception as e:
    print(f"ℹ Adapter load skipped: {e}")
    print("  Running on base model weights")

# ── Cell 5: FastAPI server ────────────────────────────────────────────────────
# In[ ]:
# OpenAI-compatible API so your local bot code stays unchanged.
# The /health endpoint prevents Kaggle's 20-min idle timeout when pinged regularly.

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
import threading
import json
import asyncio

app = FastAPI(title="Aethr Inference Server")

ACTIVE_ADAPTER_REVISION = commits[0].commit_id[:8] if 'commits' in dir() and commits else "base"


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": BASE_MODEL,
        "adapter": ACTIVE_ADAPTER_REVISION,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages  = body.get("messages", [])
    max_tokens = body.get("max_tokens", 2048)
    temperature = body.get("temperature", 0.7)
    stream = body.get("stream", False)

    # Apply Qwen3 chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if stream:
        async def _stream_tokens():
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            # vLLM streaming — yields tokens as they're generated
            for token_id in model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                streamer=None,     # TODO: add TextIteratorStreamer for true streaming
            )[0][inputs["input_ids"].shape[1]:]:
                token = tokenizer.decode([token_id], skip_special_tokens=True)
                chunk = {"choices": [{"delta": {"content": token}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_stream_tokens(), media_type="text/event-stream")

    # Non-streaming
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
    response_text = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return {
        "id": "aethr-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
        "model": BASE_MODEL,
    }


# Run server in background thread (non-blocking so notebook cells can continue)
def _run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

server_thread = threading.Thread(target=_run_server, daemon=True)
server_thread.start()

import time; time.sleep(2)   # wait for server to start
print("✓ Server running on port 8000")

# ── Cell 6: Expose via ngrok ──────────────────────────────────────────────────
# In[ ]:
# ngrok creates a public HTTPS tunnel to port 8000.
# The URL changes every session — copy it and send /seturl <url> to your bot.

from pyngrok import ngrok, conf

conf.get_default().auth_token = NGROK_AUTH_TOKEN
tunnel = ngrok.connect(8000, "http")
public_url = tunnel.public_url

print("=" * 60)
print(f"  PUBLIC URL: {public_url}")
print("=" * 60)
print(f"\nSend this to your bot:\n  /seturl {public_url}")
print("\nKeep this notebook running. Session lasts up to 12 hours.")
print("Ping /health to prevent 20-min idle timeout.")

# ── Cell 7: Health check ──────────────────────────────────────────────────────
# In[ ]:
# Run this to verify everything is working before connecting the bot.

import httpx

resp = httpx.get(f"{public_url}/health")
print(f"Health: {resp.json()}")

test_resp = httpx.post(
    f"{public_url}/v1/chat/completions",
    json={
        "messages": [
            {"role": "system", "content": "You are Aethr, a helpful assistant."},
            {"role": "user", "content": "Say hello in one sentence."},
        ],
        "max_tokens": 50,
        "temperature": 0.7,
    },
    timeout=60,
)
print(f"\nTest response: {test_resp.json()['choices'][0]['message']['content']}")
print("\n✓ Inference server is ready. Send /seturl to your bot.")

# ── Cell 8: Keep-alive (run in a separate cell, leave running) ────────────────
# In[ ]:
# Prevents Kaggle's idle timeout from killing the session.
# Run this cell and leave it — it pings /health every 10 minutes.

import time
while True:
    try:
        r = httpx.get(f"{public_url}/health", timeout=5)
        print(f"[keep-alive] {time.strftime('%H:%M:%S')} — {r.json()['status']}")
    except Exception as e:
        print(f"[keep-alive] ping failed: {e}")
    time.sleep(600)   # 10 minutes
