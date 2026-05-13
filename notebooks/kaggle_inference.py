# Aethr — Inference Notebook (Colab / Kaggle T4)
#
# Works on: Colab T4 (15GB), Kaggle T4 x2, Kaggle P100 (16GB)
# GPU budget: 4-bit Qwen3-8B ≈ 8GB loaded, ~6GB left for KV cache + generation
#
# IMPORTANT: Runtime → Restart runtime before running if you hit OOM.
# Each cell is independent — restart clears all GPU memory.

# ── Cell 1: Install ────────────────────────────────────────────────────────────
# In[ ]:

import subprocess
subprocess.run([
    "pip", "install", "-q",
    "unsloth",
    "fastapi",
    "uvicorn[standard]",
    "pyngrok",
    "huggingface_hub",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "httpx",
], check=True)
print("✓ Packages installed")

# ── Cell 2: Configuration ─────────────────────────────────────────────────────
# In[ ]:

import os

HF_TOKEN         = os.environ.get("HF_TOKEN",         "YOUR_HF_TOKEN")
HF_ADAPTER_REPO  = os.environ.get("HF_ADAPTER_REPO",  "your-hf-username/aethr-adapters")
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN",  "YOUR_NGROK_TOKEN")
BASE_MODEL       = "unsloth/Qwen3-8B-bnb-4bit"

os.environ["HF_TOKEN"] = HF_TOKEN
print(f"✓ Config loaded — model: {BASE_MODEL}")

# ── Cell 3: Load model ────────────────────────────────────────────────────────
# In[ ]:
# NO vLLM (fast_inference=False) — T4 has 15GB; after 4-bit model loads (~8GB)
# there is not enough headroom for vLLM's KV cache pre-allocation.
# Plain HuggingFace generate() is sufficient for single-user Telegram load.
# NO standby mode — that's only needed when training and inferring on the same GPU.

import torch
from unsloth import FastLanguageModel

# Clear any leftover GPU memory from previous runs
torch.cuda.empty_cache()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=4096,
    load_in_4bit=True,
    fast_inference=False,   # vLLM disabled — not enough VRAM on T4
)

# Attach LoRA slots so we can hot-swap adapters later (Phase 4+)
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing=False,  # inference only — no need for this
)
FastLanguageModel.for_inference(model)  # puts model in fast eval mode

print(f"✓ Model loaded: {BASE_MODEL}")
print(f"  GPU memory used: {torch.cuda.memory_allocated()/1e9:.1f} GB / "
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── Cell 4: Load latest LoRA adapter from HF Hub (optional) ──────────────────
# In[ ]:
# First run: no adapter exists yet — runs on base weights, that's fine.
# After Phase 4 training: adapter is pulled and loaded automatically.

from huggingface_hub import list_repo_commits, snapshot_download

ADAPTER_DIR = "/content/active_adapter"
ACTIVE_ADAPTER = "base"

try:
    commits = list(list_repo_commits(HF_ADAPTER_REPO, repo_type="model", token=HF_TOKEN))
    if commits:
        snapshot_download(
            repo_id=HF_ADAPTER_REPO,
            local_dir=ADAPTER_DIR,
            token=HF_TOKEN,
            repo_type="model",
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
        ACTIVE_ADAPTER = commits[0].commit_id[:8]
        print(f"✓ Adapter loaded from {HF_ADAPTER_REPO} @ {ACTIVE_ADAPTER}")
    else:
        print("ℹ No adapter in HF Hub yet — running on base weights")
except Exception as e:
    print(f"ℹ Adapter load skipped ({e}) — running on base weights")

# ── Cell 5: FastAPI inference server ─────────────────────────────────────────
# In[ ]:

import json
import threading
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import TextIteratorStreamer
import uvicorn

app = FastAPI(title="Aethr Inference Server")


@app.get("/health")
async def health():
    return {"status": "ok", "model": BASE_MODEL, "adapter": ACTIVE_ADAPTER}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body        = await request.json()
    messages    = body.get("messages", [])
    max_tokens  = body.get("max_tokens", 512)
    temperature = body.get("temperature", 0.7)
    do_stream   = body.get("stream", False)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )

    if do_stream:
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs["streamer"] = streamer

        # Run generation in a background thread so we can stream tokens
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        async def _token_stream():
            for token in streamer:
                chunk = {"choices": [{"delta": {"content": token}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_token_stream(), media_type="text/event-stream")

    # Non-streaming
    with torch.no_grad():
        output = model.generate(**gen_kwargs)
    response_text = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return {
        "id": "aethr-1",
        "object": "chat.completion",
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": response_text},
                     "finish_reason": "stop"}],
        "model": BASE_MODEL,
    }


def _run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

threading.Thread(target=_run_server, daemon=True).start()

import time; time.sleep(2)
print("✓ Server running on port 8000")

# ── Cell 6: Expose via ngrok ──────────────────────────────────────────────────
# In[ ]:

from pyngrok import ngrok, conf

conf.get_default().auth_token = NGROK_AUTH_TOKEN
ngrok.kill()  # kill any leftover tunnels from previous runs
tunnel = ngrok.connect(8000, "http")
public_url = tunnel.public_url

print("=" * 60)
print(f"  PUBLIC URL: {public_url}")
print("=" * 60)
print(f"\nSend to your bot:  /seturl {public_url}")

# ── Cell 7: Verify ────────────────────────────────────────────────────────────
# In[ ]:

import httpx, time

resp = httpx.get(f"{public_url}/health", timeout=10)
print(f"Health: {resp.json()}")

test = httpx.post(
    f"{public_url}/v1/chat/completions",
    json={
        "messages": [
            {"role": "system", "content": "You are Aethr, a helpful assistant."},
            {"role": "user",   "content": "Say hello in one sentence."},
        ],
        "max_tokens": 60,
        "temperature": 0.7,
    },
    timeout=120,
)
print(f"\nTest response: {test.json()['choices'][0]['message']['content']}")
print("\n✓ Ready. Send /seturl to your Telegram bot.")

# ── Cell 8: Keep-alive (leave this running) ───────────────────────────────────
# In[ ]:

import time, httpx
while True:
    try:
        r = httpx.get(f"{public_url}/health", timeout=5)
        print(f"[keep-alive] {time.strftime('%H:%M:%S')} — {r.json()['status']}")
    except Exception as e:
        print(f"[keep-alive] ping failed: {e}")
    time.sleep(600)
