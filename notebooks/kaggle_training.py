# Aethr — Kaggle Training Notebook (Notebook B)
# Phase 4+ only — run this after you have ≥64 scored conversations in Supabase.
#
# This notebook:
#   1. Pulls scored training examples from Supabase
#   2. Runs GRPO training on them using Unsloth + TRL
#   3. Pushes the resulting LoRA adapter to HF Hub
#   4. Inference Notebook A will pick it up on next health check
#
# Run manually when you have enough data, or schedule it as a Kaggle Scheduled Notebook.

# ── Cell 1: Install ────────────────────────────────────────────────────────────
# In[ ]:

import subprocess
subprocess.run([
    "pip", "install", "-q",
    "unsloth",
    "trl",
    "peft",
    "wandb",
    "asyncpg",
    "sqlalchemy[asyncio]",
    "huggingface_hub",
    "python-dotenv",
], check=True)
print("✓ Packages installed")

# ── Cell 2: Configuration ─────────────────────────────────────────────────────
# In[ ]:

import os

HF_TOKEN        = os.environ.get("HF_TOKEN", "hf_YOUR_TOKEN_HERE")
HF_ADAPTER_REPO = os.environ.get("HF_ADAPTER_REPO", "your-hf-username/aethr-adapters")
DATABASE_URL    = os.environ.get("DATABASE_URL", "postgresql+asyncpg://...")
WANDB_API_KEY   = os.environ.get("WANDB_API_KEY", "your-wandb-key")
WANDB_PROJECT   = "aethr"

BASE_MODEL   = "unsloth/Qwen3-8B-bnb-4bit"
LORA_R       = 32
LORA_ALPHA   = 64
LR           = 1e-6
GROUP_SIZE   = 8    # completions per prompt — the GRPO "group"
BUFFER_MIN   = 64   # minimum examples before training

os.environ["WANDB_API_KEY"] = WANDB_API_KEY
os.environ["HF_TOKEN"] = HF_TOKEN

import wandb
wandb.init(project=WANDB_PROJECT, name="grpo-offline-run")

# ── Cell 3: Pull training examples from Supabase ──────────────────────────────
# In[ ]:
# Gets assistant messages with composite_score, formats them for GRPO.
# GRPO needs: {prompt: str, completions: [{text: str, reward: float}]}
# For offline training we generate multiple completions per original prompt.

import asyncio
import asyncpg
import json

async def fetch_scored_messages(db_url: str, limit: int = 200):
    """Pull the top-scored conversation turns for training."""
    # Strip the asyncpg driver prefix for raw asyncpg
    raw_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(raw_url)

    rows = await conn.fetch("""
        SELECT
            m.id,
            m.content AS response,
            m.conversation_id,
            rs.composite_score,
            rs.rule_based_scores,
            rs.llm_judge_scores
        FROM messages m
        JOIN reward_scores rs ON rs.message_id = m.id
        WHERE m.role = 'assistant'
          AND rs.composite_score IS NOT NULL
        ORDER BY m.created_at DESC
        LIMIT $1
    """, limit)

    await conn.close()
    return rows


async def fetch_conversation_context(db_url: str, conversation_id: str, before_msg_id: str):
    """Get the conversation history up to a given message."""
    raw_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(raw_url)

    rows = await conn.fetch("""
        SELECT role, content
        FROM messages
        WHERE conversation_id = $1
          AND created_at < (SELECT created_at FROM messages WHERE id = $2)
        ORDER BY created_at ASC
        LIMIT 20
    """, conversation_id, before_msg_id)

    await conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


# Fetch data
scored = asyncio.run(fetch_scored_messages(DATABASE_URL, limit=BUFFER_MIN * 2))
print(f"✓ Fetched {len(scored)} scored messages from Supabase")

if len(scored) < BUFFER_MIN:
    raise ValueError(
        f"Need at least {BUFFER_MIN} scored messages to train. "
        f"Have {len(scored)}. Keep using the bot!"
    )

# ── Cell 4: Load model ────────────────────────────────────────────────────────
# In[ ]:

import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=8192,
    load_in_4bit=True,
    fast_inference=True,
    gpu_memory_utilization=0.7,   # higher than inference notebook — we need training memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.0,    # no dropout for RL
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)

print(f"✓ Model loaded with LoRA adapters attached")

# ── Cell 5: Build GRPO dataset ────────────────────────────────────────────────
# In[ ]:
# For each original conversation turn, we generate GROUP_SIZE new completions
# and score them. GRPO learns from the relative ranking within each group.
#
# Why generate new completions instead of using the originals?
# The original response is one sample. GRPO needs variance within a group to
# compute advantages. A group of identical responses → zero gradient.

from inference.prompt_templates import format_for_training

async def build_grpo_dataset(scored_rows, temperature=1.0):
    dataset = []
    for row in scored_rows[:BUFFER_MIN]:
        ctx = await fetch_conversation_context(DATABASE_URL, str(row["conversation_id"]), str(row["id"]))

        # Format the prompt in Qwen3's ChatML format
        prompt = format_for_training(ctx)

        # Generate GROUP_SIZE diverse completions at high temperature
        completions = []
        for _ in range(GROUP_SIZE):
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=temperature,
                    do_sample=True,
                )
            text = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            completions.append({"text": text, "reward": None})   # rewards computed by reward_fn

        dataset.append({"prompt": prompt, "completions": completions})

    return dataset


print("Building GRPO dataset (this generates completions — may take a while)...")
dataset = asyncio.run(build_grpo_dataset(scored))
print(f"✓ Dataset: {len(dataset)} prompts × {GROUP_SIZE} completions = {len(dataset)*GROUP_SIZE} total")

# ── Cell 6: Define reward function for GRPOTrainer ────────────────────────────
# In[ ]:
# TRL's GRPOTrainer calls this function on every batch of completions.
# Using rule-based rewards here for speed — LLM judge would be too slow inline.
# The production setup uses pre-scored examples from Supabase instead.

import re

def reward_fn(completions: list[str], prompts: list[str]) -> list[float]:
    """
    Score completions for GRPO training.
    Returns a float reward for each completion.

    For Vithos: medical guardrails are part of this scoring.
    Any violation gets a large negative reward, training the model away from it.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        score = 0.5  # start neutral

        # Length appropriateness
        n_words = len(completion.split())
        if 20 <= n_words <= 300:
            score += 0.15
        elif n_words < 10:
            score -= 0.20

        # Medical guardrails — hard penalties
        alarm_patterns = [
            r"\b(?:DANGER|CRITICAL|URGENT|EMERGENCY)\b",
            r"\bcritically (?:elevated|low|high)\b",
            r"\blife.?threatening\b",
        ]
        for pat in alarm_patterns:
            if re.search(pat, completion, re.IGNORECASE):
                score -= 0.60
                break

        diagnosis_patterns = [
            r"\byou (?:have|are suffering from|are diagnosed with)\b",
            r"\bthis (?:indicates?|confirms?) (?:you have)?\b",
        ]
        for pat in diagnosis_patterns:
            if re.search(pat, completion, re.IGNORECASE):
                score -= 1.00
                break

        treatment_patterns = [
            r"\b(?:take|start|stop|discontinue) (?:your )?(?:medication|medicine|drug)\b",
            r"\byou should (?:take|start|stop)\b",
        ]
        for pat in treatment_patterns:
            if re.search(pat, completion, re.IGNORECASE):
                score -= 1.00
                break

        # Positive: calm contextualizing language
        good_phrases = ["worth mentioning", "trending", "across your last", "your doctor"]
        for phrase in good_phrases:
            if phrase in completion.lower():
                score += 0.10
                break

        # Positive: human-sounding disclaimer
        if "not a diagnosis" in completion.lower() or "your doctor has the final word" in completion.lower():
            score += 0.15

        rewards.append(float(max(-1.0, min(1.0, score))))

    return rewards

print(f"✓ Reward function defined")
print(f"  Sample test: {reward_fn(['Your LDL is 142 — worth mentioning at your next checkup.'], ['What is my LDL?'])}")

# ── Cell 7: GRPO Training ─────────────────────────────────────────────────────
# In[ ]:
# Key hyperparameters and why:
#   lr=1e-6: RL needs much lower LR than SFT — higher causes instability
#   kl_coef=0.001: keeps model anchored to base weights; too high=no learning
#   num_generations=GROUP_SIZE: completions per prompt (the "group" in GRPO)
#   temperature=1.0: diversity in the group is essential — low temp → zero gradient

from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

# Convert to HF Dataset format
hf_dataset = Dataset.from_list([
    {"prompt": ex["prompt"]}
    for ex in dataset
])

config = GRPOConfig(
    output_dir="/kaggle/working/checkpoints",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,    # effective batch = 16 prompts
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=5,
    save_steps=50,
    save_total_limit=3,
    num_generations=GROUP_SIZE,
    max_prompt_length=1024,
    max_completion_length=512,
    temperature=1.0,
    kl_coef=0.001,
    loss_agg_mode="token-mean",   # Dr.GRPO fix: removes length bias
    report_to="wandb",
)

trainer = GRPOTrainer(
    model=model,
    config=config,
    reward_funcs=reward_fn,
    train_dataset=hf_dataset,
    tokenizer=tokenizer,
)

print("Starting GRPO training...")
trainer.train()
print("✓ Training complete")

# ── Cell 8: Push adapter to HF Hub ───────────────────────────────────────────
# In[ ]:
# Save LoRA weights and push to HF Hub.
# Inference Notebook A will pull this on its next restart or on-demand.

ADAPTER_OUTPUT = "/kaggle/working/final_adapter"
model.save_pretrained(ADAPTER_OUTPUT)
tokenizer.save_pretrained(ADAPTER_OUTPUT)
print(f"✓ Adapter saved locally: {ADAPTER_OUTPUT}")

from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)

# Get current training step for the commit message
current_step = trainer.state.global_step

result = api.upload_folder(
    folder_path=ADAPTER_OUTPUT,
    repo_id=HF_ADAPTER_REPO,
    repo_type="model",
    commit_message=f"adapter: step {current_step}",
)
print(f"✓ Adapter pushed to {HF_ADAPTER_REPO}")
print(f"  Revision: {result.oid}")
print(f"\nSend this to Inference Notebook A to hot-swap:")
print(f"  Load adapter revision: {result.oid[:8]}")

wandb.finish()
