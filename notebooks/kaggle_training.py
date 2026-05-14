# Aethr — GRPO Training Notebook
# Runs on Colab free tier (T4, 15GB) using Unsloth's GRPO recipe.
#
# Workflow:
#   1. Clone the aethr repo (so we can reuse the reward plugins)
#   2. Pull scored conversations from Supabase → format as GRPO dataset
#   3. Load Qwen3-8B-bnb-4bit with Unsloth (T4-safe config)
#   4. Run GRPO using our composite reward stack (rule_based + medical + judge)
#   5. Push the trained LoRA adapter to HuggingFace Hub
#   6. Inference notebook picks it up on next restart
#
# Run this when you have ≥64 scored conversations in Supabase.

# ── Cell 1: Install + clone repo ───────────────────────────────────────────────
# In[ ]:

import subprocess
subprocess.run([
    "pip", "install", "-q", "--upgrade",
    "unsloth",
    "vllm",
    "trl",
    "peft",
    "wandb",
    "asyncpg",
    "sqlalchemy[asyncio]",
    "huggingface_hub",
    "python-dotenv",
    "anthropic",
], check=True)

# Clone the aethr codebase so we can reuse the reward plugins
import os
if not os.path.exists("/content/aethr"):
    subprocess.run(["git", "clone", "https://github.com/adityaghai07/aethr", "/content/aethr"], check=True)

os.chdir("/content/aethr")
import sys
sys.path.insert(0, "/content/aethr")
print("✓ Packages installed, aethr repo cloned")

# ── Cell 2: Configuration ─────────────────────────────────────────────────────
# In[ ]:

# Paste your secrets here (or set them as Colab secrets and they'll be picked up)
HF_TOKEN        = os.environ.get("HF_TOKEN",        "YOUR_HF_TOKEN")
HF_ADAPTER_REPO = os.environ.get("HF_ADAPTER_REPO", "your-username/aethr-adapters")
DATABASE_URL    = os.environ.get("DATABASE_URL",    "postgresql+asyncpg://...")
WANDB_API_KEY   = os.environ.get("WANDB_API_KEY",   "your-wandb-key")
JUDGE_API_KEY   = os.environ.get("JUDGE_API_KEY",   "sk-ant-...")

# Export so the aethr config module picks them up
for k, v in {
    "HF_TOKEN": HF_TOKEN, "HF_ADAPTER_REPO": HF_ADAPTER_REPO,
    "DATABASE_URL": DATABASE_URL, "WANDB_API_KEY": WANDB_API_KEY,
    "JUDGE_API_KEY": JUDGE_API_KEY,
}.items():
    os.environ[k] = v

# ── T4-safe hyperparameters (from Unsloth's GRPO recipe) ──────────────────────
BASE_MODEL          = "unsloth/Qwen3-8B-bnb-4bit"
LORA_R              = 16     # smaller than the 32 we'd use on bigger GPUs
LORA_ALPHA          = 16     # 1x scaling for stability on T4
LR                  = 5e-6   # slightly higher than 1e-6 to compensate for smaller batch
GROUP_SIZE          = 6      # completions per prompt — Unsloth's T4 sweet spot
MAX_PROMPT_LENGTH   = 512
MAX_COMPLETION_LEN  = 512
PER_DEVICE_BATCH    = 1
GRAD_ACCUM          = 1      # effective batch = 1 prompt × 6 completions per step
BUFFER_MIN          = 32     # train as soon as we have this many scored examples

import wandb
wandb.login(key=WANDB_API_KEY)
wandb.init(project="aethr", name="grpo-colab-t4", config={
    "model": BASE_MODEL, "lora_r": LORA_R, "lr": LR,
    "group_size": GROUP_SIZE, "max_completion": MAX_COMPLETION_LEN,
})
print("✓ Config loaded")

# ── Cell 3: Pull scored data from Supabase ────────────────────────────────────
# In[ ]:

import asyncio
import asyncpg

async def fetch_training_pairs(db_url: str, limit: int = 200):
    raw_url = db_url.replace("postgresql+asyncpg://", "postgresql://").split("?")[0]
    # statement_cache_size=0 → compatible with Supabase's pgbouncer
    conn = await asyncpg.connect(raw_url, statement_cache_size=0)

    rows = await conn.fetch("""
        SELECT
            m.id           AS msg_id,
            m.conversation_id,
            m.created_at,
            rs.composite_score
        FROM messages m
        JOIN reward_scores rs ON rs.message_id = m.id
        WHERE m.role = 'assistant'
          AND rs.composite_score IS NOT NULL
        ORDER BY m.created_at DESC
        LIMIT $1
    """, limit)

    examples = []
    for r in rows:
        history = await conn.fetch("""
            SELECT role, content FROM messages
            WHERE conversation_id = $1 AND created_at < $2
            ORDER BY created_at ASC LIMIT 20
        """, r["conversation_id"], r["created_at"])
        examples.append({
            "msg_id": str(r["msg_id"]),
            "history": [{"role": h["role"], "content": h["content"]} for h in history],
            "original_reward": r["composite_score"],
        })

    await conn.close()
    return examples


examples = asyncio.run(fetch_training_pairs(DATABASE_URL, limit=200))
print(f"✓ Fetched {len(examples)} scored conversations")

if len(examples) < BUFFER_MIN:
    raise ValueError(
        f"Need ≥{BUFFER_MIN} scored examples to train. Have {len(examples)}. "
        f"Keep using the bot until the buffer fills."
    )

# ── Cell 4: Load model ────────────────────────────────────────────────────────
# In[ ]:
# T4 GRPO trick: fast_inference=True enables vLLM rollout backend during
# training. Unsloth handles the weight sharing so we don't double-load.
# gpu_memory_utilization=0.5 leaves room for training activations.

import torch
import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"   # share weights between vLLM rollout and training

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_PROMPT_LENGTH + MAX_COMPLETION_LEN,
    load_in_4bit=True,
    fast_inference=True,             # vLLM for rollouts
    max_lora_rank=LORA_R,
    gpu_memory_utilization=0.5,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.0,                # no dropout for RL
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
print(f"✓ Model loaded — GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# ── Cell 5: Build the GRPO prompt dataset ─────────────────────────────────────
# In[ ]:
# Each training example is a single prompt. GRPO's trainer generates GROUP_SIZE
# completions on-the-fly during training, scores them with reward_fn (Cell 6),
# and computes group-relative advantages.

from datasets import Dataset
from inference.prompt_templates import format_for_training

def to_prompt(example):
    # Append a synthetic "user message" if history ends with assistant — shouldn't happen
    # but safe-guard. Our examples are always "history ending at user message → assistant response"
    return {"prompt": format_for_training(example["history"])}

dataset_rows = [to_prompt(ex) for ex in examples]
train_dataset = Dataset.from_list(dataset_rows)
print(f"✓ Dataset built: {len(train_dataset)} prompts")
print(f"  Sample prompt (first 200 chars):\n{train_dataset[0]['prompt'][:200]}")

# ── Cell 6: Reward function — reuses aethr's plugin stack ─────────────────────
# In[ ]:
# We call the SAME reward plugins that score messages in production.
# This guarantees the model is trained on the same reward signal it's evaluated on.

from config import get_active_plugins
from rewards.composite import compute_composite

active_plugins = get_active_plugins()
print(f"Active reward plugins: {[p.name for p in active_plugins]}")

def reward_fn(prompts, completions, **kwargs):
    """
    TRL's GRPOTrainer calls this with a list of prompts and corresponding completions.
    We score each (prompt, completion) pair and return a list of floats.
    """
    # GRPOTrainer passes completions as list of dicts in newer versions; normalize
    flat_completions = []
    for c in completions:
        if isinstance(c, list) and c and isinstance(c[0], dict):
            flat_completions.append(c[0].get("content", ""))
        else:
            flat_completions.append(c)

    async def _score_all():
        tasks = [
            compute_composite(prompt=p, response=c, history=[], plugins=active_plugins)
            for p, c in zip(prompts, flat_completions)
        ]
        results = await asyncio.gather(*tasks)
        return [r[0] for r in results]   # (score, details, violated) → just the score

    return asyncio.run(_score_all())


# Sanity-check: score one example
test_score = reward_fn(
    prompts=["What is your LDL?"],
    completions=["Your LDL is 142 — worth mentioning at your next checkup."],
)
print(f"Test reward (should be > 0): {test_score[0]:.3f}")

# ── Cell 7: GRPO training ─────────────────────────────────────────────────────
# In[ ]:
# Key knobs documented inline:
#   lr 5e-6:               higher than 1e-6 because batch is small (1 prompt per step)
#   kl_coef 0.001:         keeps the model anchored to base — too high = no learning
#   num_generations 6:     completions per prompt (group_size). Diversity drives gradient
#   temperature 1.0:       essential — low temp = all completions identical = zero gradient
#   loss_agg_mode token-mean: Dr.GRPO fix — removes length bias
#   epsilon 0.2:           PPO clip — prevents catastrophic policy updates

from trl import GRPOTrainer, GRPOConfig

training_args = GRPOConfig(
    output_dir              = "/content/grpo_output",
    learning_rate           = LR,
    lr_scheduler_type       = "cosine",
    warmup_ratio            = 0.05,
    per_device_train_batch_size = PER_DEVICE_BATCH,
    gradient_accumulation_steps = GRAD_ACCUM,
    num_generations         = GROUP_SIZE,
    max_prompt_length       = MAX_PROMPT_LENGTH,
    max_completion_length   = MAX_COMPLETION_LEN,
    temperature             = 1.0,
    beta                    = 0.001,         # KL coefficient (TRL's name for kl_coef)
    epsilon                 = 0.2,           # PPO clipping
    loss_type               = "dr_grpo",     # length-bias-free Dr.GRPO normalization
    num_train_epochs        = 1,
    logging_steps           = 1,
    save_steps              = 50,
    save_total_limit        = 2,
    bf16                    = False,         # T4 has no bf16 — use fp16
    fp16                    = True,
    report_to               = "wandb",
)

trainer = GRPOTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    reward_funcs    = [reward_fn],
    processing_class = tokenizer,
)

print("Starting GRPO training...")
trainer.train()
print("✓ Training complete")

# ── Cell 8: Push adapter to HF Hub ────────────────────────────────────────────
# In[ ]:

from huggingface_hub import HfApi

ADAPTER_OUT = "/content/grpo_output/final_adapter"
model.save_pretrained(ADAPTER_OUT)
tokenizer.save_pretrained(ADAPTER_OUT)

api = HfApi(token=HF_TOKEN)
result = api.upload_folder(
    folder_path  = ADAPTER_OUT,
    repo_id      = HF_ADAPTER_REPO,
    repo_type    = "model",
    commit_message = f"adapter: step {trainer.state.global_step}",
)
print(f"✓ Adapter pushed to {HF_ADAPTER_REPO}")
print(f"  Revision: {result.oid}")
print(f"\nTo activate in inference notebook:")
print(f"  Restart the Colab inference notebook — it auto-pulls the latest adapter.")

# ── Cell 9: Register checkpoint in Supabase (closes the loop) ─────────────────
# In[ ]:

import asyncio
from db.connection import get_db
from db.models import Checkpoint

async def register():
    async with get_db() as session:
        ckpt = Checkpoint(
            step           = trainer.state.global_step,
            hf_repo        = HF_ADAPTER_REPO,
            hf_revision    = result.oid,
            base_model     = BASE_MODEL,
            eval_scores    = {},        # populated by eval suite in Phase 6
            is_active      = False,     # set to True after eval-gating
            training_config = {
                "lr": LR, "lora_r": LORA_R, "group_size": GROUP_SIZE,
                "max_completion": MAX_COMPLETION_LEN, "loss_type": "dr_grpo",
                "n_examples": len(examples),
            },
        )
        session.add(ckpt)
        await session.commit()
        await session.refresh(ckpt)
        return ckpt

ckpt = asyncio.run(register())
print(f"✓ Checkpoint registered in Supabase (id={ckpt.id}, step={ckpt.step})")

wandb.finish()
