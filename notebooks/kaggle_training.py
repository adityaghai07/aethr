# Aethr — GRPO Training Notebook (Colab T4)
#
# Mirrors Unsloth's official GRPO recipe for T4 free tier:
#   https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/DeepSeek_R1_0528_Qwen3_(8B)_GRPO.ipynb
#
# Only difference: our data comes from Supabase, and the reward function
# imports our production reward plugin stack so train and runtime use the
# same scoring logic.
#
# Run when ≥32 scored conversations are in Supabase.

# ── Cell 1: Install ───────────────────────────────────────────────────────────
# In[ ]:
# IMPORTANT: VLLM_USE_V1 must be set BEFORE any vllm/unsloth import — vLLM picks
# its engine version at import time and locks it. V0 is the stable backend; V1
# has a known "Duplicate layer name: model.layers.0.self_attn.attn" bug with
# bitsandbytes 4-bit models on T4.

import os
os.environ["VLLM_USE_V1"] = "0"

# torchcodec is pulled in transitively but needs FFmpeg system libs that
# Colab's image lacks — uninstall it before unsloth tries to import it.
os.system("pip uninstall -y -q torchcodec")

if "COLAB_" not in "".join(os.environ.keys()):
    os.system("pip install -q unsloth vllm")
else:
    # Core training stack — let pip resolve transitive deps (no --no-deps)
    os.system("pip install -q unsloth")
    os.system("pip install -q vllm==0.10.2")
    # trl<0.23 is required — newer versions have GRPOTrainer API breakage
    os.system('pip install -q --upgrade "trl<0.23"')
    os.system("pip install -q --upgrade pillow")

# Aethr-specific deps
os.system("pip install -q asyncpg sqlalchemy[asyncio] anthropic httpx python-dotenv wandb")

print("✓ Packages installed, VLLM_USE_V1=0, torchcodec removed")

# ── Cell 2: Clone aethr codebase ──────────────────────────────────────────────
# In[ ]:

import os, sys, subprocess
if not os.path.exists("/content/aethr"):
    subprocess.run(["git", "clone", "https://github.com/adityaghai07/aethr",
                    "/content/aethr"], check=True)
os.chdir("/content/aethr")
sys.path.insert(0, "/content/aethr")
print("✓ Aethr cloned")

# ── Cell 3: Configuration (paste your secrets here) ───────────────────────────
# In[ ]:

HF_TOKEN        = os.environ.get("HF_TOKEN",        "YOUR_HF_TOKEN")
HF_ADAPTER_REPO = os.environ.get("HF_ADAPTER_REPO", "your-username/aethr-adapters")
DATABASE_URL    = os.environ.get("DATABASE_URL",    "postgresql+asyncpg://...")
WANDB_API_KEY   = os.environ.get("WANDB_API_KEY",   "your-wandb-key")
JUDGE_API_KEY   = os.environ.get("JUDGE_API_KEY",   "sk-ant-...")

for k, v in {"HF_TOKEN": HF_TOKEN, "HF_ADAPTER_REPO": HF_ADAPTER_REPO,
             "DATABASE_URL": DATABASE_URL, "WANDB_API_KEY": WANDB_API_KEY,
             "JUDGE_API_KEY": JUDGE_API_KEY}.items():
    os.environ[k] = v

import wandb
wandb.login(key=WANDB_API_KEY)
print("✓ Config loaded")

# ── Cell 4: Load model (Unsloth's exact T4 settings) ──────────────────────────
# In[ ]:
# Unsloth's tuned T4 GRPO recipe — these specific values are what works.

from unsloth import FastLanguageModel
import torch

max_seq_length = 1024     # combined prompt + completion (T4 ceiling)
lora_rank      = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name             = "unsloth/Qwen3-8B",
    max_seq_length         = max_seq_length,
    load_in_4bit           = True,
    fast_inference         = True,
    max_lora_rank          = lora_rank,
    gpu_memory_utilization = 0.5,
)

model = FastLanguageModel.get_peft_model(
    model,
    r                          = lora_rank,
    target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
    lora_alpha                 = lora_rank,
    use_gradient_checkpointing = "unsloth",
    random_state               = 3407,
)
print(f"✓ Model loaded — GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# ── Cell 5: Pull scored conversations from Supabase ───────────────────────────
# In[ ]:

import asyncio
import asyncpg
from inference.prompt_templates import format_for_training

async def fetch_examples(db_url: str, limit: int = 200):
    raw = db_url.replace("postgresql+asyncpg://", "postgresql://").split("?")[0]
    conn = await asyncpg.connect(raw, statement_cache_size=0)

    rows = await conn.fetch("""
        SELECT m.id, m.conversation_id, m.created_at, rs.composite_score
        FROM messages m JOIN reward_scores rs ON rs.message_id = m.id
        WHERE m.role = 'assistant' AND rs.composite_score IS NOT NULL
        ORDER BY m.created_at DESC LIMIT $1
    """, limit)

    examples = []
    for r in rows:
        hist = await conn.fetch("""
            SELECT role, content FROM messages
            WHERE conversation_id = $1 AND created_at < $2
            ORDER BY created_at ASC LIMIT 20
        """, r["conversation_id"], r["created_at"])
        if not hist:
            continue
        examples.append({
            "history": [{"role": h["role"], "content": h["content"]} for h in hist],
        })
    await conn.close()
    return examples

examples = asyncio.run(fetch_examples(DATABASE_URL))
print(f"✓ Fetched {len(examples)} scored conversations")

if len(examples) < 32:
    raise ValueError(f"Need ≥32 scored examples. Have {len(examples)}.")

from datasets import Dataset
train_dataset = Dataset.from_list([
    {"prompt": format_for_training(ex["history"])} for ex in examples
])
print(f"Sample prompt:\n{train_dataset[0]['prompt'][:200]}")

# ── Cell 6: Reward function — uses our production plugin stack ────────────────
# In[ ]:
# Identical scoring to the runtime reward worker, so the model is trained
# on exactly the same signal it's evaluated on.

from config import get_active_plugins
from rewards.composite import compute_composite

active_plugins = get_active_plugins()
print(f"Active plugins: {[p.name for p in active_plugins]}")


def reward_fn(prompts, completions, **kwargs):
    """
    TRL's GRPOTrainer signature:
      prompts: list[str]
      completions: list[str] OR list[list[dict]] (depending on TRL version)
      returns: list[float]
    """
    # Normalize completions to plain strings
    flat = []
    for c in completions:
        if isinstance(c, list) and c and isinstance(c[0], dict):
            flat.append(c[0].get("content", ""))
        else:
            flat.append(c if isinstance(c, str) else str(c))

    async def _score_all():
        tasks = [
            compute_composite(prompt=p, response=c, history=[], plugins=active_plugins)
            for p, c in zip(prompts, flat)
        ]
        results = await asyncio.gather(*tasks)
        return [float(r[0]) for r in results]

    return asyncio.run(_score_all())


# Sanity check
test = reward_fn(
    prompts=["What is my LDL?"],
    completions=["Your LDL is 142 — worth mentioning at your next checkup."],
)
print(f"Sanity check reward (should be > 0): {test[0]:.3f}")

# ── Cell 7: GRPO training (Unsloth's T4 config verbatim) ──────────────────────
# In[ ]:

from trl import GRPOConfig, GRPOTrainer

max_prompt_length     = 256
max_completion_length = max_seq_length - max_prompt_length   # 768

training_args = GRPOConfig(
    learning_rate               = 5e-6,
    adam_beta1                  = 0.9,
    adam_beta2                  = 0.99,
    weight_decay                = 0.1,
    warmup_ratio                = 0.1,
    lr_scheduler_type           = "cosine",
    optim                       = "paged_adamw_8bit",
    logging_steps               = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,
    num_generations             = 8,
    max_prompt_length           = max_prompt_length,
    max_completion_length       = max_completion_length,
    num_train_epochs            = 1,
    max_steps                   = 100,        # start with 100 — bump up after sanity check
    save_steps                  = 100,
    max_grad_norm               = 0.1,
    report_to                   = "wandb",
    output_dir                  = "/content/aethr_grpo_out",
)

trainer = GRPOTrainer(
    model           = model,
    processing_class= tokenizer,
    reward_funcs    = [reward_fn],
    args            = training_args,
    train_dataset   = train_dataset,
)

wandb.init(project="aethr", name="grpo-colab-t4")
trainer.train()
print("✓ Training complete")

# ── Cell 8: Save & push adapter to HF Hub ────────────────────────────────────
# In[ ]:

from huggingface_hub import HfApi

ADAPTER_OUT = "/content/aethr_grpo_out/final_adapter"
model.save_pretrained(ADAPTER_OUT)
tokenizer.save_pretrained(ADAPTER_OUT)

api = HfApi(token=HF_TOKEN)
result = api.upload_folder(
    folder_path    = ADAPTER_OUT,
    repo_id        = HF_ADAPTER_REPO,
    repo_type      = "model",
    commit_message = f"adapter: step {trainer.state.global_step}",
)
print(f"✓ Adapter pushed: {HF_ADAPTER_REPO}@{result.oid}")

# ── Cell 9: Register checkpoint in Supabase ───────────────────────────────────
# In[ ]:

from db.connection import get_db
from db.models import Checkpoint

async def register():
    async with get_db() as session:
        ckpt = Checkpoint(
            step            = trainer.state.global_step,
            hf_repo         = HF_ADAPTER_REPO,
            hf_revision     = result.oid,
            base_model      = "unsloth/Qwen3-8B",
            eval_scores     = {},
            is_active       = False,    # set True after eval gate passes (Phase 6)
            training_config = {
                "lr": 5e-6, "lora_r": lora_rank, "group_size": 8,
                "max_completion": max_completion_length,
                "max_steps": training_args.max_steps,
                "n_examples": len(examples),
            },
        )
        session.add(ckpt)
        await session.commit()
        await session.refresh(ckpt)
        return ckpt

ckpt = asyncio.run(register())
print(f"✓ Checkpoint registered (id={ckpt.id}, step={ckpt.step})")
print(f"\nTo activate: restart the inference notebook (it auto-pulls latest adapter)")

wandb.finish()
