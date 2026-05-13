"""
Central configuration for Aethr.
All hyperparameters and environment-driven settings live here.
Change INFRA_MODE to scale from Kaggle hobby → dual-GPU production.
"""
import os
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class InfraMode(str, Enum):
    SINGLE_GPU = "single_gpu"   # Kaggle: inference + training time-sliced
    DUAL_GPU   = "dual_gpu"     # Two Kaggle notebooks via separate ngrok URLs
    ASYNC      = "async"        # Fully decoupled cluster (future)


# ── Infrastructure ────────────────────────────────────────────────────────────

INFRA_MODE: InfraMode = InfraMode(os.getenv("INFRA_MODE", "dual_gpu"))

# URL of the inference Kaggle notebook (changes every session — bot has /seturl)
INFERENCE_URL: str = os.getenv("INFERENCE_URL", "http://localhost:8000")

# ── Model ─────────────────────────────────────────────────────────────────────

# Unsloth's pre-quantized HF model — loads on any GPU, 4-bit saves VRAM
BASE_MODEL: str = os.getenv("BASE_MODEL", "unsloth/Qwen3-8B-bnb-4bit")

# HuggingFace repo where LoRA adapters are pushed after each training run
# Format: "{your-hf-username}/aethr-adapters"
HF_ADAPTER_REPO: str = os.getenv("HF_ADAPTER_REPO", "")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# ── Database (Supabase PostgreSQL) ────────────────────────────────────────────

# Get from: Supabase dashboard → Settings → Database → Connection string
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:password@db.xxx.supabase.co:5432/postgres",
)

# ── Telegram ──────────────────────────────────────────────────────────────────

TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
# Your personal Telegram user_id — used for /seturl and alert messages
ADMIN_USER_ID: int = int(os.getenv("ADMIN_USER_ID", "0"))

# ── Reward System ─────────────────────────────────────────────────────────────

# External LLM judge (Claude Sonnet or GPT-4o-mini)
JUDGE_API_KEY: str = os.getenv("JUDGE_API_KEY", "")
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "claude-sonnet-4-20250514")

# Composite reward weights (must sum to 1.0)
REWARD_WEIGHTS = {
    "rule_based": 0.25,
    "llm_judge":  0.50,
    "user_feedback": 0.25,
}

# ── Training ──────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # GRPO core
    learning_rate: float = 1e-6        # RL needs low LR — higher causes instability
    kl_coef: float = 0.001             # Anchors to base model; too high = no learning
    clip_range: float = 0.2            # PPO-style clipping
    group_size: int = 8                # Completions per prompt (4 min, 16 luxury)
    temperature: float = 1.0           # High temp for diverse rollouts (low → no gradient)

    # LoRA
    lora_r: int = 32                   # Rank — sweet spot for 8B models
    lora_alpha: int = 64               # alpha = 2 * r is standard scaling
    lora_dropout: float = 0.0          # No dropout for RL (unlike SFT)
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Batch
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8   # effective = 16 prompts
    num_train_epochs: int = 1
    max_prompt_length: int = 1024
    max_completion_length: int = 2048

    # Buffer trigger
    buffer_size: int = 64              # Train when this many examples accumulate
    replay_ratio: float = 0.25         # Static replay fraction per batch
    min_reward_variance: float = 0.1   # Skip training if rewards are all similar

    # Timing
    min_train_interval_sec: int = 3600  # At most one training run per hour

    # Checkpointing
    save_steps: int = 50
    save_total_limit: int = 5


TRAINING = TrainingConfig()

# ── Evaluation ────────────────────────────────────────────────────────────────

# Eval gate thresholds — new checkpoint must pass all to go live
EVAL_GATE = {
    "mmlu_delta_min": -0.02,       # Can't regress MMLU by more than 2%
    "persona_score_min": 0.60,     # Must score ≥ 0.60 on persona tests
    "avg_reward_delta_min": -0.05, # Can't drop average reward by more than 5%
}

# KL alert thresholds
KL_WARN  = 1.0   # Watch carefully above this
KL_STOP  = 5.0   # Stop training immediately above this

# ── Conversation ──────────────────────────────────────────────────────────────

MAX_CONTEXT_MESSAGES: int = 40
MAX_CONTEXT_TOKENS: int = 6000

SYSTEM_PROMPT = """You are Aethr, a helpful personal AI assistant. You are concise, \
friendly, and adapt to the user's communication style over time. You remember context \
from the current conversation. When uncertain, say so rather than guessing."""

# ── Reward Plugins ────────────────────────────────────────────────────────────
#
# This is the single place to control which reward functions run and at what weight.
# Import order doesn't matter — the registry handles deduplication.
#
# To add a custom plugin:
#   1. Create rewards/plugins/your_plugin.py with a class extending RewardPlugin
#   2. Call register(YourPlugin()) at the bottom of that file
#   3. Add "your_plugin_name" to ACTIVE_PLUGIN_NAMES below
#
# To disable a plugin without deleting it: remove its name from this list.
# To change a plugin's weight: set plugin.weight in its class definition.

ACTIVE_PLUGIN_NAMES: list[str] = [
    "rule_based",           # rewards/plugins/general.py — instant, free
    "medical_guardrails",   # rewards/plugins/medical.py — Vithos guardrails
    "llm_judge",            # rewards/plugins/general.py — async, ~$0.002/call
]

# Loaded lazily on first use by the reward worker
def get_active_plugins():
    """Import and return the configured reward plugin instances."""
    import rewards.plugins.general   # noqa: F401 — registers rule_based + llm_judge
    import rewards.plugins.medical   # noqa: F401 — registers medical_guardrails
    from rewards.registry import get_plugin
    return [get_plugin(name) for name in ACTIVE_PLUGIN_NAMES]


# ── wandb ─────────────────────────────────────────────────────────────────────

WANDB_PROJECT: str = os.getenv("WANDB_PROJECT", "aethr")
WANDB_API_KEY: str = os.getenv("WANDB_API_KEY", "")
