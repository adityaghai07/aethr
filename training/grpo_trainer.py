"""
GRPO training module — extracts the Kaggle training notebook's Cell 7 logic
into importable Python so it can be triggered programmatically by the flywheel.

Two usage modes:
  1. Kaggle notebook (interactive): import and call train() directly
  2. Flywheel (automated): the orchestrator calls train() when the buffer fills

Requirements:
  - unsloth, trl<0.23, vllm must be installed (Kaggle environment)
  - Model must be loaded with get_peft_model() before calling train()
  - VLLM_USE_V1=0 must be set before any import (notebook Cell 1)

The trainer re-uses the same reward_fn as the reward worker —
train and runtime scoring are identical (no reward-train gap).
"""
from __future__ import annotations
import asyncio
import logging
import os
from typing import TYPE_CHECKING

from config import TRAINING, get_active_plugins
from rewards.composite import compute_composite
from inference.prompt_templates import format_for_training

if TYPE_CHECKING:
    from training.trajectory import TrajectoryGroup

logger = logging.getLogger(__name__)


def build_reward_fn(plugins=None):
    """
    Build a TRL-compatible reward function from the active plugin stack.
    TRL's GRPOTrainer calls: reward_fn(prompts, completions) → list[float]
    """
    if plugins is None:
        plugins = get_active_plugins()

    def reward_fn(prompts: list[str], completions: list, **kwargs) -> list[float]:
        flat: list[str] = []
        for c in completions:
            if isinstance(c, list) and c and isinstance(c[0], dict):
                flat.append(c[0].get("content", ""))
            else:
                flat.append(c if isinstance(c, str) else str(c))

        async def _score_all():
            tasks = [
                compute_composite(
                    prompt=p, response=c, history=[], plugins=plugins
                )
                for p, c in zip(prompts, flat)
            ]
            results = await asyncio.gather(*tasks)
            return [float(r[0]) for r in results]

        return asyncio.run(_score_all())

    return reward_fn


def train(
    model,
    tokenizer,
    dataset,
    output_dir: str = "./aethr_grpo_out",
    max_steps: int | None = None,
    run_name: str = "aethr-grpo",
) -> "any":
    """
    Run GRPO training on the given dataset and return the trainer.

    Args:
        model:       LoRA-wrapped model (FastLanguageModel.get_peft_model output)
        tokenizer:   matching tokenizer
        dataset:     HF Dataset with "prompt" column (ChatML-formatted strings)
        output_dir:  where to save checkpoints
        max_steps:   override TRAINING.num_train_epochs worth of steps
        run_name:    wandb run name

    Returns:
        GRPOTrainer instance (trainer.state has global_step, logs, etc.)
    """
    try:
        from trl import GRPOConfig, GRPOTrainer
        import wandb
    except ImportError as e:
        raise ImportError(
            f"Training dependencies not installed: {e}. "
            "Run: pip install trl<0.23 wandb unsloth"
        )

    import torch
    torch.cuda.empty_cache()

    reward_fn = build_reward_fn()

    training_args = GRPOConfig(
        learning_rate               = TRAINING.learning_rate,
        adam_beta1                  = 0.9,
        adam_beta2                  = 0.99,
        weight_decay                = 0.1,
        warmup_ratio                = 0.1,
        lr_scheduler_type           = "cosine",
        optim                       = "paged_adamw_8bit",
        logging_steps               = 1,
        per_device_train_batch_size = TRAINING.per_device_train_batch_size,
        gradient_accumulation_steps = TRAINING.gradient_accumulation_steps,
        num_generations             = TRAINING.group_size,
        max_prompt_length           = TRAINING.max_prompt_length,
        max_completion_length       = TRAINING.max_completion_length,
        num_train_epochs            = TRAINING.num_train_epochs,
        max_steps                   = max_steps or -1,
        save_steps                  = TRAINING.save_steps,
        save_total_limit            = TRAINING.save_total_limit,
        max_grad_norm               = 0.1,
        report_to                   = "wandb",
        output_dir                  = output_dir,
        kl_coef                     = TRAINING.kl_coef,
    )

    trainer = GRPOTrainer(
        model            = model,
        processing_class = tokenizer,
        reward_funcs     = [reward_fn],
        args             = training_args,
        train_dataset    = dataset,
    )

    wandb.init(project=os.getenv("WANDB_PROJECT", "aethr"), name=run_name)
    trainer.train()
    wandb.finish()

    logger.info(f"Training complete at step {trainer.state.global_step}")
    return trainer


def build_dataset_from_groups(groups: list["TrajectoryGroup"]):
    """
    Convert TrajectoryGroups into a HuggingFace Dataset for GRPOTrainer.
    Each TrajectoryGroup contributes group_size rows.
    """
    from datasets import Dataset

    rows = []
    for group in groups:
        batch = group.to_grpo_batch()
        for prompt in batch["prompts"]:
            rows.append({"prompt": prompt})

    if not rows:
        raise ValueError("No training examples — check that rollouts produced varied rewards.")

    return Dataset.from_list(rows)


async def push_adapter(
    model,
    tokenizer,
    hf_repo: str,
    hf_token: str,
    step: int,
    output_dir: str = "./adapter_out",
) -> str:
    """
    Save and push the LoRA adapter to HuggingFace Hub.
    Returns the commit OID for DB registration.
    """
    import asyncio
    from huggingface_hub import HfApi

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    api = HfApi(token=hf_token)
    result = api.upload_folder(
        folder_path    = output_dir,
        repo_id        = hf_repo,
        repo_type      = "model",
        commit_message = f"adapter: step {step}",
    )
    logger.info(f"Adapter pushed: {hf_repo}@{result.oid}")
    return result.oid
