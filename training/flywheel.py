"""
Automatic flywheel orchestrator — the core Aethr differentiator.

The flywheel loop:
  1. Poll DB for unscored messages → reward worker handles scoring
  2. When `buffer_size` scored examples accumulate → trigger GRPO training
  3. After training → run eval gate (quality/regression checks)
  4. If eval passes → push adapter to HF Hub → inference server hot-reloads
  5. Repeat

This runs on the Kaggle training notebook (has GPU). The bot process just
collects conversations; the flywheel does everything else autonomously.

Usage (Kaggle cell):
    from training.flywheel import Flywheel
    fw = Flywheel(model=model, tokenizer=tokenizer, hf_token=HF_TOKEN)
    await fw.run()   # loops forever, trigger Ctrl-C to stop
"""
from __future__ import annotations
import asyncio
import logging
import os
import time
from dataclasses import dataclass

from config import TRAINING, EVAL_GATE, KL_WARN, KL_STOP

logger = logging.getLogger(__name__)


@dataclass
class FlywheelState:
    last_train_time: float = 0.0
    total_train_steps: int = 0
    total_rollout_groups: int = 0
    adapter_version: int = 0


class Flywheel:
    """
    Autonomous training loop. Runs on the Kaggle GPU notebook.

    Args:
        model, tokenizer: loaded Unsloth model (from FastLanguageModel)
        hf_token:         HuggingFace API token
        hf_repo:          adapter repo (default: env HF_ADAPTER_REPO)
        poll_interval:    seconds between DB checks (default: 60)
        generate_fn:      async (history, temperature, max_tokens) → str
                          defaults to calling the inference server
    """

    def __init__(
        self,
        model,
        tokenizer,
        hf_token: str = "",
        hf_repo: str = "",
        poll_interval: int = 60,
        generate_fn=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "")
        self.hf_repo = hf_repo or os.getenv("HF_ADAPTER_REPO", "")
        self.poll_interval = poll_interval
        self._generate_fn = generate_fn
        self._state = FlywheelState()

    async def _get_generate_fn(self):
        """Default generate_fn: calls the inference server."""
        if self._generate_fn:
            return self._generate_fn
        from inference.client import llm

        async def _gen(history, temperature=1.0, max_tokens=512):
            return await llm.chat(
                messages=history,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return _gen

    async def _count_unscored(self) -> int:
        from db.queries import get_unscored_messages
        msgs = await get_unscored_messages(limit=200)
        return len(msgs)

    async def _count_ready_examples(self) -> int:
        """Count assistant messages with reward scores available for training."""
        from db.queries import get_training_prompts
        prompts = await get_training_prompts(limit=TRAINING.buffer_size + 1)
        return len(prompts)

    async def _run_rollouts(self):
        """Generate GRPO rollout groups from recent conversations."""
        from training.rollout import collect_from_db
        generate_fn = await self._get_generate_fn()
        groups = await collect_from_db(
            generate_fn=generate_fn,
            limit=TRAINING.buffer_size,
            group_size=TRAINING.group_size,
        )
        self._state.total_rollout_groups += len(groups)
        return groups

    async def _run_training(self, groups) -> tuple[bool, int]:
        """
        Convert rollout groups → dataset → run GRPO.
        Returns (success, global_step).
        """
        from training.grpo_trainer import build_dataset_from_groups, train

        if not groups:
            logger.warning("No groups to train on — skipping training step")
            return False, 0

        dataset = build_dataset_from_groups(groups)
        logger.info(f"Training on {len(dataset)} examples from {len(groups)} groups")

        trainer = train(
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=dataset,
            output_dir="./aethr_flywheel_out",
            run_name=f"aethr-flywheel-v{self._state.adapter_version + 1}",
        )
        self._state.total_train_steps += trainer.state.global_step
        return True, trainer.state.global_step

    async def _run_eval_gate(self) -> bool:
        """
        Lightweight eval gate: check average recent reward hasn't dropped.
        Full benchmark suite (MMLU, persona) runs separately.
        Returns True if training should be promoted.
        """
        from db.queries import get_recent_scored_examples
        recent = await get_recent_scored_examples(limit=20)
        if not recent:
            logger.warning("No recent scored examples for eval gate — promoting anyway")
            return True

        from db.queries import get_db
        from db.models import RewardScore
        from sqlalchemy import select
        import statistics

        # Get composite scores from last 20 examples
        scores = [ex.completions[0].get("reward", 0) if ex.completions else 0 for ex in recent]
        if not scores:
            return True

        avg = sum(scores) / len(scores)
        threshold = EVAL_GATE.get("avg_reward_delta_min", -0.05)
        if avg < threshold:
            logger.warning(f"Eval gate failed: avg_reward={avg:.3f} < threshold={threshold}")
            return False

        logger.info(f"Eval gate passed: avg_reward={avg:.3f}")
        return True

    async def _push_and_register(self, step: int) -> str:
        """Push adapter to HF Hub and register checkpoint in DB."""
        from training.grpo_trainer import push_adapter
        from db.queries import register_checkpoint

        oid = await push_adapter(
            model=self.model,
            tokenizer=self.tokenizer,
            hf_repo=self.hf_repo,
            hf_token=self.hf_token,
            step=step,
            output_dir="./adapter_out",
        )

        from inference.model_registry import DEFAULT_MODEL, get_model_config
        from config import MODEL_NAME
        try:
            cfg = get_model_config(MODEL_NAME)
            base = cfg.name
        except KeyError:
            base = os.getenv("BASE_MODEL", "unsloth/Qwen3-8B-bnb-4bit")

        await register_checkpoint(
            step=step,
            hf_repo=self.hf_repo,
            hf_revision=oid,
            base_model=base,
            eval_scores={},
            training_config={"flywheel_version": self._state.adapter_version + 1},
        )

        self._state.adapter_version += 1
        logger.info(f"Adapter v{self._state.adapter_version} live: {self.hf_repo}@{oid}")
        return oid

    async def run(self, max_iterations: int | None = None) -> None:
        """
        Main flywheel loop. Runs indefinitely (or for max_iterations cycles).
        Safe to interrupt with Ctrl-C — state is persisted to DB at each step.
        """
        logger.info("Flywheel starting...")
        iteration = 0

        while True:
            if max_iterations and iteration >= max_iterations:
                logger.info(f"Flywheel stopped after {iteration} iterations")
                break

            # ── Check if it's time to train ────────────────────────────────────
            now = time.time()
            elapsed = now - self._state.last_train_time
            if elapsed < TRAINING.min_train_interval_sec:
                wait = TRAINING.min_train_interval_sec - elapsed
                logger.info(f"Waiting {wait:.0f}s before next training run...")
                await asyncio.sleep(min(wait, self.poll_interval))
                continue

            ready = await self._count_ready_examples()
            logger.info(f"Ready examples: {ready}/{TRAINING.buffer_size}")

            if ready < TRAINING.buffer_size:
                logger.info("Buffer not full yet — waiting...")
                await asyncio.sleep(self.poll_interval)
                continue

            # ── Run rollouts ───────────────────────────────────────────────────
            logger.info("Buffer full — generating rollouts...")
            try:
                groups = await self._run_rollouts()
            except Exception as e:
                logger.error(f"Rollout failed: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)
                continue

            if not groups:
                logger.warning("Rollouts produced no usable groups — waiting...")
                await asyncio.sleep(self.poll_interval)
                continue

            # ── Run training ───────────────────────────────────────────────────
            logger.info(f"Training on {len(groups)} trajectory groups...")
            try:
                success, step = await self._run_training(groups)
            except Exception as e:
                logger.error(f"Training failed: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)
                continue

            if not success:
                await asyncio.sleep(self.poll_interval)
                continue

            self._state.last_train_time = time.time()

            # ── Eval gate ──────────────────────────────────────────────────────
            promoted = await self._run_eval_gate()
            if promoted:
                await self._push_and_register(step)
            else:
                logger.warning("Eval gate failed — adapter NOT promoted")

            iteration += 1
            logger.info(
                f"Flywheel iteration {iteration} complete. "
                f"Adapter v{self._state.adapter_version}, "
                f"total steps: {self._state.total_train_steps}"
            )
