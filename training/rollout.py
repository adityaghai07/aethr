"""
Async rollout collection — generates N completions per prompt from the inference server
and scores each one, producing TrajectoryGroups ready for GRPO training.

This is the "rollout" phase of GRPO:
  1. Take a batch of prompts (conversation histories from Supabase)
  2. For each prompt, generate group_size completions (diverse, temp=1.0)
  3. Score each completion using the full reward plugin stack
  4. Return TrajectoryGroup objects (normalized advantages computed on-demand)

The inference server must be running (Kaggle /seturl). Reward scoring runs
concurrently with generation to minimize total latency.
"""
from __future__ import annotations
import asyncio
import logging
from typing import Callable

from training.trajectory import Trajectory, TrajectoryGroup, Choice, filter_groups
from rewards.composite import compute_composite
from config import TRAINING, get_active_plugins

logger = logging.getLogger(__name__)


async def _generate_completion(
    generate_fn: Callable,
    history: list[dict],
    temperature: float = 1.0,
    max_tokens: int = 512,
) -> str:
    """Call the inference server for a single completion."""
    return await generate_fn(history, temperature=temperature, max_tokens=max_tokens)


async def _score_completion(
    prompt: str,
    response: str,
    history: list[dict],
    plugins,
) -> tuple[float, dict]:
    """Score a single completion. Returns (reward, details)."""
    composite, details, _ = await compute_composite(
        prompt=prompt,
        response=response,
        history=history,
        plugins=plugins,
    )
    return composite, details


async def gather_trajectory_groups(
    prompt_histories: list[list[dict]],
    generate_fn: Callable,
    group_size: int | None = None,
    temperature: float = 1.0,
    max_tokens: int = 512,
    plugins=None,
) -> list[TrajectoryGroup]:
    """
    Generate and score rollouts for a batch of prompts.

    Args:
        prompt_histories: list of conversation histories (each is [{role, content}])
        generate_fn: async (history, temperature, max_tokens) → str
        group_size: completions per prompt (default: TRAINING.group_size)
        temperature: rollout temperature (1.0 = max diversity for GRPO)
        max_tokens: max completion length
        plugins: reward plugin list (default: get_active_plugins())

    Returns:
        list of TrajectoryGroup, one per prompt, each with group_size trajectories
    """
    if group_size is None:
        group_size = TRAINING.group_size
    if plugins is None:
        plugins = get_active_plugins()

    groups: list[TrajectoryGroup] = []

    for history in prompt_histories:
        # Extract the prompt (last user message)
        prompt = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"),
            "",
        )

        # Generate group_size completions concurrently
        completion_tasks = [
            _generate_completion(generate_fn, history, temperature, max_tokens)
            for _ in range(group_size)
        ]
        completions: list[str] = list(await asyncio.gather(*completion_tasks))

        # Score all completions concurrently
        score_tasks = [
            _score_completion(prompt, c, history, plugins)
            for c in completions
        ]
        scored = list(await asyncio.gather(*score_tasks))

        trajectories = [
            Trajectory(
                history=history,
                choice=Choice(content=completion),
                reward=reward,
                metadata=details,
            )
            for completion, (reward, details) in zip(completions, scored)
        ]

        groups.append(TrajectoryGroup(prompt_history=history, trajectories=trajectories))
        logger.debug(
            f"Rolled out {group_size} completions for prompt '{prompt[:50]}...', "
            f"rewards={[round(t.reward, 3) for t in trajectories]}"
        )

    return groups


async def collect_from_db(
    generate_fn: Callable,
    limit: int = 50,
    group_size: int | None = None,
) -> list[TrajectoryGroup]:
    """
    Pull recent conversation histories from the DB and run rollouts on them.
    Used by the flywheel orchestrator to generate fresh training data.
    """
    from db.queries import get_training_prompts
    histories = await get_training_prompts(limit=limit)
    logger.info(f"Collected {len(histories)} prompts from DB for rollout")
    groups = await gather_trajectory_groups(
        prompt_histories=histories,
        generate_fn=generate_fn,
        group_size=group_size,
    )
    # Filter out groups with no learning signal
    filtered = filter_groups(groups, min_reward_variance=TRAINING.min_reward_variance)
    logger.info(
        f"Rollout complete: {len(groups)} groups → {len(filtered)} after variance filter"
    )
    return filtered
