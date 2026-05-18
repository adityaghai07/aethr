"""
Trajectory abstractions for GRPO training — ART-inspired design.

A Trajectory is a single (prompt, completion, reward) triple.
A TrajectoryGroup is N trajectories on the same prompt — the unit GRPO trains on.

These structures decouple data collection from training:
  - The reward worker collects conversations → TrajectoryGroups
  - The GRPO trainer consumes TrajectoryGroups, normalizes group advantages,
    and updates the policy

ART reference: https://github.com/OpenPipe/ART
Key difference: Aethr's Trajectory is conversation-aware (history matters for
reward scoring) and targets personal assistant quality, not task-completion agents.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Choice:
    """
    Marks the model's generated tokens — the trainable part of a trajectory.
    The text here is what the policy produced; the reward gradient flows through it.
    """
    content: str                        # the generated text (assistant response)
    token_ids: list[int] = field(default_factory=list)   # filled during training
    log_probs: list[float] = field(default_factory=list)  # log P(token) under policy


@dataclass
class Trajectory:
    """
    One complete (prompt, response, reward) sample.

    history: conversation context (system + prior turns) — no gradient
    choice:  the model's response — gradient flows here
    reward:  final scalar reward in [-1, 1] (after WCO composite)
    metadata: anything useful to log (plugin breakdown, cost, etc.)
    """
    history: list[dict]         # [{role, content}, ...]
    choice: Choice
    reward: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def prompt_text(self) -> str:
        """Last user message in the history."""
        for msg in reversed(self.history):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    @property
    def response_text(self) -> str:
        return self.choice.content


@dataclass
class TrajectoryGroup:
    """
    N trajectories on the same prompt — the GRPO training unit.

    All members share the same conversation history/prompt; only the
    assistant's completion differs. Group-level advantage normalization
    happens here before the gradient update.

    group_size: typically 4–8 (T4 budget) or 8–16 (A100 luxury)
    """
    prompt_history: list[dict]           # shared context
    trajectories: list[Trajectory]       # N completions

    @property
    def group_size(self) -> int:
        return len(self.trajectories)

    @property
    def rewards(self) -> list[float]:
        return [t.reward for t in self.trajectories]

    def normalized_advantages(self) -> list[float]:
        """
        Group-level advantage normalization for GRPO.
        Returns zero-mean, unit-variance advantages.
        If all rewards identical (collapse), returns all zeros.
        """
        from rewards.composite import normalize_group_advantages
        return normalize_group_advantages(self.rewards)

    def best(self) -> Trajectory:
        """Trajectory with the highest reward (for rejection sampling / SFT)."""
        return max(self.trajectories, key=lambda t: t.reward)

    def worst(self) -> Trajectory:
        """Trajectory with the lowest reward."""
        return min(self.trajectories, key=lambda t: t.reward)

    def to_grpo_batch(self) -> dict[str, Any]:
        """
        Convert to a dict compatible with TRL's GRPOTrainer dataset format.
        Returns: {"prompts": [str], "completions": [str], "advantages": [float]}
        """
        from inference.prompt_templates import format_for_training
        advantages = self.normalized_advantages()
        return {
            "prompts": [format_for_training(self.prompt_history)] * self.group_size,
            "completions": [t.response_text for t in self.trajectories],
            "advantages": advantages,
        }


def filter_groups(
    groups: list[TrajectoryGroup],
    min_reward_variance: float = 0.05,
) -> list[TrajectoryGroup]:
    """
    Drop groups where all rewards are near-identical (no learning signal).
    min_reward_variance: minimum std dev of rewards in a group to keep it.
    """
    import statistics
    kept = []
    for g in groups:
        if len(g.rewards) < 2:
            continue
        try:
            std = statistics.stdev(g.rewards)
        except statistics.StatisticsError:
            std = 0.0
        if std >= min_reward_variance:
            kept.append(g)
    return kept
