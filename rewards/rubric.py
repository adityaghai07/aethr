"""
RewardRubric — declarative reward specification via natural language.

Users describe what "good" looks like for their use case in plain text,
optionally with positive/negative examples. The rubric is compiled into
a scoring prompt injected into the LLM judge or REWARDANYTHING model.

This is the FSPO-inspired personalization layer: convert user preference
pairs into a natural-language rubric that drives reward scoring.

Usage:
    rubric = RewardRubric(
        name="concise_assistant",
        criteria=["Respond in 2-3 sentences unless the question requires more",
                  "Never use bullet points for simple answers"],
        examples=[
            {"prompt": "What time is it in Tokyo?",
             "good": "Tokyo is currently 14 hours ahead of UTC.",
             "bad": "Great question! Tokyo, the capital of Japan, uses Japan Standard Time (JST), which is UTC+9..."},
        ],
    )
    prompt = rubric.to_prompt()
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PreferenceExample:
    prompt: str
    good: str           # preferred response
    bad: str            # rejected response
    note: str = ""      # optional explanation


@dataclass
class RewardRubric:
    """
    A user-defined reward rubric. Compiled to a scoring prompt at runtime.
    Store rubrics in YAML/JSON config — no code changes needed to customize.
    """
    name: str
    criteria: list[str]                             # "Be concise", "Match tone", …
    examples: list[PreferenceExample] = field(default_factory=list)
    system_context: str = ""                        # who the assistant is / the use case
    weight: float = 1.0                             # relative weight when used as a plugin

    def to_prompt(self) -> str:
        """
        Compile the rubric into a scoring prompt suitable for an LLM judge
        or REWARDANYTHING's instruction-following reward model.
        """
        lines = []

        if self.system_context:
            lines.append(f"Context: {self.system_context}\n")

        lines.append("Score the response 0.0–1.0 based on ALL of the following criteria:")
        for i, criterion in enumerate(self.criteria, 1):
            lines.append(f"  {i}. {criterion}")

        if self.examples:
            lines.append("\nReference examples (use these to calibrate your scoring):")
            for ex in self.examples:
                lines.append(f'\n  Prompt: "{ex.prompt}"')
                lines.append(f'  Good response (score ≈ 1.0): "{ex.good}"')
                lines.append(f'  Bad response  (score ≈ 0.0): "{ex.bad}"')
                if ex.note:
                    lines.append(f'  Why: {ex.note}')

        lines.append(
            '\nReturn ONLY valid JSON: {"score": <0.0–1.0>, "reasoning": "<one sentence>"}'
        )
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict) -> "RewardRubric":
        """Deserialize from a plain dict (loaded from YAML/JSON config)."""
        examples = [
            PreferenceExample(**ex) for ex in data.get("examples", [])
        ]
        return cls(
            name=data["name"],
            criteria=data["criteria"],
            examples=examples,
            system_context=data.get("system_context", ""),
            weight=data.get("weight", 1.0),
        )
