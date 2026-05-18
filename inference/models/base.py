"""
ModelConfig — declarative description of a model's requirements and capabilities.
ModelLoader — abstract backend (Unsloth, vLLM, plain HuggingFace).

This layer decouples "which model to use" from "how to load it", so swapping
backends (e.g. Unsloth → vLLM) or adding new model sizes requires only a registry
entry, not code changes.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    """Full specification for a model variant."""
    name: str                             # HuggingFace model ID
    friendly_name: str                    # Short key used in config/CLI

    # Hardware requirements
    quantization: str = "4bit"            # "4bit" | "8bit" | "none"
    tensor_parallel_size: int = 1         # GPUs required for inference
    vram_required_gb: float = 8.0         # Minimum VRAM to load model

    # Sequence limits
    max_seq_len: int = 4096

    # Model capabilities
    supports_thinking: bool = False       # Qwen3-style <think> blocks

    # LoRA defaults (overridden per training run)
    lora_r: int = 32
    lora_alpha: int = 32                  # alpha = r (not 2*r) — matches Qwen3 adapters

    # Free-form backend kwargs forwarded to the loader
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.friendly_name} ({self.quantization}, {self.tensor_parallel_size}xGPU)"


class ModelLoader(ABC):
    """
    Abstract backend for loading a model from a ModelConfig.

    Concrete implementations: UnslothLoader (training / Colab),
    VLLMLoader (production inference), HFLoader (evaluation).
    """

    @abstractmethod
    def load(
        self,
        config: ModelConfig,
        adapter_path: str | None = None,
    ) -> Any:
        """
        Load the model described by `config`, optionally with a LoRA adapter.
        Returns a backend-specific object (model+tokenizer tuple, vLLM LLM, …).
        """
        ...

    @abstractmethod
    def supports(self, config: ModelConfig) -> bool:
        """Return True if this loader can handle the given config on this machine."""
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__
