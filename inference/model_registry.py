"""
Model registry — maps friendly names to ModelConfig instances.

Usage:
    from inference.model_registry import get_model_config, MODEL_REGISTRY
    cfg = get_model_config("qwen3-8b-4bit")   # default
    cfg = get_model_config(os.getenv("MODEL_NAME", "qwen3-8b-4bit"))

To add a new model: add an entry to _CONFIGS below. No other changes needed.
The friendly_name is what goes in .env MODEL_NAME and in the Kaggle notebook.
"""
from inference.models.base import ModelConfig

_CONFIGS: list[ModelConfig] = [
    # ── Qwen3 family (Unsloth pre-quantized) ────────────────────────────────────
    ModelConfig(
        name="unsloth/Qwen3-0.6B-bnb-4bit",
        friendly_name="qwen3-0.6b-4bit",
        quantization="4bit",
        tensor_parallel_size=1,
        vram_required_gb=2.0,
        max_seq_len=32768,
        supports_thinking=True,
        lora_r=16,
        lora_alpha=16,
    ),
    ModelConfig(
        name="unsloth/Qwen3-1.7B-bnb-4bit",
        friendly_name="qwen3-1.7b-4bit",
        quantization="4bit",
        tensor_parallel_size=1,
        vram_required_gb=4.0,
        max_seq_len=32768,
        supports_thinking=True,
        lora_r=16,
        lora_alpha=16,
    ),
    ModelConfig(
        name="unsloth/Qwen3-4B-bnb-4bit",
        friendly_name="qwen3-4b-4bit",
        quantization="4bit",
        tensor_parallel_size=1,
        vram_required_gb=6.0,
        max_seq_len=32768,
        supports_thinking=True,
        lora_r=32,
        lora_alpha=32,
    ),
    ModelConfig(
        name="unsloth/Qwen3-8B-bnb-4bit",
        friendly_name="qwen3-8b-4bit",       # ← default
        quantization="4bit",
        tensor_parallel_size=1,
        vram_required_gb=8.0,
        max_seq_len=32768,
        supports_thinking=True,
        lora_r=32,
        lora_alpha=32,
    ),
    ModelConfig(
        name="unsloth/Qwen3-14B-bnb-4bit",
        friendly_name="qwen3-14b-4bit",
        quantization="4bit",
        tensor_parallel_size=1,
        vram_required_gb=12.0,
        max_seq_len=32768,
        supports_thinking=True,
        lora_r=32,
        lora_alpha=32,
    ),
    ModelConfig(
        name="unsloth/Qwen3-32B-bnb-4bit",
        friendly_name="qwen3-32b-4bit",
        quantization="4bit",
        tensor_parallel_size=2,              # needs 2x A100 80GB for 4-bit 32B
        vram_required_gb=24.0,
        max_seq_len=32768,
        supports_thinking=True,
        lora_r=64,
        lora_alpha=64,
    ),
    # ── Qwen3 full-precision (production, vLLM) ──────────────────────────────────
    ModelConfig(
        name="Qwen/Qwen3-8B",
        friendly_name="qwen3-8b",
        quantization="none",
        tensor_parallel_size=1,
        vram_required_gb=18.0,
        max_seq_len=32768,
        supports_thinking=True,
        lora_r=32,
        lora_alpha=32,
    ),
    ModelConfig(
        name="Qwen/Qwen3-32B",
        friendly_name="qwen3-32b",
        quantization="none",
        tensor_parallel_size=4,
        vram_required_gb=70.0,
        max_seq_len=32768,
        supports_thinking=True,
        lora_r=64,
        lora_alpha=64,
    ),
]

# Build lookup dict
MODEL_REGISTRY: dict[str, ModelConfig] = {cfg.friendly_name: cfg for cfg in _CONFIGS}

# Also index by HF model ID for reverse lookup
_BY_HF_NAME: dict[str, ModelConfig] = {cfg.name: cfg for cfg in _CONFIGS}

DEFAULT_MODEL = "qwen3-8b-4bit"


def get_model_config(name: str) -> ModelConfig:
    """
    Look up a ModelConfig by friendly name or HF model ID.
    Raises KeyError with a helpful message if not found.
    """
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    if name in _BY_HF_NAME:
        return _BY_HF_NAME[name]
    available = sorted(MODEL_REGISTRY.keys())
    raise KeyError(
        f"Unknown model '{name}'. Available: {available}\n"
        "Add a new ModelConfig to inference/model_registry.py to register it."
    )
