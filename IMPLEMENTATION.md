# Aethr — Implementation Plan

**One-line product**: Your personal AI assistant gets smarter every time it talks to you — automatically, no ML expertise needed.

**Moat**: The only framework that closes the full loop from live Slack/Telegram conversations to a retrained, deployed personal assistant — using SOTA reward models that work without verifiable ground truth. Every other RL training framework requires designed task scenarios with correct/wrong answers. Aethr trains on lived conversational experience.

---

## Architecture Overview

```
User (Telegram / Slack / Discord)
        ↓ message
    Bot Layer  →  Inference Server (vLLM + active LoRA)
        ↓ conversation saved
    Supabase (conversations + reward scores + checkpoints)
        ↓ background
    Reward Worker  →  3-tier reward pipeline (auto-scored)
        ↓ buffer fills (N conversations)
    Training Orchestrator  →  GRPO on GPU (Colab / RunPod / local)
        ↓ eval gate passes
    New LoRA adapter  →  hot-reloaded into vLLM
        ↓
    Better assistant → loop continues
```

---

## Technical Decisions (Rationale)

### T1 — Reward Model: Skywork-Reward-V2-Llama-3.1-8B-40M

**Why**: #1 on RM-Bench (96.0 overall, 93.5 Hard). RM-Bench correlates 0.55 with downstream RLHF improvement vs RewardBench's 0.21 (2.6× stronger). RM-Bench Hard specifically tests whether a model rewards genuine quality vs writing style — exactly what we need for conversational GRPO where all 8 rollouts look stylistically similar. The 8B size fits on the same A100 doing GRPO training. The 40M-sample training variant dominates because data quality and scale matter more than architecture.

**Do not use**: INF-ORM-Llama3.1-70B (RewardBench #1 at 95.1%, RM-Bench Hard 44.8% — worse than random), ArmoRM-Llama3-8B (outdated, superseded by Skywork-V2).

### T2 — Configurable Judge: REWARDANYTHING-8B + Prometheus-2-7B

**Why REWARDANYTHING-8B**: #3 RM-Bench (86.4, Hard 84.4), reasoning-based GenRM — generates chain-of-thought critique before scoring. Best at following custom rubrics while remaining style-resistant. This is the layer users configure via prompt + examples.

**Why Prometheus-2-7B fallback**: Supports 1,000+ natural language evaluation criteria, Pearson 0.685 with GPT-4. Use when the user wants maximum rubric customizability over raw RM-Bench performance. Self-hosted, essentially free.

### T3 — Ensemble: Worst-Case Optimization (WCO), not weighted average

**Why**: RM-Bench research (arXiv:2310.02743) shows WCO (take minimum across ensemble members) eliminates reward overoptimization and improves performance by up to 70% over single-model optimization. The model must score well on ALL three tiers to get a high reward — cannot hack any single dimension.

### T4 — Group-Level Variance Normalization

**Why**: Reward collapse is the most common silent failure in conversational GRPO. All 8 rollouts score 0.63–0.68, normalized advantages ≈ 0, training stalls. The fix: if max−min within a group < 0.15, apply rank-based spreading to force [0.1, 0.9] uniform distribution. This guarantees GRPO always has a learning signal.

### T5 — ART-Inspired Abstractions (adopted, not the full framework)

**Adopt**: `Trajectory` + `TrajectoryGroup` data model (marks trainable vs context tokens), `gather_trajectory_groups` async parallel rollout collection, vLLM + LoRA hot-reload pipeline, client/server architecture split.

**Do not adopt**: Full ART dependency. ART's serverless backend requires W&B and only supports 2 models. Its `LocalBackend` is single-GPU only. We take the abstractions, not the dependency.

### T6 — vLLM over HuggingFace generate()

**Why**: HF generate() is single-request, no batching, no KV cache. vLLM with continuous batching is 5–10× faster on A100+. vLLM also supports `--enable-lora` to serve multiple LoRA adapters simultaneously and hot-reload without server restart — essential for the automatic flywheel.

**T4 constraint**: On Colab T4 (15GB), vLLM's KV pre-allocation fails for 8B 4-bit models. Use HF generate() for T4. Auto-detect GPU at startup and select accordingly.

### T7 — Training Backend: TRL GRPOTrainer + Accelerate

**Why not ART's Megatron backend**: Megatron backend is experimental, supports only Qwen3 dense/MoE variants, requires `apex` and custom NCCL kernels (`quack-kernels`). Too much complexity for the hobbyist tier and limited model support.

**Why TRL + Accelerate**: TRL's GRPOTrainer is the reference implementation, actively maintained, Accelerate handles multi-GPU/FSDP with minimal code changes. Unsloth provides memory-optimized kernels on top. Same stack as ART's own Unsloth backend internally.

### T8 — Personalization: FSPO

**Why**: FSPO achieves 70% win rate with 4 explicit preference examples from the actual user (arXiv, fewshot-preference-optimization). The RAT enhancement (infer natural-language user profile from examples) conditions both generation and Prometheus rubric. No retraining required — works at inference time.

### T9 — Inference-First Deployment (not Colab-first)

**Why**: ngrok tunnels die every 12 hours. The hobbyist tier needs a reliably reachable inference server. Solution: generate a `Dockerfile` that runs inference server + bot together, deployable to any $5–10/month VPS. For users without a server, Hugging Face Inference Endpoints as the fallback. Colab for training only, not serving.

---

## Phase 0 — Foundation Cleanup (Current Codebase)

**Goal**: Make the existing code solid before adding anything new. No new features.

**Status**: Partially done (streaming, `enable_thinking=False`, double-PEFT fix already merged).

### 0.1 Model Abstraction Layer

Remove the 3 hardcoded locations of `BASE_MODEL`, `lora_r`, `lora_alpha`, `target_modules`.

```
inference/
  models/
    __init__.py
    base.py          # ModelConfig dataclass + ModelLoader ABC
    unsloth.py       # UnslothLoader: 4-bit, fast_inference=False, T4
    vllm_loader.py   # vLLMLoader: fast_inference=True, A100+
  model_registry.py  # name → ModelConfig, active model tracking
```

`ModelConfig`:
```python
@dataclass
class ModelConfig:
    name: str                    # display name
    hf_id: str                   # "unsloth/Qwen3-8B-bnb-4bit"
    lora_rank: int               # 32
    lora_alpha: int              # 32 (= lora_rank, as in training)
    lora_target_modules: list[str]
    max_seq_length: int          # 4096
    load_in_4bit: bool           # True for T4, False for A100 bf16
    gpu_memory_utilization: float  # 0.75
    dtype: str                   # "float16" T4, "bfloat16" A100
```

Training notebook and inference notebook both import `ModelConfig` from here. Single source of truth.

**Built-in registry entries**:
- `qwen3-0.6b` — Colab T4 free, ~3GB VRAM
- `qwen3-1.7b` — Colab T4 free, ~5GB VRAM
- `qwen3-8b-4bit` — Colab T4/A100, ~8GB VRAM (current default)
- `qwen3-14b-4bit` — A100 40GB, ~16GB VRAM
- `qwen3-32b-4bit` — 2×A100, ~32GB VRAM
- `llama-3.1-8b-4bit` — A100, ~8GB VRAM (alternative base)

### 0.2 Fix `lora_alpha` Drift

Training uses `lora_alpha = lora_rank = 32`. Old inference notebook had `lora_alpha = 64`. Both now derived from `ModelConfig`. The mismatch was causing adapter loading warnings — resolved by the double-PEFT fix but the root cause (two configs) must be permanently removed.

### 0.3 Reward Plugin Auto-Discovery

Replace manual imports in `config.py:get_active_plugins()`:

```python
# rewards/loader.py
def discover_plugins(plugin_dir: Path) -> list[RewardPlugin]:
    """Scan plugins/ directory, import all .py files, collect registered plugins."""
    for path in plugin_dir.glob("*.py"):
        if path.name.startswith("_"):
            continue
        importlib.import_module(f"rewards.plugins.{path.stem}")
    return list(_REGISTRY.values())
```

Medical guardrails moves to `rewards/plugins/medical.py` as a config-toggleable plugin, not baked into `safety.py`. `safety.py` keeps the hard-block logic but sources patterns from the plugin, not inline regex.

### 0.4 SQLite Fallback (Hobbyist Zero-Config)

Supabase is currently required. Add SQLite fallback:

```python
# db/connection.py
if DATABASE_URL.startswith("sqlite"):
    engine = create_async_engine(DATABASE_URL)  # SQLite via aiosqlite
else:
    engine = create_async_engine(DATABASE_URL, ...)  # Supabase PostgreSQL
```

Hobbyist default: `DATABASE_URL=sqlite+aiosqlite:///./aethr.db`. No account required. Supabase for anyone who wants persistent cloud storage or multi-device.

**Done when**: Single env file with 3 secrets (HF_TOKEN, TELEGRAM_TOKEN, NGROK_AUTH_TOKEN) launches a working bot on Colab T4.

---

## Phase 1 — SOTA Reward System

**Goal**: Replace current `llm_judge.py` + `rule_based.py` with the 3-tier ensemble. Produce benchmark numbers that validate the reward pipeline.

### 1.1 Tier 1 — Skywork-Reward-V2

```
rewards/
  models/
    skywork_v2.py    # SkyworkRewardV2: loads model, forward pass, returns float
    rewardanything.py  # RewardAnything8B: reasoning GenRM, chain-of-thought + score
```

`SkyworkRewardV2` wraps `Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M`:
- Load in 4-bit if VRAM < 20GB, bf16 otherwise
- Forward pass returns scalar logit (raw, not sigmoid — wider range, better variance)
- Batch scoring: score all N completions in a GRPO group in one forward pass where possible
- Important: use `attn_implementation="flash_attention_2"` or `"eager"` — SDPA has a documented bug that degrades Skywork-V2 performance

```python
class SkyworkRewardV2:
    def score_group(self, prompt: str, completions: list[str]) -> list[float]:
        # Returns raw logits for all completions
        # Caller applies group-level normalization
```

### 1.2 Tier 2 — Rubric-Based Judge

`RewardAnything8B` for standard use. `Prometheus2` as fallback when maximum rubric flexibility is needed.

**The user-facing rubric interface** (this is the "prompt + examples" product feature):

```python
@dataclass
class RewardRubric:
    criteria: str        # "A good response should be direct, concise, actionable..."
    good_examples: list[str]  # 2-3 examples of good responses
    bad_examples: list[str]   # 2-3 examples of bad responses
    hard_rules: list[str]     # ["never start with 'Certainly!'", "max 200 words for simple questions"]
```

Rubric is serialized to `rubric.yaml` in the project config. REWARDANYTHING-8B receives it as a system prompt. The rubric becomes the user's "definition of good" — the only config they need to write.

Validation step: before training starts, run the rubric on 20 sample conversations and check reward variance. If std < 0.12, warn: *"Your rubric may be too vague — all responses score similarly. Add more specific good/bad examples."*

### 1.3 Tier 3 — Rule-Based

```python
# rewards/rules.py
SYCOPHANCY_STARTERS = ["Certainly!", "Of course!", "Absolutely!", "Great question", "Sure!"]
AI_LEAKAGE = ["as an AI", "as a language model", "I cannot", "I don't have feelings"]

def rule_score(prompt: str, response: str) -> float:
    score = 1.0
    if any(response.startswith(s) for s in SYCOPHANCY_STARTERS): score -= 0.35
    if any(p in response.lower() for p in AI_LEAKAGE):           score -= 0.40
    words = len(response.split())
    if words < 8 or words > 800:                                  score -= 0.25
    if response.count("**") > 6 and is_casual(prompt):           score -= 0.15
    return max(0.0, score)
```

### 1.4 WCO Ensemble + Variance Normalization

```python
# rewards/composite.py
async def compute_reward_group(
    prompt: str,
    completions: list[str],
    rubric: RewardRubric,
) -> list[float]:
    # Run all three tiers concurrently
    t1_scores, t2_scores, t3_scores = await asyncio.gather(
        skywork.score_group(prompt, completions),
        judge.score_group(prompt, completions, rubric),
        asyncio.gather(*[rule_score(prompt, c) for c in completions]),
    )

    # WCO: take minimum across tiers per completion
    raw = [min(t1, t2, t3) for t1, t2, t3 in zip(t1_scores, t2_scores, t3_scores)]

    # Variance normalization: guarantee spread within group
    return _normalize_group(raw)


def _normalize_group(scores: list[float]) -> list[float]:
    if max(scores) - min(scores) < 0.15:
        # Rank-based uniform spread → [0.1, 0.9]
        order = sorted(range(len(scores)), key=lambda i: scores[i])
        spread = [0.1 + 0.8 * r / (len(scores) - 1) for r in range(len(scores))]
        result = [0.0] * len(scores)
        for rank, idx in enumerate(order):
            result[idx] = spread[rank]
        return result
    # Standard whitening
    mu = statistics.mean(scores)
    sigma = statistics.stdev(scores) + 1e-8
    return [(s - mu) / sigma for s in scores]
```

### 1.5 Reward Benchmark (Gate Before Phase 2)

Must pass all three before proceeding:

**B1 — Human Agreement**: Build 100 personal assistant response pairs (GPT-4o labels majority). Measure reward pipeline agreement. **Target: ≥ 75%.**

**B2 — Variance Adequacy**: Run 8 completions on 50 diverse prompts. Measure per-group std before normalization. **Target: mean std ≥ 0.12 pre-normalization** (normalization is a fallback, not the primary mechanism).

**B3 — Hacking Resistance**: Feed 20 sycophantic responses (start with "Certainly! Great question!") with good content. Measure whether the reward penalizes them despite good content. **Target: sycophantic responses score ≥ 0.2 lower than equivalent non-sycophantic responses.**

Publish all three numbers. This is the "SOTA reward system" claim substantiated.

---

## Phase 2 — ART-Inspired Training Infrastructure

**Goal**: Replace manual Colab notebook training with a proper async training infrastructure. Adopt ART's best abstractions without taking ART as a dependency.

### 2.1 Trajectory + TrajectoryGroup Data Model

```python
# training/trajectory.py
from dataclasses import dataclass, field
from typing import Union

Message = dict  # {"role": "user"|"assistant"|"system", "content": str}

@dataclass
class Choice:
    """Marks an assistant turn as trainable (gradient flows through this)."""
    content: str
    token_log_probs: list[float] | None = None  # for importance sampling

@dataclass
class Trajectory:
    messages_and_choices: list[Union[Message, Choice]]
    reward: float = 0.0
    metrics: dict = field(default_factory=dict)

    def messages(self) -> list[Message]:
        return [m if isinstance(m, dict) else {"role": "assistant", "content": m.content}
                for m in self.messages_and_choices]

    def trainable_content(self) -> str:
        return " ".join(c.content for c in self.messages_and_choices if isinstance(c, Choice))

@dataclass
class TrajectoryGroup:
    prompt_id: str
    trajectories: list[Trajectory]

    def rewards(self) -> list[float]:
        return [t.reward for t in self.trajectories]
```

**Why this matters**: The `Choice` vs `Message` distinction makes explicit which tokens the policy gradient applies to. Without this, GRPO accidentally optimizes over user messages (context) instead of only assistant responses — a subtle but critical bug in naive implementations.

### 2.2 Async Parallel Rollout Collection

```python
# training/rollout.py
async def gather_trajectory_groups(
    model_client: InferenceClient,
    prompts: list[str],
    n_completions: int = 8,
    rubric: RewardRubric | None = None,
    max_concurrent: int = 16,
) -> list[TrajectoryGroup]:
    """
    For each prompt: generate n_completions in parallel, score as a group.
    max_concurrent limits simultaneous vLLM requests to avoid OOM.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _rollout_group(prompt: str) -> TrajectoryGroup:
        async with semaphore:
            completions = await asyncio.gather(*[
                model_client.chat([{"role": "user", "content": prompt}])
                for _ in range(n_completions)
            ])
            rewards = await compute_reward_group(prompt, completions, rubric)
            return TrajectoryGroup(
                prompt_id=hashlib.md5(prompt.encode()).hexdigest()[:8],
                trajectories=[
                    Trajectory(
                        messages_and_choices=[
                            {"role": "user", "content": prompt},
                            Choice(content=c),
                        ],
                        reward=r,
                    )
                    for c, r in zip(completions, rewards)
                ],
            )

    groups = await asyncio.gather(*[_rollout_group(p) for p in prompts])
    return groups
```

This replaces the current "fetch from DB, format, pass to GRPOTrainer" pattern with a proper async rollout loop that generates live completions. The reward signal applies to *current model behavior*, not old logged responses.

### 2.3 Conversation-to-Prompt Pipeline

This is what ART doesn't have and Aethr's unique contribution.

```python
# training/data_pipeline.py
async def build_prompt_bank(
    db_session,
    limit: int = 500,
    min_reward_variance: float = 0.1,
) -> list[str]:
    """
    Pull real user prompts from conversation history.
    Filter for diversity (deduplicate near-duplicates via embedding cosine similarity).
    Filter for prompts where the model has shown variance (high and low reward responses exist).
    These become the 'scenarios' for GRPO rollouts.
    """
```

The key insight: we don't train directly on logged conversations (ART's mistake would be to do this). We *extract prompts* from logged conversations, then *re-run* those prompts N=8 times against the current model, score the new completions, and train on those. The historical data provides the prompt bank and initial reward calibration. The training data is always freshly generated.

### 2.4 Extracting Module from Notebook

```
training/
  __init__.py
  trajectory.py        # Trajectory, TrajectoryGroup, Choice
  rollout.py           # gather_trajectory_groups
  data_pipeline.py     # build_prompt_bank, conversation mining
  grpo_trainer.py      # GRPOTrainer wrapper (importable, not notebook code)
  distributed.py       # Accelerate config for multi-GPU
  eval_gate.py         # benchmark checks before activating checkpoint
  orchestrator.py      # the automatic flywheel (Phase 3)
```

`grpo_trainer.py` wraps TRL's GRPOTrainer:

```python
class AethrGRPOTrainer:
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model, self.tokenizer = self._load_model(model_config)
        self.config = training_config

    def train(self, trajectory_groups: list[TrajectoryGroup]) -> TrainingResult:
        dataset = self._groups_to_dataset(trajectory_groups)
        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[self._reward_fn],
            args=self._build_grpo_config(),
            train_dataset=dataset,
        )
        trainer.train()
        return TrainingResult(
            step=trainer.state.global_step,
            adapter_path=self._save_adapter(),
        )
```

The Colab/Kaggle notebooks become thin wrappers that call this module — no more business logic in notebook cells.

### 2.5 vLLM Inference Server

Replace the current `model.generate()` + `TextIteratorStreamer` FastAPI server:

```python
# inference/vllm_server.py
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.lora.request import LoRARequest

class VLLMInferenceServer:
    def __init__(self, model_config: ModelConfig):
        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=model_config.hf_id,
                enable_lora=True,
                max_lora_rank=model_config.lora_rank,
                gpu_memory_utilization=model_config.gpu_memory_utilization,
                dtype=model_config.dtype,
                max_model_len=model_config.max_seq_length,
            )
        )
        self.active_lora: LoRARequest | None = None

    async def reload_adapter(self, adapter_path: str):
        """Hot-reload LoRA without restarting the server. Called by orchestrator."""
        self.active_lora = LoRARequest("active", 1, adapter_path)

    async def generate_stream(self, messages, **kwargs):
        sampling_params = SamplingParams(**kwargs)
        async for output in self.engine.generate(
            prompt=self._format(messages),
            sampling_params=sampling_params,
            lora_request=self.active_lora,
        ):
            yield output.outputs[0].text
```

`/reload_adapter` endpoint added to FastAPI. The training orchestrator calls it after a checkpoint passes the eval gate. Zero downtime — vLLM swaps the LoRA while the engine keeps serving.

GPU auto-detection at startup:
```python
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
USE_VLLM = vram_gb >= 20  # A100/H100: vLLM; T4/P100: HF generate
```

---

## Phase 3 — The Automatic Flywheel

**Goal**: Training triggers itself. The developer deploys and forgets.

### 3.1 Training Orchestrator

```python
# training/orchestrator.py
class TrainingOrchestrator:
    BUFFER_TRIGGER = 100      # conversations before training
    MIN_INTERVAL_SEC = 3600   # at most once per hour

    async def run(self):
        while True:
            count = await count_untrained_conversations(self.db)
            if count >= self.BUFFER_TRIGGER and self._interval_ok():
                await self._training_cycle()
            await asyncio.sleep(300)  # check every 5 min

    async def _training_cycle(self):
        # 1. Build prompt bank from recent conversations
        prompts = await build_prompt_bank(self.db, limit=500)

        # 2. Generate rollouts against current model
        groups = await gather_trajectory_groups(
            self.inference_client, prompts, n_completions=8
        )

        # 3. Train
        result = await self.trainer.train(groups)

        # 4. Eval gate
        if await self.eval_gate.passes(result.adapter_path):
            # 5. Push adapter to HF Hub
            await push_adapter(result.adapter_path, self.hf_repo)

            # 6. Hot-reload into inference server
            await self.inference_client.reload_adapter(result.adapter_path)

            # 7. Register checkpoint
            await register_checkpoint(self.db, result)

            logger.info(f"New adapter live: step={result.step}")
        else:
            logger.warning(f"Eval gate failed at step={result.step} — rollback")
```

### 3.2 Eval Gate

Lightweight checks that run before activating a new checkpoint:

```python
# training/eval_gate.py
class EvalGate:
    async def passes(self, adapter_path: str) -> bool:
        checks = await asyncio.gather(
            self._reward_delta_check(adapter_path),  # new > old by margin
            self._sycophancy_check(adapter_path),    # sycophancy rate < 5%
            self._kl_check(adapter_path),            # KL from base < KL_STOP
            self._length_stability_check(adapter_path),
        )
        return all(checks)
```

No MMLU in the gate — too slow and correlates poorly with personal assistant quality. The gate is fast (runs on 50 held-out prompts) and domain-relevant.

### 3.3 Checkpoint Lifecycle

```
                base weights
                    │
           ┌────────┴────────┐
        step-0            step-0 (fork)
           │                  │
        step-1            step-1b (different hyperparams)
           │
        step-2  ←── currently active
           │
        step-3  ←── candidate (in eval gate)
```

`Checkpoint` table already exists in Supabase (`db/models.py`). Add:
- `is_rollback_point: bool` — auto-set on every 5th step
- `eval_gate_scores: dict` — all gate check results
- One-command rollback: `aethr rollback --to step-2`

---

## Phase 4 — Integrations

**Goal**: Telegram refined + Slack added as first-class data source.

### 4.1 Telegram (Refinement)

Already working. Additions:
- Inline reaction buttons (👍 / 👎) on every response — explicit preference signal
- Reaction stored as `user_feedback_score` in `RewardScore` table
- `/rubric` command — lets user view and update their personal rubric in-chat
- `/checkpoint` command — shows current adapter version + reward trend

### 4.2 Slack

```
bot/
  slack_handler.py   # Slack Bolt async app
```

Slack Bolt (official async SDK). Event: `message.channels` + `app_mention`. Same `handle_message` flow as Telegram — `save_message` → `chat_stream` → streaming response via `say()` with token edits.

Slack-specific: capture emoji reactions (`reaction_added` event) as explicit reward signal. A `:+1:` reaction on a message → positive reward update on that assistant turn.

### 4.3 Integration Architecture

```python
# bot/base_handler.py
class BaseAssistantHandler(ABC):
    """Shared logic for all integrations."""

    async def handle_message(self, user_id: str, text: str) -> AsyncIterator[str]:
        conv_id = await self._get_conv(user_id)
        await save_message(conv_id, "user", text)
        history = await get_conversation_history(conv_id)
        messages = build_context(history)
        async for chunk in llm.chat_stream(messages):
            yield chunk

    @abstractmethod
    async def send_streaming(self, chunk: str): ...
    @abstractmethod
    async def send_final(self, text: str): ...
```

Telegram and Slack handlers extend `BaseAssistantHandler`, override only the delivery methods. Adding Discord/WhatsApp/API later = add new handler file, no core logic changes.

---

## Phase 5 — Multi-GPU + Larger Models

**Goal**: Support 14B–32B models with multi-GPU training and inference.

### 5.1 Multi-GPU Training with Accelerate

```python
# training/distributed.py
def build_accelerate_config(n_gpus: int, model_size_b: float) -> dict:
    if n_gpus == 1:
        return {}  # single GPU, no Accelerate needed
    if model_size_b <= 8:
        return {"fsdp_config": {"fsdp_sharding_strategy": "SHARD_GRAD_OP"}}
    if model_size_b <= 32:
        return {"fsdp_config": {"fsdp_sharding_strategy": "FULL_SHARD"}}
    # 70B+
    return {
        "fsdp_config": {"fsdp_sharding_strategy": "FULL_SHARD"},
        "gradient_checkpointing": True,
        "mixed_precision": "bf16",
    }
```

Training launch changes from `python train.py` to `accelerate launch train.py`. GRPOTrainer already supports Accelerate — no changes to the trainer itself, only the launch config.

### 5.2 vLLM Tensor Parallelism for Inference

```python
AsyncEngineArgs(
    tensor_parallel_size=n_gpus,  # auto-shard model across GPUs
    ...
)
```

vLLM handles multi-GPU inference natively. For 14B 4-bit: 1×A100. For 32B 4-bit: 1×A100 80GB or 2×A100 40GB. Config derived from `ModelConfig.gpu_count`.

### 5.3 Model Tier Table

| Model | GPU Requirement | Training Setup | Inference Setup | Target User |
|---|---|---|---|---|
| Qwen3-0.6B | Colab T4 free | 1× T4, HF generate | 1× T4, HF generate | Hobbyist demo |
| Qwen3-1.7B | Colab T4 free | 1× T4, Unsloth | 1× T4, vLLM | Hobbyist |
| Qwen3-8B 4-bit | Colab A100 / Kaggle | 1× A100, Unsloth | 1× A100, vLLM | Developer |
| Qwen3-14B 4-bit | RunPod A100 40GB | 1× A100, Unsloth+Accelerate | 1× A100, vLLM | Developer+ |
| Qwen3-32B 4-bit | RunPod 2×A100 | 2× A100, FSDP | 2× A100, vLLM TP=2 | Production |

---

## Phase 6 — Personalization (FSPO)

**Goal**: Model adapts to individual user preferences from minimal explicit feedback.

### 6.1 Preference Collection UI

After the first 50 conversations, the bot prompts:
> "I've been learning as we talk. Want to help me get better? I'll show you 4 pairs of responses — just pick which you prefer. Takes ~2 minutes."

4–8 preference pairs → stored as `UserPreference` records.

### 6.2 User Profile Derivation (RAT)

```python
# rewards/personalization.py
async def derive_user_profile(preferences: list[UserPreference]) -> str:
    """Use LLM to infer a natural-language user profile from preference pairs."""
    prompt = f"""
Given these response preferences from a user, describe their communication style in 3-4 sentences.
Focus on: tone (formal/informal), length preference, what they value (directness/detail/humor), 
what they dislike (hedging/caveats/verbosity).

Preferences:
{format_preferences(preferences)}

User profile:"""
    return await llm_judge_call(prompt)
```

Profile example: *"This user prefers short, direct responses without unnecessary caveats. They use informal language and appreciate concrete examples over general principles. They dislike responses that hedge with 'it depends' without giving a concrete answer. They value actionable outputs."*

### 6.3 Rubric Injection

User profile is prepended to the Prometheus rubric automatically. No user action required after the initial preference collection.

```python
def personalize_rubric(base_rubric: RewardRubric, user_profile: str) -> RewardRubric:
    return RewardRubric(
        criteria=f"{user_profile}\n\n{base_rubric.criteria}",
        good_examples=base_rubric.good_examples,
        bad_examples=base_rubric.bad_examples,
        hard_rules=base_rubric.hard_rules,
    )
```

---

## Phase 7 — Benchmarking + Public Numbers

**Goal**: Substantiate every claim with reproducible numbers.

### Benchmark Suite

**B1 — Reward Model Validity** (Phase 1 gate)
- 100 personal assistant response pairs, GPT-4o + 3 human labels
- Metric: agreement rate with human majority vote
- Target: ≥ 75%

**B2 — Reward Variance Adequacy** (Phase 1 gate)
- 8 completions × 50 diverse prompts
- Metric: per-group std pre-normalization
- Target: mean std ≥ 0.12

**B3 — Hacking Resistance** (Phase 1 gate)
- 20 sycophantic vs 20 equivalent non-sycophantic responses
- Metric: score gap
- Target: sycophantic ≥ 0.2 lower

**B4 — GRPO Training Efficacy** (Phase 3 gate)
- 200 WildBench personal assistant prompts (planning, writing, Q&A)
- GPT-4o pairwise judge: post-training vs base model
- Metric: win rate
- Target: ≥ 60% win rate

**B5 — Reward Hacking Over Training** (Phase 3 continuous)
- Track sycophancy rate, mean response length, markdown density on held-out set over training steps
- Target: all three stable (not monotonically increasing) through 100 training steps

**B6 — Personalization Gain** (Phase 6 gate)
- 5 users × 20 prompts, personalized vs non-personalized responses, user rates which they prefer
- Metric: win rate for personalized
- Target: ≥ 65%

All benchmarks run automatically in CI and results published in `benchmarks/results/`.

---

## Deployment Story

### Hobbyist (Free)
1. Open Colab notebook (one click from README)
2. Add 3 secrets: `HF_TOKEN`, `TELEGRAM_TOKEN`, `NGROK_AUTH_TOKEN`
3. Run all cells — bot is live in ~5 minutes
4. Training triggers automatically when 100 conversations accumulate (overnight)

### Developer (VPS / RunPod)
1. `git clone + pip install`
2. Fill `.env` (6 variables)
3. `docker-compose up` — inference server + bot + reward worker + orchestrator
4. Point `INFERENCE_URL` at the container

### Production (Multi-GPU)
1. Same as Developer, set `MODEL_NAME=qwen3-32b-4bit`, `GPU_COUNT=2`
2. `accelerate launch` for training
3. vLLM with `tensor_parallel_size=2` for inference

Dockerfile generated in Phase 2. `docker-compose.yml` with services: `inference`, `bot`, `reward_worker`, `orchestrator`.

---

## Milestones

| Phase | Deliverable | Gate |
|---|---|---|
| 0 | Model abstraction, SQLite fallback, plugin auto-discovery | Single `.env` → working bot on Colab T4 |
| 1 | 3-tier reward pipeline, WCO ensemble, variance normalization | B1+B2+B3 benchmarks pass |
| 2 | Trajectory abstraction, async rollouts, vLLM server, training module | Manual training run on A100 produces measurably better model |
| 3 | Automatic flywheel, eval gate, hot-reload | 24-hour run with 3+ automatic training cycles |
| 4 | Slack integration, reaction-as-reward, base handler abstraction | Slack bot works end-to-end |
| 5 | Multi-GPU training + inference, model tier table | Qwen3-32B training run on 2×A100 |
| 6 | FSPO personalization, user profile derivation | B6 benchmark: ≥ 65% personalization win rate |
| 7 | Full benchmark suite, CI automation, published results | B4+B5+B6 pass, results in `benchmarks/results/` |

---

## What We Are Not Building

- A coding agent trainer (ART does this well)
- A math/reasoning verifier (PrimeIntellect does this well)
- A managed cloud platform (not our moat — we are the framework)
- A reward plugin ecosystem with 10+ community packages (1-2 tiers is correct)
- A custom training kernel (Unsloth handles this)
- A general-purpose fine-tuning tool (we are RL-only, self-improvement from usage only)

---

## Key Files to Create (in order)

```
inference/models/base.py                 # Phase 0
inference/models/unsloth.py              # Phase 0
inference/models/vllm_loader.py          # Phase 2
inference/model_registry.py             # Phase 0
rewards/loader.py                       # Phase 0
rewards/models/skywork_v2.py            # Phase 1
rewards/models/rewardanything.py        # Phase 1
rewards/rubric.py                       # Phase 1 (RewardRubric dataclass)
training/trajectory.py                  # Phase 2
training/rollout.py                     # Phase 2
training/data_pipeline.py              # Phase 2
training/grpo_trainer.py               # Phase 2
training/distributed.py                # Phase 5
training/eval_gate.py                  # Phase 3
training/orchestrator.py               # Phase 3
bot/base_handler.py                    # Phase 4
bot/slack_handler.py                   # Phase 4
rewards/personalization.py             # Phase 6
benchmarks/run_all.py                  # Phase 7
docker-compose.yml                     # Phase 2
Dockerfile                             # Phase 2
```
