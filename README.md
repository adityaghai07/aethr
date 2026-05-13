# Aethr

A framework for building self-improving AI assistants that learn from every conversation using online reinforcement learning (GRPO). No static datasets — the model scores its own responses and trains on the live conversation stream.

> Built as the backbone for [Vithos](https://vithos.in), but designed to be domain-agnostic. Swap in your own reward plugins and system prompt for any use case.

## How it works

```
User (Telegram)
    │
    ▼
Bot + Safety Filter (local machine)
    │  OpenAI-compatible API
    ▼
Inference Notebook (Colab / Kaggle, ngrok tunnel)
    └─ Unsloth + Qwen3-8B-4bit + active LoRA adapter

    [async, after response is sent]
    ▼
Reward Worker (local machine)
    ├─ RuleBasedPlugin   — instant, free
    ├─ YourCustomPlugin  — add any domain-specific signals
    └─ LLMJudgePlugin    — Claude/GPT judge, ~$0.002/call
    │
    ▼
Supabase (PostgreSQL)
    └─ conversations, messages, reward_scores, training_examples

    [when buffer hits 64 scored examples]
    ▼
Training Notebook (Colab / Kaggle)
    └─ GRPO → new LoRA adapter → pushed to HF Hub
    ▼
Inference Notebook loads new adapter → loop repeats
```

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/adityaghai07/aethr
cd aethr
pip install -r requirements.txt
cp .env.example .env
# Fill in: TELEGRAM_TOKEN, DATABASE_URL, HF_TOKEN, HF_ADAPTER_REPO, JUDGE_API_KEY
```

### 2. Database (Supabase — free tier)

1. Create a project at [supabase.com](https://supabase.com)
2. SQL Editor → run `db/migrations/001_initial.sql`
3. Paste the connection string into `DATABASE_URL` in `.env`

### 3. HuggingFace Hub (adapter storage)

Create a private model repo at [huggingface.co/new](https://huggingface.co/new) — e.g. `your-username/aethr-adapters`. Set `HF_ADAPTER_REPO` in `.env`.

### 4. Inference notebook

Open `notebooks/kaggle_inference.py` in Colab or Kaggle (T4 GPU). Fill in Cell 2 with your tokens. Run all cells — Cell 6 prints the ngrok URL.

### 5. Bot + Reward Worker

```bash
# Terminal 1
python -m bot.telegram_handler

# Terminal 2
python -m rewards.worker
```

Send `/seturl <ngrok-url>` to your bot, then `/health` to confirm it's connected.

## Reward Plugins

The reward stack is fully pluggable. Configure it in `config.py → ACTIVE_PLUGIN_NAMES`.

| Plugin | Weight | What it rewards |
|---|---|---|
| `rule_based` | 0.25 | Length appropriateness, format, language match — instant, free |
| `llm_judge` | 0.50 | 5-dimension rubric scoring via external LLM — most informative signal |
| `medical_guardrails` | 0.40 | Example domain plugin — ships with the repo (see below) |

**Adding your own plugin:**
1. Create `rewards/plugins/your_plugin.py`, extend `RewardPlugin`, implement `score()`
2. Call `register(YourPlugin())` at the bottom
3. Add the name to `ACTIVE_PLUGIN_NAMES` in `config.py`

See `rewards/plugins/example_coding.py` for a minimal template.

### Bundled domain plugin: Medical Guardrails

`rewards/plugins/medical.py` ships as an example of a domain-specific plugin. It enforces a "observe and contextualize, never diagnose" contract:

| Signal | Reward | Effect |
|---|---|---|
| Diagnosis / treatment language | −1.0 + `violated=True` | Response hard-blocked, regenerated |
| Alarming language | −0.6 + `violated=True` | Hard-blocked |
| Legalistic AI boilerplate | −0.4 | Penalized in training |
| Human-sounding disclaimer | +0.2 | Rewarded |
| Calm, contextualizing language | +0.3 | Rewarded |
| Doctor redirect on clinical query | +0.2 | Rewarded |

Hard violations are caught by `bot/safety.py` before delivery. The bot retries with a stricter prompt; after 2 failed retries it sends a safe fallback. Remove this plugin from `ACTIVE_PLUGIN_NAMES` if your use case doesn't need it.

## Build Phases

| Phase | What | Status |
|---|---|---|
| 0 | Inference notebook (Colab/Kaggle) | ✓ |
| 1 | Inference client + health checks | ✓ |
| 2 | Telegram bot + Supabase logging | ✓ |
| 3 | Reward plugin system | ✓ |
| 4 | Offline GRPO training notebook | ✓ |
| 5 | Online RL loop (buffer → train → eval-gate) | — |
| 6 | Anti-forgetting + eval suite | — |
| 7 | Gradio dashboard + wandb | — |

## Stack

- **Model:** Qwen3-8B (4-bit via Unsloth) — swap for any HF model
- **Training:** TRL GRPOTrainer + Unsloth
- **Inference:** HuggingFace generate (Colab T4 compatible)
- **Database:** Supabase (PostgreSQL)
- **Adapter storage:** HuggingFace Hub (versioned, rollback by commit hash)
- **Observability:** wandb
- **Bot:** python-telegram-bot v21
