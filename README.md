# Aethr

A self-improving AI assistant that learns from every conversation using online reinforcement learning (GRPO). Built for [Vithos](https://vithos.app) — a medical lab-data companion that observes and contextualizes, never diagnoses.

## Architecture

```
User (Telegram)
    │
    ▼
Bot + Safety Filter (local machine)
    │  OpenAI-compatible API
    ▼
Kaggle Inference Notebook A (ngrok tunnel)
    └─ Unsloth + vLLM + Qwen3-8B + active LoRA adapter

    [async, after response sent]
    ▼
Reward Worker (local machine)
    ├─ RuleBasedPlugin        — instant, free
    ├─ MedicalGuardrailPlugin — Vithos guardrails
    └─ LLMJudgePlugin         — Claude Sonnet judge, ~$0.002/call
    │
    ▼
Supabase (PostgreSQL)
    └─ conversations, messages, reward_scores, training_examples

    [when buffer hits 64 scored examples]
    ▼
Kaggle Training Notebook B
    └─ GRPO → new LoRA adapter → pushed to HF Hub
    ▼
Inference Notebook A loads new adapter on next restart
    └─ Loop repeats — model improves with every conversation
```

## Quick Start

### 1. Setup

```bash
git clone https://github.com/your-username/aethr
cd aethr
pip install -r requirements.txt

cp .env.example .env
# Fill in: TELEGRAM_TOKEN, DATABASE_URL, HF_TOKEN, HF_ADAPTER_REPO, JUDGE_API_KEY
```

### 2. Database (Supabase)

1. Create a free project at [supabase.com](https://supabase.com)
2. SQL Editor → run `db/migrations/001_initial.sql`
3. Paste the connection string into `DATABASE_URL` in `.env`

### 3. Inference (Kaggle Notebook A)

1. Open `notebooks/kaggle_inference.py` as a new Kaggle notebook (GPU accelerator on)
2. Add secrets as Kaggle secrets: `HF_TOKEN`, `NGROK_AUTH_TOKEN`, `HF_ADAPTER_REPO`
3. Run all cells — copy the ngrok URL it prints

### 4. Bot + Reward Worker

```bash
# Terminal 1 — Telegram bot
python -m bot.telegram_handler

# Terminal 2 — background reward scorer
python -m rewards.worker
```

Send `/seturl <ngrok-url>` to your bot, then `/health` to verify the connection.

## Reward Plugins

Configured in `config.py → ACTIVE_PLUGIN_NAMES`. Currently active:

| Plugin | Weight | Purpose |
|---|---|---|
| `rule_based` | 0.25 | Length, format, language — instant, free |
| `medical_guardrails` | 0.40 | Vithos guardrails — penalizes diagnosis/alarm/treatment language |
| `llm_judge` | 0.50 | Claude Sonnet rubric scoring — most informative |

**To add a custom plugin:** extend `RewardPlugin` in `rewards/plugins/`, implement `score()`, call `register()`, add name to `ACTIVE_PLUGIN_NAMES` in `config.py`. See `rewards/plugins/example_coding.py` for a template.

### Medical Guardrails (`rewards/plugins/medical.py`)

| Signal | Reward | Action |
|---|---|---|
| Diagnosis language | −1.0 + `violated=True` | Hard-blocked, response regenerated |
| Treatment recommendation | −1.0 + `violated=True` | Hard-blocked |
| Alarming language | −0.6 + `violated=True` | Hard-blocked |
| Legalistic AI disclaimer | −0.4 | Penalized (shapes training) |
| Human disclaimer | +0.2 | Rewarded |
| Calm contextualizing language | +0.3 | Rewarded |
| Doctor redirect on clinical query | +0.2 | Rewarded |

Hard violations (`violated=True`) are caught by `bot/safety.py` before the message reaches the user. The bot retries with a stricter system prompt; after 2 failed retries it sends a safe fallback message.

## Build Phases

| Phase | What | Status |
|---|---|---|
| 0 | Kaggle inference server (Notebook A) | ✓ |
| 1 | Inference client + health checks | ✓ |
| 2 | Telegram bot + Supabase logging | ✓ |
| 3 | Reward system (all plugins) | ✓ |
| 4 | Offline GRPO training (Notebook B) | ✓ |
| 5 | Online RL loop (buffer → train → eval-gate) | — |
| 6 | Anti-forgetting + eval suite | — |
| 7 | Gradio dashboard + wandb | — |

## Guardrails (Vithos)

The AI observes and contextualizes. It never diagnoses, prescribes, or recommends stopping/starting treatment.

✓ *"Your LDL is 142 mg/dL — above the 130 threshold and rising across your last 3 tests. Worth mentioning at your next checkup."*

✗ *"DANGER: Your LDL is critically elevated."*
✗ *"You have hyperlipidemia."*
✗ *"You should start a statin."*

Disclaimer sounds human: *"These are observations from your data, not a diagnosis. Your doctor has the final word."*
