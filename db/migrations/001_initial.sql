-- Run this once in the Supabase SQL editor (or via psql).
-- Creates all tables for Aethr.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- for gen_random_uuid()

-- ── Conversations ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS conversations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now(),
    metadata    JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);

-- ── Messages ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS messages (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id     UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role                TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content             TEXT NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT now(),
    tokens_used         INTEGER,
    generation_params   JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);

-- ── Reward Scores ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS reward_scores (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id              UUID REFERENCES messages(id) ON DELETE CASCADE UNIQUE,
    helpfulness             FLOAT,
    tone_match              FLOAT,
    conciseness             FLOAT,
    factuality              FLOAT,
    instruction_following   FLOAT,
    composite_score         FLOAT NOT NULL,
    rule_based_scores       JSONB DEFAULT '{}',
    llm_judge_scores        JSONB DEFAULT '{}',
    user_feedback_scores    JSONB DEFAULT '{}',
    scored_at               TIMESTAMPTZ DEFAULT now(),
    judge_model             TEXT,
    judge_cost_usd          FLOAT DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_reward_scores_composite ON reward_scores(composite_score);

-- ── Training Examples ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS training_examples (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt          TEXT NOT NULL,
    completions     JSONB NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now(),
    used_in_step    INTEGER,
    source          TEXT DEFAULT 'live'
);

CREATE INDEX IF NOT EXISTS idx_training_examples_unused
    ON training_examples(used_in_step)
    WHERE used_in_step IS NULL;

-- ── Checkpoints ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS checkpoints (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    step            INTEGER NOT NULL,
    hf_repo         TEXT NOT NULL,
    hf_revision     TEXT NOT NULL,
    base_model      TEXT NOT NULL,
    eval_scores     JSONB NOT NULL,
    is_active       BOOLEAN DEFAULT FALSE,
    is_merged       BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT now(),
    training_config JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_active
    ON checkpoints(is_active)
    WHERE is_active = TRUE;

-- ── Eval Runs ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS eval_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    checkpoint_id   UUID REFERENCES checkpoints(id),
    benchmark       TEXT NOT NULL,
    score           FLOAT NOT NULL,
    details         JSONB DEFAULT '{}',
    run_at          TIMESTAMPTZ DEFAULT now()
);
