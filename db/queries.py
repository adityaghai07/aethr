"""
Async query functions used across the bot, reward worker, and training pipeline.
All functions accept an optional `session` — pass one from get_db() context.
"""
import uuid
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from db.connection import get_db
from db.models import Conversation, Message, RewardScore, TrainingExample, Checkpoint


# ── Conversations ─────────────────────────────────────────────────────────────

async def create_conversation(user_id: str) -> Conversation:
    async with get_db() as session:
        conv = Conversation(user_id=user_id)
        session.add(conv)
        await session.commit()
        await session.refresh(conv)
        return conv


# ── Messages ──────────────────────────────────────────────────────────────────

async def save_message(
    conversation_id: uuid.UUID,
    role: str,
    content: str,
    tokens_used: int | None = None,
    generation_params: dict | None = None,
) -> Message:
    async with get_db() as session:
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tokens_used=tokens_used,
            generation_params=generation_params or {},
        )
        session.add(msg)
        await session.commit()
        await session.refresh(msg)
        return msg


async def get_conversation_history(
    conversation_id: uuid.UUID,
    limit: int = 40,
) -> list[dict]:
    """Return last N messages as {role, content} dicts for LLM context."""
    async with get_db() as session:
        result = await session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        messages = result.scalars().all()
        return [{"role": m.role, "content": m.content} for m in reversed(messages)]


# ── Reward Scores ─────────────────────────────────────────────────────────────

async def get_unscored_messages(limit: int = 20) -> list[Message]:
    """Assistant messages that haven't been scored yet."""
    async with get_db() as session:
        result = await session.execute(
            select(Message)
            .outerjoin(RewardScore, Message.id == RewardScore.message_id)
            .where(
                and_(
                    Message.role == "assistant",
                    RewardScore.id.is_(None),
                )
            )
            .limit(limit)
        )
        return result.scalars().all()


async def save_reward_score(
    message_id: uuid.UUID,
    composite_score: float,
    rule_based_scores: dict,
    llm_judge_scores: dict,
    user_feedback_scores: dict,
    judge_model: str | None = None,
    judge_cost_usd: float = 0.0,
    **dimension_scores,
) -> RewardScore:
    async with get_db() as session:
        score = RewardScore(
            message_id=message_id,
            composite_score=composite_score,
            rule_based_scores=rule_based_scores,
            llm_judge_scores=llm_judge_scores,
            user_feedback_scores=user_feedback_scores,
            judge_model=judge_model,
            judge_cost_usd=judge_cost_usd,
            **dimension_scores,
        )
        session.add(score)
        await session.commit()
        await session.refresh(score)
        return score


async def get_message_context(message_id: uuid.UUID) -> dict:
    """Return the conversation context needed for reward scoring."""
    async with get_db() as session:
        msg = await session.get(Message, message_id)
        history = await get_conversation_history(msg.conversation_id, limit=10)

        # The last user message before this assistant response
        last_user = next(
            (m for m in reversed(history) if m["role"] == "user"), {"content": ""}
        )
        return {
            "message": msg,
            "history": history,
            "last_user_message": last_user["content"],
        }


# ── Training ──────────────────────────────────────────────────────────────────

async def get_training_prompts(limit: int = 50) -> list[list[dict]]:
    """
    Pull recent conversation histories suitable for GRPO rollouts.
    Returns a list of conversation histories — each is [{role, content}].
    Picks conversations with at least one scored assistant turn (quality signal exists).
    """
    async with get_db() as session:
        result = await session.execute(
            select(Message.conversation_id)
            .join(RewardScore, Message.id == RewardScore.message_id)
            .where(Message.role == "assistant")
            .group_by(Message.conversation_id)
            .order_by(func.max(Message.created_at).desc())
            .limit(limit)
        )
        conv_ids = [row[0] for row in result.all()]

    histories = []
    for conv_id in conv_ids:
        history = await get_conversation_history(conv_id, limit=20)
        if len(history) >= 2:   # need at least one user+assistant exchange
            # Return history up to but not including the last assistant turn
            # (so rollout generates fresh completions for the last user prompt)
            user_turns = [i for i, m in enumerate(history) if m["role"] == "user"]
            if user_turns:
                histories.append(history[:user_turns[-1] + 1])
    return histories


async def get_recent_scored_examples(limit: int = 64) -> list[TrainingExample]:
    async with get_db() as session:
        result = await session.execute(
            select(TrainingExample)
            .where(TrainingExample.used_in_step.is_(None))
            .order_by(TrainingExample.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()


async def mark_examples_used(example_ids: list[uuid.UUID], step: int) -> None:
    from sqlalchemy import update
    async with get_db() as session:
        await session.execute(
            update(TrainingExample)
            .where(TrainingExample.id.in_(example_ids))
            .values(used_in_step=step)
        )
        await session.commit()


# ── Checkpoints ───────────────────────────────────────────────────────────────

async def get_active_checkpoint() -> Checkpoint | None:
    async with get_db() as session:
        result = await session.execute(
            select(Checkpoint).where(Checkpoint.is_active.is_(True))
        )
        return result.scalar_one_or_none()


async def register_checkpoint(
    step: int,
    hf_repo: str,
    hf_revision: str,
    base_model: str,
    eval_scores: dict,
    training_config: dict,
) -> Checkpoint:
    async with get_db() as session:
        ckpt = Checkpoint(
            step=step,
            hf_repo=hf_repo,
            hf_revision=hf_revision,
            base_model=base_model,
            eval_scores=eval_scores,
            training_config=training_config,
        )
        session.add(ckpt)
        await session.commit()
        await session.refresh(ckpt)
        return ckpt
