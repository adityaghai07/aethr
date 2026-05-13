"""
SQLAlchemy ORM models — maps directly to the Supabase PostgreSQL schema.
Run db/migrations/001_initial.sql once in the Supabase SQL editor to create the tables.
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    String, Text, Float, Integer, Boolean,
    ForeignKey, DateTime, UniqueConstraint, Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from db.connection import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    messages: Mapped[list["Message"]] = relationship(back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)   # user | assistant | system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    tokens_used: Mapped[int | None] = mapped_column(Integer)
    generation_params: Mapped[dict] = mapped_column(JSONB, default=dict)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
    reward_score: Mapped["RewardScore | None"] = relationship(back_populates="message")

    __table_args__ = (
        Index("idx_messages_conversation", "conversation_id", "created_at"),
    )


class RewardScore(Base):
    __tablename__ = "reward_scores"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("messages.id"), unique=True, nullable=False)
    # Dimension scores (0.0 – 1.0)
    helpfulness: Mapped[float | None] = mapped_column(Float)
    tone_match: Mapped[float | None] = mapped_column(Float)
    conciseness: Mapped[float | None] = mapped_column(Float)
    factuality: Mapped[float | None] = mapped_column(Float)
    instruction_following: Mapped[float | None] = mapped_column(Float)
    # Composite (−1.0 to 1.0, centered at 0)
    composite_score: Mapped[float] = mapped_column(Float, nullable=False)
    # Raw scores from each tier
    rule_based_scores: Mapped[dict] = mapped_column(JSONB, default=dict)
    llm_judge_scores: Mapped[dict] = mapped_column(JSONB, default=dict)
    user_feedback_scores: Mapped[dict] = mapped_column(JSONB, default=dict)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    judge_model: Mapped[str | None] = mapped_column(String)
    judge_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    message: Mapped["Message"] = relationship(back_populates="reward_score")

    __table_args__ = (
        Index("idx_reward_scores_composite", "composite_score"),
    )


class TrainingExample(Base):
    __tablename__ = "training_examples"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    completions: Mapped[list] = mapped_column(JSONB, nullable=False)  # [{text, reward}]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    used_in_step: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String, default="live")  # live | replay | synthetic

    __table_args__ = (
        Index("idx_training_examples_unused", "used_in_step",
              postgresql_where="used_in_step IS NULL"),
    )


class Checkpoint(Base):
    __tablename__ = "checkpoints"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    # HF Hub repo + revision where the adapter lives
    hf_repo: Mapped[str] = mapped_column(String, nullable=False)
    hf_revision: Mapped[str] = mapped_column(String, nullable=False)
    base_model: Mapped[str] = mapped_column(String, nullable=False)
    eval_scores: Mapped[dict] = mapped_column(JSONB, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    is_merged: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    training_config: Mapped[dict] = mapped_column(JSONB, default=dict)

    __table_args__ = (
        Index("idx_checkpoints_active", "is_active", postgresql_where="is_active = TRUE"),
    )


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    checkpoint_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("checkpoints.id"), nullable=False)
    benchmark: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    details: Mapped[dict] = mapped_column(JSONB, default=dict)
    run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
