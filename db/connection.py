"""
Async SQLAlchemy engine connected to Supabase (PostgreSQL).
All DB access goes through get_db() — a session factory used as an async context manager.
"""
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from config import DATABASE_URL

engine = create_async_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,   # detect stale connections before using them
    echo=False,
    # Supabase's transaction-mode pooler (port 6543) doesn't support
    # prepared statements. Disable asyncpg's statement cache.
    connect_args={
        "statement_cache_size": 0,
        "prepared_statement_cache_size": 0,
    },
)

get_db = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass
