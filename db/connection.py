"""
Async SQLAlchemy engine — supports both Supabase (PostgreSQL) and SQLite.

PostgreSQL (default): set DATABASE_URL to a postgresql+asyncpg:// URL.
SQLite (hobbyist/dev): set DATABASE_URL=sqlite+aiosqlite:///./aethr.db

The _is_sqlite flag is exported so db/models.py can pick the right column types.
"""
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from config import DATABASE_URL

_is_sqlite: bool = DATABASE_URL.startswith("sqlite")

if _is_sqlite:
    # aiosqlite needs check_same_thread=False; no connection pool for SQLite
    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        connect_args={"check_same_thread": False},
    )
else:
    # Supabase transaction-mode pooler (port 6543) doesn't support prepared
    # statements — disable asyncpg's statement cache.
    engine = create_async_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=False,
        connect_args={
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
        },
    )

get_db = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass
