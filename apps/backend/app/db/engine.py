"""Async engine and session factory construction.

Built once per process in the app lifespan and shared via ``app.state``;
tests build their own against in-memory sqlite. Nothing here opens a
connection - SQLAlchemy connects lazily on first use, which keeps boot
resilient when the database is still warming up.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings


def build_engine(settings: Settings) -> AsyncEngine:
    """Create the process-wide async engine from application settings."""
    return create_async_engine(
        settings.database_url,
        echo=settings.database_echo,
        # Validate pooled connections before handing them out: postgres restarts
        # (or docker pauses) would otherwise surface as mid-request failures.
        pool_pre_ping=True,
    )


def build_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create the session factory bound to ``engine``.

    ``expire_on_commit=False`` keeps ORM objects readable after commit -
    required because responses are serialized from entities after the
    transaction ends (an expired attribute would trigger a sync lazy-load).
    """
    return async_sessionmaker(engine, expire_on_commit=False)
