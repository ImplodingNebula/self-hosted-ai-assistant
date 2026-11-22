"""Alembic environment: async engine, metadata from the SQLModel tables."""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

import app.models  # noqa: F401  - importing registers every table on SQLModel.metadata
from alembic import context
from app.core.config import get_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata


def _database_url() -> str:
    """Resolve the database URL: explicit DATABASE_URL env wins, then app settings.

    The explicit env check keeps migration behavior obvious even though the
    settings object would honor the same variable - migrations are exactly the
    place where "which database?" must never be a surprise.
    """
    return os.environ.get("DATABASE_URL") or get_settings().database_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode: emit SQL to stdout, no DB connection."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def _run_async_migrations() -> None:
    engine = create_async_engine(_database_url(), poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    """Run migrations against the live database via the async driver."""
    asyncio.run(_run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
