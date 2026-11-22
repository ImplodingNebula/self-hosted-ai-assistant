"""initial multi-tenant schema.

Revision ID: 5022b7cb3271
Revises:
Create Date: 2026-07-04 17:52:06.684365
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel  # noqa: F401  - autogenerate renders sqlmodel.sql.sqltypes.* columns

from alembic import op

revision: str = "5022b7cb3271"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this revision."""
    op.create_table(
        "collections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_collections_tenant_id"), "collections", ["tenant_id"], unique=False)
    op.create_table(
        "conversations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("title", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("settings", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_conversations_tenant_updated",
        "conversations",
        ["tenant_id", "updated_at"],
        unique=False,
    )
    op.create_table(
        "mcp_servers",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("base_url", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "name", name="uq_mcp_servers_tenant_name"),
    )
    op.create_table(
        "secrets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("last4", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "name", name="uq_secrets_tenant_name"),
    )
    op.create_table(
        "tenants",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "attachments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=True),
        sa.Column("filename", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("content_type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("storage_key", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("status", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_attachments_tenant_id"), "attachments", ["tenant_id"], unique=False)
    op.create_table(
        "messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("role", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("thinking", sa.Text(), nullable=True),
        sa.Column("model", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("meta", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_messages_conversation_created",
        "messages",
        ["conversation_id", "created_at"],
        unique=False,
    )
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("email", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("display_name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("role", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_tenant_id"), "users", ["tenant_id"], unique=False)
    op.create_table(
        "workspace_settings",
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("settings", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
        ),
        sa.PrimaryKeyConstraint("tenant_id"),
    )


def downgrade() -> None:
    """Revert this revision."""
    op.drop_table("workspace_settings")
    op.drop_index(op.f("ix_users_tenant_id"), table_name="users")
    op.drop_table("users")
    op.drop_index("ix_messages_conversation_created", table_name="messages")
    op.drop_table("messages")
    op.drop_index(op.f("ix_attachments_tenant_id"), table_name="attachments")
    op.drop_table("attachments")
    op.drop_table("tenants")
    op.drop_table("secrets")
    op.drop_table("mcp_servers")
    op.drop_index("ix_conversations_tenant_updated", table_name="conversations")
    op.drop_table("conversations")
    op.drop_index(op.f("ix_collections_tenant_id"), table_name="collections")
    op.drop_table("collections")
