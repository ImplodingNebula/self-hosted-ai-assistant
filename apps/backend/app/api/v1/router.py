"""Aggregates the versioned (/api/v1) routers."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import (
    attachments,
    auth,
    chat,
    conversations,
    integrations,
    knowledge,
    members,
    models,
    rag,
    workspace,
)

router = APIRouter(prefix="/api/v1")
router.include_router(attachments.router)
router.include_router(auth.router)
router.include_router(chat.router)
router.include_router(conversations.router)
router.include_router(integrations.router)
router.include_router(knowledge.router)
router.include_router(members.router)
router.include_router(models.router)
router.include_router(rag.router)
router.include_router(workspace.router)
