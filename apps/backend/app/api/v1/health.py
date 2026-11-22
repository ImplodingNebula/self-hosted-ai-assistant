"""Liveness and readiness probes.

Both are unauthenticated and mounted at the root rather than under ``/api/v1``:
an orchestrator's probe config outlives any API version, and the two answer
questions about the PROCESS, not about the API surface.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.deps import get_job_queue, get_model_provider, get_sessionmaker, get_vector_store
from app.core.config import Settings, get_settings
from app.providers.base import ModelProvider
from app.providers.rag.base import VectorStore
from app.schemas.health import ReadinessReport
from app.services.readiness import check_readiness
from app.workers.queue import JobQueue

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@router.get("/readyz", response_model=ReadinessReport)
async def readyz(
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
    job_queue: Annotated[JobQueue, Depends(get_job_queue)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    model_provider: Annotated[ModelProvider, Depends(get_model_provider)],
) -> ReadinessReport:
    """Readiness probe: report every backing service, and 503 if a required one is down.

    Answers 200 when the database, the job queue and the vector store are all
    reachable, and 503 otherwise. The model server, the search sidecar and the
    reranker are reported for visibility but never change the status code:
    the model server is routinely remote and slow to load, and the other two
    are optional by design, so a failure there degrades the app rather than
    disabling it.
    """
    report = await check_readiness(
        sessionmaker=sessionmaker,
        job_queue=job_queue,
        vector_store=vector_store,
        model_provider=model_provider,
        # A deployment that switched web search off has no search sidecar to
        # report on, whatever the URL still says - so ask the same question the
        # factory asks, not just whether a URL is present.
        searxng_url=settings.searxng_url if settings.search_provider != "none" else None,
        reranker_url=settings.reranker_url,
    )
    if report.status != "ready":
        response.status_code = 503
    return report
