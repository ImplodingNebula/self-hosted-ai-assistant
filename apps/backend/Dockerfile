# syntax=docker/dockerfile:1
# Multi-stage build for the FastAPI backend. `dev` target hot-reloads; `prod` is lean.
FROM python:3.13-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy
# Tesseract: the OCR engine that reads scanned PDF pages (a system binary, not a
# pip package). Installed in the base so every stage has it - the worker (prod
# target) is where files are actually parsed, and OCR degrades gracefully to
# text-layer-only if it is ever absent. English data only; add tesseract-ocr-<lang>
# packages to widen OCR_LANGUAGE.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Resolve dependencies from the committed lockfile (cached layer).
FROM base AS deps
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

FROM base AS dev
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY . .
EXPOSE 8000
# Migrations run before the server so the schema is always current when the
# API starts accepting requests; `exec` keeps uvicorn as PID 1 for signals.
# The dev compose overlay (infra/docker-compose.override.yml) repeats this
# command so stale local images still honor it - keep the two in sync.
CMD ["sh", "-c", "uv run alembic upgrade head && exec uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"]

FROM base AS prod
COPY --from=deps /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
# Make `uv run` work for the unprivileged user. The server itself never calls uv
# (it runs the baked venv straight off PATH), but every operational command does
# - migrations, the perf smoke test, the tool check, and the warm and status
# probes, which in a deployment can only reach the API from inside this
# container - and each hit a different permission wall, so BOTH settings are
# load-bearing:
#
#   UV_CACHE_DIR  uv keeps its cache under $HOME, and the user below is created
#     with --no-create-home, so the default path cannot be created at all.
#   UV_NO_SYNC    the venv copied in above was resolved with --no-dev, so `uv
#     run` sees the dev group missing and tries to install it into /app/.venv -
#     which is root-owned, and must stay that way. The image's environment is
#     already complete and immutable by design; there is nothing to sync.
ENV UV_CACHE_DIR=/tmp/uv-cache \
    UV_NO_SYNC=1
COPY . .
# Drop root: the server only reads /app (root-owned, world-readable) and writes
# nothing inside the image, so a passwordless system user is enough.
#
# Attachment storage is the one path it must write. The directory is created
# here, owned by appuser, because Docker seeds an EMPTY named volume from the
# image's mountpoint - ownership included. Without this the volume would be
# created root-owned and every upload would fail with a permission error that
# no test catches, since the dev stage runs as root.
RUN useradd --system --no-create-home appuser \
    && mkdir -p /data/storage \
    && chown appuser:appuser /data/storage
USER appuser
EXPOSE 8000
# Same migrate-then-serve contract as dev, via the baked venv (no uv at runtime).
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
