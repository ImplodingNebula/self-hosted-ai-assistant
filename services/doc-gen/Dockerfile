# syntax=docker/dockerfile:1
# The doc-gen sidecar: one MCP server that renders structured specs into real
# files (see app/main.py). Mirrors the backend image's conventions: slim base,
# uv from the official image, committed lockfile, non-root user.
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    # matplotlib writes a font cache on first use. Without a writable location
    # it warns on every process start; the non-root user below has no home, so
    # it is pointed at a tmpfs-friendly path explicitly.
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Resolve dependencies from the committed lockfile (cached layer).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app

# Drop root: the server only reads /app and writes nothing inside the image, so
# a passwordless system user is enough.
RUN useradd --system --no-create-home appuser
USER appuser

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8765
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8765"]
