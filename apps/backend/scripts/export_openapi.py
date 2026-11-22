"""Export the backend OpenAPI schema as deterministic JSON.

Builds the FastAPI app offline via ``create_app()`` - construction wires
routes and middleware only; database, Redis, and Ollama connections happen
in the lifespan, which never runs here - and serializes ``app.openapi()``
with sorted keys and fixed indentation so repeated runs are byte-identical.
The committed copy in ``packages/api-types/openapi.json`` feeds the
TypeScript type generation; CI regenerates both and fails on drift.

Usage (from ``apps/backend``)::

    uv run python scripts/export_openapi.py                # print to stdout
    uv run python scripts/export_openapi.py --out FILE     # write to a file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The backend is not an installed package (tool.uv package = false), so the
# backend root must be on sys.path for `app.*` imports when this file runs as
# a plain script (sys.path[0] is scripts/, not the backend root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    """Build the app, serialize its OpenAPI schema, and emit it.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="Export the OpenAPI schema as stable JSON.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the schema to this path (LF line endings) instead of stdout.",
    )
    args = parser.parse_args()

    # Deferred import: it needs the sys.path fix above.
    from app.main import create_app

    schema = create_app().openapi()
    text = json.dumps(schema, sort_keys=True, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # Explicit LF keeps the file byte-identical across Windows and Linux.
        args.out.write_text(text, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
