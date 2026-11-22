"""Guard the remote-model-server guard itself.

``scripts/remote_check.sh check`` is the single enforcement point for the
PROFILE=remote contract: invoked from the Makefile's ``env-check`` recipe, it
is what stops ``make up``/``rebuild``/``warm`` on a profile
that would start a local Ollama container nothing uses (or a remote profile
with nothing to dial), and what warns on an unprotected remote endpoint.
Nothing else exercises that wiring - deleting the invocation would break no
test and no boot - so these tests pin it the same way test_compose_env.py pins
the compose environment lists: by reading the repo's own files.

The host list the classifier keys on exists twice on purpose (shell for make,
Python for the request path and the /models payload); the sync test here is
what turns "change one, change both" from a comment into a failure.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.core.config import _LOCAL_OLLAMA_HOSTS


def _repo_root() -> Path | None:
    """Walk up for the repo root, identified by the Makefile + scripts pair.

    Mirrors test_compose_env.py: counting parents breaks inside the backend
    image, where the backend directory is the filesystem root.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "Makefile").is_file() and (parent / "scripts" / "remote_check.sh").is_file():
            return parent
    return None


_ROOT = _repo_root()

pytestmark = pytest.mark.skipif(_ROOT is None, reason="no Makefile/scripts above this checkout")


def _recipe(makefile: str, target: str) -> str:
    """Return the recipe body of one make target (its tab-indented lines)."""
    lines = makefile.splitlines()
    body: list[str] = []
    collecting = False
    for line in lines:
        if collecting:
            if line.startswith("\t"):
                body.append(line)
                continue
            if line.strip() == "" or line.lstrip().startswith("#"):
                continue
            break
        if re.match(rf"^{re.escape(target)}\s*:", line):
            collecting = True
    return "\n".join(body)


def test_env_check_invokes_the_remote_guard() -> None:
    """env-check must call remote_check.sh - it is the guard's only caller.

    env-check is a prerequisite of up/rebuild/warm, so this one
    line is what makes the refusal rules real; without it the script is dead
    code and every mismatched profile boots anyway.
    """
    assert _ROOT is not None
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "scripts/remote_check.sh check" in _recipe(makefile, "env-check")


def test_cool_branches_on_the_remote_classification() -> None:
    """make cool must consult the classifier before trusting a local exec.

    Without the branch, a remote deployment's ``make cool`` finds no local
    container and reports "nothing to unload" while the remote GPU stays
    pinned by keep_alive - a false all-clear, which is the bug the remote
    branch exists to fix.
    """
    assert _ROOT is not None
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "remote_check.sh classify-env" in _recipe(makefile, "cool")


def test_local_host_lists_stay_in_sync() -> None:
    """The shell and Python local-host lists must be identical.

    The two classifiers answer the same question in different languages; if
    the lists drift, the same OLLAMA_URL gets one verdict from the make
    targets and the opposite one from the backend.
    """
    assert _ROOT is not None
    script = (_ROOT / "scripts" / "remote_check.sh").read_text(encoding="utf-8")
    match = re.search(r'^LOCAL_HOSTS="([^"]+)"', script, re.MULTILINE)
    assert match is not None, "LOCAL_HOSTS assignment not found in remote_check.sh"
    shell_hosts = frozenset(match.group(1).split())
    assert shell_hosts == _LOCAL_OLLAMA_HOSTS
