"""How the operator targets reach the API once login is configured.

`make warm` and its siblings ask the backend over HTTP, and with an issuer set
every ``/api/v1`` route wants a session they have no way to hold. The one thing
the app accepts instead is a request made from INSIDE the backend container - it
arrives on the container's own loopback interface, which the principal
resolution grants the operator context (see
:func:`app.services.auth.principal.is_local_operator`).

So the Makefile has to switch to the in-container form whenever login is on,
whatever compose layout is running. Curl from the host arrives from the bridge
gateway instead, which is not loopback and is answered 401 - and the dev layout
is exactly where somebody tries login first, so the failure would land there.

This is asserted by asking make what it WOULD run (``make -n``), because the
selection is Makefile conditional logic and reading the file's text back would
only assert that the text is the text.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def _repo_root() -> Path | None:
    """Walk up for the repo root, identified by the Makefile + scripts pair.

    Counting parents breaks inside the backend image, where the backend
    directory is the filesystem root - the same reason the backup-script tests
    walk instead.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "Makefile").is_file() and (parent / "scripts" / "backend_http.py").is_file():
            return parent
    return None


_ROOT = _repo_root()

pytestmark = pytest.mark.skipif(
    _ROOT is None or shutil.which("make") is None,
    reason="needs the repo checkout and make",
)


def _would_run(target: str, *overrides: str) -> str:
    """Return the commands make would run for a target, without running them."""
    assert _ROOT is not None
    result = subprocess.run(  # noqa: S603  - fixed argv, no shell
        ["make", "-n", "--no-print-directory", target, *overrides],
        capture_output=True,
        text=True,
        cwd=_ROOT,
        check=False,
    )
    return result.stdout + result.stderr


@pytest.mark.parametrize("target", ["warm-chat", "warm-embed", "warm-search"])
def test_a_configured_login_moves_the_warm_probes_inside_the_container(target: str) -> None:
    with_login = _would_run(target, "OIDC_ISSUER=https://accounts.google.com")

    assert "exec -T backend" in with_login, (
        f"{target} still probes from the host with login configured; that request "
        "arrives from the bridge gateway, not loopback, and is answered 401."
    )
    assert "scripts/backend_http.py" in with_login


@pytest.mark.parametrize("target", ["warm-chat", "warm-embed", "warm-search"])
def test_without_login_the_dev_probes_stay_the_cheap_host_curl(target: str) -> None:
    # The other direction, and it matters: exec needs a container healthy enough
    # to run an interpreter in, while curl works on a stack too broken to exec
    # into. Nothing should pay that cost for a deployment with no login.
    without_login = _would_run(target, "OIDC_ISSUER=", "DEPLOY_ENV=dev")

    assert "curl" in without_login
    assert "exec -T backend" not in without_login


def test_the_production_layout_probes_inside_the_container_regardless() -> None:
    # Unchanged behaviour, pinned here because the login switch now shares the
    # conditional that decides it: nothing is published but the proxy there, so
    # there is no host port to curl whether or not login is on.
    in_production = _would_run("warm-embed", "OIDC_ISSUER=", "DEPLOY_ENV=prod")

    assert "exec -T backend" in in_production
