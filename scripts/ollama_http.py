"""Speak the model server's HTTP API from inside the backend container.

WHY THIS EXISTS. Several make targets reach the model server by running the
`ollama` CLI inside the local Ollama container. That is exact and fast while
the server IS that container - and quietly wrong the moment OLLAMA_URL points
somewhere else. A remote deployment starts no local container, so those targets
either fail with a confusing "not running" or, worse, exec into a stale
container left over from an earlier run and report the wrong machine's models:
`make cool` then announces that nothing needed unloading while the remote GPU
stays pinned.

The fix is to stop assuming a container and speak the HTTP API instead. This
script does that from INSIDE the backend container, which is where the address
and the credentials already live: nothing has to be re-derived on the host, and
no secret transits a host shell or a process list. Run it with `make`, which
pipes this file in:

    docker compose exec -T backend uv run python - ps < scripts/ollama_http.py

Subcommands: version, tags, ps, pull <tag>, unload-all.

REDIRECTS ARE NEVER FOLLOWED, here or anywhere else in this stack. A cloud
pod's HTTP proxy answers plain http with a 301 to its https address, and
following it would replay the request - with the credential attached to it - at
an address nobody configured. Reporting the redirect and stopping turns that
into a one-line fix (correct the scheme in OLLAMA_URL) instead of a mystery.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, NoReturn, Sequence

import httpx

# Connecting is either quick or hopeless; reading tolerates a busy server
# thinking between lines. Neither has to cover a cold model LOAD: nothing here
# generates, and a pull streams progress continuously, so the read ceiling only
# has to outlast a quiet stretch.
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 60.0
PULL_READ_TIMEOUT = 300.0

# Shared wording: the backend reports the same two failures on the request path,
# and an operator should not have to recognise two descriptions of one problem.
REDIRECT_MESSAGE = (
    "The model server redirected to '{location}'. Set OLLAMA_URL to the address the "
    "server expects (a cloud proxy usually needs https:// instead of http://)."
)
UNAUTHORIZED_MESSAGE = (
    "The model server rejected the request as unauthorized (HTTP {status}). Set "
    "OLLAMA_AUTH_HEADERS to the credentials the server's proxy expects."
)


def _fail(message: str, code: int = 1) -> NoReturn:
    """Report a message on stderr and end the run.

    Args:
        message: What went wrong, in terms an operator can act on.
        code: Exit status; 2 marks a configuration fault the caller can fix.
    """
    print(message, file=sys.stderr)
    raise SystemExit(code)


def _base_url() -> str:
    """Return the model server's base URL, without a trailing slash."""
    return (os.environ.get("OLLAMA_URL") or "http://ollama:11434").rstrip("/")


def _auth_headers() -> dict[str, str]:
    """Return the extra headers every request carries.

    OLLAMA_AUTH_HEADERS is a JSON object mapping header name to value, and its
    values are credentials - so a parse failure describes the SHAPE expected and
    never quotes back what it was given.

    Returns:
        The parsed headers, or an empty mapping when the variable is unset.
    """
    raw = (os.environ.get("OLLAMA_AUTH_HEADERS") or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        _fail(
            "OLLAMA_AUTH_HEADERS is not valid JSON. It must be an object mapping header "
            'name to value, e.g. {"Authorization": "Bearer <token>"}.',
            2,
        )
    if not isinstance(parsed, dict):
        _fail(
            "OLLAMA_AUTH_HEADERS must be a JSON OBJECT mapping header name to value, "
            'e.g. {"Authorization": "Bearer <token>"}.',
            2,
        )
    return {str(key): str(value) for key, value in parsed.items()}


def _client(read_timeout: float = READ_TIMEOUT) -> httpx.Client:
    """Build a client bound to OLLAMA_URL that never follows redirects.

    Args:
        read_timeout: Seconds to wait for the next bytes of a response.

    Returns:
        A client carrying the configured auth headers.
    """
    return httpx.Client(
        base_url=_base_url(),
        headers=_auth_headers(),
        timeout=httpx.Timeout(read_timeout, connect=CONNECT_TIMEOUT),
        follow_redirects=False,
    )


def _check(response: httpx.Response) -> None:
    """Turn the failures a proxied endpoint produces into plain guidance.

    Args:
        response: The response to judge; its body is not read, so this is safe
            on a streaming request.
    """
    if 300 <= response.status_code < 400:
        # Same fallback wording as the backend provider's redirect message, so
        # the operator meets one sentence from either side.
        location = response.headers.get("location") or "an unnamed address"
        _fail(REDIRECT_MESSAGE.format(location=location), 2)
    if response.status_code in (401, 403):
        _fail(UNAUTHORIZED_MESSAGE.format(status=response.status_code), 2)
    if response.status_code >= 400:
        _fail(
            f"The model server answered HTTP {response.status_code} for "
            f"{response.request.url.path}."
        )


def _human_size(num_bytes: Any) -> str:
    """Format a byte count in the largest unit that keeps it above one.

    Args:
        num_bytes: A byte count; None and 0 are both rendered as zero bytes.

    Returns:
        A short string such as "2.4 GB".
    """
    size = float(num_bytes or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.0f} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    """Print rows in columns sized to their widest cell.

    Args:
        headers: Column titles.
        rows: One sequence of cells per row, each as long as `headers`.
    """
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)).rstrip())
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())


def _cmd_version(args: Sequence[str]) -> None:
    """Report the model server's version, which doubles as a reachability check."""
    with _client() as client:
        response = client.get("/api/version")
        _check(response)
        version = response.json().get("version") or "unknown"
    print(f"Model server at {_base_url()} answered - version {version}.")


def _cmd_tags(args: Sequence[str]) -> None:
    """List the models installed on the server."""
    with _client() as client:
        response = client.get("/api/tags")
        _check(response)
        models = response.json().get("models") or []
    if not models:
        print("No models are installed on the model server.")
        return
    rows = []
    for model in sorted(models, key=lambda entry: entry.get("name") or ""):
        # Some servers omit `details` entirely; the size is always there, and a
        # missing parameter count is reported as absent rather than guessed.
        details = model.get("details") or {}
        facts = [
            str(part)
            for part in (details.get("parameter_size"), details.get("quantization_level"))
            if part
        ]
        rows.append(
            [
                str(model.get("name") or "?"),
                _human_size(model.get("size")),
                " ".join(facts) or "-",
            ]
        )
    _print_table(["NAME", "SIZE", "PARAMS / QUANT"], rows)


def _cmd_ps(args: Sequence[str]) -> None:
    """List the models resident on the server, and where each one is running."""
    with _client() as client:
        response = client.get("/api/ps")
        _check(response)
        models = response.json().get("models") or []
    if not models:
        print("Nothing loaded - no model is resident on the model server.")
        return
    rows = []
    for model in models:
        size = int(model.get("size") or 0)
        vram = int(model.get("size_vram") or 0)
        # No VRAM at all is the reading that matters: it means the server is
        # running that model entirely on the CPU, which is many times slower and
        # reports no error of its own.
        if vram <= 0:
            processor = "100% CPU"
        elif size <= 0 or vram >= size:
            processor = "100% GPU"
        else:
            gpu_share = round(vram * 100 / size)
            processor = f"{100 - gpu_share}%/{gpu_share}% CPU/GPU"
        rows.append([str(model.get("name") or "?"), _human_size(size), processor])
    _print_table(["NAME", "SIZE", "PROCESSOR"], rows)


def _cmd_pull(args: Sequence[str]) -> None:
    """Download a model tag onto the server, reporting progress as it arrives.

    Args:
        args: The tag to pull, as a single argument.
    """
    if not args:
        _fail("usage: pull <tag>", 2)
    tag = args[0]
    print(f"Pulling {tag} ...")
    last_status = None
    # The byte-level progress arrives many times a second. Only a change of
    # phase, or another ten percent of the current layer, is worth a line.
    last_percent = -10
    with _client(PULL_READ_TIMEOUT) as client:
        with client.stream("POST", "/api/pull", json={"model": tag, "stream": True}) as response:
            _check(response)
            for line in response.iter_lines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("error"):
                    _fail(f"The model server refused the pull: {event['error']}")
                status = str(event.get("status") or "")
                if status != last_status:
                    last_status = status
                    last_percent = -10
                    print(f"  {status}")
                total = int(event.get("total") or 0)
                if total > 0:
                    percent = int(int(event.get("completed") or 0) * 100 / total)
                    if percent >= last_percent + 10:
                        last_percent = percent - percent % 10
                        print(f"    {percent}% of {_human_size(total)}")
    print(f"Pulled {tag}.")


def _cmd_unload_all(args: Sequence[str]) -> None:
    """Evict every resident model, freeing the server's VRAM and RAM."""
    with _client() as client:
        response = client.get("/api/ps")
        _check(response)
        loaded = [
            str(model.get("name"))
            for model in (response.json().get("models") or [])
            if model.get("name")
        ]
        if not loaded:
            print("Nothing to unload - no model is resident on the model server.")
            return
        stuck = []
        for name in loaded:
            # The documented way to evict a model NOW instead of waiting out its
            # keep_alive: a generate call with no prompt and a zero keep_alive,
            # which is exactly what the `ollama stop` command sends.
            result = client.post("/api/generate", json={"model": name, "keep_alive": 0})
            if result.status_code == 200:
                print(f"Unloaded {name}.")
            else:
                stuck.append(name)
                print(f"Could not unload {name} (HTTP {result.status_code}).")
    if stuck:
        _fail("Still resident: " + ", ".join(stuck))
    print("All resident models unloaded.")


COMMANDS: dict[str, Callable[[Sequence[str]], None]] = {
    "version": _cmd_version,
    "tags": _cmd_tags,
    "ps": _cmd_ps,
    "pull": _cmd_pull,
    "unload-all": _cmd_unload_all,
}

USAGE = "usage: ollama_http.py <version | tags | ps | pull <tag> | unload-all>"


def main(argv: Sequence[str]) -> None:
    """Dispatch a subcommand, reducing transport failures to one plain line.

    Args:
        argv: The arguments after the script itself.
    """
    if not argv:
        _fail(USAGE, 2)
    if argv[0] in ("-h", "--help"):
        print(USAGE)
        return
    handler = COMMANDS.get(argv[0])
    if handler is None:
        _fail(f"Unknown subcommand '{argv[0]}'. {USAGE}", 2)
    try:
        handler(argv[1:])
    except httpx.HTTPError as exc:
        # The exception CLASS is the useful part (a connect refusal reads very
        # differently from a read timeout); its message adds little and is not
        # worth the risk of it having captured request detail.
        _fail(
            f"Could not reach the model server at {_base_url()}: "
            f"{exc.__class__.__name__}. Check OLLAMA_URL and that the server is up."
        )


if __name__ == "__main__":
    main(sys.argv[1:])
