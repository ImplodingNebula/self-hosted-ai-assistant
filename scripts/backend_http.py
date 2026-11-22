r"""Call the backend's own HTTP API from inside the backend container.

WHY THIS EXISTS. The targets that prime and inspect a running stack (`warm`,
`warm-chat`, `warm-embed`, `warm-search`, `status`) reach the backend over HTTP.
In the development layout the backend publishes a host port, so plain curl on
the host is the cheapest possible way to ask - no container, no interpreter, no
dependency on the stack being exec-able. That path is kept exactly as it was.

A deployment publishes nothing but the reverse proxy, and neither remaining host
address works. There is no backend port to curl. Dialling the proxy instead
fails too, and not for a reason worth working around: its site address is the
deployment's real hostname, so a request to `localhost` matches no site and is
refused before it ever reaches a route. The one honest address left is the
backend's own port on the compose network, which is what this script uses - the
same move `scripts/ollama_http.py` makes for the model server, for the same
reason.

It is piped in on stdin, so nothing has to be mounted into or baked into the
image:

    docker compose exec -T backend uv run python - GET /healthz < scripts/backend_http.py

`uv run` works here only because the production image sets UV_CACHE_DIR and
UV_NO_SYNC; without those it cannot start at all for the unprivileged user.

THREE OUTPUT MODES, each mirroring the curl invocation it stands in for, so the
make recipes keep ONE parsing path across both layouts rather than growing a
second, subtly different one:

    (default)   print the body; exit non-zero unless the status is 2xx
                            <- curl -fsS
    --coded     print the body, then a `__HTTP__<status>` marker; ALWAYS exit 0,
                because the caller reads the code itself and treats 404 as a
                real answer ("web search is not configured") rather than a
                failure
                            <- curl -sS -w '\n__HTTP__%{http_code}'
    --quiet     print nothing; the exit code is the entire answer
                            <- curl -fs -o /dev/null

THE PATH ARGUMENT IS WRITTEN WITHOUT ITS LEADING SLASH, and that is not a style
choice. Under Git Bash on Windows, an argument that looks like an absolute Unix
path is rewritten into a Windows path on its way to a native executable, and
`docker` is one - so `/healthz` arrives here as `C:/Program Files/Git/healthz`
and the request fails on a filesystem path nobody typed. Passing `healthz` gives
that rewriting nothing to match. A leading slash is still accepted, for anyone
running this by hand on a platform where it is harmless.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request

# The backend's own IN-CONTAINER port, deliberately not BACKEND_PORT: this runs
# inside the container, so how (or whether) the port is published on the host
# changes nothing here.
BASE_URL = "http://localhost:8000"

# A warm is the heaviest thing the stack does - a cold model load on a contended
# host can run for many minutes - so this has to outlast the model server's own
# give-up ceiling (OLLAMA_LOAD_TIMEOUT, 15m). Same rule as the backend's
# OLLAMA_READ_TIMEOUT_SECONDS: whoever hangs up first turns a slow success into
# a reported failure, and it should never be the side doing the asking.
TIMEOUT_SECONDS = 960.0

# curl's exit codes for the two failures, so a caller that logs the status of a
# failed probe reads something familiar rather than an invention of this script.
_EXIT_HTTP_ERROR = 22
_EXIT_UNREACHABLE = 7

_USAGE = "usage: backend_http.py [--coded|--quiet] <METHOD> <PATH> [JSON_BODY]"


def main(argv: list[str]) -> int:
    """Perform one request and report it in the mode the caller asked for.

    Args:
        argv: Command-line arguments after the script name.

    Returns:
        A process exit status: 0 on success, 2 on a usage error, and otherwise
        curl's code for the corresponding failure.
    """
    args = list(argv)
    mode = "strict"
    if args and args[0] in ("--coded", "--quiet"):
        mode = args.pop(0).lstrip("-")
    if len(args) < 2:
        print(_USAGE, file=sys.stderr)
        return 2

    method, path = args[0].upper(), args[1]
    body = args[2] if len(args) > 2 else None
    # Callers pass `healthz`, not `/healthz` - see the note in the module
    # docstring about Git Bash rewriting leading-slash arguments.
    path = path if path.startswith("/") else "/" + path

    data = None
    headers = {}
    if body is not None:
        data = body.encode()
        headers["content-type"] = "application/json"
    request = urllib.request.Request(  # noqa: S310 - fixed localhost scheme
        BASE_URL + path, data=data, headers=headers, method=method
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            status, payload = response.status, response.read()
    except urllib.error.HTTPError as exc:
        # An answer, just not a successful one - and for --coded that is the
        # whole point of asking, so it must not be treated as no answer at all.
        status, payload = exc.code, exc.read()
    except OSError as exc:
        # No answer: the stack is down, still starting, or not exec-able.
        print(f"backend unreachable at {BASE_URL}: {exc}", file=sys.stderr)
        if mode == "coded":
            # Exactly what curl leaves in its output when it never connected, so
            # the caller's status parsing needs no special case.
            sys.stdout.write("\n__HTTP__000")
            return 0
        return _EXIT_UNREACHABLE

    ok = 200 <= status < 300
    if mode == "coded":
        sys.stdout.write(payload.decode(errors="replace"))
        sys.stdout.write(f"\n__HTTP__{status}")
        return 0
    if mode == "quiet":
        return 0 if ok else _EXIT_HTTP_ERROR
    if ok:
        sys.stdout.write(payload.decode(errors="replace"))
        return 0
    # Mirrors `curl -f`: the body of a failure goes to stderr, where it explains
    # the non-zero exit without polluting output the caller may be capturing.
    print(payload.decode(errors="replace"), file=sys.stderr)
    return _EXIT_HTTP_ERROR


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
