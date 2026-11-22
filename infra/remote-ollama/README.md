# Running the model server on a remote GPU

The stack normally starts its own model server as a container and reaches it
over the private compose network. This directory covers the other arrangement:
the models run on a cloud GPU pod somewhere else, and the stack reaches them
across the internet.

Everything here is optional. A deployment with a local model server needs none
of it.

- [Why a proxy stops being optional](#why-a-proxy-stops-being-optional)
- [On the pod](#on-the-pod)
- [On the machine running the stack](#on-the-machine-running-the-stack)
- [Checking it before trusting it](#checking-it-before-trusting-it)
- [Caveats](#caveats)

## Why a proxy stops being optional

Ollama has no authentication of any kind. There are no accounts, no tokens, and
no way for it to refuse a request. That is a sound design for a process
listening on loopback, and it is why the local setup needs nothing extra: the
container publishes its port to the host's loopback address, so only the compose
network can address it.

A cloud GPU pod changes that completely. The pod provider's HTTP proxy is what
makes a port reachable at all, and what it publishes is a public URL. An
unprotected model server behind one lets anyone who finds the address run
generations on the rented GPU, list the installed models, pull new ones onto the
pod's disk, and delete the ones already there. It works perfectly for its owner
the whole time, which is exactly why it goes unnoticed.

The remedy is a reverse proxy on the pod that checks a credential before letting
anything through, with the model server bound to loopback so the proxy is the
only way in. [`Caddyfile.example`](Caddyfile.example) is a minimal working
version of that.

## On the pod

**1. Bind the model server to loopback**, on a port the proxy will forward to:

```sh
OLLAMA_HOST=127.0.0.1:11435 ollama serve
```

The binding is what makes the proxy unavoidable. Left on all interfaces, the
pod's exposed port would still reach the model server directly and the proxy
would be decoration.

**2. Generate a token** and keep it somewhere safe:

```sh
openssl rand -hex 32
```

**3. Run the proxy** on the port the pod exposes, with that token in its
environment:

```sh
OLLAMA_PROXY_TOKEN=<the token> caddy run --config Caddyfile
```

The example config refuses anything without a matching
`Authorization: Bearer <token>` header and reverse-proxies everything else to
`127.0.0.1:11435`.

Nothing else has to happen on the pod. Model weights are downloaded onto its
disk from the client side - the Model Manager page pulls into a remote server
exactly as it pulls into a local container, and shows the same progress.

## On the machine running the stack

Three settings work together, all documented in
[`infra/.env.example`](../.env.example):

- **`OLLAMA_URL`** - the address the pod provider's proxy publishes, with the
  `https://` scheme. The scheme is not cosmetic: those proxies answer plain
  `http` with a redirect, and the backend does not follow redirects. It reports
  the redirect and stops, rather than replaying a request - and the credential
  attached to it - at an address nobody chose.

- **`PROFILE=remote`** - the mode meaning "no local model server container".
  Add it as a `PROFILE=remote` line in `.env`, beside the `OLLAMA_URL` it
  belongs with: pointing at a remote server IS this deployment's standing
  configuration, and the two settings only make sense together. Every `make`
  target then picks the mode up on its own. The stack starts everything except
  the model server, and the model targets speak HTTP to `OLLAMA_URL` instead
  of running commands inside a container that is not there. (Passing it per
  command - `make up PROFILE=remote` - also works and overrides the `.env`
  line; useful for a one-off, and easy to forget on the next command, which
  is why the standing line is the recommended form.)

- **`OLLAMA_AUTH_HEADERS`** - a JSON object mapping header name to value, sent
  on every backend call to the model server:

  ```
  OLLAMA_AUTH_HEADERS={"Authorization": "Bearer <the token>"}
  ```

  A literal `$` inside the value must be written `$$`, or compose reads it as a
  variable reference and substitutes it away. The value is a secret: it belongs
  in `.env` and nowhere else.

With those in place - `OLLAMA_URL`, `PROFILE=remote` and the credential all
standing in `.env` - the usual commands behave as they always did, with no
extra arguments to remember:

```sh
make up
make gpu-check       # reachability, and what is resident
make models          # installed on the pod, and loaded right now
make warm
make cool            # unloads everything on the pod
```

Downloading weights onto the pod is done from the Model Manager page rather
than from a `make` target, so the choice is made against the inventory the pod
actually reports. There is no headless shortcut here the way there is for a
local container: with no Ollama service in this stack, `docker compose exec` has
nothing to reach, and the pull has to travel over HTTP to `OLLAMA_URL`.

`make env-check`, which `up`, `rebuild` and `warm` all run first, refuses the
two combinations that cannot work - a remote URL with a profile that starts a
local container, and `PROFILE=remote` with a local URL - and warns about a
remote endpoint reached over plain `http` or with no credential set.

## Checking it before trusting it

A proxy that was meant to authenticate and does not looks identical to one that
works, from the inside. Two requests settle it. Without the header, the endpoint
should refuse:

```sh
curl -s -o /dev/null -w '%{http_code}\n' https://<the proxied address>/api/version
# expect 401
```

With it, the model server should answer:

```sh
curl -s -H 'Authorization: Bearer <the token>' \
  https://<the proxied address>/api/version
# expect a version payload
```

A `200` from the first command means the proxy is passing everything through and
the endpoint is open.

## Caveats

**A first-byte deadline usually sits in front of the pod.** Pod HTTP proxies
commonly drop a request that has produced no response bytes within roughly 100
seconds. Almost everything this stack sends is far quicker than that. The
exception is the first request after a model is evicted, which waits out the
whole load - and on a large model that can exceed the ceiling and surface as a
timeout that reads like a broken endpoint.

Two things keep it to a first-load concern rather than a standing one:

- `OLLAMA_KEEP_ALIVE=-1`, the default in `infra/.env.example`, keeps a model
  resident until something unloads it, so the load is paid once per session
  rather than after every idle gap.
- Warming deliberately with `make warm` puts that one slow request somewhere it
  can be waited on, instead of under someone's first message.

**Where the pod provider also offers a direct TCP port, the deadline can be
removed entirely - at a price that has to be named.** A direct TCP mapping
publishes a bare public address (`IP:port`) with no HTTP proxy in the path: no
first-byte rule, and also none of the things that proxy quietly provided.

- **The traffic is cleartext unless the pod's own proxy terminates TLS.** The
  provider's HTTP proxy wrapped every call in HTTPS; a raw TCP port wraps
  nothing, so the credential header and every prompt and retrieved passage
  cross the internet readable - a token sent in the clear has to be treated as
  observable, and rotated accordingly. The reverse proxy on the pod remains
  mandatory either way (it is still the only authentication), and it can also
  terminate TLS - but a certificate for a bare IP cannot be publicly signed,
  so real TLS needs a hostname pointed at the pod's address for the proxy to
  hold a certificate for.
- **The address is less stable than the proxied URL.** External port mappings
  are commonly reassigned when a pod resets, and on some provider tiers the IP
  itself can change when a pod migrates - `OLLAMA_URL` then needs updating
  where the proxied hostname would have survived.

A reasonable division of labour: the provider's HTTP proxy for everyday use -
it is encrypted, and its deadline only bites cold loads and long silent
operations - with the direct port reserved for exactly those operations (a
very large first load), fronted with TLS where the data crossing it matters.

**A capability check is the one operation that must NOT be moved to the other
address.** Tool-check verdicts bind to the server that earned them, alongside
the weights digest and the runtime version: throughput is the host's hardware
and tool-calling behaviour belongs to the runtime as that host runs it, so a
verdict recorded against one address certifies nothing for another. Running a
check over the direct port records `http://<ip>:<port>` while everyday turns
are served over the proxied hostname, and the resulting verdict can never
apply - the model manager shows the run and marks it as measured elsewhere.
Run checks over whichever address `OLLAMA_URL` names, and warm the model first
(`make warm`) so the check is not also paying for the cold load.

The same rule has a consequence for the address instability noted above: when
a pod reset changes the port, or a migration changes the IP, the endpoint
changes with it and every verdict earned at the old address stops counting.
That is the intended behaviour - a new machine has proven nothing - but it
means a re-check belongs in the same routine as updating `OLLAMA_URL`.

**Every model call is now a network round trip.** Embedding during ingest,
retrieval queries and chat generation all cross the same link, so a slow or
distant connection shows up as latency on every turn, not only on generation.

**The local model volume goes unused.** `PROFILE=remote` starts no model server
container, so nothing writes to the `ollama_models` volume on the local machine;
the weights live on the pod. The reranker sidecar is unaffected - it is a
separate service and still runs locally when its profile is enabled.
