# reranker

Infinity sidecar serving `bge-reranker-v2-m3` (a cross-encoder). Exists because Ollama has no rerank endpoint.

Runs as an opt-in compose service (`--profile reranker`) using the pinned `michaelf34/infinity`
image; about 1.1 GB of model weights download into a named cache volume on first start, so that boot
is slow and later ones are not. `RERANKER_URL` is the switch: it starts the service and it is what
the backend reads, so the container and the code cannot disagree about whether reranking is
available. (`RERANKER=0` and `RERANKER=1` force the compose profile off or on and outrank the URL -
an escape hatch, not the everyday control.) A small card is meant to leave the URL unset and retrieve
dense-only. Any sidecar failure falls back to the dense ranking - reranking costs quality when
absent, never availability - and a workspace can switch it off for itself from the console.

Which weights are served is declared in two places that have to name the same model: the sidecar's
own `--model-id` in [`../../infra/docker-compose.yml`](../../infra/docker-compose.yml), and
`RERANKER_MODEL` in the environment file, which is the id the backend puts in each `/rerank`
request. Swapping the sidecar's weights without moving `RERANKER_MODEL` with them leaves the backend
asking for a model the sidecar is not serving, and every rerank call is rejected - gracefully, as a
fall back to the dense ranking, so the symptom is quieter results rather than an error.

See [../../docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) sections 4, 6, and 11.
