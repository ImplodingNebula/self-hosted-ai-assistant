"""The REMOTE model registry: what a tag would cost before anything is fetched.

Deliberately not part of :class:`~app.providers.base.ModelProvider`. That seam
speaks to the LOCAL model server about weights it already holds; this speaks to
the distribution registry about weights nobody has yet. Folding the two together
would put a public-internet dependency behind an interface every turn uses, and
would give a provider that has no registry at all (a runtime loading one model
at process start) a method it could only refuse.

Two questions are answered, both on demand and never while rendering a list:
does the registry know this tag, and how many bytes would arrive. Everything
else a download involves belongs to the pull itself.

The registry BASE URL comes from settings and nowhere else. A caller supplies a
NAME, whose segments have already been through
:func:`app.services.model_names.parse_model_name`, so nothing a user types can
redirect this client at a host of their choosing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# Registries speak the OCI distribution API. Asking for the manifest media types
# explicitly keeps a content-negotiating registry from answering with an index
# (a list of per-platform manifests) whose entries carry no layer sizes.
_MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/json",
    )
)


class RegistryUnavailableError(Exception):
    """The registry could not answer. Never means "the model does not exist".

    The distinction is the whole point of having two error types here: a
    network blip reported as a missing model sends someone to fix a spelling
    that was correct all along.
    """


class RegistryManifestNotFoundError(Exception):
    """The registry answered, and it has no such tag."""


@dataclass(frozen=True)
class TagFacts:
    """What a registry can say about a tag before anything is downloaded.

    The size is the ANSWER; the other two are context for it, and they are what
    turn "4.7 GB" into a decision. Two tags of one model can be a gigabyte apart
    purely on quantization, and a size with no parameter count beside it cannot
    be compared against the card it has to fit on.

    Both are optional and independently so. They are read from the config blob
    the manifest points at, which is a convention of the model registry rather
    than part of the OCI spec, so a registry that does not publish one - or
    publishes a shape this client does not recognise - still answers the size
    question completely. A missing fact is reported as missing; nothing here is
    ever inferred from the tag string, where "q4_K_M" in a name is a naming
    habit and not a declaration.

    Attributes:
        download_size_bytes: Total bytes the download would transfer.
        quantization: How hard the weights were quantized, in the registry's
            own wording ("Q4_K_M"), or None.
        parameter_size: The parameter count as the registry words it ("8.2B"),
            or None.
    """

    download_size_bytes: int
    quantization: str | None = None
    parameter_size: str | None = None


class OllamaRegistryClient:
    """Reads tag manifests from an Ollama-flavoured OCI registry over HTTP."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the client for a registry at ``base_url``.

        Args:
            base_url: Root URL of the registry (settings-supplied).
            timeout: Read timeout in seconds. Short on purpose: a preflight
                runs while somebody waits on a button, so an unreachable
                registry must become an honest "unknown" quickly rather than
                hanging the page for the length of a generation timeout.
            client: Optional externally-owned HTTP client (tests inject a mock
                transport here). When omitted, the client is created lazily and
                owned by this instance; call :meth:`aclose` on shutdown.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(connect=5.0, read=timeout, write=5.0, pool=5.0)
        self._client = client
        self._owns_client = client is None

    async def aclose(self) -> None:
        """Close the owned HTTP client (no-op for an injected one)."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return the shared client, creating it lazily on first use."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def tag_facts(self, *, namespace: str, repo: str, tag: str) -> TagFacts:
        """Return the size this tag would cost, plus what the registry says it is.

        Two requests at most, and the second one cannot cost the first its
        answer: the manifest decides success or failure, and the config blob
        behind it is read best-effort. A registry that serves manifests and
        refuses blobs still answers "4.7 GB to download" rather than "the
        registry could not be reached", which would be both wrong and the more
        alarming of the two.

        Args:
            namespace: Publisher segment (the registry's own library namespace
                for a bare name).
            repo: Repository segment.
            tag: Version segment.

        Returns:
            The assembled :class:`TagFacts`.

        Raises:
            RegistryManifestNotFoundError: The registry has no such tag.
            RegistryUnavailableError: The registry could not be reached, or
                answered with something this client cannot read.
        """
        manifest = await self._manifest(namespace=namespace, repo=repo, tag=tag)
        total = manifest_size(manifest)
        if total is None:
            raise RegistryUnavailableError("The registry manifest listed no downloadable layers.")
        quantization, parameter_size = await self._config_facts(
            namespace=namespace, repo=repo, manifest=manifest
        )
        return TagFacts(
            download_size_bytes=total,
            quantization=quantization,
            parameter_size=parameter_size,
        )

    async def _config_facts(
        self, *, namespace: str, repo: str, manifest: object
    ) -> tuple[str | None, str | None]:
        """Read (quantization, parameter size) off the manifest's config blob.

        Best-effort throughout, and deliberately silent about its own failures
        at anything above debug level: this is decoration on an answer that is
        already complete, and a warning per preflight for a registry that
        simply does not publish a config blob would be noise that trains people
        to ignore the log.
        """
        digest = _config_digest(manifest)
        if digest is None:
            return None, None
        url = f"{self._base_url}/v2/{namespace}/{repo}/blobs/{digest}"
        try:
            resp = await self._get_client().get(url)
            if resp.status_code != 200:
                return None, None
            # A config blob is a few hundred bytes of JSON. The URL is built
            # from a REMOTE document's digest, so what answers is not
            # necessarily what the registry meant to serve - and this decoration
            # must not be able to pull an arbitrarily large body into memory
            # while somebody waits on a button. Read the bytes, bound them, and
            # only then parse.
            if len(resp.content) > _MAX_CONFIG_BYTES:
                logger.debug(
                    "Registry config blob for %s/%s exceeded %d bytes; ignoring it.",
                    namespace,
                    repo,
                    _MAX_CONFIG_BYTES,
                )
                return None, None
            config = resp.json()
        except (httpx.HTTPError, ValueError):
            logger.debug("Registry config blob for %s/%s was unreadable.", namespace, repo)
            return None, None
        if not isinstance(config, dict):
            return None, None
        # The model registry's own field names. `file_type` is the quantization
        # ("Q4_K_M") and `model_type` is the parameter count ("8.2B") - not
        # obvious from either name, which is exactly why they are read in one
        # place and handed on under names that say what they are.
        return _text(config.get("file_type")), _text(config.get("model_type"))

    async def _manifest(self, *, namespace: str, repo: str, tag: str) -> object:
        """Fetch and decode one tag manifest, or raise the typed failure."""
        url = f"{self._base_url}/v2/{namespace}/{repo}/manifests/{tag}"
        try:
            resp = await self._get_client().get(url, headers={"Accept": _MANIFEST_ACCEPT})
        except httpx.HTTPError as exc:
            raise RegistryUnavailableError("The model registry could not be reached.") from exc
        if resp.status_code == 404:
            raise RegistryManifestNotFoundError(f"The registry has no tag '{repo}:{tag}'.")
        if resp.status_code != 200:
            # Redirects included: manifests are served directly, so a 3xx here
            # means something is intercepting the request, not that the answer
            # is one hop away. Following it would be following a stranger.
            raise RegistryUnavailableError(f"The model registry answered HTTP {resp.status_code}.")
        try:
            return resp.json()
        except ValueError as exc:
            raise RegistryUnavailableError(
                "The model registry sent an unreadable manifest."
            ) from exc


# What a content digest looks like, as a positive match. A denylist of the
# characters that hurt (slashes, dot segments) is the wrong shape here: the
# value is interpolated into a URL from a REMOTE document, and a query string,
# a fragment or a control character all reach an address this client never
# meant to request without containing either. The grammar is narrow and known,
# so matching it is both simpler and complete.
_DIGEST = re.compile(r"^[a-z0-9]+:[0-9a-f]{16,}$")

# How much of a config blob this client will hold in memory. Real ones are a
# few hundred bytes; 64 KiB is generous by two orders of magnitude and still
# bounds what a misbehaving (or redirected) host can hand back on a request
# somebody is waiting on.
_MAX_CONFIG_BYTES = 64 * 1024


def _config_digest(manifest: object) -> str | None:
    """The digest of the manifest's config blob, when it declares a usable one."""
    if not isinstance(manifest, dict):
        return None
    config = manifest.get("config")
    if not isinstance(config, dict):
        return None
    digest = config.get("digest")
    if not isinstance(digest, str) or not _DIGEST.match(digest):
        return None
    return digest


def _text(value: object) -> str | None:
    """A non-empty string from an untrusted JSON value, or None."""
    return value.strip() or None if isinstance(value, str) else None


def manifest_size(manifest: object) -> int | None:
    """Sum the blob sizes an OCI manifest declares, or None when it cannot.

    All or nothing, by design. A partial sum would be reported as an EXACT
    download size while being off by orders of magnitude - a confidently wrong
    number, which is the one answer this whole path exists to avoid. One layer
    without a readable size makes the whole document unusable, and "unknown" is
    the honest result.

    The config blob counts alongside the layers because it is downloaded too -
    the number this feeds is "how much traffic will this cost", not "how large
    are the weights" - but it is optional, since a manifest without one is
    still fully described by its layers.

    Args:
        manifest: The decoded manifest document.

    Returns:
        Total bytes, or None when the document cannot be read as a size.
    """
    if not isinstance(manifest, dict):
        return None
    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        return None
    total = 0
    for blob in layers:
        size = _blob_size(blob)
        if size is None:
            return None
        total += size
    return total + (_blob_size(manifest.get("config")) or 0)


def _blob_size(blob: object) -> int | None:
    """Read one blob's declared size, or None when it declares no usable one."""
    if not isinstance(blob, dict):
        return None
    size = blob.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        return None
    return size
