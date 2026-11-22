"""Construction of the process-wide singletons, free of any web framework.

Three processes need the same objects built the same way: the API lifespan, the
API's lazy request-scoped fallbacks, and the arq worker. These factories are
that single definition. They live in ``core`` rather than beside the FastAPI
dependencies because the worker has no requests and must not import the API
layer to build an embedder.

Every factory here is cheap and performs no network I/O. Connections open on
first use, so a slow Qdrant, a missing storage volume, or a SearXNG that is
still booting can never take a process down at startup.
"""

from __future__ import annotations

from app.core.config import Settings
from app.providers.base import ModelProvider
from app.providers.fake_provider import FakeProvider
from app.providers.ollama_provider import OllamaProvider
from app.providers.search.base import SearchProvider
from app.providers.search.searxng import SearXNGSearchProvider
from app.services.model_resolution import resolve_chat_models
from app.storage.base import StorageBackend
from app.storage.local import LocalFsStorage
from app.storage.s3 import S3Storage


def build_model_provider(settings: Settings) -> ModelProvider:
    """Construct the configured model provider (Ollama by default; Fake for tests).

    The provider's default model is the deployment's own configured tag, and
    empty where nothing configures one - which the seam already treats as "the
    caller names the model", the shape every turn uses anyway. No I/O.
    """
    if settings.model_provider == "fake":
        return FakeProvider()
    default = resolve_chat_models(settings).default
    return OllamaProvider(
        base_url=settings.ollama_url,
        default_model=default,
        timeout=settings.ollama_read_timeout_seconds,
        auth_headers=settings.ollama_auth_headers,
    )


def build_storage_backend(settings: Settings) -> StorageBackend:
    """Construct the configured object-storage backend (local-fs by default).

    Construction writes nothing and creates nothing: the storage root (or the
    bucket) is made on the first upload, so a missing volume or a service that
    is still booting surfaces when a file is stored rather than taking the
    process down at boot.

    Broken S3 CREDENTIALS are the one exception, and they fail here on
    purpose. They are not a transient condition that a retry clears, and the
    alternative is a stack that boots green and then rejects the first upload a
    user ever attempts.

    Raises:
        ValueError: STORAGE_BACKEND is "s3" against a custom endpoint with no
            credentials configured, or with only half a static key pair on any
            endpoint.
    """
    if settings.storage_backend == "s3":
        # Half a static key pair is never a valid state on ANY endpoint:
        # botocore raises PartialCredentialsError at signing time, which would
        # surface as the first upload failing on a stack that booted green.
        #
        # `is None`, not truthiness: an empty value would slip past a
        # truthiness test and skip this check entirely. Settings normalises a
        # blank to None precisely so the two readings cannot diverge.
        if (settings.s3_access_key_id is None) != (settings.s3_secret_access_key is None):
            raise ValueError(
                "STORAGE_BACKEND=s3 with only one of S3_ACCESS_KEY_ID / "
                "S3_SECRET_ACCESS_KEY set - static keys come as a pair "
                "(values withheld - they are credentials)."
            )
        # BOTH missing is refused only against a custom endpoint: on AWS
        # itself, botocore's credential chain (instance role, shared config)
        # is the normal way to authenticate, and demanding static keys would
        # forbid the better practice.
        if settings.s3_endpoint_url is not None and settings.s3_access_key_id is None:
            raise ValueError(
                "STORAGE_BACKEND=s3 with S3_ENDPOINT_URL set needs both "
                "S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY (values withheld - "
                "they are credentials)."
            )
        return S3Storage(
            settings.s3_bucket,
            endpoint_url=settings.s3_endpoint_url,
            region=settings.s3_region,
            access_key_id=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key,
        )
    return LocalFsStorage(settings.storage_local_dir)


def build_search_provider(settings: Settings) -> SearchProvider | None:
    """Construct the configured search provider, or None when web search is off.

    ``SEARCH_PROVIDER`` selects the implementation the same way
    ``MODEL_PROVIDER`` does, so the seam has a name in the configuration
    instead of being inferred. ``none`` refuses outright; ``searxng`` (the
    default) still yields None when ``SEARXNG_URL`` is unset, which is the
    pre-existing behaviour and the same graceful-degrade contract as the
    reranker - so an existing deployment is unaffected either way.

    Construction performs no network I/O, so this is safe to call at startup.
    Shared by the lifespan (which pre-builds and warms it) and the request
    dependency (which builds it lazily), so both use one instance and one
    connection pool.
    """
    if settings.search_provider == "none":
        return None
    if not settings.searxng_url:
        return None
    return SearXNGSearchProvider(settings.searxng_url, timeout=settings.web_search_timeout_seconds)
