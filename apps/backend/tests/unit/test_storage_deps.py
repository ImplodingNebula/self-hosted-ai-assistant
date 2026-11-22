"""Tests for the storage-backend dep and factory: always built, override-first."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.api.deps import get_storage_backend
from app.core.config import Settings
from app.core.factories import build_storage_backend
from app.storage.local import FakeStorageBackend, LocalFsStorage
from app.storage.s3 import S3Storage


def _request() -> SimpleNamespace:
    """A minimal stand-in exposing request.app.state, as the dep reads."""
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))


def _settings(**over) -> Settings:
    return Settings(_env_file=None, **over)


def test_default_root_matches_the_compose_mountpoint() -> None:
    # The default is a contract with infra/docker-compose.yml, which mounts the
    # attachment_data volume at this exact path on the backend and the worker.
    # Changing one without the other silently writes uploads into the container
    # layer, where the next recreate discards them.
    assert _settings().storage_local_dir == "/data/storage"


def test_constructs_and_caches(tmp_path: Path) -> None:
    req = _request()
    settings = _settings(storage_local_dir=str(tmp_path))

    backend = get_storage_backend(req, settings)

    assert isinstance(backend, LocalFsStorage)
    # Cached on app.state so every request shares one backend.
    assert req.app.state.storage is backend
    assert get_storage_backend(req, settings) is backend


def test_never_returns_none(tmp_path: Path) -> None:
    # Unlike the reranker and the search provider, storage has no off switch:
    # an attachment whose bytes cannot be reached cannot be answered about.
    assert get_storage_backend(_request(), _settings(storage_local_dir=str(tmp_path))) is not None


def test_override_wins(tmp_path: Path) -> None:
    req = _request()
    fake = FakeStorageBackend()
    req.app.state.storage = fake

    assert get_storage_backend(req, _settings(storage_local_dir=str(tmp_path))) is fake


async def test_the_built_backend_round_trips_under_the_configured_root(tmp_path: Path) -> None:
    backend = build_storage_backend(_settings(storage_local_dir=str(tmp_path)))

    await backend.put("tenant/conv/file.bin", b"payload")

    assert (tmp_path / "tenant/conv/file.bin").read_bytes() == b"payload"
    assert await backend.get("tenant/conv/file.bin") == b"payload"


# ------------------------------------------------------------ the s3 selector


def test_the_default_backend_is_still_local() -> None:
    # STORAGE_BACKEND grew a second value; a deployment that never set it must
    # keep writing exactly where it always did.
    assert _settings().storage_backend == "local"
    assert isinstance(build_storage_backend(_settings()), LocalFsStorage)


def test_an_unknown_backend_name_fails_at_boot() -> None:
    # A Literal, so "minio" or a typo is a startup failure rather than a value
    # that is quietly ignored - which is what STORAGE_BACKEND used to be.
    with pytest.raises(ValidationError):
        _settings(storage_backend="minio")


def test_the_selector_builds_the_s3_backend() -> None:
    backend = build_storage_backend(
        _settings(
            storage_backend="s3",
            s3_endpoint_url="http://minio:9000",
            s3_bucket="uploads",
            s3_access_key_id="key",
            s3_secret_access_key="secret",
        )
    )

    assert isinstance(backend, S3Storage)
    # No client, and therefore no connection, until something is stored: a
    # storage service still booting must not be able to fail the API's boot.
    assert backend._client is None


def test_a_custom_endpoint_without_credentials_refuses_to_boot() -> None:
    """Missing MinIO credentials must stop the process, not the first upload.

    They are not a transient condition a retry clears. Left to surface later,
    the stack boots green and every attachment upload fails, which reads as a
    broken feature rather than a missing setting.
    """
    with pytest.raises(ValueError, match="S3_ACCESS_KEY_ID"):
        build_storage_backend(_settings(storage_backend="s3", s3_endpoint_url="http://minio:9000"))

    with pytest.raises(ValueError, match="S3_ACCESS_KEY_ID"):
        build_storage_backend(
            _settings(
                storage_backend="s3",
                s3_endpoint_url="http://minio:9000",
                s3_access_key_id="key",  # half a credential is still no credential
            )
        )


def test_the_credential_error_never_quotes_the_credential() -> None:
    # A boot-time failure goes straight to stderr and the container log.
    with pytest.raises(ValueError) as caught:
        build_storage_backend(
            _settings(
                storage_backend="s3",
                s3_endpoint_url="http://minio:9000",
                s3_secret_access_key="hunter2",
            )
        )

    assert "hunter2" not in str(caught.value)


def test_an_empty_s3_value_reads_as_unset() -> None:
    """Compose forwards an unset key as an empty string, and the template ships blanks.

    Read literally, an empty value is poison three times over: the bucket and
    region defaults are DEFEATED (a request signed for no region against no
    bucket), an empty access key is still an explicit credential so botocore
    never consults the instance role, and the factory's credential guard skips
    itself because an empty endpoint is falsy. The deployment boots clean and
    every single upload fails.
    """
    settings = _settings(
        storage_backend="s3",
        s3_endpoint_url="",
        s3_access_key_id="",
        s3_secret_access_key="",
        s3_bucket="",
        s3_region="  ",
    )

    assert settings.s3_endpoint_url is None
    assert settings.s3_access_key_id is None
    assert settings.s3_secret_access_key is None
    assert settings.s3_bucket == "attachments"
    assert settings.s3_region == "us-east-1"

    # And so the blanks now read as "AWS with an instance role", which builds.
    assert isinstance(build_storage_backend(settings), S3Storage)


def test_a_blank_endpoint_with_a_real_one_beside_it_still_guards() -> None:
    # The dangerous half of the same bug: a real endpoint whose credentials are
    # present-but-empty must still refuse to boot, not sign with an empty key.
    with pytest.raises(ValueError, match="S3_ACCESS_KEY_ID"):
        build_storage_backend(
            _settings(
                storage_backend="s3",
                s3_endpoint_url="http://minio:9000",
                s3_access_key_id="",
                s3_secret_access_key="",
            )
        )


def test_aws_without_static_credentials_is_allowed() -> None:
    # On AWS the preferred credential is the instance role the container already
    # runs under, resolved by botocore's own chain. Demanding a static key pair
    # here would forbid the better practice and push a long-lived secret into
    # .env for no reason.
    backend = build_storage_backend(_settings(storage_backend="s3"))

    assert isinstance(backend, S3Storage)


def test_half_a_key_pair_refuses_to_boot_even_without_an_endpoint() -> None:
    """A lone key half is never valid on ANY endpoint - AWS included.

    The no-endpoint path deliberately admits ZERO static keys (the instance
    role is the better practice), but one-of-two is a typo, not a credential
    chain: botocore would raise PartialCredentialsError at signing time, i.e.
    on the first upload of a stack that booted green.
    """
    with pytest.raises(ValueError, match="S3_ACCESS_KEY_ID"):
        build_storage_backend(_settings(storage_backend="s3", s3_access_key_id="key"))

    with pytest.raises(ValueError, match="S3_SECRET_ACCESS_KEY"):
        build_storage_backend(_settings(storage_backend="s3", s3_secret_access_key="secret"))
