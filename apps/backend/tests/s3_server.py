"""A real S3 service for the storage tests, and adapters wired to it.

moto's in-process ``mock_aws`` patches botocore's SYNCHRONOUS HTTP path, which
aiobotocore does not use, so it would silently mock nothing here. moto's HTTP
server does work: the adapter signs a real request, sends it over a real
socket, and parses a real response - the same wire path a MinIO deployment
takes, which is the whole reason the contract suite is worth running against S3
at all.

The server is session-scoped because starting one costs about as much as the
tests that use it. Isolation comes from a fresh BUCKET per adapter instead.

Imported rather than living in conftest, and requested lazily inside the
contract fixture's s3 branch, so that the only test runs paying for moto are
the ones that actually talk to it.
"""

from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest

from app.storage.s3 import S3Storage

# Accepted by moto without checking, and by botocore's signer, which refuses to
# sign with nothing. Deliberately not read from the environment: a developer's
# real AWS credentials must never be what these tests pick up.
ACCESS_KEY = "test-access-key"
SECRET_KEY = "test-secret-key"

# Small enough that ordinary listing tests cross a page boundary, so the paging
# loop is exercised by the whole suite rather than by one test that remembers to.
TEST_PAGE_SIZE = 2


@pytest.fixture(scope="session")
def s3_endpoint() -> Iterator[str]:
    """Start one moto S3 server for the session and yield its base URL."""
    from moto.server import ThreadedMotoServer

    # Port 0 lets the OS pick a free one: a fixed port collides with whatever
    # else is listening on a developer's machine, and with a parallel test run.
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    try:
        yield f"http://{host}:{port}"
    finally:
        server.stop()


def make_s3_storage(endpoint: str, **overrides: object) -> S3Storage:
    """Return an S3Storage against ``endpoint`` with a bucket of its own.

    Args:
        endpoint: Base URL of the S3 service, from the ``s3_endpoint`` fixture.
        **overrides: Constructor arguments to replace the defaults.

    Returns:
        A backend nobody else in the session shares.
    """
    kwargs: dict[str, object] = {
        "endpoint_url": endpoint,
        "access_key_id": ACCESS_KEY,
        "secret_access_key": SECRET_KEY,
        "page_size": TEST_PAGE_SIZE,
    }
    kwargs.update(overrides)
    # A bucket nobody else uses, so one test's objects can never answer for
    # another's - the server outlives every test in the session.
    return S3Storage(f"test-{uuid4().hex}", **kwargs)  # type: ignore[arg-type]
