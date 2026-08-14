import asyncio

import pytest
from starlette.datastructures import MutableHeaders

from h3_gradio import (
    PublicOriginMiddleware,
    add_missing_public_port,
    configured_public_port,
    configured_public_proto,
)


def test_public_port_is_added_only_when_forwarded_host_omits_it():
    assert add_missing_public_port("example.com", "8443") == "example.com:8443"
    assert add_missing_public_port("example.com:9443", "8443") == "example.com:9443"
    assert add_missing_public_port(
        "example.com, proxy.example.com", "8443"
    ) == "example.com:8443, proxy.example.com:8443"


def test_public_port_accepts_ipv6_hosts():
    assert add_missing_public_port("[2001:db8::1]", "8443") == "[2001:db8::1]:8443"
    assert (
        add_missing_public_port("[2001:db8::1]:8080", "8443")
        == "[2001:db8::1]:8080"
    )


def test_middleware_patches_gradio_forwarded_origin_scope():
    seen = {}

    async def app(scope, receive, send):
        seen["headers"] = MutableHeaders(scope=scope)

    scope = {
        "type": "http",
        "headers": [
            (b"x-forwarded-host", b"example.com"),
            (b"x-forwarded-proto", b"https"),
        ],
    }
    middleware = PublicOriginMiddleware(app, "8443", "https")
    asyncio.run(middleware(scope, None, None))
    assert seen["headers"]["x-forwarded-host"] == "example.com:8443"
    assert seen["headers"]["x-forwarded-proto"] == "https"


def test_explicit_public_proto_overrides_proxy_header():
    seen = {}

    async def app(scope, receive, send):
        seen["headers"] = MutableHeaders(scope=scope)

    scope = {
        "type": "http",
        "headers": [
            (b"x-forwarded-host", b"example.com"),
            (b"x-forwarded-proto", b"http"),
        ],
    }
    middleware = PublicOriginMiddleware(app, "8443", "https")
    asyncio.run(middleware(scope, None, None))
    assert seen["headers"]["x-forwarded-host"] == "example.com:8443"
    assert seen["headers"]["x-forwarded-proto"] == "https"


def test_public_proto_can_be_used_without_a_port_override():
    seen = {}

    async def app(scope, receive, send):
        seen["headers"] = MutableHeaders(scope=scope)

    scope = {
        "type": "http",
        "headers": [(b"x-forwarded-host", b"example.com:9443")],
    }
    middleware = PublicOriginMiddleware(app, public_proto="https")
    asyncio.run(middleware(scope, None, None))
    assert seen["headers"]["x-forwarded-host"] == "example.com:9443"
    assert seen["headers"]["x-forwarded-proto"] == "https"


def test_public_port_configuration_is_optional_and_validated():
    assert configured_public_port(None) is None
    assert configured_public_port("") is None
    assert configured_public_port(" 8443 ") == "8443"
    with pytest.raises(ValueError, match="from 1 to 65535"):
        configured_public_port("https")
    assert configured_public_proto(None) is None
    assert configured_public_proto(" HTTPS ") == "https"
    with pytest.raises(ValueError, match="http or https"):
        configured_public_proto("ftp")
