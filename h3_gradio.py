"""Small Gradio proxy helpers shared by application deployments."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from starlette.datastructures import MutableHeaders


ASGIApp = Callable[
    [dict[str, Any], Callable[..., Awaitable[None]], Callable[..., Awaitable[None]]],
    Awaitable[None],
]


def configured_public_port(value: str | None) -> str | None:
    """Validate GRADIO_PUBLIC_PORT without assuming a specific proxy."""
    if value is None or not value.strip():
        return None
    port = value.strip()
    if not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError(
            f"GRADIO_PUBLIC_PORT must be a port from 1 to 65535, got {value!r}"
        )
    return port


def configured_public_proto(value: str | None) -> str | None:
    """Validate an explicit external protocol override."""
    if value is None or not value.strip():
        return None
    protocol = value.strip().lower()
    if protocol not in {"http", "https"}:
        raise ValueError(f"GRADIO_PUBLIC_PROTO must be http or https, got {value!r}")
    return protocol


def _has_explicit_port(host: str) -> bool:
    bracket_end = host.rfind("]")
    if bracket_end != -1:
        return host.find(":", bracket_end) != -1
    return host.count(":") == 1


def add_missing_public_port(forwarded_host: str, public_port: str) -> str:
    """Append a port to forwarded hosts that do not already declare one."""
    hosts = [host.strip() for host in forwarded_host.split(",")]
    return ", ".join(
        host if not host or _has_explicit_port(host) else f"{host}:{public_port}"
        for host in hosts
    )


class PublicOriginMiddleware:
    """Make proxied request origins match the configured public origin."""

    def __init__(
        self,
        app: ASGIApp,
        public_port: str | None = None,
        public_proto: str | None = None,
    ) -> None:
        self.app = app
        self.public_port = configured_public_port(public_port)
        self.public_proto = configured_public_proto(public_proto)

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Awaitable[None]],
        send: Callable[..., Awaitable[None]],
    ) -> None:
        if scope["type"] == "http" and (
            self.public_port is not None or self.public_proto is not None
        ):
            headers = MutableHeaders(scope=scope)
            forwarded_host = headers.get("x-forwarded-host")
            if forwarded_host:
                if self.public_port is not None:
                    headers["x-forwarded-host"] = add_missing_public_port(
                        forwarded_host, self.public_port
                    )
                if self.public_proto is not None:
                    headers["x-forwarded-proto"] = self.public_proto
        await self.app(scope, receive, send)
