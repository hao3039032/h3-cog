"""Bounded, authenticated operator tuning for H3 quality/speed sweeps."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any

TUNING_SECRET_ENV = "H3_TUNING_SECRET"
MAX_TUNING_LIFETIME_SECONDS = 3600
_IDENTIFIER = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_ALLOWED_KEYS = {
    "profile",
    "reuse_threshold",
    "start_percent",
    "end_percent",
    "verbose",
    "sweep_id",
    "candidate_id",
    "expires_at",
}

PROFILES = {
    "conservative": (0.08, 0.20, 0.80),
    "balanced": (0.12, 0.15, 0.90),
    "aggressive": (0.20, 0.15, 0.95),
}


@dataclass(frozen=True)
class CacheTuning:
    profile: str
    reuse_threshold: float
    start_percent: float
    end_percent: float
    verbose: bool = False
    sweep_id: str = "manual"
    candidate_id: str = "manual"

    def public_dict(self) -> dict[str, Any]:
        """Return telemetry-safe values. Authentication data is never included."""
        return asdict(self)


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sign_tuning(payload: dict[str, Any], secret: str) -> str:
    if not secret:
        raise ValueError("tuning secret must not be empty")
    digest = hmac.new(secret.encode(), _canonical(payload), hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _number(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc


def validate_tuning(payload: dict[str, Any]) -> CacheTuning:
    if not isinstance(payload, dict):
        raise ValueError("operator tuning must be a JSON object")
    unexpected = sorted(set(payload) - _ALLOWED_KEYS)
    if unexpected:
        raise ValueError("unsupported operator tuning fields: " + ", ".join(unexpected))

    profile = str(payload.get("profile", "balanced"))
    if profile == "custom":
        defaults = (0.12, 0.15, 0.90)
    elif profile in PROFILES:
        defaults = PROFILES[profile]
    else:
        raise ValueError("tuning profile must be conservative, balanced, aggressive, or custom")

    threshold = _number(payload, "reuse_threshold", defaults[0])
    start = _number(payload, "start_percent", defaults[1])
    end = _number(payload, "end_percent", defaults[2])
    if not 0 <= threshold <= 0.30:
        raise ValueError("reuse_threshold must be between 0 and 0.30")
    if not 0 <= start < end <= 1:
        raise ValueError("cache window must satisfy 0 <= start_percent < end_percent <= 1")

    sweep_id = str(payload.get("sweep_id", "manual"))
    candidate_id = str(payload.get("candidate_id", profile))
    if not _IDENTIFIER.fullmatch(sweep_id) or not _IDENTIFIER.fullmatch(candidate_id):
        raise ValueError("sweep_id and candidate_id must be 1-64 letters, digits, dots, dashes, or underscores")
    verbose = payload.get("verbose", False)
    if not isinstance(verbose, bool):
        raise ValueError("verbose must be a boolean")
    return CacheTuning(profile, threshold, start, end, verbose, sweep_id, candidate_id)


def authorize_tuning(
    payload: dict[str, Any] | None,
    signature: str | None,
    *,
    secret: str | None = None,
    now: int | None = None,
) -> CacheTuning | None:
    """Authenticate an optional tuning envelope and return bounded cache settings."""
    if payload is None and signature is None:
        return None
    if not isinstance(payload, dict) or not isinstance(signature, str):
        raise ValueError("operator tuning authorization failed")
    configured_secret = secret if secret is not None else os.getenv(TUNING_SECRET_ENV, "")
    expected = sign_tuning(payload, configured_secret) if configured_secret else ""
    if not expected or not hmac.compare_digest(expected, signature):
        raise ValueError("operator tuning authorization failed")

    current = int(time.time()) if now is None else int(now)
    expires_at = payload.get("expires_at")
    if isinstance(expires_at, bool):
        raise ValueError("operator tuning authorization failed")
    try:
        expiry = int(expires_at)
    except (TypeError, ValueError) as exc:
        raise ValueError("operator tuning authorization failed") from exc
    if expiry < current or expiry > current + MAX_TUNING_LIFETIME_SECONDS:
        raise ValueError("operator tuning authorization failed")
    return validate_tuning(payload)
