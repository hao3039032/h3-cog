import pytest

from h3_sweep import build_matrix
from h3_tuning import CacheTuning, authorize_tuning, sign_tuning, validate_tuning


def test_signed_tuning_is_bounded_and_authorized():
    payload = {
        "profile": "custom",
        "reuse_threshold": 0.11,
        "start_percent": 0.2,
        "end_percent": 0.85,
        "sweep_id": "quality-01",
        "candidate_id": "cache-011",
        "expires_at": 1300,
    }
    tuning = authorize_tuning(payload, sign_tuning(payload, "secret"), secret="secret", now=1000)
    assert tuning == CacheTuning("custom", 0.11, 0.2, 0.85, False, "quality-01", "cache-011")


def test_tuning_rejects_tampering_expiry_and_unsafe_ranges():
    payload = {"profile": "balanced", "expires_at": 1300}
    signature = sign_tuning(payload, "secret")
    with pytest.raises(ValueError, match="authorization failed"):
        authorize_tuning({**payload, "profile": "aggressive"}, signature, secret="secret", now=1000)
    with pytest.raises(ValueError, match="authorization failed"):
        authorize_tuning(payload, signature, secret="secret", now=1400)
    with pytest.raises(ValueError, match="reuse_threshold"):
        validate_tuning({"profile": "custom", "reuse_threshold": 0.31})


def test_missing_tuning_is_lossless_default_and_partial_auth_fails():
    assert authorize_tuning(None, None, secret="secret", now=1000) is None
    with pytest.raises(ValueError, match="authorization failed"):
        authorize_tuning({"profile": "balanced"}, None, secret="secret", now=1000)


def test_sweep_matrix_keeps_fixed_input_and_adds_baseline():
    jobs = build_matrix(
        {"prompt": "same", "seed": 42},
        ["conservative", "balanced"],
        secret="secret",
        sweep_id="sweep-1",
        expires_at=1300,
    )
    assert [job["candidate_id"] for job in jobs] == ["off", "conservative", "balanced"]
    assert all(job["input"]["seed"] == 42 for job in jobs)
    assert "_tuning" not in jobs[0]["input"]
    signed = jobs[2]["input"]
    assert authorize_tuning(
        signed["_tuning"], signed["_tuning_signature"], secret="secret", now=1000
    ).profile == "balanced"
