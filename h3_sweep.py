"""Create short-lived, signed RunPod request matrices for H3 cache A/B tests."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

from h3_tuning import PROFILES, TUNING_SECRET_ENV, sign_tuning, validate_tuning


def build_matrix(
    base_input: dict,
    profiles: list[str],
    *,
    secret: str,
    sweep_id: str,
    expires_at: int,
) -> list[dict]:
    jobs = [{"input": dict(base_input), "candidate_id": "off"}]
    for profile in profiles:
        payload = {
            "profile": profile,
            "sweep_id": sweep_id,
            "candidate_id": profile,
            "expires_at": expires_at,
        }
        validate_tuning(payload)
        values = dict(base_input)
        values["_tuning"] = payload
        values["_tuning_signature"] = sign_tuning(payload, secret)
        jobs.append({"input": values, "candidate_id": profile})
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON file containing the common RunPod input object")
    parser.add_argument("--profiles", nargs="+", default=list(PROFILES), choices=list(PROFILES))
    parser.add_argument("--sweep-id", default=f"h3-{uuid.uuid4().hex[:10]}")
    parser.add_argument("--expires-in", type=int, default=900, choices=range(60, 3601), metavar="SECONDS")
    args = parser.parse_args()
    secret = os.getenv(TUNING_SECRET_ENV, "")
    if not secret:
        parser.error(f"{TUNING_SECRET_ENV} must be set")
    base = json.loads(args.input.read_text())
    if not isinstance(base, dict):
        parser.error("input file must contain one JSON object")
    jobs = build_matrix(
        base,
        args.profiles,
        secret=secret,
        sweep_id=args.sweep_id,
        expires_at=int(time.time()) + args.expires_in,
    )
    print(json.dumps({"sweep_id": args.sweep_id, "jobs": jobs}, indent=2))


if __name__ == "__main__":
    main()
