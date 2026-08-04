"""Mirror only the tested H3 5090 files to a public Cloudflare R2 prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from weights import FILES, REPO, license_accepted


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=os.getenv("R2_BUCKET", "appstatic"))
    parser.add_argument("--prefix", default="models")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not license_accepted():
        parser.error("set MINIMAX_H3_LICENSE_ACCEPTED=1 after reviewing MiniMax's license")

    from huggingface_hub import hf_hub_download

    entries = []
    local_files = []
    for relative in FILES:
        downloaded = Path(hf_hub_download(REPO, relative))
        entry = {"path": relative, "size": downloaded.stat().st_size, "sha256": sha256(downloaded)}
        entries.append(entry)
        local_files.append((relative, downloaded))
        print(json.dumps(entry), flush=True)
    if args.dry_run:
        return 0

    import boto3

    endpoint = os.getenv("R2_ENDPOINT") or os.getenv("R2_ENDPOINT_URL")
    if not endpoint:
        parser.error("R2_ENDPOINT is required")
    client = boto3.client("s3", endpoint_url=endpoint)
    root = f"{args.prefix.rstrip('/')}/{REPO}"
    for relative, filename in local_files:
        client.upload_file(
            str(filename),
            args.bucket,
            f"{root}/{relative}",
            ExtraArgs={"ContentType": "application/octet-stream", "CacheControl": "public, max-age=31536000, immutable"},
        )
    body = (json.dumps({entry["path"]: entry for entry in entries}, indent=2) + "\n").encode()
    client.put_object(
        Bucket=args.bucket,
        Key=f"{root}/manifest.json",
        Body=body,
        ContentType="application/json",
        CacheControl="public, max-age=300",
    )
    print(f"mirrored {len(entries)} files to s3://{args.bucket}/{root}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
