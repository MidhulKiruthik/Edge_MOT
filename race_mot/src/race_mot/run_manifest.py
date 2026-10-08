"""Private run-manifest helpers with source and credential redaction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from race_mot.stream_probe import redact_source


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    run_id: str,
    config_path: Path,
    source: str,
    source_kind: str,
    device: Mapping[str, Any] | None = None,
    model: Mapping[str, Any] | None = None,
    tracker: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a JSON-safe manifest without storing raw URLs or images."""
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    if not config_path.is_file():
        raise ValueError("config_path must point to a file")
    return {
        "schema_version": 1,
        "run_id": run_id,
        "source": redact_source(source),
        "source_kind": source_kind,
        "config_path": config_path.name,
        "config_sha256": file_sha256(config_path),
        "device": dict(device or {}),
        "model": dict(model or {}),
        "tracker": dict(tracker or {}),
        "privacy": {
            "raw_video_persistence": False,
            "annotated_export": False,
            "persistent_identity": False,
            "face_or_embedding_data": False,
        },
    }


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write once and refuse to overwrite an existing run record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
