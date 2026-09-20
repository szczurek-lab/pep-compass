"""Shared filesystem, receipt, and HTTP helpers for serum-source acquisition."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


DEFAULT_TIMEOUT_SECONDS = 60


def repository_root() -> Path:
    """Return the repository root inferred from this module location.

    :return: Absolute repository root.
    """

    return Path(__file__).resolve().parents[4]


def acquisition_root() -> Path:
    """Return the root directory for serum-calibration source artifacts.

    :return: Absolute serum-calibration data root.
    """

    return repository_root() / "data" / "serum_proteolysis_calibration"


def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format.

    :return: Timestamp with an explicit UTC offset.
    """

    return datetime.now(UTC).isoformat()


def sha256sum(path: Path) -> str:
    """Return the SHA-256 checksum of one file.

    :param path: Existing file to hash.
    :return: Lowercase hexadecimal checksum.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically with deterministic formatting.

    :param path: Destination JSON file.
    :param payload: JSON-serializable object.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".part")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_bytes(path: Path, payload: bytes) -> None:
    """Write source bytes atomically without altering their representation.

    :param path: Destination path.
    :param payload: Unmodified source bytes.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".part")
    temporary_path.write_bytes(payload)
    temporary_path.replace(path)


def download_url(
    url: str,
    destination: Path,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    accept: str = "*/*",
) -> str:
    """Download one public URL atomically and return its checksum.

    :param url: Source HTTP(S) URL.
    :param destination: Destination file path.
    :param timeout_seconds: Socket timeout in seconds.
    :param accept: HTTP ``Accept`` header.
    :return: SHA-256 checksum of the stored response.
    :raises OSError: If the response or local write fails.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    request = Request(
        url,
        headers={
            "Accept": accept,
            "User-Agent": "pep-compass-serum-acquisition/1.0",
        },
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response, temporary_path.open("wb") as handle:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(chunk)
                digest.update(chunk)
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def raw_file_record(path: Path, source_url: str, status: str) -> dict[str, object]:
    """Create one receipt entry for an existing source file.

    :param path: Existing source artifact.
    :param source_url: URL or manual-source identifier.
    :param status: Acquisition outcome.
    :return: Receipt-file metadata.
    """

    return {
        "path": str(path),
        "source_url": source_url,
        "sha256": sha256sum(path),
        "bytes": path.stat().st_size,
        "status": status,
    }


def write_receipt(
    path: Path,
    source: str,
    source_version_or_accession: str,
    request_or_url: str,
    files: list[dict[str, object]],
    status: str = "complete",
) -> None:
    """Write a standardized acquisition receipt.

    :param path: Receipt destination.
    :param source: Source-system name.
    :param source_version_or_accession: Version or accession identifier.
    :param request_or_url: Query or source URL.
    :param files: Downloaded-file records.
    :param status: Overall source status.
    """

    write_json(
        path,
        {
            "source": source,
            "retrieved_at": utc_now(),
            "source_version_or_accession": source_version_or_accession,
            "request_or_url": request_or_url,
            "status": status,
            "files": files,
        },
    )


def merge_receipt(
    path: Path,
    source: str,
    source_version_or_accession: str,
    request_or_url: str,
    files: list[dict[str, object]],
) -> None:
    """Merge files into one source-level receipt without dropping earlier files.

    :param path: Existing or new receipt path.
    :param source: Shared source-system name.
    :param source_version_or_accession: Source version or accession label.
    :param request_or_url: Current request URL.
    :param files: New or refreshed source-file records.
    """

    existing_files: list[dict[str, object]] = []
    existing_requests: list[str] = []
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("source") == source and isinstance(payload.get("files"), list):
                existing_files = [item for item in payload["files"] if isinstance(item, dict)]
                old_request = payload.get("request_or_url")
                if isinstance(old_request, str) and old_request:
                    existing_requests.extend(value for value in old_request.split("; ") if value)
        except (OSError, json.JSONDecodeError):
            existing_files = []
            existing_requests = []

    merged_by_path = {
        str(record.get("path")): record
        for record in existing_files
        if isinstance(record.get("path"), str)
    }
    merged_by_path.update(
        {
            str(record.get("path")): record
            for record in files
            if isinstance(record.get("path"), str)
        }
    )
    requests = list(dict.fromkeys([*existing_requests, request_or_url]))
    write_receipt(
        path,
        source,
        source_version_or_accession,
        "; ".join(requests),
        sorted(merged_by_path.values(), key=lambda record: str(record["path"])),
    )
