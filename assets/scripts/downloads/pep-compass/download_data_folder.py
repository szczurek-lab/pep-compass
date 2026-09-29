#!/usr/bin/env python3
"""Download and install the shared pep-compass data archive."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


FILE_ID = "1qjL2CX47s-2T_TTl5-fmoTFIVP81381c"
DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download"
    f"?id={FILE_ID}&export=download&confirm=t"
)
EXPECTED_DIRECTORIES = {
    "merops",
    "peptides_data",
    "reference",
    "serum_proteolysis_calibration",
}
EXPECTED_FILE = "data/peptides_data/peptides_raw/peptides.csv"
CHUNK_SIZE = 1024 * 1024
PROGRESS_INTERVAL = 25 * 1024 * 1024


def _repository_root() -> Path:
    """Return the repository root from this script's location.

    :returns: Absolute path to the repository root.
    :rtype: pathlib.Path
    """
    return Path(__file__).resolve().parents[4]


def _download_archive(destination: Path) -> None:
    """Stream the shared Drive archive to ``destination`` and report progress.

    :param destination: Temporary path where the ZIP archive is written.
    :type destination: pathlib.Path
    :returns: ``None``.
    :raises OSError: If the archive cannot be written locally.
    :raises urllib.error.URLError: If the Drive download fails.
    """
    request = urllib.request.Request(
        DOWNLOAD_URL,
        headers={"User-Agent": "pep-compass-data-download/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        total_size = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        next_report = PROGRESS_INTERVAL
        with destination.open("wb") as archive_file:
            while chunk := response.read(CHUNK_SIZE):
                archive_file.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    downloaded_mib = downloaded / (1024 * 1024)
                    if total_size:
                        total_mib = total_size / (1024 * 1024)
                        print(
                            f"Downloaded {downloaded_mib:.0f} / "
                            f"{total_mib:.0f} MiB"
                        )
                    else:
                        print(f"Downloaded {downloaded_mib:.0f} MiB")
                    next_report += PROGRESS_INTERVAL

    print(f"Downloaded {downloaded / (1024 * 1024):.0f} MiB")


def _validate_archive(archive_path: Path) -> None:
    """Reject unexpected, unsafe, or incomplete archives before extraction.

    :param archive_path: Path to the downloaded ZIP archive.
    :type archive_path: pathlib.Path
    :returns: ``None``.
    :raises ValueError: If an archive member is unsafe or the layout is wrong.
    :raises zipfile.BadZipFile: If the file is not a valid ZIP archive.
    """
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        file_members = [member for member in members if not member.is_dir()]
        roots: set[str] = set()
        directories: set[str] = set()

        for member in members:
            member_path = PurePosixPath(member.filename)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or "\\" in member.filename
            ):
                raise ValueError(f"Unsafe path in archive: {member.filename}")
            if member_path.parts:
                roots.add(member_path.parts[0])
                if len(member_path.parts) > 1:
                    directories.add(member_path.parts[1])

        file_names = {member.filename for member in file_members}
        if roots != {"data"}:
            raise ValueError("Archive must contain a single top-level 'data/' folder.")
        if directories != EXPECTED_DIRECTORIES:
            raise ValueError(
                "Archive data folders do not match the expected pep-compass layout."
            )
        if EXPECTED_FILE not in file_names:
            raise ValueError(f"Required dataset file is missing: {EXPECTED_FILE}")


def _install_archive(archive_path: Path, data_directory: Path) -> None:
    """Extract validated archive contents and merge them into ``data_directory``.

    Existing paths from the archive are replaced; other files in the destination
    are preserved.

    :param archive_path: Path to the validated ZIP archive.
    :type archive_path: pathlib.Path
    :param data_directory: Destination corresponding to the archive's ``data/``.
    :type data_directory: pathlib.Path
    :returns: ``None``.
    :raises OSError: If extraction or copying fails.
    :raises zipfile.BadZipFile: If the archive data are corrupt.
    """
    with tempfile.TemporaryDirectory(prefix="pep-compass-data-extract-") as temporary:
        extraction_directory = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extraction_directory)

        source_directory = extraction_directory / "data"
        data_directory.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_directory, data_directory, dirs_exist_ok=True)


def main() -> int:
    """Download and install the data archive under the repository.

    :returns: Process exit status (zero on success).
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_repository_root() / "data",
        help="destination data directory (default: <repository>/data)",
    )
    args = parser.parse_args()
    data_directory = args.data_dir.expanduser().resolve()

    try:
        with tempfile.TemporaryDirectory(prefix="pep-compass-data-download-") as temporary:
            archive_path = Path(temporary) / "data.zip"
            print("Downloading data.zip from Google Drive…")
            _download_archive(archive_path)
            _validate_archive(archive_path)
            print(f"Extracting data into {data_directory}")
            _install_archive(archive_path, data_directory)
    except (OSError, ValueError, urllib.error.URLError, zipfile.BadZipFile) as error:
        print(f"Data download failed: {error}", file=sys.stderr)
        return 1

    print(f"Data installed in {data_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
