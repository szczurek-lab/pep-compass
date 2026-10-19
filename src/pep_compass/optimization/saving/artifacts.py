"""Direct artifact validation and build-scoped model provenance."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pep_compass.utils.logger import get_custom_logger

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Never


logger = get_custom_logger(__name__)
_artifact_capture = ContextVar("pep_compass_artifact_capture", default=None)


@dataclass(frozen=True)
class FileArtifact:
    relative_path: str


@dataclass(frozen=True)
class GlobArtifact:
    pattern: str
    expected_count: int


@dataclass(frozen=True)
class DirectoryArtifact:
    relative_path: str
    required_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelArtifacts:
    provider: str
    name: str
    directory: str
    artifacts: tuple
    download_script: str | None = None
    base_directory: Path | None = None

    def resolve(
        self, model: str | None = None, *, root: str | Path | None = None
    ) -> tuple[Path, ...]:
        """Validate this declared variant against its model-owned default root.

        :param model: Optional expected variant name.
        :param root: Optional external root replacing base_directory.
        :return: Deterministically ordered validated artifact paths.
        :raises ValueError: If the variant or base directory is invalid.

        :type model: str | None
        :type root: str | Path | None
        :rtype: tuple[Path, ...]
        """
        if model is not None and model != self.name:
            raise ValueError(f"Unknown model variant: {model!r}.")
        root = self.base_directory if root is None else root
        if root is None:
            raise ValueError("Model artifacts require a base directory.")
        return resolve_artifacts(self, root=root)


@contextmanager
def capture_model_artifacts() -> Iterator[list[object]]:
    """Collect files actually opened while preparing an executable graph.

    :return: Result of this operation.
    :rtype: Iterator[list[object]]
    """
    records = []
    token = _artifact_capture.set(records)
    try:
        yield records
    finally:
        _artifact_capture.reset(token)


def _missing(descriptor: ModelArtifacts, detail: str) -> Never:
    """Missing.

    :param descriptor: Descriptor.
    :type descriptor: ModelArtifacts
    :param detail: Detail.
    :type detail: str
    :return: Result of this operation.
    :rtype: Never
    """
    hint = (
        f" Run {descriptor.download_script} to install the model."
        if descriptor.download_script
        else ""
    )
    raise FileNotFoundError(f"{detail}.{hint}")


def resolve_artifacts(
    descriptor: ModelArtifacts, *, root: str | Path | None
) -> tuple[Path, ...]:
    """Validate explicit artifact requirements and capture file SHA-256 values.

    :param descriptor: ModelArtifacts supplied directly by a model implementation.
    :param root: Base directory for its relative paths.
    :return: Deterministically ordered file/directory paths.
    :raises FileNotFoundError: If any declared artifact is missing/incomplete.

    :type descriptor: ModelArtifacts
    :type root: str | Path | None
    :rtype: tuple[Path, ...]
    """
    provider, name = descriptor.provider, descriptor.name
    directory = Path(root) / descriptor.directory
    if not directory.is_dir():
        _missing(descriptor, f"model directory not found: {directory}")
    paths: list[Path] = []
    for artifact in descriptor.artifacts:
        if isinstance(artifact, DirectoryArtifact):
            path = directory / artifact.relative_path
            if not path.is_dir():
                _missing(descriptor, f"model directory not found: {path}")
            for required_file in artifact.required_files:
                required_path = path / required_file
                if not required_path.is_file():
                    _missing(
                        descriptor,
                        f"model directory is incomplete; file not found: "
                        f"{required_path}",
                    )
            paths.append(path)
        elif isinstance(artifact, FileArtifact):
            path = directory / artifact.relative_path
            if not path.is_file():
                _missing(descriptor, f"model file not found: {path}")
            paths.append(path)
        elif isinstance(artifact, GlobArtifact):
            matches = tuple(sorted(directory.glob(artifact.pattern)))
            if len(matches) != artifact.expected_count:
                _missing(
                    descriptor,
                    f"expected {artifact.expected_count} files matching "
                    f"{artifact.pattern!r}, found {len(matches)} under {directory}",
                )
            paths.extend(matches)
        else:
            raise TypeError("Unsupported model artifact requirement.")
    logger.debug(
        "Resolved model provider=%s name=%s artifacts=%s directory=%s.",
        provider,
        name,
        len(paths),
        directory,
    )
    capture = _artifact_capture.get()
    if capture is not None:
        files = []
        for artifact in paths:
            for path in (
                sorted(artifact.rglob("*")) if artifact.is_dir() else (artifact,)
            ):
                if path.is_file():
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    files.append(
                        {"path": str(path.resolve()), "sha256": digest.hexdigest()}
                    )
        capture.append({"provider": provider, "model": name, "files": files})
    return tuple(paths)
