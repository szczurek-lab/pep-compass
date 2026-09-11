"""Shared declarations and filesystem validation for named model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pep_compass.utils.logger import get_custom_logger


logger = get_custom_logger(__name__)


@dataclass(frozen=True, slots=True)
class FileArtifact:
    """Require one model file relative to a resolved model directory."""

    relative_path: str


@dataclass(frozen=True, slots=True)
class GlobArtifact:
    """Require an exact number of files matching a pattern in one directory."""

    pattern: str
    expected_count: int


@dataclass(frozen=True, slots=True)
class DirectoryArtifact:
    """Require one model directory and selected files below it.

    :param relative_path: Directory relative to the model descriptor directory.
    :param required_files: Files whose presence identifies a complete directory
        artifact, for example the metadata and variables of a SavedModel.
    """

    relative_path: str
    required_files: tuple[str, ...] = ()


ModelArtifact = DirectoryArtifact | FileArtifact | GlobArtifact


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """Declare named files required by a selectable model variant."""

    provider: str
    name: str
    directory: str
    artifacts: tuple[ModelArtifact, ...]
    parameters: dict[str, Any]
    download_script: str | None = None


class ModelRegistry:
    """Store named model descriptors and validate their artifact sets."""

    def __init__(self) -> None:
        self._descriptors: dict[tuple[str, str], ModelDescriptor] = {}

    def register(self, descriptor: ModelDescriptor) -> None:
        """Register one unique ``(provider, name)`` descriptor."""
        key = (descriptor.provider, descriptor.name)
        if not descriptor.provider or not descriptor.name:
            raise ValueError("Model provider and name cannot be empty.")
        if key in self._descriptors:
            raise ValueError(
                f"Duplicate model registration: {descriptor.provider}/{descriptor.name}."
            )
        self._descriptors[key] = descriptor

    def names(self, provider: str) -> tuple[str, ...]:
        """Return model names registered under one provider."""
        return tuple(
            sorted(
                name
                for registered_provider, name in self._descriptors
                if registered_provider == provider
            )
        )

    def descriptor(self, provider: str, name: str) -> ModelDescriptor:
        """Return one model descriptor.

        :raises ValueError: If the provider/model pair is unknown.
        """
        try:
            return self._descriptors[(provider, name)]
        except KeyError as error:
            raise ValueError(
                f"Unknown model {provider}/{name}. Available: {list(self.names(provider))}."
            ) from error

    def resolve(
        self,
        provider: str,
        name: str,
        *,
        root: str | Path,
    ) -> tuple[ModelDescriptor, tuple[Path, ...]]:
        """Resolve and validate files for one registered model.

        :param provider: Model provider namespace.
        :param name: Model variant name.
        :param root: Root containing descriptor-relative model directories.
        :return: Descriptor and deterministically ordered artifact paths.
        """
        descriptor = self.descriptor(provider, name)
        directory = Path(root) / descriptor.directory
        if not directory.is_dir():
            self._missing(descriptor, f"model directory not found: {directory}")
        paths: list[Path] = []
        for artifact in descriptor.artifacts:
            if isinstance(artifact, DirectoryArtifact):
                path = directory / artifact.relative_path
                if not path.is_dir():
                    self._missing(descriptor, f"model directory not found: {path}")
                for required_file in artifact.required_files:
                    required_path = path / required_file
                    if not required_path.is_file():
                        self._missing(
                            descriptor,
                            f"model directory is incomplete; file not found: "
                            f"{required_path}",
                        )
                paths.append(path)
            elif isinstance(artifact, FileArtifact):
                path = directory / artifact.relative_path
                if not path.is_file():
                    self._missing(descriptor, f"model file not found: {path}")
                paths.append(path)
            else:
                matches = tuple(sorted(directory.glob(artifact.pattern)))
                if len(matches) != artifact.expected_count:
                    self._missing(
                        descriptor,
                        f"expected {artifact.expected_count} files matching "
                        f"{artifact.pattern!r}, found {len(matches)} under {directory}",
                    )
                paths.extend(matches)
        logger.debug(
            "Resolved model provider=%s name=%s artifacts=%s directory=%s.",
            provider,
            name,
            len(paths),
            directory,
        )
        return descriptor, tuple(paths)

    @staticmethod
    def _missing(descriptor: ModelDescriptor, detail: str) -> None:
        hint = (
            f" Run {descriptor.download_script} to install the model."
            if descriptor.download_script
            else ""
        )
        raise FileNotFoundError(f"{detail}.{hint}")


# One catalog shared by autoencoders and all model-backed components.
model_catalog = ModelRegistry()
