"""Tests for strict shared registration infrastructure."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from pep_compass.registry import (
    ComponentCatalog,
    DirectoryArtifact,
    FileArtifact,
    GlobArtifact,
    ModelDescriptor,
    ModelRegistry,
    Registry,
)


class Base:
    """Small test interface for registry type validation."""


class Implementation(Base):
    """Concrete implementation returned by the test factory."""


def test_registry_rejects_duplicate_registration() -> None:
    """A duplicate name must never depend on import order."""
    registry: Registry[Base] = Registry("test", expected_type=Base)

    @registry.register("method")
    def build_first() -> Base:
        return Implementation()

    with pytest.raises(ValueError, match="Duplicate registration"):

        @registry.register("method")
        def build_second() -> Base:
            return Implementation()


def test_registry_validates_services_and_factory_result() -> None:
    """Services are injected only by the composition root."""
    registry: Registry[Base] = Registry("test", expected_type=Base)

    @registry.register("method", services={"dependency"})
    def build(*, dependency: object, value: int = 1) -> Base:
        del dependency, value
        return Implementation()

    registry.validate("method", {"value": 2})
    assert isinstance(
        registry.build("method", {"value": 2}, services={"dependency": object()}),
        Implementation,
    )
    with pytest.raises(ValueError, match="cannot override services"):
        registry.validate("method", {"dependency": object()})


def test_component_catalog_applies_family_validation_before_build() -> None:
    """Family validation must also run for direct composition-root builds."""
    registry: Registry[Base] = Registry("test", expected_type=Base)
    validated: list[tuple[str, dict[str, object]]] = []

    @registry.register("method")
    def build(*, value: int = 1) -> Base:
        del value
        return Implementation()

    def validate(method: str, parameters: Mapping[str, object]) -> None:
        validated.append((method, dict(parameters)))

    catalog = ComponentCatalog()
    catalog.register_family("example", registry, validator=validate)
    result = catalog.build("example", "method", {"value": 2})

    assert isinstance(result, Implementation)
    assert validated == [("method", {"value": 2})]
    with pytest.raises(ValueError, match="Duplicate component family"):
        catalog.register_family("example", registry)


def test_model_registry_validates_file_and_ensemble_artifacts(tmp_path) -> None:
    """One resolver must support both fixed files and ensemble patterns."""
    registry = ModelRegistry()
    registry.register(
        ModelDescriptor(
            provider="oracle.example",
            name="default",
            directory="default",
            artifacts=(
                FileArtifact("metadata.json"),
                GlobArtifact("weights_*", expected_count=2),
            ),
            parameters={},
        )
    )
    directory = tmp_path / "default"
    directory.mkdir()
    (directory / "metadata.json").touch()
    (directory / "weights_0").touch()

    with pytest.raises(FileNotFoundError, match="expected 2.*found 1"):
        registry.resolve("oracle.example", "default", root=tmp_path)

    (directory / "weights_1").touch()
    _, paths = registry.resolve("oracle.example", "default", root=tmp_path)
    assert [path.name for path in paths] == [
        "metadata.json",
        "weights_0",
        "weights_1",
    ]


def test_model_registry_validates_directory_artifact_contents(tmp_path) -> None:
    """A directory model is complete only when its identifying files exist."""
    registry = ModelRegistry()
    registry.register(
        ModelDescriptor(
            provider="oracle.example",
            name="saved_model",
            directory=".",
            artifacts=(
                DirectoryArtifact(
                    "model",
                    required_files=("saved_model.pb", "variables/index"),
                ),
            ),
            parameters={},
        )
    )
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    (model_directory / "saved_model.pb").touch()

    with pytest.raises(FileNotFoundError, match="directory is incomplete"):
        registry.resolve("oracle.example", "saved_model", root=tmp_path)

    (model_directory / "variables").mkdir()
    (model_directory / "variables" / "index").touch()
    _, paths = registry.resolve("oracle.example", "saved_model", root=tmp_path)
    assert paths == (model_directory,)


def test_bundled_oracle_models_have_complete_registered_artifacts() -> None:
    """Every bundled oracle model declaration must resolve without loading it."""
    from pep_compass.optimization.components.oracles.model_registry import (
        resolve_oracle_model,
    )

    expected_artifact_counts = {
        "battleamp": 1,
        "eipred": 2,
        "mbc_attention": 1,
        "toxipep": 1,
    }
    for strategy, expected_count in expected_artifact_counts.items():
        descriptor, paths = resolve_oracle_model(strategy)
        assert descriptor.name == "default"
        assert len(paths) == expected_count
