"""Shared registration infrastructure for PepCompass components and models."""

from pep_compass.registry.core import Registry, RegistryEntry
from pep_compass.registry.bootstrap import load_builtin_registrations
from pep_compass.registry.components import ComponentCatalog, component_catalog
from pep_compass.registry.models import (
    DirectoryArtifact,
    FileArtifact,
    GlobArtifact,
    ModelArtifact,
    ModelDescriptor,
    ModelRegistry,
    model_catalog,
)

__all__ = [
    "DirectoryArtifact",
    "FileArtifact",
    "ComponentCatalog",
    "GlobArtifact",
    "ModelDescriptor",
    "ModelArtifact",
    "ModelRegistry",
    "model_catalog",
    "Registry",
    "RegistryEntry",
    "load_builtin_registrations",
    "component_catalog",
]
