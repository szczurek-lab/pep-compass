"""Canonical locations for half-life data and centralized manifests."""
from pathlib import Path


def download_manifest(root, source, *, legacy_fallback=False):
    path=Path(root)/'manifests'/f'download_{source}.json'
    legacy=Path(root)/'raw'/source/'download_manifest.json'
    return legacy if legacy_fallback and not path.exists() and legacy.exists() else path


def protocol_manifest(root, environment, source):
    return Path(root)/'manifests'/f'protocol_{environment}_{source}.json'


def aggregation_manifest(root, environment):
    return Path(root)/'manifests'/f'aggregation_{environment}.json'


def source_csv(root, environment, source):
    return Path(root)/'processed'/environment/'sources'/f'{source}.csv'
