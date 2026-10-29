"""Locate the shared raw receipt and downstream processing manifests."""
import json
from pathlib import Path


def download_manifest(root, source, *, legacy_fallback=False):
    path = Path(root) / 'raw/manifest.json'
    if legacy_fallback and not path.exists():
        for legacy in (Path(root) / 'manifests' / f'download_{source}.json',
                       Path(root) / 'raw' / source / 'download_manifest.json'):
            if legacy.exists():
                return legacy
    return path


def read_download_receipt(root, source, *, missing_ok=False):
    """Read one source from the shared receipt or an existing legacy snapshot.

    :param root: Data-series root containing ``raw``.
    :param str source: Source identifier.
    :param bool missing_ok: Return an empty mapping when no receipt exists.
    :return: Source receipt with its file records.
    :rtype: dict
    :raises FileNotFoundError: If no receipt exists and ``missing_ok`` is false.
    """
    path = download_manifest(root, source, legacy_fallback=True)
    if not path.exists() and missing_ok:
        return {}
    payload = json.loads(path.read_text())
    if 'sources' in payload:
        if missing_ok:
            return payload['sources'].get(source, {})
        return payload['sources'][source]
    return payload


def protocol_manifest(root, environment, source):
    return Path(root)/'manifests'/f'protocol_{environment}_{source}.json'


def aggregation_manifest(root, environment):
    return Path(root)/'manifests'/f'aggregation_{environment}.json'


def source_csv(root, environment, source):
    return Path(root)/'processed'/environment/'sources'/f'{source}.csv'
