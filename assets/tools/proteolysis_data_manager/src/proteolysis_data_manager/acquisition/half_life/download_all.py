"""Acquire all or selected raw half-life sources."""
from pathlib import Path

from proteolysis_data_manager.acquisition.half_life import peplife2, pepmsnd, plifepred, dramp, cavaco
from proteolysis_data_manager.utils.paths import default_data_root
from proteolysis_data_manager.utils.storage import read_download_receipt

MODULES = {module.__name__.rsplit('.', 1)[-1]: module for module in (peplife2, pepmsnd, plifepred, dramp, cavaco)}
SOURCES = {name: module.FILES for name, module in MODULES.items()}
AUDIT_FILES = {('peplife2', 'natural.json'): 'peplife_natural.json',
 ('peplife2', 'modified.json'): 'peplife_modified.json',
 ('peplife2', 'complete.fasta'): 'peplife_fasta.txt',
 ('pepmsnd', 'dataset.xlsx'): 'pepmsnd.xlsx',
 ('plifepred', 'natural.xlsx'): 'plifepred_natural.xlsx',
 ('plifepred', 'mixed.xlsx'): 'plifepred_modified.xlsx',
 ('cavaco', 'article.xml'): 'PMC8301568.1.xml',
 ('cavaco', 'article.pdf'): 'PMC8301568.1.pdf',
 ('cavaco', 'supplement.docx'): 'CTS-14-1349-s001.docx',
 ('cavaco', 'supplement.pdf'): 'CTS-14-1349-s003.pdf'}


def acquire(source, root=None, *, force=False, import_audit=None, import_legacy=None):
    """Route one source to its unchanged raw destination.

    :param str source: One of the five half-life database identifiers.
    :param root: Series root; defaults to ``data/proteolysis/half_time_datasets``.
    :param bool force: Replace registered artifacts.
    :param import_audit: Optional preceding local audit directory.
    :param import_legacy: Optional preceding serum data root for DRAMP.
    :return: Source receipt.
    :rtype: dict
    :raises ValueError: If the source is unsupported or validation fails.
    """
    if source not in MODULES:
        raise ValueError(f'Unknown half-life source: {source}')
    root = Path(root) if root is not None else default_data_root()
    local_files, provenance = {}, {}
    if import_audit is not None:
        for (database, name), original in AUDIT_FILES.items():
            candidate = Path(import_audit) / original
            if database == source and candidate.is_file():
                local_files[name] = candidate
    if source == 'dramp' and import_legacy is not None:
        candidate = Path(import_legacy) / 'raw/dramp/stability_amps.xlsx'
        if candidate.is_file():
            local_files['stability_amps.xlsx'] = candidate
            old = read_download_receipt(import_legacy, source, missing_ok=True)
            entries = old.get('files', [])
            if isinstance(entries, list):
                provenance['stability_amps.xlsx'] = next((entry | {'retrieved_at': old.get('retrieved_at')} for entry in entries if Path(entry.get('path', '')).name == candidate.name), {})
            else:
                provenance = entries
    return MODULES[source].acquire(root / 'raw' / source, force=force, local_files=local_files, provenance=provenance)


def acquire_all(root=None, *, sources=None, **options):
    """Acquire all requested half-life sources in the same data series.

    :param root: Series data root.
    :param sources: Optional database identifiers; all are acquired by default.
    :param options: Options forwarded to :func:`acquire`.
    :return: Source receipts indexed by database.
    :rtype: dict
    """
    return {source: acquire(source, root, **options) for source in (sources if sources is not None else MODULES)}
