"""Acquire all raw kinetics sources."""
from pathlib import Path
from proteolysis_data_manager.acquisition.kinetics import brenda, sabio

MODULES = {'brenda': brenda, 'sabio': sabio}


def acquire_all(root=None, *, sources=None, archive=None, receipt=None, release=None,
                force=False, query=sabio.DEFAULT_QUERY, page_size=sabio.DEFAULT_PAGE_SIZE):
    """Acquire selected kinetics sources without scientific integration.

    :param root: Kinetics series root.
    :param archive: Locally obtained official BRENDA archive.
    :param receipt: Original download provenance.
    :param release: Required BRENDA release.
    :param sources: Optional database selection; defaults to BRENDA and SABIO-RK.
    :param bool force: Replace registered snapshots.
    :param str query: Explicit SABIO-RK source query.
    :param int page_size: Requested SABIO-RK page size.
    :return: Source receipts indexed by database.
    :rtype: dict
    """
    root = Path(root) if root is not None else brenda.default_root()
    selected = list(MODULES if sources is None else sources)
    if unknown := set(selected) - MODULES.keys():
        raise ValueError(f'Unknown kinetics sources: {sorted(unknown)}')
    results = {}
    for source in selected:
        if source == 'brenda':
            results[source] = brenda.acquire(root / 'raw/brenda', archive=archive,
                                             receipt=receipt, release=release, force=force)
        else:
            results[source] = sabio.acquire(root / 'raw/sabio', query=query,
                                            page_size=page_size, force=force)
    return results
