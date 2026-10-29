"""Route original cleavage-neighbourhood sources to their raw directories."""
from pathlib import Path

from proteolysis_data_manager.acquisition.window_neighbourhood import merops, mmp_pics, dpp4_qpisa
from proteolysis_data_manager.utils.paths import repository_root

MODULES = {'merops': merops, 'mmp_pics': mmp_pics, 'dpp4_qpisa': dpp4_qpisa}


def default_root():
    """Return the configured window-neighbourhood dataset root.

    :return: Default series directory inside the checkout.
    :rtype: pathlib.Path
    """
    return repository_root() / 'data/proteolysis/window_neighbourhood_datasets'


def acquire_all(root=None, *, sources=None, force=False, merops_source=None):
    """Acquire selected window sources without adapters or matrix construction.

    :param root: Dataset root, with original source files stored under ``raw``.
    :param sources: Optional source identifiers; defaults to all three sources.
    :param bool force: Replace registered source files.
    :param merops_source: Optional directory containing the original MEROPS dictionaries.
    :return: Original source receipts indexed by source name.
    :rtype: dict
    :raises ValueError: If an unsupported source is requested.
    """
    root = Path(root) if root is not None else default_root()
    selected = list(MODULES if sources is None else sources)
    if unknown := set(selected) - MODULES.keys():
        raise ValueError(f'Unknown window sources: {sorted(unknown)}')
    return {name: MODULES[name].acquire(root / 'raw' / name, force=force,
                                       **({'source_root': merops_source} if name == 'merops' else {}))
            for name in selected}
