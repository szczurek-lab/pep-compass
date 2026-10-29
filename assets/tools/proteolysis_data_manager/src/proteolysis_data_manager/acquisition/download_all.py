"""Acquire independent raw-data series using their source orchestrators."""
from proteolysis_data_manager.acquisition.half_life.download_all import acquire_all as acquire_half_life
from proteolysis_data_manager.acquisition.kinetics.download_all import acquire_all as acquire_kinetics
from proteolysis_data_manager.acquisition.window_neighbourhood.download_all import acquire_all as acquire_windows


def acquire_all(*, half_life_root=None, kinetics_root=None, brenda_archive=None,
                brenda_receipt=None, brenda_release=None, force=False,
                data_types=('half_life', 'window_neighbourhood', 'kinetics'), sources=None,
                window_root=None, window_sources=None, kinetics_sources=None):
    """Acquire selected independent source series without scientific adaptation.

    :param half_life_root: Half-life data-series root.
    :param kinetics_root: Kinetics data-series root.
    :param brenda_archive: Locally obtained licensed archive.
    :param brenda_receipt: Original BRENDA acquisition provenance.
    :param brenda_release: Required BRENDA release.
    :param bool force: Reacquire selected sources, retaining BRENDA import provenance.
    :param data_types: Independent data types to acquire.
    :param sources: Optional half-life database selection.
    :param window_root: Original cleavage-neighbourhood dataset root.
    :param window_sources: Optional window source selection.
    :param kinetics_sources: Optional kinetics database selection.
    :return: Receipts grouped by independent data type.
    :rtype: dict
    """
    unknown = set(data_types) - {'half_life', 'window_neighbourhood', 'kinetics'}
    if unknown:
        raise ValueError(f'Unsupported data types: {sorted(unknown)}')
    receipts = {}
    if 'half_life' in data_types:
        receipts['half_life'] = acquire_half_life(half_life_root, sources=sources, force=force)
    if 'window_neighbourhood' in data_types:
        receipts['window_neighbourhood'] = acquire_windows(window_root, sources=window_sources, force=force)
    if 'kinetics' in data_types:
        receipts['kinetics'] = acquire_kinetics(kinetics_root, archive=brenda_archive,
                                              receipt=brenda_receipt, release=brenda_release,
                                              sources=kinetics_sources, force=force)
    return receipts
