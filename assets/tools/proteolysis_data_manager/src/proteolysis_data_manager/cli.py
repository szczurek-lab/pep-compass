"""Command entry points; source parsing and data writes live in the library."""
import argparse
from pathlib import Path
from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def download_half_life(arguments=None):
    from proteolysis_data_manager.acquisition.half_life import SOURCES, acquire
    from proteolysis_data_manager.utils.paths import default_data_root
    parser = argparse.ArgumentParser(description='Download or import raw half-life datasets.')
    parser.add_argument('--source', choices=sorted(SOURCES), action='append')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--import-audit', type=Path, help='Import files from an earlier local audit.')
    parser.add_argument('--import-legacy', type=Path, help='Reuse the preceding DRAMP snapshot with provenance.')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(arguments)
    root = args.data_root if args.data_root is not None else default_data_root()
    failures = []
    for source in args.source or SOURCES:
        try:
            acquire(source, root, force=args.force, import_audit=args.import_audit, import_legacy=args.import_legacy)
        except (OSError, ValueError) as error:
            failures.append(source)
            print(f'ERROR {source}: {error}')
    return int(bool(failures))


def download_raw(arguments=None):
    """Acquire raw data series through their source orchestrators.

    :param arguments: Optional command-line argument list.
    :return: Zero if all selected sources were acquired successfully.
    :rtype: int
    """
    import logging
    from proteolysis_data_manager.acquisition.half_life.download_all import MODULES
    from proteolysis_data_manager.acquisition.download_all import acquire_all
    from proteolysis_data_manager.acquisition.window_neighbourhood.download_all import MODULES as WINDOWS
    parser = argparse.ArgumentParser(description='Acquire unchanged raw proteolysis source artifacts.')
    parser.add_argument('--type', choices=['half_life', 'window_neighbourhood', 'kinetics', 'all'], default='all')
    parser.add_argument('--source', choices=sorted(MODULES), action='append', help='Limit half-life sources.')
    parser.add_argument('--half-life-root', type=Path)
    parser.add_argument('--kinetics-root', type=Path)
    parser.add_argument('--window-root', type=Path)
    parser.add_argument('--window-source', choices=sorted(WINDOWS), action='append')
    parser.add_argument('--kinetics-source', choices=['brenda', 'sabio'], action='append')
    parser.add_argument('--brenda-archive', type=Path, help='Official archive obtained after licence acceptance.')
    parser.add_argument('--brenda-receipt', type=Path)
    parser.add_argument('--brenda-release')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(arguments)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    if args.source and args.type not in {'half_life', 'all'}:
        parser.error('--source selects half-life databases; use --window-source or --kinetics-source')
    data_types = ('half_life', 'window_neighbourhood', 'kinetics') if args.type == 'all' else (args.type,)
    try:
        acquire_all(half_life_root=args.half_life_root, kinetics_root=args.kinetics_root,
                    brenda_archive=args.brenda_archive, brenda_receipt=args.brenda_receipt,
                    brenda_release=args.brenda_release, force=args.force,
                    data_types=data_types, sources=args.source, window_root=args.window_root,
                    window_sources=args.window_source, kinetics_sources=args.kinetics_source)
    except (OSError, ValueError):
        logger.error('Raw acquisition failed', exc_info=True)
        return 1
    return 0


def download_window_neighbourhood(arguments=None):
    """Acquire original MEROPS, MMP/PICS and DPP4/qPISA artifacts.

    :param arguments: Optional command-line argument list.
    :return: Zero after successful source acquisition.
    :rtype: int
    """
    import logging
    from proteolysis_data_manager.acquisition.window_neighbourhood.download_all import MODULES, acquire_all
    parser = argparse.ArgumentParser(description='Download original cleavage-neighbourhood source files.')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--merops-source', type=Path, help='Original MEROPS dictionaries; defaults to data/merops.')
    parser.add_argument('--source', choices=sorted(MODULES), action='append')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(arguments)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    try:
        acquire_all(args.data_root, sources=args.source, force=args.force, merops_source=args.merops_source)
    except (OSError, ValueError):
        logger.error('Window-neighbourhood acquisition failed', exc_info=True)
        return 1
    return 0


def merge_merops(arguments=None):
    """Write the CSV view of the acquired original MEROPS dictionaries and matrices.

    :param arguments: Optional command-line argument list.
    :return: Zero after a complete CSV export.
    :rtype: int
    """
    import logging
    from proteolysis_data_manager.integration.window_neighbourhood.merops import merge_merops as export
    parser = argparse.ArgumentParser(description='Join original MEROPS matrix cells and dictionaries into CSV.')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-path', type=Path)
    args = parser.parse_args(arguments)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    try:
        print(export(args.data_root, output_path=args.output_path))
    except (OSError, ValueError):
        logger.error('MEROPS CSV export failed', exc_info=True)
        return 1
    return 0


def download_kinetics(arguments=None):
    """Acquire the original BRENDA archive and all SABIO-RK JSON pages.

    :param arguments: Optional command-line argument list.
    :return: Zero after successful source acquisition.
    :rtype: int
    """
    import logging
    from proteolysis_data_manager.acquisition.kinetics.download_all import MODULES, acquire_all
    from proteolysis_data_manager.acquisition.kinetics.sabio import DEFAULT_QUERY, DEFAULT_PAGE_SIZE
    parser = argparse.ArgumentParser(description='Acquire original BRENDA and SABIO-RK kinetics sources.')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--source', choices=sorted(MODULES), action='append')
    parser.add_argument('--brenda-archive', type=Path)
    parser.add_argument('--brenda-receipt', type=Path)
    parser.add_argument('--brenda-release')
    parser.add_argument('--sabio-query', default=DEFAULT_QUERY)
    parser.add_argument('--sabio-page-size', type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(arguments)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    try:
        acquire_all(args.data_root, sources=args.source, archive=args.brenda_archive,
                    receipt=args.brenda_receipt, release=args.brenda_release, force=args.force,
                    query=args.sabio_query, page_size=args.sabio_page_size)
    except (OSError, ValueError):
        logger.error('Kinetics acquisition failed', exc_info=True)
        return 1
    return 0


def normalize_half_life(arguments=None):
    """Build the peptide registry and experimental half-life observations.

    :param list arguments: Optional command-line arguments.
    :return: Zero after both datasets are written successfully.
    :rtype: int
    """
    import logging
    from proteolysis_data_manager.integration.half_life.datasets import build_half_time_datasets
    from proteolysis_data_manager.utils.paths import default_data_root
    parser = argparse.ArgumentParser(description='Build linked peptide and half-life experiment datasets.')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-dir', type=Path, help='Output directory; defaults to DATA_ROOT/processed.')
    args = parser.parse_args(arguments)
    root = args.data_root if args.data_root is not None else default_data_root()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    dataset = build_half_time_datasets(root, output_dir=args.output_dir)
    destination = args.output_dir if args.output_dir is not None else root / 'processed'
    print(f'{len(dataset.sequences_half_time)} peptide descriptions; {len(dataset.half_time_experiments)} source observations.')
    print(f'Output: {destination / "sequences_half_time.tsv"}')
    print(f'Output: {destination / "half_time_experiments.tsv"}')
    return 0


def import_brenda(arguments=None):
    from proteolysis_data_manager.acquisition.kinetics.brenda import default_root, import_snapshot
    parser = argparse.ArgumentParser(description='Import a locally acquired BRENDA archive with its original receipt.')
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--release')
    parser.add_argument('--data-root', type=Path)
    args = parser.parse_args(arguments)
    manifest = import_snapshot(args.archive, args.data_root or default_root(), receipt=args.receipt, release=args.release)
    print(f"BRENDA {manifest['source_version']}: {manifest['raw_path']} (verified import)")
    return 0


def normalize_kinetics(arguments=None):
    from proteolysis_data_manager.acquisition.kinetics.brenda import default_root
    from proteolysis_data_manager.integration.kinetics.normalization import normalize_brenda
    parser = argparse.ArgumentParser(description='Build all relational kinetics tables from the BRENDA raw snapshot.')
    parser.add_argument('--data-root', type=Path)
    args = parser.parse_args(arguments)
    dataset, manifest = normalize_brenda(args.data_root or default_root())
    print(f"{manifest['raw_kinetic_entries']} raw entries -> {len(dataset['kinetic_measurements'])} measurements; {len(dataset.tables)} tables")
    for name, frame in dataset.tables.items():
        print(f'{name}: {len(frame)} rows')
    return 0
