"""Command entry points; source parsing and data writes live in the library."""
import argparse
from pathlib import Path


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


def normalize_half_life(arguments=None):
    from proteolysis_data_manager.integration.half_life.normalization import build_normalized_dataset
    from proteolysis_data_manager.utils.paths import default_data_root
    parser = argparse.ArgumentParser(description='Build the compact half-life Parquet and source remarks.')
    parser.add_argument('--data-root', type=Path)
    args = parser.parse_args(arguments)
    root = args.data_root if args.data_root is not None else default_data_root()
    manifest = build_normalized_dataset(root)
    print(f'{manifest["raw_representations"]} source representations -> {manifest["rows"]} normalized measurements; {len(manifest["columns"])} columns.')
    print(f'Output: {root / "normalized/observations.parquet"}; notes: normalized/remarks_<source>.md')
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
