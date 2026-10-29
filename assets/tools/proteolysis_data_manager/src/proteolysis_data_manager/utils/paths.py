"""Locate repository defaults without coupling the library to a fixed depth."""
from pathlib import Path


def repository_root(start=None):
    """Find checkout defaults; callers outside a checkout should pass data paths."""
    starts = [Path(start)] if start is not None else [Path.cwd(), Path(__file__).resolve().parent]
    for candidate in starts:
        candidate = candidate.resolve()
        for parent in [candidate, *candidate.parents]:
            if (parent / 'src/pep_compass').is_dir() and (parent / 'assets/tools').is_dir():
                return parent
    raise FileNotFoundError('Repository defaults unavailable; pass an explicit data-root path.')


def default_data_root():
    return repository_root() / 'data/proteolysis/half_time_datasets'
