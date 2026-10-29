"""Explicit source selection; adding a dataset requires one parser registration."""

from importlib import import_module
from pathlib import Path

from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

SOURCES = {
    **{
        name: "half_time_datasets"
        for name in ("peplife2", "pepmsnd", "dramp", "plifepred", "cavaco")
    },
    **{name: "kinetics_datasets" for name in ("brenda", "sabio")},
    **{
        name: "window_neighbourhood_datasets"
        for name in (
            "merops",
            "mmp_pics",
            "mmp_pics_qf",
            "qpisa_ev1",
            "qpisa_ev4",
            "qpisa_ev6",
        )
    },
}
PARSER_GROUPS = {
    "peplife2": "half_life",
    "pepmsnd": "half_life",
    "dramp": "half_life",
    "plifepred": "half_life",
    "cavaco": "half_life",
    "brenda": "kinetics",
    "sabio": "kinetics",
    "merops": "window_neighbourhood",
    "mmp_pics": "window_neighbourhood",
    "mmp_pics_qf": "window_neighbourhood",
    "qpisa_ev1": "window_neighbourhood",
    "qpisa_ev4": "window_neighbourhood",
    "qpisa_ev6": "window_neighbourhood",
}

HALF_LIFE_SOURCES = ("peplife2", "pepmsnd", "dramp", "plifepred", "cavaco")


def parse_dataset(name, data_root, *, data=None, verify=True):
    """Call the one dataset-specific parser through the common interface.

    :param str name: Registered dataset name.
    :param data_root: Parent directory containing the acquired series roots.
    :param data: Optional already-loaded source data accepted by that parser.
    :param bool verify: Check original source bytes before conversion.
    :return: A validated common dataset.
    :raises ValueError: If the dataset is not registered.
    """
    if name not in SOURCES:
        raise ValueError(f"Unknown dataset {name}; available: {sorted(SOURCES)}")
    if verify:
        verify_source(name, data_root)
    logger.info("Converting dataset %s", name)
    module = import_module(
        f"proteolysis_data_manager.integration.merge.parsers.{PARSER_GROUPS[name]}.{name}"
    )
    result = module.parse(Path(data_root) / SOURCES[name], data=data)
    logger.debug(
        "Parsed %s table sizes: %s",
        name,
        {table: len(frame) for table, frame in result.tables.items()},
    )
    return result


def verify_source(name, data_root):
    """Verify the acquired bytes against the one existing raw-series manifest.

    :param str name: Registered parser name.
    :param data_root: Parent directory of the acquired series roots.
    :return: SHA-256 of the acquisition manifest.
    :raises ValueError: If a registered source artifact changed.
    """
    import json

    from proteolysis_data_manager.utils.io import sha256

    source = (
        "dpp4_qpisa"
        if name.startswith("qpisa_")
        else ("mmp_pics" if name == "mmp_pics_qf" else name)
    )
    root = Path(data_root) / SOURCES[name]
    manifest = root / "raw/manifest.json"
    receipt = json.loads(manifest.read_text())["sources"][source]
    for filename, record in receipt["files"].items():
        if sha256(root / "raw" / source / filename) != record["sha256"]:
            raise ValueError(f"Acquired source changed: {source}/{filename}")
    return sha256(manifest)
