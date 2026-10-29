"""Import Karol's original marginal profiles without inventing individual events."""

import json
from pathlib import Path

import numpy as np

from proteolysis_data_manager.integration.merge.contract.builder import ContractBuilder
from proteolysis_data_manager.integration.merge.contract.schema import stable_id
from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def add_profile(b, root, index, entry, enzyme):
    """Add one original code profile and every original axis-labelled cell.

    :param b: Common builder.
    :param root: Original raw MEROPS root.
    :param index: Original source index.
    :param entry: Dataset dictionary.
    :param enzyme: Original per-code dictionary.
    :return: Stable profile ID.
    """
    folder = Path(root) / entry["folder"]
    row = enzyme["row"]
    profile_id = stable_id("profile", "merops", entry["folder"], row)
    if (profile_id,) in b.rows["specificity_profiles"]:
        return profile_id
    metadata = json.loads((folder / "codes.json").read_text())
    counts = np.load(folder / "matrices_counts.npy", allow_pickle=False)[row]  # (P, A)
    logprob = np.load(folder / "matrices_logprob.npy", allow_pickle=False)[
        row
    ]  # (P, A)
    if counts.shape != logprob.shape or counts.shape != (8, 20):
        raise ValueError("Incorrect MEROPS axes")
    if (
        metadata["positions"] != index["positions"]
        or metadata["amino_acids"] != index["amino_acids"]
    ):
        raise ValueError("MEROPS dictionary axes disagree")
    rid = b.source(
        "merops",
        str(entry["dataset_index"]),
        enzyme["code"],
        folder / "matrices_counts.npy",
        f"row/{row}",
        raw={
            "enzyme": enzyme,
            "dataset": entry,
            "axes": {
                "positions": metadata["positions"],
                "amino_acids": metadata["amino_acids"],
            },
            "log_probability_file": str(folder / "matrices_logprob.npy"),
            "dictionary_file": str(folder / "codes.json"),
        },
    )
    eid = b.enzyme(
        rid,
        ("merops", entry["folder"], enzyme["code"]),
        name=enzyme["name"],
        code=enzyme["code"],
        organism=entry.get("scientific"),
    )
    b.add(
        "specificity_profiles",
        profile_id=profile_id,
        dataset_index=entry["dataset_index"],
        folder=entry["folder"],
        merops_code=enzyme["code"],
        enzyme_id=eid,
        n_cleavages=enzyme["n_cleavages"],
        source_record_id=rid,
    )
    for p in range(8):
        for a, residue in enumerate(metadata["amino_acids"]):
            b.add(
                "profile_cells",
                profile_id=profile_id,
                position_index=p,
                residue_index=a,
                residue=residue,
                count=float(counts[p, a]),
                log_probability=float(logprob[p, a]),
            )
    return profile_id


def parse(root, *, data=None):
    """Return all original profile memberships; never sum overlapping datasets.

    :param root: Acquired window-neighbourhood series root.
    :param data: Optional set/list of original integer dataset IDs.
    :return: Common dataset containing profiles and their original cells.
    """
    logger.info("Mapping merops source records to the common contract")
    root = Path(root)
    raw = root / "raw/merops"
    index = json.loads((raw / "index.json").read_text())
    b = ContractBuilder(root.parent)
    selected = set(map(int, data)) if data is not None else None
    for key, entry in sorted(index["by_index"].items(), key=lambda item: int(item[0])):
        if selected is not None and int(key) not in selected:
            continue
        metadata = json.loads((raw / entry["folder"] / "codes.json").read_text())
        for enzyme in metadata["peptidases"]:
            add_profile(b, raw, index, entry, enzyme)
    return b.finish()
