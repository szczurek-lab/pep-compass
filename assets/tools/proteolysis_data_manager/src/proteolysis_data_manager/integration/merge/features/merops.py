"""Source-defined MEROPS probability features on explicitly mapped cut contexts."""

import json
import math
from pathlib import Path

from proteolysis_data_manager.integration.merge.contract.builder import ContractBuilder
from proteolysis_data_manager.integration.merge.contract.schema import (
    QUANTITIES,
    stable_id,
)
from proteolysis_data_manager.integration.merge.parsers.window_neighbourhood.merops import (
    add_profile,
)
from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def add_merops_scores(dataset, raw_root, *, dataset_index=34):
    """Add named probability scores without background correction or residue guessing.

    :param dataset: Common dataset containing context and enzyme mappings.
    :param raw_root: Original acquired ``raw/merops`` directory.
    :param int dataset_index: Explicit source population to score against.
    :return: New validated dataset with profile inputs and probability features.

    Probability sum is the requested primary score. Missing positions contribute
    no terms and their number is retained. A recognized nonstandard residue not
    represented by the source alphabet makes the primary score unresolved;
    it is never silently replaced or omitted. No EC-to-MEROPS mapping is inferred.
    """
    logger.info(
        "Scoring %s contexts against MEROPS population %s",
        len(dataset.table("contexts")),
        dataset_index,
    )
    raw_root = Path(raw_root)
    index = json.loads((raw_root / "index.json").read_text())
    entry = index["by_index"][str(dataset_index)]
    metadata = json.loads((raw_root / entry["folder"] / "codes.json").read_text())
    enzymes = {row["code"]: row for row in metadata["peptidases"]}
    # Extend the existing contract without changing any source observations.
    b = ContractBuilder(raw_root.parents[2])
    for name, frame in dataset.tables.items():
        for record in frame.to_dict("records"):
            b.add(name, **record)
    enzyme_codes = dataset.table("enzymes").set_index("enzyme_id").merops_code.to_dict()
    participant_codes = {
        r.participant_id: enzyme_codes.get(r.enzyme_id)
        for r in dataset.table("participants").itertuples()
    }
    profiles = {}
    for context in dataset.table("contexts").itertuples():
        code = participant_codes.get(context.enzyme_participant_id)
        profile_id = None
        value_sum = value_log = value_product = None
        residues = json.loads(context.residues_json)
        observed = [(i, r) for i, r in enumerate(residues) if r is not None]
        n_observed = len(observed)
        n_scored = 0
        if code not in enzymes:
            status = "no_explicit_profile_mapping"
        else:
            if code not in profiles:
                pid = add_profile(b, raw_root, index, entry, enzymes[code])
                cells = [
                    r
                    for r in b.rows["profile_cells"].values()
                    if r["profile_id"] == pid
                ]
                profiles[code] = (
                    pid,
                    {
                        (r["position_index"], r["residue"]): r["log_probability"]
                        for r in cells
                    },
                )
            profile_id, lookup = profiles[code]
            terms = [lookup[(i, r)] for i, r in observed if (i, r) in lookup]
            n_scored = len(terms)
            if n_scored != n_observed:
                status = "unsupported_residue"
            elif not terms:
                status = "no_observed_positions"
            else:
                value_sum = math.fsum(math.exp(value) for value in terms)
                value_log = math.fsum(terms)
                value_product = math.exp(value_log)
                status = "complete_window" if n_scored == 8 else "partial_window"
        for quantity, value in [
            ("merops_probability_sum", value_sum),
            ("merops_log_probability_sum", value_log),
            ("merops_probability_product", value_product),
        ]:
            b.add(
                "context_features",
                feature_id=stable_id(
                    "feature", context.context_id, profile_id, quantity, dataset_index
                ),
                context_id=context.context_id,
                profile_id=profile_id,
                quantity=quantity,
                value=value,
                unit=QUANTITIES[quantity][0],
                n_observed_positions=n_observed,
                n_scored_positions=n_scored,
                status=status,
                method=f"original_merops_dataset_{dataset_index}_available_positions_v1",
                source_record_id=context.source_record_id,
            )
    result = b.finish()
    result.metadata.update(dataset.metadata)
    return result
