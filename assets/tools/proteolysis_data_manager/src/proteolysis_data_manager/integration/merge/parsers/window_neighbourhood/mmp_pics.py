"""Parse source-reported PICS occurrence rows into the common positional contract."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from proteolysis_data_manager.analysis.window_neighbourhood.pics import MMP_TO_MEROPS
from proteolysis_data_manager.integration.merge.contract.builder import ContractBuilder
from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def parse(root, *, data=None):
    """Keep the original occurrence representation and its source cell identities.

    :param root: Acquired window-neighbourhood series root.
    :param data: Optional original records with file/sheet/row/raw_window fields.
    :return: Common dataset with reported occurrence counts and masked windows.

    ``PICS Output - unique`` remains an alternate analysis of the same deposit,
    not additional independent evidence. Its workbook bytes remain available.
    """
    logger.info("Mapping mmp_pics source records to the common contract")
    root = Path(root)
    records = []
    if data is None:
        for filename, suffix, library in [
            ("mmc5.xlsx", "_T", "trypsin"),
            ("mmc6.xlsx", "_G", "GluC"),
        ]:
            with pd.ExcelFile(root / "raw/mmp_pics" / filename) as workbook:
                for sheet in workbook.sheet_names:
                    if not sheet.endswith(suffix):
                        continue
                    frame = pd.read_excel(workbook, sheet_name=sheet, header=None)
                    locations = np.argwhere(frame.eq("PICS Output").to_numpy())
                    candidates = []
                    for header, column in locations:
                        values = frame.iloc[header + 1 :, column].dropna()
                        if (
                            values.astype(str)
                            .str.fullmatch("[ACDEFGHIKLMNPQRSTVWY-]{12}")
                            .any()
                        ):
                            candidates.append((column, values))
                    if len(candidates) != 1:
                        raise ValueError("Ambiguous PICS output column")
                    column, values = candidates[0]
                    counters = [
                        frame.iat[int(row + 1), int(col)]
                        for row, col in locations
                        if isinstance(
                            frame.iat[int(row + 1), int(col)], (int, float, np.number)
                        )
                    ]
                    if len(counters) != 1 or counters[0] != len(values):
                        raise ValueError("PICS source window counter mismatch")
                    for row, window in values.items():
                        if not isinstance(window, str) or not re.fullmatch(
                            "[ACDEFGHIKLMNPQRSTVWY-]{12}", window
                        ):
                            raise ValueError("Unsupported PICS window")
                        records.append(
                            {
                                "file": filename,
                                "sheet": sheet,
                                "excel_row": row + 1,
                                "excel_column": int(column + 1),
                                "enzyme": sheet[: -len(suffix)],
                                "library": library,
                                "raw_window": window,
                            }
                        )
    else:
        records = data.to_dict("records") if hasattr(data, "to_dict") else list(data)
        records = [
            r
            for r in records
            if r.get("representation", "PICS Output") == "PICS Output"
        ]
    b = ContractBuilder(root.parent)
    for r in records:
        enzyme = r["enzyme"]
        library = r["library"]
        file = r["file"]
        rid = b.source(
            "mmp_pics",
            library,
            f"{r['sheet']}:{r['excel_row']}",
            root / "raw/mmp_pics" / file,
            f"{r['sheet']}/row/{r['excel_row']}/column/{r.get('excel_column', 'PICS Output')}",
            raw=r,
            publications=["DOI:10.1016/j.dib.2016.02.036"],
        )
        assay = b.assay(
            rid,
            ("PICS", enzyme, library),
            conditions={"method": "PICS", "mode": "in vitro"},
            basis="source_enzyme_library",
        )
        eid = b.enzyme(
            rid,
            ("PICS", enzyme),
            name=enzyme,
            code=MMP_TO_MEROPS[enzyme],
            organism="Homo sapiens",
        )
        ep = b.participant(assay, eid, "enzyme", enzyme=True)
        residues = [None if token == "-" else token for token in r["raw_window"][2:10]]
        context = b.context(
            rid,
            assay,
            None,
            residues,
            enzyme=ep,
            evidence="source_PICS_reconstructed_window",
        )
        mid = b.observation(
            rid,
            assay,
            None,
            "cleavage_occurrence",
            1.0,
            measurement_scope="site_specific",
            raw_value="one reported PICS output row",
            derivation="source_occurrence",
        )
        b.link(mid, context, "reported_occurrence")
    return b.finish()
