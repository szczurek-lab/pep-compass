"""Preservation and current-API contracts for the tutorial migration."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = ROOT / "docs/tutorial"


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def source_spans(text: str) -> dict[int, tuple[int, int]]:
    """Locate cell source values without normalizing any other JSON bytes."""
    decoder = json.JSONDecoder()

    def whitespace(position: int) -> int:
        while position < len(text) and text[position].isspace():
            position += 1
        return position

    def object_fields(position: int) -> Iterator[tuple[str, object, int, int]]:
        position = whitespace(position)
        assert text[position] == "{"
        position = whitespace(position + 1)
        while text[position] != "}":
            key, position = decoder.raw_decode(text, position)
            position = whitespace(position)
            assert text[position] == ":"
            start = whitespace(position + 1)
            value, end = decoder.raw_decode(text, start)
            yield key, value, start, end
            position = whitespace(end)
            if text[position] == ",":
                position = whitespace(position + 1)

    for key, cells, start, end in object_fields(0):
        if key != "cells":
            continue
        position = whitespace(start + 1)
        spans = {}
        for index in range(len(cells)):
            _, cell_end = decoder.raw_decode(text, position)
            for field, value, source_start, source_end in object_fields(position):
                if field == "source":
                    spans[index] = (source_start, source_end)
            position = whitespace(cell_end)
            if text[position] == ",":
                position = whitespace(position + 1)
        return spans
    raise AssertionError("Notebook has no cells array.")


def masked_document_digest(text: str, editable: list[int]) -> str:
    spans = source_spans(text)
    for index in sorted(editable, key=lambda item: spans[item][0], reverse=True):
        start, end = spans[index]
        text = text[:start] + f"<EDITABLE_SOURCE_{index}>" + text[end:]
    return hashlib.sha256(text.encode()).hexdigest()


def test_methodology_preserves_every_byte_outside_allowlisted_api_sources() -> None:
    specification = json.loads((TUTORIAL / "tutorial_preservation.json").read_text())[
        "methodology"
    ]
    path = TUTORIAL / specification["destination_filename"]
    text = path.read_bytes().decode("utf-8")
    assert (
        masked_document_digest(text, specification["editable_code_cells"])
        == specification["raw_document_outside_editable_sources_sha256"]
    )
    notebook = json.loads(text)
    assert len(notebook["cells"]) == specification["cell_count"]
    assert digest(notebook["metadata"]) == specification["notebook_metadata_sha256"]
    for index, cell in enumerate(notebook["cells"]):
        assert (
            digest({key: value for key, value in cell.items() if key != "source"})
            == specification["cell_non_source_sha256"][str(index)]
        )
        if index not in specification["editable_code_cells"]:
            assert (
                digest(cell["source"])
                == specification["source_sha256_by_cell"][str(index)]
            )


def test_wrapper_preserves_all_non_todo_code_cells() -> None:
    specification = json.loads((TUTORIAL / "tutorial_preservation.json").read_text())
    wrapper = json.loads((TUTORIAL / "wrapper_tutorial.ipynb").read_text())
    assert len(wrapper["cells"]) == specification["wrapper"]["cell_count"]
    for index, expected in specification["wrapper"]["protected_code_cells"].items():
        assert digest(wrapper["cells"][int(index)]) == expected
    for filename, expected in specification["protected_backups"].items():
        assert (
            hashlib.sha256((TUTORIAL / filename).read_bytes()).hexdigest() == expected
        )


def test_methodology_edits_keep_plot_inputs_parameters_and_missing_algorithm_explicit() -> (
    None
):
    specification = json.loads((TUTORIAL / "tutorial_preservation.json").read_text())[
        "methodology"
    ]
    notebook = json.loads((TUTORIAL / "methodology_tutorial.ipynb").read_text())
    sources = {i: "".join(c["source"]) for i, c in enumerate(notebook["cells"])}
    original = specification["original_edited_code"]
    assert (
        sources[11].replace(
            "data/peptides_data/peptides_raw/peptides.csv", "data/peptides.csv"
        )
        == original["11"]
    )
    assert sources[25] == original["25"].replace(',\n    models_dir="models"', "")
    assert sources[63] == original["63"].replace(
        "MutationEnumerationInTangentSpace(", "ps.mutations.Mutang("
    )
    assert sources[64].endswith(
        original["64"][
            original["64"].index('print("Position-wise Mutation Candidates:') :
        ]
    )
    assert sources[42].endswith(original["42"])
    assert "raise NotImplementedError" in sources[42]
    assert '"oracle_calls": 100' in sources[90]
    assert "seed=42" in sources[90]
    assert "recorded_scores.cummin()" in sources[90]
    assert "ml-in-pl-tutorial-main" not in sources[3].split("%pip", 1)[1]


def test_all_tutorial_code_parses_and_imports_use_current_paths() -> None:
    transform = TransformerManager()
    obsolete = (
        "pep_compass.models",
        "pep_compass.local_enumeration",
        "pep_compass.optimization.black_box",
        "pep_compass.optimization.lebo",
        "pep_compass.geodesic_search",
    )
    for filename in (
        "wrapper_tutorial.ipynb",
        "reader_tutorial.ipynb",
        "orchestrator_tutorial.ipynb",
        "methodology_tutorial.ipynb",
    ):
        notebook = json.loads((TUTORIAL / filename).read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse(transform.transform_cell("".join(cell["source"])))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith(obsolete)
                elif isinstance(node, ast.Import):
                    assert all(
                        not name.name.startswith(obsolete) for name in node.names
                    )


def test_wrapper_markdown_explains_construction_validation_and_composition() -> None:
    notebook = json.loads((TUTORIAL / "wrapper_tutorial.ipynb").read_text())
    markdown = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    for required in [
        "### Validation",
        "NodeSpec",
        "lightweight",
        "leaf Step",
        "composite Step",
        "immutable",
        "feedback",
        "latent_origins",
        "reader_tutorial.ipynb",
        "orchestrator_tutorial.ipynb",
    ]:
        assert required in markdown
    assert "#TODO" not in markdown
