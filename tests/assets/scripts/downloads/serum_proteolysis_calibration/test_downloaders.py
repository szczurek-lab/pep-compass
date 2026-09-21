"""Behavioural tests for serum proteolysis raw-data acquisition scripts."""

from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
from io import BytesIO
from pathlib import Path
from types import ModuleType
from urllib.parse import parse_qs, urlsplit

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = REPOSITORY_ROOT / "assets" / "scripts" / "downloads" / "serum_proteolysis_calibration"


def load_script(module_name: str) -> ModuleType:
    """Load one standalone downloader script as a test module.

    :param module_name: Downloader filename without the ``.py`` suffix.
    :return: Imported module.
    """

    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    specification = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / f"{module_name}.py")
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class MockResponse(BytesIO):
    """Provide the context-manager protocol expected from ``urlopen`` responses."""

    def __enter__(self) -> "MockResponse":
        """Return this in-memory response.

        :return: This response.
        """

        return self

    def __exit__(self, *arguments: object) -> None:
        """Close the in-memory response after use."""

        self.close()


def write_serum_manifest(path: Path, rows: list[list[str]]) -> None:
    """Write a minimal valid serum manifest for downloader tests.

    :param path: Manifest path.
    :param rows: Ordered row values.
    """

    header = [
        "merops_code",
        "merops_name",
        "merops_family",
        "n_merops_cleavages",
        "serum_category",
        "include_in_hydrolysis_model",
        "uniprot_accession",
        "gene_name",
        "ensembl_gene_id",
        "ec_number",
        "mapping_status",
    ]
    path.write_text(
        "\t".join(header) + "\n" + "".join("\t".join(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_pride_downloader_stores_manifest_and_selected_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A selected PRIDE file and its complete manifest are stored with provenance."""

    downloader = load_script("download_pride_project")

    def mock_urlopen(request: object, timeout: int) -> MockResponse:
        url = request.full_url
        if url == "https://example.test/pride/projects/PXD123/files/all":
            return MockResponse(
                json.dumps(
                    [
                        {
                            "fileName": "selected.tsv",
                            "publicFileLocations": [{"name": "HTTP Protocol", "value": "https://example.test/files/selected.tsv"}],
                        },
                        {
                            "fileName": "unselected.raw",
                            "publicFileLocations": [{"name": "FTP Protocol", "value": "ftp://example.test/unselected.raw"}],
                        },
                    ]
                ).encode("utf-8")
            )
        if url == "https://example.test/files/selected.tsv":
            return MockResponse(b"peptide\tturnover\nAA\t1.0\n")
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(downloader, "urlopen", mock_urlopen)
    exit_status = downloader.main(
        [
            "PXD123",
            "--base-url",
            "https://example.test/pride",
            "--output-dir",
            str(tmp_path),
            "--include",
            r"\.tsv$",
        ]
    )

    assert exit_status == 0
    assert (tmp_path / "PXD123" / "files" / "selected.tsv").read_text(encoding="utf-8").startswith("peptide")
    manifest = json.loads((tmp_path / "PXD123" / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 2
    download_manifest = json.loads((tmp_path / "PXD123" / "download_manifest.json").read_text(encoding="utf-8"))
    assert download_manifest["source_version_or_accession"] == "PXD123"
    assert download_manifest["files"][1]["status"] == "downloaded"
    assert len(download_manifest["files"][1]["sha256"]) == 64


def test_ad_hoc_sabio_downloader_is_disabled() -> None:
    """Unconstrained SABIO-RK name queries cannot bypass the serum manifest."""

    downloader = load_script("download_sabio_rk")

    assert downloader.main() == 2


def test_brenda_installer_preserves_archive_and_release_provenance(tmp_path: Path) -> None:
    """A manually downloaded BRENDA archive is retained without extraction."""

    installer = load_script("install_brenda_bulk_export")
    source_archive = tmp_path / "source.tar.gz"
    source_member = tmp_path / "source.txt"
    source_member.write_text("BRENDA source artifact\n", encoding="utf-8")
    with tarfile.open(source_archive, "w:gz") as archive:
        archive.add(source_member, arcname="source.txt")

    output_dir = tmp_path / "installed"
    exit_status = installer.main(
        [
            str(source_archive),
            "--format",
            "json",
            "--release",
            "test-2026.1",
            "--output-dir",
            str(output_dir),
        ]
    )

    installed_archive = output_dir / "brenda_test-2026.1_json.tar.gz"
    assert exit_status == 0
    assert installed_archive.read_bytes() == source_archive.read_bytes()
    provenance = json.loads((output_dir / "brenda_test-2026.1_json.tar.gz.provenance.json").read_text(encoding="utf-8"))
    assert provenance["release"] == "test-2026.1"
    assert provenance["format"] == "json"


def test_serum_panel_downloader_builds_ec_query_for_mapped_entries(tmp_path: Path) -> None:
    """The SABIO panel only plans mapped hydrolases and queries by EC number."""

    downloader = load_script("download_sabio_serum_panel")
    serum_manifest = tmp_path / "serum_proteases.tsv"
    serum_manifest.write_text(
        "\t".join(
            [
                "merops_code",
                "merops_name",
                "merops_family",
                "n_merops_cleavages",
                "serum_category",
                "include_in_hydrolysis_model",
                "uniprot_accession",
                "gene_name",
                "ensembl_gene_id",
                "ec_number",
                "mapping_status",
            ]
        )
        + "\n"
        + "S01.002\tbeta trypsin\tS01\t1\tcoagulation\ttrue\tP00760\tPRSS1\tENSG1\t3.4.21.4\tmapped_reviewed\n"
        + "C111.001\tfactor XIIIa\tC111\t1\tcoagulation\tfalse\tP00488\tF13A1\tENSG2\t2.3.2.13\tmapped_reviewed\n"
        + "M01.001\taminopeptidase N\tM01\t1\tplasma\ttrue\t\t\t\t\tunmapped\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "sabio"

    exit_status = downloader.main(
        [
            "--manifest",
            str(serum_manifest),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert exit_status == 0
    report = json.loads((output_dir / "serum_panel_download_manifest.json").read_text(encoding="utf-8"))
    assert [entry["merops_code"] for entry in report["queries"]] == ["S01.002"]
    assert report["queries"][0]["query"] == 'Organism:"Homo sapiens" AND ECNumber:3.4.21.4'


def test_manifest_builder_excludes_factor_xiiia(tmp_path: Path) -> None:
    """The generated manifest retains all MEROPS rows but excludes C111.001."""

    builder = load_script("build_serum_manifest")
    source = tmp_path / "codes.json"
    source.write_text(
        json.dumps(
            {
                "peptidases": [
                    {"row": 1, "code": "S01.002", "name": "beta trypsin", "n_cleavages": 4, "serum_category": "coagulation"},
                    {"row": 0, "code": "C111.001", "name": "factor XIIIa", "n_cleavages": 1, "serum_category": "coagulation"},
                ]
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "serum_proteases.tsv"

    assert builder.main(["--merops-codes", str(source), "--output", str(destination)]) == 0

    records = destination.read_text(encoding="utf-8").splitlines()
    assert records[1].split("\t")[0] == "C111.001"
    assert records[1].split("\t")[5] == "false"
    assert records[2].split("\t")[5] == "true"


def test_uniprot_candidate_extracts_ec_number_from_catalytic_activity() -> None:
    """EC annotations in UniProt catalytic-activity comments become manifest query keys."""

    mapper = load_script("map_uniprot_serum_proteases")
    candidate = mapper.candidate_from_uniprot(
        "M10.001",
        {
            "primaryAccession": "P03956",
            "entryType": "UniProtKB reviewed (Swiss-Prot)",
            "organism": {"taxonId": 9606},
            "comments": [{"commentType": "CATALYTIC ACTIVITY", "reaction": {"ecNumber": "3.4.24.7"}}],
            "uniProtKBCrossReferences": [{"database": "Ensembl", "id": "ENST00001"}],
        },
    )

    assert candidate is not None
    assert candidate["ec_number"] == "3.4.24.7"


def test_uniprot_merops_cross_reference_fallback_uses_codes_not_names() -> None:
    """Fallback candidates derive from explicit MEROPS cross-references only."""

    mapper = load_script("map_uniprot_serum_proteases")
    grouped: dict[str, list[dict[str, str]]] = {}
    mapper.add_merops_cross_reference_candidates(
        {
            "results": [
                {
                    "primaryAccession": "P03956",
                    "entryType": "UniProtKB reviewed (Swiss-Prot)",
                    "organism": {"taxonId": 9606},
                    "comments": [{"commentType": "CATALYTIC ACTIVITY", "reaction": {"ecNumber": "3.4.24.7"}}],
                    "uniProtKBCrossReferences": [{"database": "MEROPS", "id": "M10.001"}],
                }
            ]
        },
        {"M10.001"},
        grouped,
    )

    assert grouped["M10.001"][0]["uniprot_accession"] == "P03956"


def test_sabio_panel_fetches_ids_then_normalized_sbml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A mapped protease is acquired through entry IDs before SBML retrieval."""

    downloader = load_script("download_sabio_serum_panel")
    manifest = tmp_path / "serum_proteases.tsv"
    write_serum_manifest(
        manifest,
        [["S01.002", "beta trypsin", "S01", "1", "coagulation", "true", "P00760", "PRSS1", "ENSG1", "3.4.21.4", "mapped_reviewed"]],
    )
    requested_urls: list[str] = []

    def mock_get_bytes(url: str, timeout_seconds: int) -> bytes:
        requested_urls.append(url)
        if "entryIDs" in url:
            return b"12\n34\n"
        return b"<?xml version='1.0'?><sbml level='3' version='1'/>"

    monkeypatch.setattr(downloader, "get_bytes", mock_get_bytes)
    output_dir = tmp_path / "sabio"
    assert downloader.main(["--manifest", str(manifest), "--output-dir", str(output_dir)]) == 0

    assert len(requested_urls) == 2
    query = parse_qs(urlsplit(requested_urls[0]).query)
    assert query["q"] == ['Organism:"Homo sapiens" AND ECNumber:3.4.21.4']
    assert parse_qs(urlsplit(requested_urls[1]).query)["kinlawids"] == ["12,34"]
    assert (output_dir / "S01.002" / "kinetic_laws.sbml").read_text(encoding="utf-8").endswith("/>")


def test_sabio_panel_writes_empty_result_without_sbml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid no-match response does not create a fake SBML source file."""

    downloader = load_script("download_sabio_serum_panel")
    manifest = tmp_path / "serum_proteases.tsv"
    write_serum_manifest(
        manifest,
        [["S01.002", "beta trypsin", "S01", "1", "coagulation", "true", "P00760", "PRSS1", "ENSG1", "3.4.21.4", "mapped_reviewed"]],
    )
    monkeypatch.setattr(downloader, "get_bytes", lambda url, timeout_seconds: b"\n")
    output_dir = tmp_path / "sabio"

    assert downloader.main(["--manifest", str(manifest), "--output-dir", str(output_dir)]) == 0
    assert (output_dir / "S01.002" / "entry_ids.txt").is_file()
    assert not (output_dir / "S01.002" / "kinetic_laws.sbml").exists()


def test_sabio_panel_rejects_html_without_writing_an_entry_id_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An HTML service page cannot be retained as a SABIO entry-ID response."""

    downloader = load_script("download_sabio_serum_panel")
    manifest = tmp_path / "serum_proteases.tsv"
    write_serum_manifest(
        manifest,
        [["S01.002", "beta trypsin", "S01", "1", "coagulation", "true", "P00760", "PRSS1", "ENSG1", "3.4.21.4", "mapped_reviewed"]],
    )
    monkeypatch.setattr(downloader, "get_bytes", lambda url, timeout_seconds: b"<!DOCTYPE html><html></html>")
    output_dir = tmp_path / "sabio"

    assert downloader.main(["--manifest", str(manifest), "--output-dir", str(output_dir)]) == 1
    assert not (output_dir / "S01.002" / "entry_ids.txt").exists()


def test_pride_rejects_raw_ms_without_explicit_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The generic PRIDE command cannot silently retrieve vendor RAW files."""

    downloader = load_script("download_pride_project")

    def mock_urlopen(request: object, timeout: int) -> MockResponse:
        return MockResponse(
            json.dumps(
                [{"fileName": "instrument.raw", "publicFileLocations": [{"value": "https://example.test/instrument.raw"}]}]
            ).encode("utf-8")
        )

    monkeypatch.setattr(downloader, "urlopen", mock_urlopen)
    assert downloader.main(["PXD123", "--output-dir", str(tmp_path), "--include", r"\.raw$"]) == 2


def test_pmc_selector_downloads_only_the_exact_adamts13_spreadsheet() -> None:
    """The ADAMTS13 selector refuses article media and unrelated spreadsheets."""

    downloader = load_script("download_pmc_supplements")
    selected = downloader.select_objects(
        [
            "PMC4522773.1/supplement/pnas.1511328112.sd01.xlsx",
            "PMC4522773.1/supplement/other.xlsx",
            "PMC4522773.1/figure/image.png",
        ],
        downloader.DATASET_PROFILES["adamts13"]["patterns"],
    )

    assert selected == ["PMC4522773.1/supplement/pnas.1511328112.sd01.xlsx"]


def test_peplife2_refuses_a_not_found_medium_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PEPlife2 status=404 is a source failure, not an invitation to broaden the query."""

    downloader = load_script("download_peplife2")

    def mock_urlopen(request: object, timeout: int) -> MockResponse:
        return MockResponse(json.dumps({"status": 404, "count": 0, "data": []}).encode("utf-8"))

    monkeypatch.setattr(downloader, "urlopen", mock_urlopen)
    assert downloader.main(["human_serum", "--output-dir", str(tmp_path)]) == 1
