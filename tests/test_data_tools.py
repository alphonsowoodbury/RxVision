from __future__ import annotations

import io

from scripts.prepare_dataset import make_folder_safe, normalize_drug_name, parse_args
from scripts.summarize_manifest import summarize_manifest


def test_manifest_summary_counts_without_retaining_records():
    stream = io.StringIO(
        "0001|1|a.jpg|REFERENCE|DRUG A\n"
        "0001|1|b.jpg|TEST|DRUG A\n"
        "0002|1|c.jpg|TEST|DRUG B\n"
        "malformed\n"
    )

    summary = summarize_manifest(stream)

    assert summary["row_count"] == 3
    assert summary["malformed_row_count"] == 1
    assert summary["unique_ndc_field_values"] == 2
    assert summary["image_type_row_counts"] == {"REFERENCE": 1, "TEST": 2}


def test_drug_name_normalization_is_stable():
    assert normalize_drug_name("  Example Drug Tablets, USP ") == "EXAMPLE DRUG"
    assert make_folder_safe("EXAMPLE DRUG 10 MG") == "example_drug_10_mg"


def test_reference_table_is_required():
    try:
        parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError("missing reference table was accepted")


def test_parse_args_accepts_explicit_reference_table():
    args = parse_args(["--reference-table", "manifest.txt", "--list-drugs"])
    assert args.reference_table == "manifest.txt"
    assert args.list_drugs is True
