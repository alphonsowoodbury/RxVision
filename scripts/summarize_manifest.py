#!/usr/bin/env python3
"""Produce aggregate evidence from a pipe-delimited image manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import TextIO


def summarize_manifest(stream: TextIO) -> dict[str, object]:
    """Count rows and field cardinality without retaining source values."""

    row_count = 0
    malformed_rows = 0
    ndcs: set[str] = set()
    image_types: Counter[str] = Counter()

    for line in stream:
        stripped = line.rstrip("\n")
        if not stripped:
            continue
        fields = stripped.split("|")
        if len(fields) != 5:
            malformed_rows += 1
            continue
        ndc, _part, _path, image_type, _drug_name = fields
        row_count += 1
        ndcs.add(ndc)
        image_types[image_type] += 1

    return {
        "row_count": row_count,
        "malformed_row_count": malformed_rows,
        "unique_ndc_field_values": len(ndcs),
        "unique_image_type_labels": len(image_types),
        "image_type_row_counts": dict(sorted(image_types.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8", errors="replace") as stream:
        summary = summarize_manifest(stream)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
