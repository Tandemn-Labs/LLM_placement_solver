#!/usr/bin/env python3
"""
Collect all solution_summary.csv files under a root directory into one CSV.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Dict


def find_solution_summaries(root: Path) -> List[Path]:
    return sorted(root.rglob("solution_summary.csv"))


def merge_fieldnames(fieldnames_order: List[str], new_fields: Iterable[str]) -> List[str]:
    for field in new_fields:
        if field not in fieldnames_order:
            fieldnames_order.append(field)
    return fieldnames_order


def collect_rows(files: Iterable[Path], root: Path) -> (List[str], List[Dict[str, str]]):
    fieldnames_order: List[str] = ["source_path"]
    rows: List[Dict[str, str]] = []

    for path in files:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            fieldnames_order = merge_fieldnames(fieldnames_order, reader.fieldnames)
            for row in reader:
                row = dict(row)
                row["source_path"] = str(path.relative_to(root))
                rows.append(row)

    return fieldnames_order, rows


def write_output(output_path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect solution_summary.csv files into one CSV.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/projects/LLM_placement_solver/config"),
        help="Root directory to search (default: /mnt/projects/LLM_placement_solver/config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/projects/LLM_placement_solver/solution_summaries_merged.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_path = args.output.resolve()

    files = find_solution_summaries(root)
    if not files:
        raise SystemExit(f"No solution_summary.csv files found under {root}")

    fieldnames, rows = collect_rows(files, root)
    write_output(output_path, fieldnames, rows)
    print(f"Wrote {len(rows)} rows from {len(files)} files to {output_path}")


if __name__ == "__main__":
    main()




