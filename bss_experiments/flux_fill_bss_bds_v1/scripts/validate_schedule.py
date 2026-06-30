#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common import METHOD_SPECS, read_csv, read_json
from schedule_utils import compare_uniform_and_bss, row_schedule_payload, validate_schedule_payload


def load_or_build(row: Dict[str, str], require_files: bool) -> Tuple[Dict[str, Any] | None, List[str]]:
    path = Path(row["schedule_json_path"])
    if path.exists():
        try:
            return read_json(path), []
        except Exception as exc:
            return None, [f"{row['run_id']}: failed to read schedule JSON {path}: {exc!r}"]
    if require_files:
        return None, [f"{row['run_id']}: missing schedule JSON {path}"]
    try:
        return row_schedule_payload(row), []
    except Exception as exc:
        return None, [f"{row['run_id']}: failed to build expected schedule: {exc!r}"]


def validate_manifest(path: Path, require_files: bool = False) -> Tuple[List[str], List[str]]:
    rows = read_csv(path)
    errors: List[str] = []
    warnings: List[str] = []
    payload_by_run: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        payload, row_errors = load_or_build(row, require_files=require_files)
        errors.extend(row_errors)
        if payload is None:
            continue
        payload_by_run[row["run_id"]] = payload
        errors.extend([f"{row['run_id']}: {msg}" for msg in validate_schedule_payload(payload)])
        expected = METHOD_SPECS[row["method"]]
        if int(row["actual_nfe"]) != int(expected["actual_nfe"]):
            errors.append(f"{row['run_id']}: manifest actual_nfe mismatch")
        if row["sampler_mode"] != expected["sampler_mode"]:
            errors.append(f"{row['run_id']}: manifest sampler_mode mismatch")

    rows_by_case: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_case[row["case_id"]].append(row)
    for case_id, case_rows in rows_by_case.items():
        prompts = {row["prompt_hash"] for row in case_rows}
        sources = {row["source_image_path"] for row in case_rows}
        masks = {row["mask_image_path"] for row in case_rows}
        seeds = {row["seed"] for row in case_rows}
        if len(prompts) != 1 or len(sources) != 1 or len(masks) != 1 or len(seeds) != 1:
            errors.append(f"{case_id}: prompt/source/mask/seed are not fixed across methods")

        by_method = {row["method"]: payload_by_run.get(row["run_id"]) for row in case_rows}
        for nfe in [10, 20, 30, 40]:
            uniform = by_method.get(f"uniform{nfe}")
            bss = by_method.get(f"bss{nfe}")
            if uniform is None or bss is None:
                continue
            if compare_uniform_and_bss(uniform, bss):
                errors.append(f"{case_id}: bss{nfe} schedule unexpectedly equals uniform{nfe}")
    if not rows:
        warnings.append(f"{path}: manifest has no rows")
    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate FLUX Fill manifest schedules.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--require_schedule_files", action="store_true")
    args = parser.parse_args()
    errors, warnings = validate_manifest(Path(args.manifest), require_files=args.require_schedule_files)
    for warning in warnings:
        print("warning:", warning)
    if errors:
        for error in errors:
            print("error:", error)
        raise SystemExit(f"schedule validation failed with {len(errors)} error(s)")
    print(f"schedule validation passed: {args.manifest}")


if __name__ == "__main__":
    main()
