#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from common import (
    LOW_BASELINE_METHOD,
    METHOD_SPECS,
    MODEL_ID,
    REFERENCE_METHOD,
    REFERENCE_NFE,
    SETTING,
    TASK,
    MODALITY,
    PROTOCOL,
    REPO_ROOT,
    git_commit,
    git_dirty_status,
    sha256_file,
    sha256_text,
    utc_timestamp,
    write_json,
)


def make_default_sigmas(num_steps: int) -> List[float]:
    """Small deterministic fallback schedule for dry-run validation.

    Real inference should prefer the scheduler coordinates produced by
    diffusers. This fallback exists so manifest and BSS invariants can be
    validated before FLUX weights are available.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    return [float(num_steps - i) / float(num_steps) for i in range(num_steps)]


def normalize_split_intervals(split_pairs: str | Sequence[int] | None, base_steps: int) -> List[int]:
    if base_steps <= 0:
        raise ValueError("base_steps must be positive")
    if split_pairs is None or split_pairs == "":
        return []
    if isinstance(split_pairs, str):
        raw = [item.strip() for item in split_pairs.split(",") if item.strip()]
        values = [int(item) for item in raw]
    else:
        values = [int(item) for item in split_pairs]
    out: List[int] = []
    for value in values:
        idx = base_steps - 1 if value == -1 else value
        if idx < 0 or idx >= base_steps:
            raise ValueError(f"split interval {value} resolves to {idx}, outside [0, {base_steps - 1}]")
        if idx not in out:
            out.append(idx)
    return out


def build_boundary_split_coords(
    base_coords: Sequence[float],
    split_pairs: str | Sequence[int] | None = "0,-1",
    terminal_coord: float = 0.0,
) -> Tuple[List[float], List[Dict[str, Any]]]:
    base = [float(x) for x in base_coords]
    if not base:
        raise ValueError("base_coords must not be empty")
    intervals = normalize_split_intervals(split_pairs, len(base))
    interval_set = set(intervals)
    final: List[float] = []
    inserted: List[Dict[str, Any]] = []
    for idx, left in enumerate(base):
        final.append(float(left))
        if idx not in interval_set:
            continue
        right = float(base[idx + 1]) if idx + 1 < len(base) else float(terminal_coord)
        midpoint = (float(left) + right) * 0.5
        final.append(midpoint)
        inserted.append({"interval": idx, "left": float(left), "right": right, "midpoint": midpoint})
    return final, inserted


def schedule_for_method(method: str, base_coords: Sequence[float] | None = None, terminal_coord: float = 0.0) -> Dict[str, Any]:
    if method not in METHOD_SPECS:
        raise ValueError(f"unknown method: {method}")
    spec = METHOD_SPECS[method]
    base_steps = int(spec["base_sample_steps"])
    coords = list(base_coords) if base_coords is not None else make_default_sigmas(base_steps)
    if len(coords) != base_steps:
        raise ValueError(f"{method} expected {base_steps} base coords, got {len(coords)}")
    if spec["sampler_mode"] == "bss":
        final_coords, inserted = build_boundary_split_coords(coords, spec["split_pairs"], terminal_coord=terminal_coord)
    else:
        final_coords, inserted = [float(x) for x in coords], []
    actual_nfe = int(spec["actual_nfe"])
    if len(final_coords) != actual_nfe:
        raise ValueError(f"{method} expected {actual_nfe} final coords, got {len(final_coords)}")
    return {
        "method": method,
        "sampler_mode": spec["sampler_mode"],
        "method_family": spec["method_family"],
        "actual_nfe": actual_nfe,
        "base_sample_steps": base_steps,
        "num_inference_steps": int(spec["num_inference_steps"]),
        "split_pairs": spec["split_pairs"],
        "coordinate_type": "sigmas",
        "base_coords": [float(x) for x in coords],
        "final_coords": [float(x) for x in final_coords],
        "inserted_midpoints": inserted,
        "terminal_coord": float(terminal_coord),
    }


def row_schedule_payload(row: Dict[str, Any], base_coords: Sequence[float] | None = None, terminal_coord: float = 0.0) -> Dict[str, Any]:
    method = str(row["method"])
    schedule = schedule_for_method(method, base_coords=base_coords, terminal_coord=terminal_coord)
    source_path = Path(str(row.get("source_image_path", "")))
    mask_path = Path(str(row.get("mask_image_path", "")))
    schedule.update(
        {
            "model_id": MODEL_ID,
            "setting": SETTING,
            "task": TASK,
            "modality": MODALITY,
            "protocol": PROTOCOL,
            "reference_method": REFERENCE_METHOD,
            "reference_nfe": REFERENCE_NFE,
            "low_baseline_method": LOW_BASELINE_METHOD,
            "case_id": row.get("case_id", ""),
            "seed": int(row.get("seed", 0)),
            "prompt_hash": row.get("prompt_hash") or sha256_text(str(row.get("prompt", ""))),
            "source_image_hash": sha256_file(source_path) if source_path.exists() else "",
            "mask_hash": sha256_file(mask_path) if mask_path.exists() else "",
            "height": int(row.get("height", 0) or 0),
            "width": int(row.get("width", 0) or 0),
            "guidance_scale": float(row.get("guidance_scale", 0) or 0),
            "max_sequence_length": int(row.get("max_sequence_length", 0) or 0),
            "output_path": row.get("output_path", ""),
            "git_commit": row.get("git_commit") or git_commit(REPO_ROOT),
            "dirty_status": row.get("dirty_status") or git_dirty_status(REPO_ROOT),
            "timestamp": utc_timestamp(),
        }
    )
    return schedule


def validate_schedule_payload(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    method = str(payload.get("method", ""))
    if method not in METHOD_SPECS:
        return [f"unknown method: {method}"]
    spec = METHOD_SPECS[method]
    actual = int(spec["actual_nfe"])
    base_steps = int(spec["base_sample_steps"])
    final_coords = payload.get("final_coords") or []
    base_coords = payload.get("base_coords") or []
    if len(final_coords) != actual:
        errors.append(f"{method}: expected {actual} final coords, got {len(final_coords)}")
    if len(base_coords) != base_steps:
        errors.append(f"{method}: expected {base_steps} base coords, got {len(base_coords)}")
    if str(spec["sampler_mode"]) == "bss":
        inserted = payload.get("inserted_midpoints") or []
        if len(inserted) != 2:
            errors.append(f"{method}: expected 2 inserted midpoint records, got {len(inserted)}")
        intervals = [item.get("interval") for item in inserted if isinstance(item, dict)]
        if intervals != [0, base_steps - 1]:
            errors.append(f"{method}: expected split intervals [0, {base_steps - 1}], got {intervals}")
    else:
        if payload.get("inserted_midpoints"):
            errors.append(f"{method}: uniform/reference schedule should not have inserted midpoints")
    return errors


def compare_uniform_and_bss(uniform_payload: Dict[str, Any], bss_payload: Dict[str, Any]) -> bool:
    return list(uniform_payload.get("final_coords") or []) == list(bss_payload.get("final_coords") or [])


def write_schedule_for_row(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = row_schedule_payload(row)
    path = Path(str(row["schedule_json_path"]))
    write_json(path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Write or print a fallback schedule JSON for a method.")
    parser.add_argument("--method", required=True, choices=sorted(METHOD_SPECS))
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    payload = schedule_for_method(args.method)
    if args.output:
        write_json(Path(args.output), payload)
    else:
        import json

        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
