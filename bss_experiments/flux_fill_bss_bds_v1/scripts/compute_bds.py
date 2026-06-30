#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from common import RUN_ROOT, fmt_float, markdown_table, read_csv, safe_float, write_csv, write_json, write_markdown_table

BDS_FIELDS = [
    "split_name",
    "tset_name",
    "calibration_cases",
    "calibration_observations",
    "calibration_mean_mask_gain",
    "calibration_mask_lcb95",
    "calibration_mean_full_gain",
    "mean_preservation_delta",
    "preservation_guard_ok",
    "predicted_verdict",
    "holdout_cases",
    "holdout_observations",
    "holdout_mean_mask_gain",
    "holdout_win_rate",
    "notes",
]
CROSS_FIELDS = [
    "model_id",
    "setting",
    "protocol",
    "reference_method",
    "reference_nfe",
    "cases",
    "tested_nfe_points",
    "bds_low_mean_over_splits",
    "bds_low_lcb_min_over_splits",
    "bds_all_mean_over_splits",
    "bds_all_lcb_min_over_splits",
    "final_verdict",
    "notes",
]


def mean(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def win_rate(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return sum(1 for v in vals if v > 0) / len(vals)


def percentile(sorted_values: Sequence[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def bootstrap_lcb(values: Sequence[float], resamples: int, seed: int) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    if len(vals) == 1 or resamples <= 0:
        return mean(vals)
    rng = random.Random(seed)
    boot: List[float] = []
    n = len(vals)
    for _ in range(resamples):
        boot.append(sum(vals[rng.randrange(n)] for _i in range(n)) / n)
    boot.sort()
    return percentile(boot, 0.05)


def load_gain_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for row in read_csv(path):
        mask_gain = safe_float(row.get("mask_rgb_l1_closure_gain"))
        full_gain = safe_float(row.get("full_rgb_l1_closure_gain"))
        nfe = safe_float(row.get("actual_nfe"))
        if nfe is None:
            continue
        out = dict(row)
        out["actual_nfe"] = int(round(nfe))
        out["mask_rgb_l1_closure_gain"] = mask_gain
        out["full_rgb_l1_closure_gain"] = full_gain
        out["preservation_delta_bss_minus_uniform"] = safe_float(row.get("preservation_delta_bss_minus_uniform"))
        rows.append(out)
    return rows


def build_splits(cases: Sequence[str]) -> Dict[str, Dict[str, str]]:
    ordered = sorted(cases)
    half = max(1, len(ordered) // 2)
    first = set(ordered[:half])
    alt = {case for i, case in enumerate(ordered) if i % 2 == 0}
    shuffled = list(ordered)
    random.Random(0).shuffle(shuffled)
    rand = set(shuffled[:half])
    return {
        "first_half_split": {case: ("calibration" if case in first else "holdout") for case in ordered},
        "alternating_split": {case: ("calibration" if case in alt else "holdout") for case in ordered},
        "random_split_seed0": {case: ("calibration" if case in rand else "holdout") for case in ordered},
    }


def observations(rows: Sequence[Dict[str, Any]], split: Dict[str, str], split_value: str, tset: Sequence[int]) -> List[Dict[str, Any]]:
    wanted = set(int(x) for x in tset)
    return [row for row in rows if split.get(str(row["case_id"])) == split_value and int(row["actual_nfe"]) in wanted]


def verdict(mask_lcb: float | None, low_lcb: float | None, tset_name: str, guard_ok: bool, provisional: bool) -> str:
    if mask_lcb is None or low_lcb is None:
        base = "Need more data"
    elif tset_name == "all" and mask_lcb > 0 and guard_ok:
        base = "Green"
    elif low_lcb > 0 and guard_ok:
        base = "Low-only"
    elif low_lcb <= 0:
        base = "Reject"
    else:
        base = "Need more data"
    return f"Provisional {base}" if provisional and base not in {"Need more data"} else base


def compute_rows(gain_rows: Sequence[Dict[str, Any]], resamples: int, tolerance: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cases = sorted({str(row["case_id"]) for row in gain_rows})
    provisional = len(cases) < 8
    available_nfes = sorted({int(row["actual_nfe"]) for row in gain_rows})
    tsets = {"low": [10] if 10 in available_nfes else available_nfes[:1], "all": [x for x in [10, 20, 30, 40] if x in available_nfes]}
    splits = build_splits(cases)
    rows: List[Dict[str, Any]] = []
    lcb_by_tset: Dict[str, List[float]] = defaultdict(list)
    mean_by_tset: Dict[str, List[float]] = defaultdict(list)
    for split_name, mapping in splits.items():
        low_obs = observations(gain_rows, mapping, "calibration", tsets["low"])
        low_lcb = bootstrap_lcb([float(r["mask_rgb_l1_closure_gain"]) for r in low_obs if r["mask_rgb_l1_closure_gain"] is not None], resamples, 7)
        for tset_name, tset in tsets.items():
            cal = observations(gain_rows, mapping, "calibration", tset)
            hold = observations(gain_rows, mapping, "holdout", tset)
            mask_values = [safe_float(r.get("mask_rgb_l1_closure_gain")) for r in cal]
            full_values = [safe_float(r.get("full_rgb_l1_closure_gain")) for r in cal]
            preservation_values = [safe_float(r.get("preservation_delta_bss_minus_uniform")) for r in cal]
            mask_lcb = bootstrap_lcb([float(v) for v in mask_values if v is not None], resamples, len(split_name) + len(tset_name))
            mask_mean = mean(mask_values)
            guard_value = mean(preservation_values)
            guard_ok = guard_value is None or guard_value <= tolerance
            pred = verdict(mask_lcb, low_lcb, tset_name, guard_ok, provisional)
            if mask_lcb is not None:
                lcb_by_tset[tset_name].append(mask_lcb)
            if mask_mean is not None:
                mean_by_tset[tset_name].append(mask_mean)
            rows.append(
                {
                    "split_name": split_name,
                    "tset_name": tset_name,
                    "calibration_cases": sum(1 for value in mapping.values() if value == "calibration"),
                    "calibration_observations": len(cal),
                    "calibration_mean_mask_gain": fmt_float(mask_mean),
                    "calibration_mask_lcb95": fmt_float(mask_lcb),
                    "calibration_mean_full_gain": fmt_float(mean(full_values)),
                    "mean_preservation_delta": fmt_float(guard_value),
                    "preservation_guard_ok": guard_ok,
                    "predicted_verdict": pred,
                    "holdout_cases": sum(1 for value in mapping.values() if value == "holdout"),
                    "holdout_observations": len(hold),
                    "holdout_mean_mask_gain": fmt_float(mean([safe_float(r.get("mask_rgb_l1_closure_gain")) for r in hold])),
                    "holdout_win_rate": fmt_float(win_rate([safe_float(r.get("mask_rgb_l1_closure_gain")) for r in hold])),
                    "notes": f"Tset={{{','.join(str(x) for x in tset)}}}; {'provisional_4case' if provisional else '8case_or_more'}",
                    "_mask_lcb_num": mask_lcb,
                    "_mask_mean_num": mask_mean,
                }
            )
    low_min = min(lcb_by_tset["low"]) if lcb_by_tset["low"] else None
    all_min = min(lcb_by_tset["all"]) if lcb_by_tset["all"] else None
    if all_min is not None and all_min > 0:
        final = "Green"
    elif low_min is not None and low_min > 0:
        final = "Low-only"
    elif low_min is not None and low_min <= 0:
        final = "Reject"
    else:
        final = "Need more data"
    if provisional and final not in {"Need more data"}:
        final = f"Provisional {final}"
    cross = {
        "model_id": "FLUX.1 Fill-dev",
        "setting": "Image Fill",
        "protocol": "fixed_fill_suite_diffusers",
        "reference_method": "reference_uniform50",
        "reference_nfe": 50,
        "cases": len(cases),
        "tested_nfe_points": ",".join(str(x) for x in available_nfes),
        "bds_low_mean_over_splits": fmt_float(mean(mean_by_tset["low"])),
        "bds_low_lcb_min_over_splits": fmt_float(low_min),
        "bds_all_mean_over_splits": fmt_float(mean(mean_by_tset["all"])),
        "bds_all_lcb_min_over_splits": fmt_float(all_min),
        "final_verdict": final,
        "notes": "Primary metric is mask RGB-L1 closure gain; 4-case results are provisional.",
    }
    return [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows], cross


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute provisional FLUX Fill BDS tables.")
    parser.add_argument("--run_root", default=str(RUN_ROOT))
    parser.add_argument("--gain_csv", default="")
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    parser.add_argument("--preservation_tolerance", type=float, default=0.005)
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()
    run_root = Path(args.run_root)
    gain_csv = Path(args.gain_csv) if args.gain_csv else run_root / "metrics" / "same_compute_gain_long.csv"
    gain_rows = load_gain_rows(gain_csv)
    if not gain_rows:
        reason = f"No same-compute gain rows found at {gain_csv}."
        write_csv(run_root / "tables" / "tableA_flux_fill_bds_by_split.csv", [], BDS_FIELDS)
        write_markdown_table(run_root / "tables" / "tableA_flux_fill_bds_by_split.md", [], BDS_FIELDS)
        cross = {
            "model_id": "FLUX.1 Fill-dev",
            "setting": "Image Fill",
            "protocol": "fixed_fill_suite_diffusers",
            "reference_method": "reference_uniform50",
            "reference_nfe": 50,
            "cases": 0,
            "tested_nfe_points": "",
            "bds_low_mean_over_splits": "N/A",
            "bds_low_lcb_min_over_splits": "N/A",
            "bds_all_mean_over_splits": "N/A",
            "bds_all_lcb_min_over_splits": "N/A",
            "final_verdict": "Need more data",
            "notes": reason,
        }
        write_csv(run_root / "tables" / "cross_model_bds_row.csv", [cross], CROSS_FIELDS)
        write_markdown_table(run_root / "tables" / "cross_model_bds_row.md", [cross], CROSS_FIELDS)
        (run_root / "reports" / "03_bds_report.md").write_text(f"# BDS Report\n\n{reason}\n", encoding="utf-8")
        if not args.allow_missing:
            raise SystemExit(reason)
        print(reason)
        return
    rows, cross = compute_rows(gain_rows, args.bootstrap_resamples, args.preservation_tolerance)
    write_csv(run_root / "tables" / "tableA_flux_fill_bds_by_split.csv", rows, BDS_FIELDS)
    write_markdown_table(run_root / "tables" / "tableA_flux_fill_bds_by_split.md", rows, BDS_FIELDS)
    write_csv(run_root / "tables" / "cross_model_bds_row.csv", [cross], CROSS_FIELDS)
    write_markdown_table(run_root / "tables" / "cross_model_bds_row.md", [cross], CROSS_FIELDS)
    write_json(run_root / "metrics" / "bds_summary.json", {"cross_model_bds_row": cross})
    report = [
        "# BDS Report",
        "",
        "Primary score: mask RGB-L1 closure gain. Secondary full-image closure is reported in the split table.",
        "",
        "## Split Results",
        "",
        markdown_table(rows, BDS_FIELDS),
        "",
        "## Cross-Model Row",
        "",
        markdown_table([cross], CROSS_FIELDS),
    ]
    (run_root / "reports" / "03_bds_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"BDS table: {run_root / 'tables/tableA_flux_fill_bds_by_split.csv'}")
    print(f"final verdict: {cross['final_verdict']}")


if __name__ == "__main__":
    main()
