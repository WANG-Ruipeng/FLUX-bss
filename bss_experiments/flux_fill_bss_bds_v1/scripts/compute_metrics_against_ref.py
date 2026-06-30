#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from common import (
    LOW_BASELINE_METHOD,
    REFERENCE_METHOD,
    REFERENCE_NFE,
    RUN_ROOT,
    fmt_float,
    read_csv,
    runtime_sec_from_row,
    safe_float,
    write_csv,
    write_json,
)


def load_image(path: Path):
    try:
        from PIL import Image
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Pillow and numpy are required for metrics.") from exc
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype("float32") / 255.0


def load_mask(path: Path, shape: Tuple[int, int]):
    from PIL import Image
    import numpy as np

    mask = Image.open(path).convert("L").resize((shape[1], shape[0]))
    arr = np.asarray(mask).astype("float32") / 255.0
    return arr > 0.5


def align(a, b):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    c = min(a.shape[2], b.shape[2])
    return a[:h, :w, :c], b[:h, :w, :c]


def mean_l1(a, b, mask=None) -> float:
    import numpy as np

    a, b = align(a, b)
    diff = np.abs(a - b)
    if mask is not None:
        m = mask[: diff.shape[0], : diff.shape[1]]
        if not bool(m.any()):
            return float("nan")
        diff = diff[m]
    return float(np.mean(diff))


def mean_l2(a, b, mask=None) -> float:
    import numpy as np

    a, b = align(a, b)
    diff = (a - b) ** 2
    if mask is not None:
        m = mask[: diff.shape[0], : diff.shape[1]]
        if not bool(m.any()):
            return float("nan")
        diff = diff[m]
    return float(np.mean(diff))


def psnr(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return float(-10.0 * math.log10(mse))


def closure(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline == 0 or not math.isfinite(float(value)) or not math.isfinite(float(baseline)):
        return None
    return 1.0 - float(value) / float(baseline)


def method_nfe(method: str) -> int | None:
    if method == REFERENCE_METHOD:
        return REFERENCE_NFE
    for prefix in ["uniform", "bss"]:
        if method.startswith(prefix):
            return int(method[len(prefix) :])
    return None


def write_empty(run_root: Path, reason: str) -> None:
    write_csv(run_root / "metrics" / "master_long_metrics.csv", [], [])
    write_csv(run_root / "metrics" / "per_case_metrics.csv", [], [])
    write_csv(run_root / "metrics" / "same_compute_gain_long.csv", [], [])
    write_json(run_root / "metrics" / "metrics_status.json", {"status": "missing_outputs", "reason": reason})
    (run_root / "reports" / "metrics_status.md").write_text(f"# Metrics Status\n\n{reason}\n", encoding="utf-8")


def build_metric_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    by_case: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_case[row["case_id"]][row["method"]] = row

    out: List[Dict[str, Any]] = []
    for case_id, methods in sorted(by_case.items()):
        ref_row = methods.get(REFERENCE_METHOD)
        low_row = methods.get(LOW_BASELINE_METHOD)
        if not ref_row or not low_row:
            continue
        ref_path = Path(ref_row["output_path"])
        low_path = Path(low_row["output_path"])
        if not ref_path.exists() or not low_path.exists():
            continue
        ref = load_image(ref_path)
        low = load_image(low_path)
        source = load_image(Path(ref_row["source_image_path"]))
        mask = load_mask(Path(ref_row["mask_image_path"]), ref.shape[:2])
        unmask = ~mask
        low_full_l1 = mean_l1(low, ref)
        low_mask_l1 = mean_l1(low, ref, mask=mask)
        uniform_by_nfe: Dict[int, Dict[str, Any]] = {}

        for method, row in sorted(methods.items(), key=lambda item: (method_nfe(item[0]) or 999, item[0])):
            out_path = Path(row["output_path"])
            if not out_path.exists():
                continue
            cur = load_image(out_path)
            full_l1 = mean_l1(cur, ref)
            full_l2 = mean_l2(cur, ref)
            mask_l1 = mean_l1(cur, ref, mask=mask)
            unmasked_l1 = mean_l1(cur, source, mask=unmask)
            item = {
                "case_id": case_id,
                "category": row.get("category", ""),
                "method": method,
                "method_family": row.get("method_family", ""),
                "sampler_mode": row.get("sampler_mode", ""),
                "actual_nfe": int(row["actual_nfe"]),
                "compute_fraction": float(row["actual_nfe"]) / float(REFERENCE_NFE),
                "rgb_l1_to_ref": full_l1,
                "rgb_l2_to_ref": full_l2,
                "psnr_to_ref": psnr(full_l2),
                "rgb_l1_closure": closure(full_l1, low_full_l1),
                "mask_rgb_l1_to_ref": mask_l1,
                "mask_rgb_l1_closure": closure(mask_l1, low_mask_l1),
                "unmasked_rgb_l1_to_source": unmasked_l1,
                "unmasked_preservation_delta_vs_uniform": None,
                "runtime_sec": runtime_sec_from_row(row),
                "source_image_path": row["source_image_path"],
                "mask_image_path": row["mask_image_path"],
                "output_path": row["output_path"],
                "schedule_json_path": row["schedule_json_path"],
            }
            if method.startswith("uniform"):
                uniform_by_nfe[int(row["actual_nfe"])] = item
            out.append(item)

        for item in out:
            if item["case_id"] != case_id:
                continue
            nfe = int(item["actual_nfe"])
            uniform = uniform_by_nfe.get(nfe)
            if uniform is not None and safe_float(item["unmasked_rgb_l1_to_source"]) is not None:
                item["unmasked_preservation_delta_vs_uniform"] = float(item["unmasked_rgb_l1_to_source"]) - float(
                    uniform["unmasked_rgb_l1_to_source"]
                )
    return out


def build_gain_rows(metric_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_case: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in metric_rows:
        by_case[str(row["case_id"])][str(row["method"])] = row
    gains: List[Dict[str, Any]] = []
    for case_id, methods in sorted(by_case.items()):
        for nfe in [10, 20, 30, 40]:
            uniform = methods.get(f"uniform{nfe}")
            bss = methods.get(f"bss{nfe}")
            if not uniform or not bss:
                continue
            mask_gain = None
            full_gain = None
            if safe_float(uniform.get("mask_rgb_l1_closure")) is not None and safe_float(bss.get("mask_rgb_l1_closure")) is not None:
                mask_gain = float(bss["mask_rgb_l1_closure"]) - float(uniform["mask_rgb_l1_closure"])
            if safe_float(uniform.get("rgb_l1_closure")) is not None and safe_float(bss.get("rgb_l1_closure")) is not None:
                full_gain = float(bss["rgb_l1_closure"]) - float(uniform["rgb_l1_closure"])
            gains.append(
                {
                    "case_id": case_id,
                    "category": uniform.get("category", ""),
                    "actual_nfe": nfe,
                    "compute_fraction": float(nfe) / float(REFERENCE_NFE),
                    "uniform_method": f"uniform{nfe}",
                    "bss_method": f"bss{nfe}",
                    "uniform_mask_rgb_l1_closure": uniform.get("mask_rgb_l1_closure"),
                    "bss_mask_rgb_l1_closure": bss.get("mask_rgb_l1_closure"),
                    "mask_rgb_l1_closure_gain": mask_gain,
                    "uniform_full_rgb_l1_closure": uniform.get("rgb_l1_closure"),
                    "bss_full_rgb_l1_closure": bss.get("rgb_l1_closure"),
                    "full_rgb_l1_closure_gain": full_gain,
                    "preservation_delta_bss_minus_uniform": bss.get("unmasked_preservation_delta_vs_uniform"),
                    "bss_win_mask": mask_gain is not None and mask_gain > 0,
                    "bss_win_full": full_gain is not None and full_gain > 0,
                    "reference_method": REFERENCE_METHOD,
                    "reference_nfe": REFERENCE_NFE,
                }
            )
    return gains


def summarize(rows: Sequence[Dict[str, Any]], gains: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "metric_rows": len(rows),
        "same_compute_gain_rows": len(gains),
        "cases": len({row["case_id"] for row in rows}),
        "nfe_points": sorted({int(row["actual_nfe"]) for row in gains}),
        "mean_mask_gain": mean([safe_float(row.get("mask_rgb_l1_closure_gain")) for row in gains]),
        "mean_full_gain": mean([safe_float(row.get("full_rgb_l1_closure_gain")) for row in gains]),
    }


def mean(values: Sequence[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FLUX Fill metrics against reference_uniform50.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run_root", default=str(RUN_ROOT))
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()
    run_root = Path(args.run_root)
    rows = read_csv(Path(args.manifest))
    if not rows:
        write_empty(run_root, "Manifest has no rows.")
        return
    metric_rows = build_metric_rows(rows)
    if not metric_rows:
        reason = "No complete case has both reference_uniform50, uniform8, and output images. Run smoke/mini first."
        write_empty(run_root, reason)
        if not args.allow_missing:
            raise SystemExit(reason)
        print(reason)
        return
    gains = build_gain_rows(metric_rows)
    write_csv(run_root / "metrics" / "master_long_metrics.csv", metric_rows)
    write_csv(run_root / "metrics" / "per_case_metrics.csv", metric_rows)
    write_csv(run_root / "metrics" / "same_compute_gain_long.csv", gains)
    write_json(run_root / "metrics" / "metrics_summary.json", summarize(metric_rows, gains))
    print(f"metric rows: {len(metric_rows)}")
    print(f"same-compute gain rows: {len(gains)}")
    print(f"mean mask gain: {fmt_float(summarize(metric_rows, gains)['mean_mask_gain'])}")


if __name__ == "__main__":
    main()
