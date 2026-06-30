#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from common import RUN_ROOT, fmt_float, markdown_table, read_csv, safe_float, write_csv, write_markdown_table

FIELDS = ["Model", "Setting", "Few-step prior", "Ref.", "Cases", "BDS", "Low", "Middle Low", "Middle High", "High", "Mean Delta", "Win"]


def mean(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def cell(rows: Sequence[Dict[str, str]], nfe: int, gain_key: str, closure_key: str) -> str:
    subset = [row for row in rows if int(float(row.get("actual_nfe", 0) or 0)) == nfe]
    if not subset:
        return "--"
    gain = mean([safe_float(row.get(gain_key)) for row in subset])
    closure = mean([safe_float(row.get(closure_key)) for row in subset])
    if gain is None and closure is None:
        return "--"
    return f"{fmt_float(closure, 3)} / {fmt_float(gain, 3)} [{nfe} NFE]"


def win_rate(rows: Sequence[Dict[str, str]], gain_key: str) -> str:
    vals = [safe_float(row.get(gain_key)) for row in rows]
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return "N/A"
    return fmt_float(sum(1 for v in vals if v > 0) / len(vals), 3)


def latex_table(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> str:
    lines = ["\\begin{tabular}{llllllllllll}", "\\toprule", " & ".join(fields) + " \\\\", "\\midrule"]
    for row in rows:
        vals = [str(row.get(field, "")).replace("_", "\\_") for field in fields]
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def load_bds_verdict(run_root: Path) -> str:
    path = run_root / "tables" / "cross_model_bds_row.csv"
    if not path.exists():
        return "Need more data"
    rows = read_csv(path)
    if not rows:
        return "Need more data"
    return rows[0].get("final_verdict", "Need more data")


def build_row(gain_rows: Sequence[Dict[str, str]], verdict: str, metric_prefix: str) -> Dict[str, Any]:
    if metric_prefix == "mask":
        gain_key = "mask_rgb_l1_closure_gain"
        closure_key = "bss_mask_rgb_l1_closure"
    else:
        gain_key = "full_rgb_l1_closure_gain"
        closure_key = "bss_full_rgb_l1_closure"
    gains = [safe_float(row.get(gain_key)) for row in gain_rows]
    gains = [float(v) for v in gains if v is not None]
    return {
        "Model": "FLUX.1 Fill-dev",
        "Setting": "Image Fill",
        "Few-step prior": "Guidance-distilled / not step-distilled",
        "Ref.": "50",
        "Cases": len({row.get("case_id") for row in gain_rows}) if gain_rows else 0,
        "BDS": verdict,
        "Low": cell(gain_rows, 10, gain_key, closure_key),
        "Middle Low": cell(gain_rows, 20, gain_key, closure_key),
        "Middle High": cell(gain_rows, 30, gain_key, closure_key),
        "High": cell(gain_rows, 40, gain_key, closure_key),
        "Mean Delta": fmt_float(mean(gains), 3),
        "Win": win_rate(gain_rows, gain_key),
    }


def write_table_set(run_root: Path, name: str, rows: Sequence[Dict[str, Any]]) -> None:
    csv_path = run_root / "tables" / f"{name}.csv"
    md_path = run_root / "tables" / f"{name}.md"
    tex_path = run_root / "tables" / f"{name}.tex"
    write_csv(csv_path, rows, FIELDS)
    write_markdown_table(md_path, rows, FIELDS)
    tex_path.write_text(latex_table(rows, FIELDS), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-style FLUX Fill same-compute table rows.")
    parser.add_argument("--run_root", default=str(RUN_ROOT))
    parser.add_argument("--gain_csv", default="")
    args = parser.parse_args()
    run_root = Path(args.run_root)
    gain_csv = Path(args.gain_csv) if args.gain_csv else run_root / "metrics" / "same_compute_gain_long.csv"
    gain_rows = read_csv(gain_csv) if gain_csv.exists() else []
    verdict = load_bds_verdict(run_root)
    mask_row = build_row(gain_rows, verdict, "mask")
    full_row = build_row(gain_rows, verdict, "full")
    write_table_set(run_root, "table_cross_model_same_compute_flux_fill_row", [mask_row])
    write_table_set(run_root, "table_cross_model_same_compute_flux_fill_row_full_rgb", [full_row])
    print(f"mask closure row: {run_root / 'tables/table_cross_model_same_compute_flux_fill_row.csv'}")
    print(f"full closure row: {run_root / 'tables/table_cross_model_same_compute_flux_fill_row_full_rgb.csv'}")


if __name__ == "__main__":
    main()
