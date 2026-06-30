#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from common import REFERENCE_METHOD, RUN_ROOT, read_csv, safe_float


def plot_metric(run_root: Path, rows: List[Dict[str, str]], metric: str, output_name: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        (run_root / "figures" / f"{output_name}.txt").write_text("matplotlib unavailable; figure not generated\n", encoding="utf-8")
        return
    by_method: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        value = safe_float(row.get(metric))
        if value is not None:
            by_method[row["method"]].append(value)
    if not by_method:
        return
    order = ["uniform8", "uniform10", "bss10", "uniform20", "bss20", "uniform40", "bss40", REFERENCE_METHOD]
    labels = [m for m in order if m in by_method]
    values = [sum(by_method[m]) / len(by_method[m]) for m in labels]
    plt.figure(figsize=(9, 4))
    colors = ["#777777" if "uniform" in m else "#2c7fb8" if "bss" in m else "#222222" for m in labels]
    plt.bar(labels, values, color=colors)
    plt.ylabel(metric)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = run_root / "figures" / f"{output_name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160)
    plt.close()


def rel(path: str, index_path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(index_path.parent.resolve()).as_posix()
    except Exception:
        return Path(path).as_posix()


def write_side_by_side(run_root: Path, manifest_rows: List[Dict[str, str]]) -> None:
    rows_by_case: Dict[str, Dict[str, str]] = defaultdict(dict)
    meta_by_case: Dict[str, Dict[str, str]] = {}
    for row in manifest_rows:
        rows_by_case[row["case_id"]][row["method"]] = row.get("output_path", "")
        meta_by_case[row["case_id"]] = row
    index_path = run_root / "figures" / "side_by_side" / "index.html"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ["source", "mask", "uniform8", "uniform10", "bss10", REFERENCE_METHOD],
        ["source", "mask", "uniform10", "bss10", "uniform20", "bss20", "uniform40", "bss40", REFERENCE_METHOD],
    ]
    lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>FLUX Fill BSS/BDS Side-by-Side</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:32px}.cell{border:1px solid #ddd;padding:8px}.cell img{max-width:100%;display:block}.label{font-size:12px;font-weight:700;margin-bottom:6px}</style>",
        "<h1>FLUX Fill BSS/BDS Side-by-Side</h1>",
    ]
    for case_id, methods in sorted(rows_by_case.items()):
        meta = meta_by_case[case_id]
        lines.append(f"<h2>{html.escape(case_id)} - {html.escape(meta.get('category',''))}</h2>")
        for panel in panels:
            lines.append("<div class='grid'>")
            for method in panel:
                if method == "source":
                    path = meta.get("source_image_path", "")
                elif method == "mask":
                    path = meta.get("mask_image_path", "")
                else:
                    path = methods.get(method, "")
                exists = path and Path(path).exists()
                lines.append("<div class='cell'>")
                lines.append(f"<div class='label'>{html.escape(method)}</div>")
                if exists:
                    lines.append(f"<img src='{html.escape(rel(path, index_path))}' alt='{html.escape(method)}'>")
                else:
                    lines.append("<p>missing</p>")
                lines.append("</div>")
            lines.append("</div>")
    index_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create FLUX Fill metrics figures and side-by-side HTML.")
    parser.add_argument("--run_root", default=str(RUN_ROOT))
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root)
    metric_path = run_root / "metrics" / "per_case_metrics.csv"
    metric_rows = read_csv(metric_path) if metric_path.exists() else []
    if metric_rows:
        plot_metric(run_root, metric_rows, "mask_rgb_l1_closure", "compute_quality_mask_rgb_closure")
        plot_metric(run_root, metric_rows, "rgb_l1_closure", "compute_quality_full_rgb_closure")
    write_side_by_side(run_root, read_csv(Path(args.manifest)))
    print(f"side-by-side: {run_root / 'figures/side_by_side/index.html'}")


if __name__ == "__main__":
    main()
