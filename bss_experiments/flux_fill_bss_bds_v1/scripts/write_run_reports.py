#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List

from common import REFERENCE_METHOD, RUN_ROOT, markdown_table, read_csv


def status_summary(rows: List[Dict[str, str]]) -> Dict[str, int]:
    return dict(Counter(row.get("status", "") or "unknown" for row in rows))


def output_count(rows: List[Dict[str, str]]) -> int:
    return sum(1 for row in rows if Path(row.get("output_path", "")).exists())


def schedule_count(rows: List[Dict[str, str]]) -> int:
    return sum(1 for row in rows if Path(row.get("schedule_json_path", "")).exists())


def write_smoke_report(run_root: Path, rows: List[Dict[str, str]]) -> None:
    lines = [
        "# FLUX Fill Smoke Report",
        "",
        "This smoke checks case001 with uniform8, uniform10, bss10, and reference_uniform50.",
        "",
        f"- rows: `{len(rows)}`",
        f"- status summary: `{status_summary(rows)}`",
        f"- output images present: `{output_count(rows)}`",
        f"- schedule JSON present: `{schedule_count(rows)}`",
        f"- smoke pass: `{output_count(rows) == len(rows) and schedule_count(rows) == len(rows)}`",
        "",
        "## Rows",
        "",
        markdown_table(rows, ["run_id", "case_id", "method", "actual_nfe", "status", "error_message", "output_path"]),
    ]
    (run_root / "reports" / "01_flux_fill_smoke_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_mini_report(run_root: Path, rows: List[Dict[str, str]]) -> None:
    completed = output_count(rows)
    cases = sorted({row["case_id"] for row in rows})
    lines = [
        "# FLUX Fill Mini-Suite Run Report",
        "",
        "Mini-suite rows are only intended to run after smoke passes.",
        "",
        f"- cases: `{len(cases)}`",
        f"- rows: `{len(rows)}`",
        f"- completed output images: `{completed}`",
        f"- schedule JSON present: `{schedule_count(rows)}`",
        f"- status summary: `{status_summary(rows)}`",
        "",
        "## Rows",
        "",
        markdown_table(rows, ["run_id", "case_id", "method", "actual_nfe", "status", "error_message", "output_path"]),
    ]
    (run_root / "reports" / "02_mini_suite_run_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write smoke and mini run reports from manifests.")
    parser.add_argument("--run_root", default=str(RUN_ROOT))
    parser.add_argument("--smoke_manifest", default="")
    parser.add_argument("--mini_manifest", default="")
    args = parser.parse_args()
    run_root = Path(args.run_root)
    smoke = Path(args.smoke_manifest) if args.smoke_manifest else run_root / "manifests" / "flux_fill_smoke_manifest.csv"
    mini = Path(args.mini_manifest) if args.mini_manifest else run_root / "manifests" / "flux_fill_mini_manifest.csv"
    if smoke.exists():
        write_smoke_report(run_root, read_csv(smoke))
        print(f"smoke report: {run_root / 'reports/01_flux_fill_smoke_report.md'}")
    if mini.exists():
        write_mini_report(run_root, read_csv(mini))
        print(f"mini report: {run_root / 'reports/02_mini_suite_run_report.md'}")


if __name__ == "__main__":
    main()
