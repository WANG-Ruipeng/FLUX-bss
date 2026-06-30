#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import DRIVE_WEIGHTS_ROOT, RUN_ROOT, read_csv


def exists_text(path: Path) -> str:
    return str(path) if path.exists() else f"{path} (missing)"


def first_value(path: Path, key: str, default: str = "Need more data") -> str:
    if not path.exists():
        return default
    rows = read_csv(path)
    if not rows:
        return default
    return rows[0].get(key, default)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write final FLUX Fill BSS/BDS report.")
    parser.add_argument("--run_root", default=str(RUN_ROOT))
    parser.add_argument("--notebook_path", default="notebooks/flux_fill_colab_bss_bds.ipynb")
    args = parser.parse_args()
    run_root = Path(args.run_root)
    verdict = first_value(run_root / "tables" / "cross_model_bds_row.csv", "final_verdict")
    lines = [
        "# Final FLUX Fill BSS/BDS Report",
        "",
        "## 1. Purpose",
        "",
        "Condition-anchoring smoke after Sana-0.6B text-only Reject. This is not a FLUX SOTA benchmark and not a universal BSS claim.",
        "",
        "## 2. Model / License / Hardware Audit",
        "",
        f"- audit report: `{exists_text(run_root / 'reports/00_repo_model_hardware_audit.md')}`",
        f"- Drive weight path: `{DRIVE_WEIGHTS_ROOT}`",
        "",
        "## 3. Notebook",
        "",
        f"- Colab notebook path: `{args.notebook_path}`",
        "",
        "## 4. Source Image / Mask Suite",
        "",
        f"- asset manifest: `{exists_text(run_root / 'assets/asset_manifest.md')}`",
        "",
        "## 5. BSS Schedule Implementation",
        "",
        "BSS constructs a base schedule with T-2 scheduler coordinates, splits the first and last intervals, and passes custom `sigmas` to `FluxFillPipeline` when supported. If the installed diffusers pipeline does not support custom `sigmas`, this scaffold stops instead of faking BSS.",
        "",
        "## 6. Smoke Result",
        "",
        f"- smoke report: `{exists_text(run_root / 'reports/01_flux_fill_smoke_report.md')}`",
        "",
        "## 7. Mini-Suite Result",
        "",
        f"- mini-suite report: `{exists_text(run_root / 'reports/02_mini_suite_run_report.md')}`",
        "",
        "## 8. Same-Compute Closure Tables",
        "",
        f"- mask-region row: `{exists_text(run_root / 'tables/table_cross_model_same_compute_flux_fill_row.csv')}`",
        f"- full-image row: `{exists_text(run_root / 'tables/table_cross_model_same_compute_flux_fill_row_full_rgb.csv')}`",
        f"- LaTeX row: `{exists_text(run_root / 'tables/table_cross_model_same_compute_flux_fill_row.tex')}`",
        "",
        "## 9. Unmasked Preservation",
        "",
        "Unmasked preservation is reported as BSS minus same-NFE uniform unmasked RGB-L1 to source. Positive values trigger the preservation guard if they exceed tolerance.",
        "",
        "## 10. BDS Calibration / Holdout",
        "",
        f"- BDS table: `{exists_text(run_root / 'tables/tableA_flux_fill_bds_by_split.csv')}`",
        f"- BDS report: `{exists_text(run_root / 'reports/03_bds_report.md')}`",
        "",
        "## 11. Figures",
        "",
        f"- mask closure figure: `{exists_text(run_root / 'figures/compute_quality_mask_rgb_closure.png')}`",
        f"- full closure figure: `{exists_text(run_root / 'figures/compute_quality_full_rgb_closure.png')}`",
        f"- side-by-side index: `{exists_text(run_root / 'figures/side_by_side/index.html')}`",
        "",
        "## 12. Final Verdict",
        "",
        f"`{verdict}`",
        "",
        "## 13. Caveats",
        "",
        "- FLUX.1 Fill-dev is gated and uses a non-commercial dev license.",
        "- Fixed synthetic/public assets are a smoke suite, not a benchmark.",
        "- Reference is uniform50, not ground truth.",
        "- NFE is used as the compute proxy.",
        "- Fill/inpainting metrics depend on mask definition.",
        "- The model card describes guidance distillation; do not label this as path-consistency distillation unless separately audited.",
        "",
        "## 14. Next Steps",
        "",
        "- Expand to 8 cases if 4-case smoke/mini trends are promising.",
        "- Try FLUX.1 Canny-dev after fill tools pass.",
        "- Try Qwen-Image-Edit only after FLUX tooling is stable.",
    ]
    path = run_root / "reports" / "FINAL_FLUX_FILL_BSS_BDS_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"final report: {path}")
    print(f"final verdict: {verdict}")


if __name__ == "__main__":
    main()
