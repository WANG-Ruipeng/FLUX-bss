# FLUX Fill BSS/BDS V1

This folder contains the code and schemas for a condition-anchoring smoke and
mini-suite around `FluxFillPipeline`.

Run order:

```bash
python scripts/audit_flux_fill.py
python scripts/make_flux_fill_assets_and_manifest.py
python scripts/validate_schedule.py --manifest <run_root>/manifests/flux_fill_smoke_manifest.csv
python scripts/run_manifest.py --manifest <run_root>/manifests/flux_fill_smoke_manifest.csv --resume
python scripts/compute_metrics_against_ref.py --manifest <run_root>/manifests/flux_fill_smoke_manifest.csv
python scripts/compute_bds.py
python scripts/make_tables_flux_fill.py
python scripts/make_figures.py --manifest <run_root>/manifests/flux_fill_mini_manifest.csv
python scripts/write_final_report.py
```

The Colab notebook wires these steps together and keeps weights plus generated
outputs in Drive.
