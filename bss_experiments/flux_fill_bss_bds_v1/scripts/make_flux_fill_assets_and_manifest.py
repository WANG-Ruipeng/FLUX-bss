#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence

from common import (
    DRIVE_EXPERIMENT_ROOT,
    LOW_BASELINE_METHOD,
    MANIFEST_FIELDS,
    METHOD_SPECS,
    MINI_METHODS,
    MODEL_ID,
    MODALITY,
    PROTOCOL,
    REFERENCE_METHOD,
    REFERENCE_NFE,
    REPO_EXPERIMENT_ROOT,
    REPO_ROOT,
    RUN_ROOT,
    SETTING,
    SMOKE_METHODS,
    TASK,
    ensure_layout,
    git_commit,
    git_dirty_status,
    sanitize_slug,
    sha256_file,
    sha256_text,
    utc_timestamp,
    write_csv,
    write_json,
    write_jsonl,
)


def require_pillow():
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover - exercised only in missing deps envs
        raise SystemExit("Pillow is required to create synthetic fill assets. Install `pillow`.") from exc
    return Image, ImageDraw


CASE_DEFS = [
    {
        "case_id": "case001",
        "category": "object_replacement_on_table",
        "expected_edit_region": "center tabletop object",
        "prompt": "Replace the masked red cup on the table with a small blue ceramic vase, realistic lighting, keep the table and background unchanged.",
    },
    {
        "case_id": "case002",
        "category": "indoor_object_remove_replace",
        "expected_edit_region": "right side floor object",
        "prompt": "Replace the masked cardboard box with a low green indoor plant in a white pot, preserve the room layout and shadows.",
    },
    {
        "case_id": "case003",
        "category": "outdoor_missing_region_fill",
        "expected_edit_region": "middle foreground path interruption",
        "prompt": "Fill the masked missing region with a natural stone path continuing through the grass, matching the outdoor scene.",
    },
    {
        "case_id": "case004",
        "category": "local_texture_lighting_edit",
        "expected_edit_region": "warm illuminated wall patch",
        "prompt": "Turn the masked wall patch into a warm window-shaped light reflection with subtle texture, keep unmasked areas stable.",
    },
]


def draw_case(case_id: str, size: int, source_path: Path, mask_path: Path) -> None:
    Image, ImageDraw = require_pillow()
    source_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (size, size), (230, 230, 225))
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    md = ImageDraw.Draw(mask)
    s = size

    if case_id == "case001":
        draw.rectangle([0, 0, s, int(s * 0.58)], fill=(210, 220, 228))
        draw.rectangle([0, int(s * 0.58), s, s], fill=(150, 104, 70))
        draw.rectangle([int(s * 0.18), int(s * 0.64), int(s * 0.82), int(s * 0.72)], fill=(115, 74, 48))
        draw.ellipse([int(s * 0.43), int(s * 0.39), int(s * 0.57), int(s * 0.57)], fill=(180, 35, 35))
        draw.rectangle([int(s * 0.44), int(s * 0.47), int(s * 0.56), int(s * 0.64)], fill=(165, 32, 32))
        md.ellipse([int(s * 0.39), int(s * 0.36), int(s * 0.61), int(s * 0.67)], fill=255)
    elif case_id == "case002":
        draw.rectangle([0, 0, s, int(s * 0.62)], fill=(218, 214, 205))
        draw.rectangle([0, int(s * 0.62), s, s], fill=(126, 111, 93))
        draw.rectangle([int(s * 0.16), int(s * 0.16), int(s * 0.42), int(s * 0.40)], outline=(95, 75, 60), width=max(3, s // 80))
        draw.rectangle([int(s * 0.18), int(s * 0.18), int(s * 0.40), int(s * 0.38)], fill=(128, 154, 170))
        draw.rectangle([int(s * 0.66), int(s * 0.60), int(s * 0.84), int(s * 0.78)], fill=(145, 98, 58))
        draw.line([int(s * 0.66), int(s * 0.60), int(s * 0.75), int(s * 0.52), int(s * 0.84), int(s * 0.60)], fill=(170, 124, 76), width=max(2, s // 100))
        md.rectangle([int(s * 0.62), int(s * 0.50), int(s * 0.88), int(s * 0.82)], fill=255)
    elif case_id == "case003":
        draw.rectangle([0, 0, s, int(s * 0.45)], fill=(148, 198, 235))
        draw.polygon([(0, int(s * 0.45)), (int(s * 0.30), int(s * 0.22)), (int(s * 0.56), int(s * 0.45))], fill=(105, 126, 118))
        draw.polygon([(int(s * 0.35), int(s * 0.45)), (int(s * 0.70), int(s * 0.20)), (s, int(s * 0.45))], fill=(95, 118, 109))
        draw.rectangle([0, int(s * 0.45), s, s], fill=(92, 151, 76))
        draw.polygon([(int(s * 0.40), s), (int(s * 0.47), int(s * 0.62)), (int(s * 0.56), int(s * 0.62)), (int(s * 0.66), s)], fill=(138, 123, 96))
        draw.rectangle([int(s * 0.42), int(s * 0.61), int(s * 0.62), int(s * 0.75)], fill=(225, 225, 215))
        md.rectangle([int(s * 0.40), int(s * 0.58), int(s * 0.66), int(s * 0.78)], fill=255)
    elif case_id == "case004":
        draw.rectangle([0, 0, s, s], fill=(96, 105, 112))
        for x in range(0, s, max(12, s // 32)):
            shade = 92 + (x // max(1, s // 16)) % 22
            draw.line([x, 0, x, s], fill=(shade, shade + 8, shade + 15), width=max(1, s // 180))
        draw.rectangle([int(s * 0.18), int(s * 0.22), int(s * 0.42), int(s * 0.72)], outline=(70, 78, 86), width=max(3, s // 90))
        draw.rectangle([int(s * 0.58), int(s * 0.32), int(s * 0.82), int(s * 0.68)], fill=(112, 118, 124))
        md.rectangle([int(s * 0.55), int(s * 0.28), int(s * 0.85), int(s * 0.72)], fill=255)
    else:
        raise ValueError(f"unknown case_id: {case_id}")

    img.save(source_path)
    mask.save(mask_path)


def build_rows(
    cases: Sequence[Dict[str, Any]],
    methods: Sequence[str],
    run_root: Path,
    asset_root: Path,
    seed: int,
    height: int,
    width: int,
    guidance_scale: float,
    max_sequence_length: int,
    dtype: str,
) -> List[Dict[str, Any]]:
    commit = git_commit(REPO_ROOT)
    dirty = git_dirty_status(REPO_ROOT)
    rows: List[Dict[str, Any]] = []
    for case in cases:
        case_id = case["case_id"]
        source_path = asset_root / case_id / "source.png"
        mask_path = asset_root / case_id / "mask.png"
        prompt = case["prompt"]
        prompt_hash = sha256_text(prompt)
        for method in methods:
            spec = METHOD_SPECS[method]
            run_id = f"{sanitize_slug(case_id)}__{method}__seed{seed}__{height}x{width}"
            output_name = f"flux_fill__{sanitize_slug(case_id)}__{method}__seed{seed}__nfe{spec['actual_nfe']}.png"
            row = {
                "run_id": run_id,
                "model_id": "FLUX.1-Fill-dev",
                "setting": SETTING,
                "task": TASK,
                "modality": MODALITY,
                "protocol": PROTOCOL,
                "case_id": case_id,
                "category": case["category"],
                "expected_edit_region": case["expected_edit_region"],
                "prompt": prompt,
                "prompt_hash": prompt_hash,
                "source_image_path": str(source_path),
                "mask_image_path": str(mask_path),
                "method": method,
                "method_family": spec["method_family"],
                "sampler_mode": spec["sampler_mode"],
                "actual_nfe": spec["actual_nfe"],
                "num_inference_steps": spec["num_inference_steps"],
                "base_sample_steps": spec["base_sample_steps"],
                "split_pairs": spec["split_pairs"],
                "guidance_scale": guidance_scale,
                "max_sequence_length": max_sequence_length,
                "seed": seed,
                "height": height,
                "width": width,
                "dtype": dtype,
                "reference_method": REFERENCE_METHOD,
                "reference_nfe": REFERENCE_NFE,
                "low_baseline_method": LOW_BASELINE_METHOD,
                "output_path": str(run_root / "outputs" / method / case_id / output_name),
                "schedule_json_path": str(run_root / "schedules" / f"{run_id}.schedule.json"),
                "stdout_log_path": str(run_root / "logs" / f"{run_id}.stdout.log"),
                "stderr_log_path": str(run_root / "logs" / f"{run_id}.stderr.log"),
                "runtime_json_path": str(run_root / "logs" / f"{run_id}.runtime.json"),
                "status": "pending",
                "error_message": "",
                "git_commit": commit,
                "dirty_status": dirty,
            }
            rows.append(row)
    return rows


def write_asset_manifest(asset_root: Path, cases: Sequence[Dict[str, Any]], size: int, repo_copy: bool) -> None:
    lines = [
        "# FLUX Fill Synthetic Asset Manifest",
        "",
        f"- generated_at: `{utc_timestamp()}`",
        f"- asset_root: `{asset_root}`",
        f"- resolution: `{size}x{size}`",
        "- source: deterministic synthetic PIL drawings generated by `make_flux_fill_assets_and_manifest.py`",
        "- license: project-generated synthetic assets; no external copyrighted image sources",
        "",
        "| case_id | category | source | mask | prompt |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        case_id = case["case_id"]
        lines.append(
            f"| {case_id} | {case['category']} | `{asset_root / case_id / 'source.png'}` | `{asset_root / case_id / 'mask.png'}` | {case['prompt']} |"
        )
    text = "\n".join(lines) + "\n"
    (asset_root.parent / "asset_manifest.md").write_text(text, encoding="utf-8")
    if repo_copy:
        (REPO_EXPERIMENT_ROOT / "assets" / "asset_manifest.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic FLUX Fill cases and smoke/mini manifests.")
    parser.add_argument("--run_root", default=str(RUN_ROOT))
    parser.add_argument("--asset_root", default="", help="Default: <run_root>/assets/fill_cases")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=30.0)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--repo_asset_manifest_copy", action="store_true", default=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    asset_root = Path(args.asset_root) if args.asset_root else run_root / "assets" / "fill_cases"
    ensure_layout(run_root)
    for case in CASE_DEFS:
        draw_case(case["case_id"], args.size, asset_root / case["case_id"] / "source.png", asset_root / case["case_id"] / "mask.png")

    write_asset_manifest(asset_root, CASE_DEFS, args.size, repo_copy=bool(args.repo_asset_manifest_copy))
    smoke_rows = build_rows(
        CASE_DEFS[:1],
        SMOKE_METHODS,
        run_root=run_root,
        asset_root=asset_root,
        seed=args.seed,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        dtype=args.dtype,
    )
    mini_rows = build_rows(
        CASE_DEFS,
        MINI_METHODS,
        run_root=run_root,
        asset_root=asset_root,
        seed=args.seed,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        dtype=args.dtype,
    )
    smoke_csv = run_root / "manifests" / "flux_fill_smoke_manifest.csv"
    mini_csv = run_root / "manifests" / "flux_fill_mini_manifest.csv"
    write_csv(smoke_csv, smoke_rows, MANIFEST_FIELDS)
    write_csv(mini_csv, mini_rows, MANIFEST_FIELDS)
    write_jsonl(run_root / "manifests" / "flux_fill_smoke_manifest.jsonl", smoke_rows)
    write_jsonl(run_root / "manifests" / "flux_fill_mini_manifest.jsonl", mini_rows)
    write_json(
        run_root / "manifests" / "manifest_summary.json",
        {
            "model_id": MODEL_ID,
            "run_root": str(run_root),
            "drive_experiment_root": str(DRIVE_EXPERIMENT_ROOT),
            "asset_root": str(asset_root),
            "smoke_manifest": str(smoke_csv),
            "mini_manifest": str(mini_csv),
            "smoke_rows": len(smoke_rows),
            "mini_rows": len(mini_rows),
            "height": args.height,
            "width": args.width,
            "guidance_scale": args.guidance_scale,
            "max_sequence_length": args.max_sequence_length,
            "seed": args.seed,
            "dtype": args.dtype,
        },
    )
    print(f"wrote synthetic assets: {asset_root}")
    print(f"wrote smoke manifest ({len(smoke_rows)} rows): {smoke_csv}")
    print(f"wrote mini manifest ({len(mini_rows)} rows): {mini_csv}")


if __name__ == "__main__":
    main()
