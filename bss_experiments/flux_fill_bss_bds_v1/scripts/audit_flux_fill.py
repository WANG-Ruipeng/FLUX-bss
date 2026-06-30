#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from common import (
    DRIVE_EXPERIMENT_ROOT,
    DRIVE_WEIGHTS_ROOT,
    MODEL_ID,
    REPO_ROOT,
    RUN_ROOT,
    ensure_layout,
    git_audit,
    is_colab_runtime,
    utc_timestamp,
    write_json,
)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "not installed"


def command_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=20).strip()
    except Exception as exc:
        return f"unavailable: {exc!r}"


def import_ok(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def torch_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "pytorch_version": package_version("torch"),
        "cuda_version": "unknown",
        "cuda_available": False,
        "gpu_name": "",
        "gpu_vram_gb": "",
    }
    try:
        import torch

        info["cuda_version"] = getattr(torch.version, "cuda", "") or ""
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_vram_gb"] = round(props.total_memory / (1024**3), 2)
    except Exception as exc:
        info["torch_error"] = repr(exc)
    return info


def weights_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "file_count": 0, "has_model_index": False}
    files = [p for p in path.rglob("*") if p.is_file()]
    return {
        "exists": True,
        "file_count": len(files),
        "has_model_index": (path / "model_index.json").exists(),
        "path": str(path),
    }


def try_pipeline_load(path: Path, dtype: str, cpu_offload: bool) -> Dict[str, Any]:
    try:
        import torch
        from diffusers import FluxFillPipeline

        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16 if dtype == "float16" else torch.float32
        pipe = FluxFillPipeline.from_pretrained(str(path), torch_dtype=torch_dtype, local_files_only=True)
        scheduler = getattr(pipe, "scheduler", None)
        if cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        return {
            "attempted": True,
            "loaded": True,
            "scheduler_class": scheduler.__class__.__name__ if scheduler else "",
            "has_set_timesteps": hasattr(scheduler, "set_timesteps") if scheduler else False,
            "has_sigmas": hasattr(scheduler, "sigmas") if scheduler else False,
            "dtype": dtype,
            "cpu_offload_requested": cpu_offload,
        }
    except Exception as exc:
        return {"attempted": True, "loaded": False, "error": repr(exc), "dtype": dtype, "cpu_offload_requested": cpu_offload}


def write_report(path: Path, audit: Dict[str, Any]) -> None:
    lines = [
        "# FLUX Fill Repo / Model / Hardware Audit",
        "",
        f"- generated_at: `{audit['generated_at']}`",
        f"- current repo path: `{audit['repo_path']}`",
        f"- git remote: `{audit['git'].get('remote') or 'N/A'}`",
        f"- current branch: `{audit['git'].get('branch') or 'N/A'}`",
        f"- commit hash: `{audit['git'].get('commit') or 'N/A'}`",
        f"- dirty status: `{audit['git'].get('dirty_status') or 'clean or not a git repo'}`",
        f"- Python version: `{audit['python_version']}`",
        f"- platform: `{audit['platform']}`",
        f"- PyTorch version: `{audit['torch'].get('pytorch_version')}`",
        f"- CUDA version: `{audit['torch'].get('cuda_version')}`",
        f"- CUDA available: `{audit['torch'].get('cuda_available')}`",
        f"- diffusers version: `{audit['packages']['diffusers']}`",
        f"- transformers version: `{audit['packages']['transformers']}`",
        f"- accelerate version: `{audit['packages']['accelerate']}`",
        f"- GPU name: `{audit['torch'].get('gpu_name') or 'N/A'}`",
        f"- GPU VRAM GB: `{audit['torch'].get('gpu_vram_gb') or 'N/A'}`",
        f"- RTX PRO 6000 Blackwell: `{audit['is_rtx_pro_6000_blackwell']}`",
        f"- running in Colab: `{audit['running_in_colab']}`",
        f"- Google Drive mounted: `{audit['google_drive_mounted']}`",
        f"- HF token available in env: `{audit['hf_token_available']}`",
        f"- user accepted FLUX license: `{audit['license_accepted']}`",
        f"- Drive weights path: `{audit['drive_weights_path']}`",
        f"- Drive weights exist: `{audit['weights'].get('exists')}`",
        f"- Drive weights file count: `{audit['weights'].get('file_count')}`",
        f"- FluxFillPipeline importable: `{audit['flux_fill_pipeline_importable']}`",
        f"- model gated / non-commercial dev license: `yes`",
        f"- exact model path loaded: `{audit['pipeline_load'].get('model_path_loaded', 'N/A')}`",
        f"- dtype requested: `{audit['dtype']}`",
        f"- CPU offload requested: `{audit['cpu_offload']}`",
        f"- pipeline load attempted: `{audit['pipeline_load'].get('attempted')}`",
        f"- pipeline loaded from local Drive path: `{audit['pipeline_load'].get('loaded')}`",
        f"- scheduler class: `{audit['pipeline_load'].get('scheduler_class', 'N/A')}`",
        f"- attention settings: `torch SDPA default unless diffusers/pytorch changes it; xformers not enabled by this scaffold`",
        "",
        "## nvidia-smi",
        "",
        "```text",
        audit["nvidia_smi"],
        "```",
        "",
        "## License / Download Instructions",
        "",
        "FLUX.1 Fill-dev is gated. If the local Drive weights are missing or the license is not accepted, stop after this audit and download only after accepting the Hugging Face license:",
        "",
        "```bash",
        "huggingface-cli download black-forest-labs/FLUX.1-Fill-dev \\",
        "  --local-dir /content/drive/MyDrive/ModelWeights/FLUX/FLUX.1-Fill-dev \\",
        "  --local-dir-use-symlinks False",
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit repo, hardware, license, packages, and local FLUX weights.")
    parser.add_argument("--run_root", default=str(RUN_ROOT))
    parser.add_argument("--drive_weights_root", default=str(DRIVE_WEIGHTS_ROOT))
    parser.add_argument("--license_accepted", default="unknown", choices=["yes", "no", "unknown"])
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--try_load_pipeline", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    weights_root = Path(args.drive_weights_root)
    ensure_layout(run_root)
    torch = torch_info()
    gpu_name = str(torch.get("gpu_name") or "")
    pipeline_load: Dict[str, Any] = {"attempted": False, "loaded": False, "model_path_loaded": ""}
    if args.try_load_pipeline and weights_root.exists():
        pipeline_load = try_pipeline_load(weights_root, args.dtype, bool(args.cpu_offload))
        if pipeline_load.get("loaded"):
            pipeline_load["model_path_loaded"] = str(weights_root)
    audit = {
        "generated_at": utc_timestamp(),
        "repo_path": str(REPO_ROOT),
        "run_root": str(run_root),
        "drive_experiment_root": str(DRIVE_EXPERIMENT_ROOT),
        "drive_weights_path": str(weights_root),
        "git": git_audit(REPO_ROOT),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "torch": package_version("torch"),
            "diffusers": package_version("diffusers"),
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
        },
        "torch": torch,
        "nvidia_smi": command_text(["nvidia-smi"]),
        "is_rtx_pro_6000_blackwell": "RTX PRO 6000" in gpu_name and "Blackwell" in gpu_name,
        "running_in_colab": is_colab_runtime(),
        "google_drive_mounted": Path("/content/drive").exists(),
        "hf_token_available": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")),
        "license_accepted": args.license_accepted,
        "weights": weights_summary(weights_root),
        "flux_fill_pipeline_importable": import_ok("diffusers"),
        "model_id": MODEL_ID,
        "dtype": args.dtype,
        "cpu_offload": bool(args.cpu_offload),
        "pipeline_load": pipeline_load,
    }
    report_path = run_root / "reports" / "00_repo_model_hardware_audit.md"
    write_json(run_root / "reports" / "00_repo_model_hardware_audit.json", audit)
    write_report(report_path, audit)
    print(f"audit report: {report_path}")
    print(f"weights exist: {audit['weights'].get('exists')} at {weights_root}")
    if not audit["weights"].get("exists") or args.license_accepted != "yes":
        print("audit note: gated weights/license not fully confirmed; do not run inference until resolved.")


if __name__ == "__main__":
    main()
