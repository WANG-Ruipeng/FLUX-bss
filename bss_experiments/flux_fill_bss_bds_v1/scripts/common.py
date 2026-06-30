#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

EXPERIMENT_NAME = "flux_fill_bss_bds_v1"
EXPERIMENT_REL = Path("bss_experiments") / EXPERIMENT_NAME
MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev"
MODEL_NAME = "FLUX.1 Fill-dev"
SETTING = "Image Fill"
TASK = "image_inpainting"
MODALITY = "image"
PROTOCOL = "fixed_fill_suite_diffusers"
REFERENCE_METHOD = "reference_uniform50"
REFERENCE_NFE = 50
LOW_BASELINE_METHOD = "uniform8"


def repo_root() -> Path:
    env = os.environ.get("BSS_CONDITION_REPO_DIR") or os.environ.get("REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


REPO_ROOT = repo_root()
REPO_EXPERIMENT_ROOT = REPO_ROOT / EXPERIMENT_REL
DRIVE_PROJECT_ROOT = Path(
    os.environ.get("DRIVE_PROJECT_ROOT", "/content/drive/MyDrive/Colab_Projects/FLUX-bss")
)
DRIVE_RUNS_ROOT = Path(os.environ.get("DRIVE_RUNS_ROOT", str(DRIVE_PROJECT_ROOT / "runs")))
DRIVE_MODELS_ROOT = Path(os.environ.get("DRIVE_MODELS_ROOT", str(DRIVE_PROJECT_ROOT / "models")))
DRIVE_WEIGHTS_ROOT = Path(os.environ.get("DRIVE_WEIGHTS_ROOT", str(DRIVE_MODELS_ROOT / "FLUX.1-Fill-dev")))
DRIVE_EXPERIMENT_ROOT = Path(os.environ.get("DRIVE_EXPERIMENT_ROOT", str(DRIVE_RUNS_ROOT / EXPERIMENT_NAME)))


def is_colab_runtime() -> bool:
    return "google.colab" in sys.modules or Path("/content").exists() or bool(os.environ.get("COLAB_RELEASE_TAG"))


def default_run_root() -> Path:
    env = os.environ.get("EXPERIMENT_ROOT") or os.environ.get("FLUX_FILL_EXPERIMENT_ROOT")
    if env:
        return Path(env).expanduser()
    if is_colab_runtime():
        return Path("/content/FLUX-bss-Runs") / EXPERIMENT_NAME
    return REPO_ROOT / "_local_runs" / EXPERIMENT_NAME


RUN_ROOT = default_run_root()

REPO_SUBDIRS = [
    "assets",
    "figures/side_by_side",
    "logs",
    "manifests",
    "metrics",
    "patches",
    "reports",
    "schedules",
    "scripts",
    "splits",
    "tables",
]

RUN_SUBDIRS = [
    "assets/fill_cases",
    "figures/side_by_side",
    "logs",
    "manifests",
    "metrics",
    "outputs/uniform8",
    "outputs/uniform10",
    "outputs/uniform20",
    "outputs/uniform40",
    "outputs/reference_uniform50",
    "outputs/bss10",
    "outputs/bss20",
    "outputs/bss40",
    "reports",
    "schedules",
    "splits",
    "tables",
]

METHOD_SPECS: Dict[str, Dict[str, Any]] = {
    "uniform8": {
        "method_family": "uniform",
        "sampler_mode": "uniform",
        "actual_nfe": 8,
        "num_inference_steps": 8,
        "base_sample_steps": 8,
        "split_pairs": "",
    },
    "uniform10": {
        "method_family": "uniform",
        "sampler_mode": "uniform",
        "actual_nfe": 10,
        "num_inference_steps": 10,
        "base_sample_steps": 10,
        "split_pairs": "",
    },
    "uniform20": {
        "method_family": "uniform",
        "sampler_mode": "uniform",
        "actual_nfe": 20,
        "num_inference_steps": 20,
        "base_sample_steps": 20,
        "split_pairs": "",
    },
    "uniform40": {
        "method_family": "uniform",
        "sampler_mode": "uniform",
        "actual_nfe": 40,
        "num_inference_steps": 40,
        "base_sample_steps": 40,
        "split_pairs": "",
    },
    REFERENCE_METHOD: {
        "method_family": "reference",
        "sampler_mode": "uniform",
        "actual_nfe": 50,
        "num_inference_steps": 50,
        "base_sample_steps": 50,
        "split_pairs": "",
    },
    "bss10": {
        "method_family": "bss",
        "sampler_mode": "bss",
        "actual_nfe": 10,
        "num_inference_steps": 10,
        "base_sample_steps": 8,
        "split_pairs": "0,-1",
    },
    "bss20": {
        "method_family": "bss",
        "sampler_mode": "bss",
        "actual_nfe": 20,
        "num_inference_steps": 20,
        "base_sample_steps": 18,
        "split_pairs": "0,-1",
    },
    "bss40": {
        "method_family": "bss",
        "sampler_mode": "bss",
        "actual_nfe": 40,
        "num_inference_steps": 40,
        "base_sample_steps": 38,
        "split_pairs": "0,-1",
    },
}

SMOKE_METHODS = ["uniform8", "uniform10", "bss10", REFERENCE_METHOD]
MINI_METHODS = ["uniform8", "uniform10", "uniform20", "uniform40", REFERENCE_METHOD, "bss10", "bss20", "bss40"]

MANIFEST_FIELDS = [
    "run_id",
    "model_id",
    "setting",
    "task",
    "modality",
    "protocol",
    "case_id",
    "category",
    "expected_edit_region",
    "prompt",
    "prompt_hash",
    "source_image_path",
    "mask_image_path",
    "method",
    "method_family",
    "sampler_mode",
    "actual_nfe",
    "num_inference_steps",
    "base_sample_steps",
    "split_pairs",
    "guidance_scale",
    "max_sequence_length",
    "seed",
    "height",
    "width",
    "dtype",
    "reference_method",
    "reference_nfe",
    "low_baseline_method",
    "output_path",
    "schedule_json_path",
    "stdout_log_path",
    "stderr_log_path",
    "runtime_json_path",
    "status",
    "error_message",
    "git_commit",
    "dirty_status",
]


def ensure_layout(run_root: Path = RUN_ROOT, repo_experiment_root: Path = REPO_EXPERIMENT_ROOT) -> None:
    for subdir in REPO_SUBDIRS:
        (repo_experiment_root / subdir).mkdir(parents=True, exist_ok=True)
    for subdir in RUN_SUBDIRS:
        (run_root / subdir).mkdir(parents=True, exist_ok=True)


def sanitize_slug(value: Any) -> str:
    text = str(value).replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_") or "item"


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_fields: List[str] = list(fields or [])
    seen = set(out_fields)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                out_fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in out_fields})


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(clean_json(row), sort_keys=True) + "\n")


def markdown_table(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        vals = [str(row.get(field, "")) for field in fields]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_table(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_table(rows, fields), encoding="utf-8")


def git_text(args: Sequence[str], cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(list(args), cwd=str(cwd), stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def git_commit(repo_dir: Path = REPO_ROOT) -> str:
    return git_text(["git", "rev-parse", "HEAD"], repo_dir)


def git_dirty_status(repo_dir: Path = REPO_ROOT) -> str:
    return git_text(["git", "status", "--short"], repo_dir)


def git_audit(repo_dir: Path = REPO_ROOT) -> Dict[str, str]:
    return {
        "remote": git_text(["git", "remote", "-v"], repo_dir),
        "branch": git_text(["git", "branch", "--show-current"], repo_dir),
        "commit": git_commit(repo_dir),
        "dirty_status": git_dirty_status(repo_dir),
    }


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"expected boolean, got {value!r}")


def parse_methods(text: str | None, default: Sequence[str] = MINI_METHODS) -> List[str]:
    if text is None or str(text).strip() == "":
        return list(default)
    out = [item.strip() for item in str(text).split(",") if item.strip()]
    invalid = [item for item in out if item not in METHOD_SPECS]
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}. Valid methods: {sorted(METHOD_SPECS)}")
    return out


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.lower() in {"none", "nan", "null"}:
            return None
        return float(text)
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    number = safe_float(value)
    if number is None:
        return None
    return int(round(number))


def fmt_float(value: Any, digits: int = 6) -> str:
    number = safe_float(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}g}"


def runtime_sec_from_row(row: Dict[str, Any]) -> float | None:
    path = Path(str(row.get("runtime_json_path", "")))
    if not path.exists():
        return None
    try:
        return safe_float(read_json(path).get("runtime_sec"))
    except Exception:
        return None


def mirror_to_drive_if_available(run_root: Path = RUN_ROOT, drive_root: Path = DRIVE_EXPERIMENT_ROOT) -> bool:
    if not is_colab_runtime() or not Path("/content/drive").exists():
        return False
    drive_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_root, drive_root, dirs_exist_ok=True)
    return True
