#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from common import (
    DRIVE_EXPERIMENT_ROOT,
    DRIVE_WEIGHTS_ROOT,
    MANIFEST_FIELDS,
    RUN_ROOT,
    ensure_layout,
    mirror_to_drive_if_available,
    parse_bool,
    parse_methods,
    read_csv,
    write_csv,
    write_json,
)
from schedule_utils import write_schedule_for_row


def read_log_excerpt(path: str, max_chars: int = 4000) -> str:
    log_path = Path(str(path))
    if not log_path.exists():
        return f"missing log: {log_path}"
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"failed to read {log_path}: {exc!r}"
    if len(text) > max_chars:
        return "...\n" + text[-max_chars:]
    return text


def artifacts_ready(row: Dict[str, str]) -> bool:
    return Path(row["output_path"]).exists() and Path(row["schedule_json_path"]).exists()


def completed_ready(row: Dict[str, str]) -> bool:
    runtime_path = Path(str(row.get("runtime_json_path", "")))
    if not artifacts_ready(row) or not runtime_path.exists():
        return False
    try:
        import json

        payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        return payload.get("status") == "done"
    except Exception:
        return False


def dry_run_one(row: Dict[str, str]) -> Dict[str, Any]:
    Path(row["output_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(row["stdout_log_path"]).parent.mkdir(parents=True, exist_ok=True)
    schedule = write_schedule_for_row(row)
    command = [
        "FluxFillPipeline",
        "--model",
        "<local-drive-weights>",
        "--method",
        row["method"],
        "--case",
        row["case_id"],
        "--nfe",
        str(row["actual_nfe"]),
    ]
    Path(row["stdout_log_path"]).write_text("dry-run: " + " ".join(shlex.quote(part) for part in command) + "\n", encoding="utf-8")
    Path(row["stderr_log_path"]).write_text("dry-run: no stderr\n", encoding="utf-8")
    return {
        "status": "dry_run",
        "error_message": "",
        "command": command,
        "schedule_json_path": row["schedule_json_path"],
        "expected_nfe": schedule["actual_nfe"],
    }


def run_one(row: Dict[str, str], runner: Any, force: bool, dry_run: bool) -> Dict[str, Any]:
    Path(row["output_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(row["stdout_log_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(row["runtime_json_path"]).parent.mkdir(parents=True, exist_ok=True)
    if completed_ready(row) and not force:
        return {"status": "done", "skipped": True, "error_message": "", "runtime_sec": None}
    if dry_run:
        result = dry_run_one(row)
        result["skipped"] = True
        result["runtime_sec"] = None
        return result

    started = time.perf_counter()
    try:
        result = runner.run_row(row)
        runtime = time.perf_counter() - started
        if not artifacts_ready(row):
            return {
                "status": "failed",
                "skipped": False,
                "runtime_sec": runtime,
                "error_message": "inference returned but output/schedule artifacts are missing",
            }
        result.update({"skipped": False, "runtime_sec": runtime, "error_message": ""})
        return result
    except Exception as exc:
        runtime = time.perf_counter() - started
        Path(row["stderr_log_path"]).write_text(repr(exc) + "\n", encoding="utf-8")
        return {"status": "failed", "skipped": False, "runtime_sec": runtime, "error_message": repr(exc)}


def select_rows(rows: Sequence[Dict[str, str]], methods: Sequence[str], limit: int) -> List[Dict[str, str]]:
    method_set = set(methods)
    selected = [row for row in rows if row["method"] in method_set]
    if limit > 0:
        return selected[:limit]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FLUX Fill manifest rows with resume support.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run_root", default=str(RUN_ROOT))
    parser.add_argument("--drive_run_root", default=str(DRIVE_EXPERIMENT_ROOT))
    parser.add_argument("--model_path", default=str(DRIVE_WEIGHTS_ROOT))
    parser.add_argument("--methods", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--sync_drive", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu_offload", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    ensure_layout(run_root)
    manifest_path = Path(args.manifest)
    rows = read_csv(manifest_path)
    default_methods = sorted({row["method"] for row in rows})
    methods = parse_methods(args.methods, default=default_methods)
    selected = select_rows(rows, methods, int(args.limit))
    selected_ids = {row["run_id"] for row in selected}

    runner = None
    if not args.dry_run:
        from flux_fill_adapter import FluxFillRunner

        runner = FluxFillRunner(
            model_path=args.model_path,
            dtype=args.dtype,
            device=args.device,
            cpu_offload=bool(args.cpu_offload),
            local_files_only=True,
        )

    processed = 0
    failed_run_ids: List[str] = []
    for row in rows:
        if row["run_id"] not in selected_ids:
            continue
        if args.resume and completed_ready(row) and not args.force:
            continue
        result = run_one(row, runner, force=bool(args.force), dry_run=bool(args.dry_run))
        row["status"] = result["status"]
        row["error_message"] = result.get("error_message", "")
        runtime_payload = {
            "run_id": row["run_id"],
            "status": result["status"],
            "runtime_sec": result.get("runtime_sec"),
            "skipped": result.get("skipped"),
            "error_message": result.get("error_message", ""),
            "output_path": row["output_path"],
            "schedule_json_path": row["schedule_json_path"],
            "model_path": args.model_path,
            "dry_run": bool(args.dry_run),
            "timestamp": time.time(),
        }
        write_json(Path(row["runtime_json_path"]), runtime_payload)
        write_csv(manifest_path, rows, MANIFEST_FIELDS)
        processed += 1
        if args.sync_drive:
            mirror_to_drive_if_available(run_root, Path(args.drive_run_root))
        if result["status"] == "failed":
            failed_run_ids.append(row["run_id"])
            print(f"[failed] {row['run_id']}: {result.get('error_message')}")
            print(f"[failed] stdout log: {row.get('stdout_log_path')}")
            print(read_log_excerpt(row.get("stdout_log_path", "")))
            print(f"[failed] stderr log: {row.get('stderr_log_path')}")
            print(read_log_excerpt(row.get("stderr_log_path", "")))
            break
    print(f"processed {processed} selected row(s)")
    if failed_run_ids:
        raise SystemExit(f"run_manifest failed for {len(failed_run_ids)} row(s): {', '.join(failed_run_ids)}")


if __name__ == "__main__":
    main()
