#!/usr/bin/env python3
from __future__ import annotations

import copy
import inspect
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from common import MODEL_ID, REPO_ROOT, git_commit, git_dirty_status, sha256_file, sha256_text, utc_timestamp, write_json
from schedule_utils import build_boundary_split_coords, row_schedule_payload


def import_runtime_deps():
    try:
        import torch
        from diffusers import FluxFillPipeline
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("FLUX Fill runtime dependencies are missing. Install diffusers, torch, and pillow.") from exc
    return torch, FluxFillPipeline, Image


def dtype_from_text(torch: Any, text: str) -> Any:
    value = str(text).lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def tensor_to_float_list(value: Any) -> List[float]:
    if value is None:
        return []
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().float().tolist()
        elif hasattr(value, "cpu"):
            value = value.cpu().tolist()
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return [float(x) for x in list(value)]


def scheduler_coords_for_steps(pipe: Any, num_steps: int, device: str | None = None) -> Tuple[List[float], float, str, Dict[str, Any]]:
    scheduler = copy.deepcopy(pipe.scheduler)
    kwargs: Dict[str, Any] = {}
    signature = inspect.signature(scheduler.set_timesteps)
    if "device" in signature.parameters and device:
        kwargs["device"] = device
    scheduler.set_timesteps(num_steps, **kwargs)
    sigmas = tensor_to_float_list(getattr(scheduler, "sigmas", None))
    timesteps = tensor_to_float_list(getattr(scheduler, "timesteps", None))
    info = {
        "scheduler_class": scheduler.__class__.__name__,
        "set_timesteps_signature": str(signature),
        "sigmas_len": len(sigmas),
        "timesteps_len": len(timesteps),
        "supports_sigmas_call_arg": "sigmas" in inspect.signature(pipe.__call__).parameters,
        "supports_timesteps_call_arg": "timesteps" in inspect.signature(pipe.__call__).parameters,
    }
    if sigmas:
        if len(sigmas) == num_steps + 1:
            return sigmas[:-1], float(sigmas[-1]), "sigmas", info
        if len(sigmas) == num_steps:
            return sigmas, 0.0, "sigmas", info
    if timesteps:
        if len(timesteps) == num_steps:
            return timesteps, 0.0, "timesteps", info
        if len(timesteps) == num_steps + 1:
            return timesteps[:-1], float(timesteps[-1]), "timesteps", info
    raise RuntimeError(f"Could not extract {num_steps} scheduler coordinates. Info: {info}")


def build_schedule_from_pipe(pipe: Any, row: Dict[str, Any], device: str | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    method = str(row["method"])
    base_steps = int(row["base_sample_steps"])
    base_coords, terminal, coord_type, scheduler_info = scheduler_coords_for_steps(pipe, base_steps, device=device)
    if str(row["sampler_mode"]) == "bss":
        if coord_type != "sigmas":
            raise RuntimeError(
                f"BSS custom schedule requires scheduler sigmas for this scaffold; got coordinate_type={coord_type}. "
                "Stop and inspect the diffusers scheduler instead of faking BSS."
            )
        final_coords, inserted = build_boundary_split_coords(base_coords, row.get("split_pairs") or "0,-1", terminal_coord=terminal)
    else:
        final_coords, inserted = list(base_coords), []

    payload = row_schedule_payload(row, base_coords=base_coords, terminal_coord=terminal)
    payload.update(
        {
            "coordinate_type": coord_type,
            "base_coords": [float(x) for x in base_coords],
            "final_coords": [float(x) for x in final_coords],
            "inserted_midpoints": inserted,
            "terminal_coord": terminal,
            "scheduler_info": scheduler_info,
            "custom_sigmas_used": str(row["sampler_mode"]) == "bss",
            "actual_model_evaluations_counted": None,
        }
    )
    if int(row["actual_nfe"]) != len(final_coords):
        raise RuntimeError(f"{method}: final schedule has {len(final_coords)} coords, expected {row['actual_nfe']}")
    return payload, scheduler_info


class FluxFillRunner:
    def __init__(
        self,
        model_path: str | Path,
        dtype: str = "bfloat16",
        device: str = "cuda",
        cpu_offload: bool = False,
        local_files_only: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.dtype = dtype
        self.device = device
        self.cpu_offload = cpu_offload
        self.local_files_only = local_files_only
        self._pipe = None

    def load(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        torch, FluxFillPipeline, _Image = import_runtime_deps()
        torch_dtype = dtype_from_text(torch, self.dtype)
        pipe = FluxFillPipeline.from_pretrained(str(self.model_path), torch_dtype=torch_dtype, local_files_only=self.local_files_only)
        if self.cpu_offload:
            if not hasattr(pipe, "enable_model_cpu_offload"):
                raise RuntimeError("Pipeline does not expose enable_model_cpu_offload().")
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(self.device)
        self._pipe = pipe
        return pipe

    def run_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        torch, _FluxFillPipeline, Image = import_runtime_deps()
        pipe = self.load()
        output_path = Path(str(row["output_path"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        source = Image.open(str(row["source_image_path"])).convert("RGB").resize((int(row["width"]), int(row["height"])))
        mask = Image.open(str(row["mask_image_path"])).convert("L").resize((int(row["width"]), int(row["height"])))
        payload, scheduler_info = build_schedule_from_pipe(pipe, row, device=self.device if not self.cpu_offload else None)
        call_signature = inspect.signature(pipe.__call__)
        supports_sigmas = "sigmas" in call_signature.parameters
        if str(row["sampler_mode"]) == "bss" and not supports_sigmas:
            raise RuntimeError(
                "Current FluxFillPipeline.__call__ does not accept `sigmas`; custom BSS schedule injection is unsupported. "
                "Stop after audit/report instead of faking BSS."
            )

        generator_device = self.device if self.device == "cuda" and torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(row["seed"]))
        step_counter = {"count": 0}

        def on_step_end(_pipe, _step_index, _timestep, callback_kwargs):
            step_counter["count"] += 1
            return callback_kwargs

        kwargs: Dict[str, Any] = {
            "prompt": str(row["prompt"]),
            "image": source,
            "mask_image": mask,
            "height": int(row["height"]),
            "width": int(row["width"]),
            "num_inference_steps": int(row["num_inference_steps"]),
            "guidance_scale": float(row["guidance_scale"]),
            "max_sequence_length": int(row["max_sequence_length"]),
            "generator": generator,
        }
        if str(row["sampler_mode"]) == "bss":
            kwargs["sigmas"] = [float(x) for x in payload["final_coords"]]
        if "callback_on_step_end" in call_signature.parameters:
            kwargs["callback_on_step_end"] = on_step_end

        result = pipe(**kwargs)
        image = result.images[0]
        image.save(output_path)
        payload.update(
            {
                "actual_model_evaluations_counted": step_counter["count"] or len(payload["final_coords"]),
                "model_path_loaded": str(self.model_path),
                "pipeline_call_signature": str(call_signature),
                "source_image_hash": sha256_file(Path(row["source_image_path"])),
                "mask_hash": sha256_file(Path(row["mask_image_path"])),
                "prompt_hash": row.get("prompt_hash") or sha256_text(str(row.get("prompt", ""))),
                "git_commit": git_commit(REPO_ROOT),
                "dirty_status": git_dirty_status(REPO_ROOT),
                "timestamp": utc_timestamp(),
            }
        )
        write_json(Path(str(row["schedule_json_path"])), payload)
        return {
            "status": "done",
            "output_path": str(output_path),
            "schedule_json_path": row["schedule_json_path"],
            "scheduler_info": scheduler_info,
            "nfe_counted": payload["actual_model_evaluations_counted"],
            "model_id": MODEL_ID,
        }
