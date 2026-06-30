#!/usr/bin/env python3
from __future__ import annotations

import copy
import inspect
import math
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


def config_value(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def calculate_flux_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    if max_seq_len == base_seq_len:
        return float(max_shift)
    slope = (float(max_shift) - float(base_shift)) / float(max_seq_len - base_seq_len)
    intercept = float(base_shift) - slope * float(base_seq_len)
    return float(image_seq_len) * slope + intercept


def pipeline_default_sigmas(num_steps: int) -> List[float]:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if num_steps == 1:
        return [1.0]
    start = 1.0
    stop = 1.0 / float(num_steps)
    step = (stop - start) / float(num_steps - 1)
    return [start + step * idx for idx in range(num_steps)]


def flux_shift_context(pipe: Any, height: int, width: int) -> Dict[str, Any]:
    scheduler = pipe.scheduler
    config = getattr(scheduler, "config", {})
    vae_scale_factor = int(getattr(pipe, "vae_scale_factor", 8) or 8)
    image_seq_len = (int(height) // vae_scale_factor // 2) * (int(width) // vae_scale_factor // 2)
    base_seq_len = int(config_value(config, "base_image_seq_len", 256))
    max_seq_len = int(config_value(config, "max_image_seq_len", 4096))
    base_shift = float(config_value(config, "base_shift", 0.5))
    max_shift = float(config_value(config, "max_shift", 1.15))
    return {
        "mu": calculate_flux_shift(image_seq_len, base_seq_len, max_seq_len, base_shift, max_shift),
        "image_seq_len": image_seq_len,
        "vae_scale_factor": vae_scale_factor,
        "base_image_seq_len": base_seq_len,
        "max_image_seq_len": max_seq_len,
        "base_shift": base_shift,
        "max_shift": max_shift,
    }


def scheduler_shift_factor(scheduler: Any, mu: float | None) -> float:
    config = getattr(scheduler, "config", {})
    if bool(config_value(config, "use_dynamic_shifting", False)):
        if mu is None:
            raise RuntimeError("scheduler uses dynamic shifting but no mu is available")
        shift_type = str(config_value(config, "time_shift_type", "exponential"))
        if shift_type == "exponential":
            return math.exp(float(mu))
        if shift_type == "linear":
            return float(mu)
        raise RuntimeError(f"Unsupported FlowMatch time_shift_type={shift_type!r}; inspect scheduler before running BSS.")
    return float(getattr(scheduler, "shift", config_value(config, "shift", 1.0)))


def inverse_scheduler_shift_sigmas(scheduler: Any, shifted_sigmas: Sequence[float], mu: float | None = None) -> List[float]:
    config = getattr(scheduler, "config", {})
    unsupported_flags = [
        "use_karras_sigmas",
        "use_exponential_sigmas",
        "use_beta_sigmas",
        "invert_sigmas",
    ]
    active_flags = [name for name in unsupported_flags if bool(config_value(config, name, False))]
    if active_flags:
        raise RuntimeError(f"Unsupported scheduler sigma transform(s) for BSS inversion: {active_flags}")
    if config_value(config, "shift_terminal", None) is not None:
        raise RuntimeError("Unsupported scheduler shift_terminal for BSS inversion; inspect scheduler before running BSS.")

    factor = scheduler_shift_factor(scheduler, mu)
    raw: List[float] = []
    for value in shifted_sigmas:
        y = float(value)
        if y < -1e-8 or y > 1.0 + 1e-8:
            raise RuntimeError(f"shifted sigma outside [0, 1]: {y}")
        y = min(1.0, max(0.0, y))
        if y == 0.0:
            raw.append(0.0)
        else:
            raw.append(y / (factor * (1.0 - y) + y))
    return raw


def scheduler_coords_for_steps(
    pipe: Any,
    num_steps: int,
    device: str | None = None,
    height: int | None = None,
    width: int | None = None,
    input_sigmas: Sequence[float] | None = None,
) -> Tuple[List[float], float, str, Dict[str, Any]]:
    scheduler = copy.deepcopy(pipe.scheduler)
    kwargs: Dict[str, Any] = {}
    signature = inspect.signature(scheduler.set_timesteps)
    config = getattr(scheduler, "config", {})
    scheduler_input_sigmas: List[float] = []
    shift_context: Dict[str, Any] = {}

    if "device" in signature.parameters and device:
        kwargs["device"] = device
    if input_sigmas is not None:
        if len(input_sigmas) != num_steps:
            raise RuntimeError(f"expected {num_steps} input sigmas, got {len(input_sigmas)}")
        if "sigmas" not in signature.parameters:
            raise RuntimeError("scheduler.set_timesteps does not accept custom sigmas")
        scheduler_input_sigmas = [float(x) for x in input_sigmas]
        kwargs["sigmas"] = scheduler_input_sigmas
    elif "sigmas" in signature.parameters:
        scheduler_input_sigmas = pipeline_default_sigmas(num_steps)
        kwargs["sigmas"] = scheduler_input_sigmas

    if "mu" in signature.parameters and bool(config_value(config, "use_dynamic_shifting", False)):
        if height is None or width is None:
            raise RuntimeError("scheduler requires dynamic shift `mu`; height and width are needed to compute it")
        shift_context = flux_shift_context(pipe, int(height), int(width))
        kwargs["mu"] = shift_context["mu"]

    scheduler.set_timesteps(num_steps, **kwargs)
    sigmas = tensor_to_float_list(getattr(scheduler, "sigmas", None))
    timesteps = tensor_to_float_list(getattr(scheduler, "timesteps", None))
    call_signature = inspect.signature(pipe.__call__)
    info = {
        "scheduler_class": scheduler.__class__.__name__,
        "set_timesteps_signature": str(signature),
        "sigmas_len": len(sigmas),
        "timesteps_len": len(timesteps),
        "supports_sigmas_call_arg": "sigmas" in call_signature.parameters,
        "supports_timesteps_call_arg": "timesteps" in call_signature.parameters,
        "scheduler_input_coord_type": "raw_sigmas" if scheduler_input_sigmas else "scheduler_default",
        "scheduler_input_sigmas": scheduler_input_sigmas,
        "use_dynamic_shifting": bool(config_value(config, "use_dynamic_shifting", False)),
        **shift_context,
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
    sampler_mode = str(row["sampler_mode"])
    base_steps = int(row["base_sample_steps"])
    height = int(row["height"])
    width = int(row["width"])
    base_coords, terminal, coord_type, scheduler_info = scheduler_coords_for_steps(
        pipe, base_steps, device=device, height=height, width=width
    )
    base_input_sigmas = [float(x) for x in scheduler_info.get("scheduler_input_sigmas", [])]
    runtime_scheduler_info = scheduler_info
    scheduler_input_sigmas = list(base_input_sigmas)
    requested_final_coords: List[float]
    custom_scheduler_info: Dict[str, Any] = {}

    if sampler_mode == "bss":
        if coord_type != "sigmas":
            raise RuntimeError(
                f"BSS custom schedule requires scheduler sigmas for this scaffold; got coordinate_type={coord_type}. "
                "Stop and inspect the diffusers scheduler instead of faking BSS."
            )
        if not scheduler_info.get("supports_sigmas_call_arg"):
            raise RuntimeError("Current FluxFillPipeline.__call__ does not accept custom `sigmas`; cannot run BSS.")
        requested_final_coords, inserted = build_boundary_split_coords(
            base_coords, row.get("split_pairs") or "0,-1", terminal_coord=terminal
        )
        scheduler_input_sigmas = inverse_scheduler_shift_sigmas(
            pipe.scheduler, requested_final_coords, mu=scheduler_info.get("mu")
        )
        final_coords, custom_terminal, custom_coord_type, custom_scheduler_info = scheduler_coords_for_steps(
            pipe,
            len(scheduler_input_sigmas),
            device=device,
            height=height,
            width=width,
            input_sigmas=scheduler_input_sigmas,
        )
        if custom_coord_type != coord_type:
            raise RuntimeError(f"BSS custom schedule coordinate type changed from {coord_type} to {custom_coord_type}")
        terminal = custom_terminal
        runtime_scheduler_info = custom_scheduler_info
        if len(final_coords) != len(requested_final_coords):
            raise RuntimeError(f"BSS final schedule length changed from {len(requested_final_coords)} to {len(final_coords)}")
        max_abs_error = max(abs(float(a) - float(b)) for a, b in zip(final_coords, requested_final_coords))
        if max_abs_error > 1e-5:
            raise RuntimeError(
                f"BSS scheduler inversion failed; max abs error {max_abs_error:.3g}. "
                "Stop instead of running a double-shifted custom schedule."
            )
    else:
        final_coords, inserted = list(base_coords), []
        requested_final_coords = list(final_coords)

    payload = row_schedule_payload(row, base_coords=base_coords, terminal_coord=terminal)
    payload.update(
        {
            "coordinate_type": coord_type,
            "scheduler_input_coord_type": "raw_sigmas" if scheduler_input_sigmas else "scheduler_default",
            "base_scheduler_input_sigmas": base_input_sigmas,
            "scheduler_input_sigmas": [float(x) for x in scheduler_input_sigmas],
            "base_coords": [float(x) for x in base_coords],
            "requested_final_coords": [float(x) for x in requested_final_coords],
            "final_coords": [float(x) for x in final_coords],
            "inserted_midpoints": inserted,
            "terminal_coord": terminal,
            "scheduler_info": runtime_scheduler_info,
            "base_scheduler_info": scheduler_info,
            "custom_scheduler_info": custom_scheduler_info,
            "custom_sigmas_used": sampler_mode == "bss",
            "actual_model_evaluations_counted": None,
        }
    )
    if int(row["actual_nfe"]) != len(final_coords):
        raise RuntimeError(f"{method}: final schedule has {len(final_coords)} coords, expected {row['actual_nfe']}")
    return payload, runtime_scheduler_info


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
            kwargs["sigmas"] = [float(x) for x in payload["scheduler_input_sigmas"]]
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
