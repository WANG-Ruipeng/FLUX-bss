from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "bss_experiments" / "flux_fill_bss_bds_v1" / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

from flux_fill_adapter import (  # noqa: E402
    build_schedule_from_pipe,
    calculate_flux_shift,
    pipeline_default_sigmas,
    scheduler_coords_for_steps,
)


class FakeFlowMatchScheduler:
    def __init__(self) -> None:
        self.config = {
            "use_dynamic_shifting": True,
            "time_shift_type": "exponential",
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096,
            "base_shift": 0.5,
            "max_shift": 1.15,
            "shift_terminal": None,
            "use_karras_sigmas": False,
            "use_exponential_sigmas": False,
            "use_beta_sigmas": False,
            "invert_sigmas": False,
        }
        self.sigmas = []
        self.timesteps = []

    @staticmethod
    def shift_sigma(value: float, mu: float) -> float:
        factor = math.exp(mu)
        return factor / (factor + (1.0 / value - 1.0))

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None):
        if self.config["use_dynamic_shifting"] and mu is None:
            raise ValueError("`mu` must be passed when `use_dynamic_shifting` is set to be `True`")
        if sigmas is None:
            sigmas = pipeline_default_sigmas(int(num_inference_steps))
        shifted = [self.shift_sigma(float(value), float(mu)) for value in sigmas]
        self.sigmas = shifted + [0.0]
        self.timesteps = [value * 1000.0 for value in shifted]


class FakeFluxFillPipe:
    vae_scale_factor = 8

    def __init__(self) -> None:
        self.scheduler = FakeFlowMatchScheduler()

    def __call__(self, sigmas=None, timesteps=None):
        return None


def bss10_row() -> dict[str, str]:
    return {
        "method": "bss10",
        "sampler_mode": "bss",
        "actual_nfe": "10",
        "base_sample_steps": "8",
        "split_pairs": "0,-1",
        "case_id": "case001",
        "source_image_path": "__missing_source__.png",
        "mask_image_path": "__missing_mask__.png",
        "prompt": "fill the masked area",
        "prompt_hash": "",
        "seed": "0",
        "height": "1024",
        "width": "1024",
        "guidance_scale": "30",
        "max_sequence_length": "512",
        "output_path": "out.png",
    }


class FluxFillAdapterScheduleTest(unittest.TestCase):
    def test_calculate_flux_shift_matches_max_at_1024_square_fake_pipe(self) -> None:
        self.assertAlmostEqual(calculate_flux_shift(4096, 256, 4096, 0.5, 1.15), 1.15)

    def test_scheduler_coords_pass_dynamic_shift_mu(self) -> None:
        coords, terminal, coord_type, info = scheduler_coords_for_steps(
            FakeFluxFillPipe(), 8, device="cuda", height=1024, width=1024
        )
        self.assertEqual(len(coords), 8)
        self.assertEqual(terminal, 0.0)
        self.assertEqual(coord_type, "sigmas")
        self.assertAlmostEqual(info["mu"], 1.15)
        self.assertEqual(info["image_seq_len"], 4096)
        self.assertEqual(len(info["scheduler_input_sigmas"]), 8)

    def test_bss_schedule_uses_raw_input_sigmas_and_records_shifted_coords(self) -> None:
        payload, scheduler_info = build_schedule_from_pipe(FakeFluxFillPipe(), bss10_row(), device="cuda")
        self.assertEqual(len(payload["final_coords"]), 10)
        self.assertEqual(len(payload["scheduler_input_sigmas"]), 10)
        self.assertTrue(payload["custom_sigmas_used"])
        self.assertNotEqual(payload["scheduler_input_sigmas"], payload["final_coords"])
        self.assertAlmostEqual(scheduler_info["mu"], 1.15)
        for raw, shifted in zip(payload["scheduler_input_sigmas"], payload["final_coords"]):
            self.assertAlmostEqual(FakeFlowMatchScheduler.shift_sigma(raw, scheduler_info["mu"]), shifted)
        self.assertEqual(payload["requested_final_coords"], payload["final_coords"])


if __name__ == "__main__":
    unittest.main()
