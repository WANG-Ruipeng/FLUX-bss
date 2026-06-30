from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "bss_experiments" / "flux_fill_bss_bds_v1" / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

from schedule_utils import (  # noqa: E402
    build_boundary_split_coords,
    compare_uniform_and_bss,
    make_default_sigmas,
    normalize_split_intervals,
    row_schedule_payload,
    schedule_for_method,
    validate_schedule_payload,
)


class ScheduleUtilsTest(unittest.TestCase):
    def test_normalize_split_intervals_first_and_last_alias(self) -> None:
        self.assertEqual(normalize_split_intervals("0,-1", 8), [0, 7])

    def test_bss10_splits_base8_first_and_last(self) -> None:
        base = make_default_sigmas(8)
        final, inserted = build_boundary_split_coords(base, "0,-1", terminal_coord=0.0)
        self.assertEqual(len(final), 10)
        self.assertEqual([item["interval"] for item in inserted], [0, 7])
        expected = [
            base[0],
            (base[0] + base[1]) * 0.5,
            base[1],
            base[2],
            base[3],
            base[4],
            base[5],
            base[6],
            base[7],
            base[7] * 0.5,
        ]
        self.assertEqual(final, expected)

    def test_method_schedule_counts(self) -> None:
        for method, expected_nfe in [
            ("uniform8", 8),
            ("uniform10", 10),
            ("uniform20", 20),
            ("uniform40", 40),
            ("reference_uniform50", 50),
            ("bss10", 10),
            ("bss20", 20),
            ("bss40", 40),
        ]:
            payload = schedule_for_method(method)
            self.assertEqual(len(payload["final_coords"]), expected_nfe)
            self.assertEqual(validate_schedule_payload(payload), [])

    def test_bss_differs_from_same_nfe_uniform(self) -> None:
        for nfe in [10, 20, 40]:
            uniform = schedule_for_method(f"uniform{nfe}")
            bss = schedule_for_method(f"bss{nfe}")
            self.assertFalse(compare_uniform_and_bss(uniform, bss))

    def test_row_schedule_payload_records_hashes_when_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.txt"
            mask = root / "mask.txt"
            source.write_text("source", encoding="utf-8")
            mask.write_text("mask", encoding="utf-8")
            row = {
                "method": "bss10",
                "case_id": "case001",
                "source_image_path": str(source),
                "mask_image_path": str(mask),
                "prompt": "replace object",
                "prompt_hash": "",
                "seed": "0",
                "height": "1024",
                "width": "1024",
                "guidance_scale": "30",
                "max_sequence_length": "512",
                "output_path": str(root / "out.png"),
            }
            payload = row_schedule_payload(row)
            self.assertEqual(payload["actual_nfe"], 10)
            self.assertTrue(payload["source_image_hash"])
            self.assertTrue(payload["mask_hash"])


if __name__ == "__main__":
    unittest.main()
