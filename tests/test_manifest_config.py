from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "bss_experiments" / "flux_fill_bss_bds_v1" / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

from common import MINI_METHODS, SMOKE_METHODS  # noqa: E402
from make_flux_fill_assets_and_manifest import CASE_DEFS  # noqa: E402


class ManifestConfigTest(unittest.TestCase):
    def test_expanded_suite_is_8_cases_x_10_methods(self) -> None:
        case_ids = [case["case_id"] for case in CASE_DEFS]
        self.assertEqual(len(case_ids), 8)
        self.assertEqual(len(set(case_ids)), 8)
        self.assertEqual(len(MINI_METHODS), 10)
        self.assertIn("uniform30", MINI_METHODS)
        self.assertIn("bss30", MINI_METHODS)
        self.assertEqual(len(CASE_DEFS) * len(MINI_METHODS), 80)

    def test_smoke_remains_single_case_quick_probe(self) -> None:
        self.assertEqual(SMOKE_METHODS, ["uniform8", "uniform10", "bss10", "reference_uniform50"])


if __name__ == "__main__":
    unittest.main()
