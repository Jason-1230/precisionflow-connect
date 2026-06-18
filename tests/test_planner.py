import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precisionflow_lab.planner import ManifestError, build_plan


class PlannerTests(unittest.TestCase):
    def test_builds_precision_streams_and_machine_groups(self):
        manifest = {
            "job_name": "test",
            "world_size": 5,
            "ranks": [
                {"rank": 0, "machine": "a", "device": "cuda:0", "precision": "fp32"},
                {"rank": 1, "machine": "a", "device": "cuda:1", "precision": "bf16"},
                {"rank": 2, "machine": "b", "device": "cuda:0", "precision": "fp16"},
                {"rank": 3, "machine": "b", "device": "cuda:1", "precision": "int8"},
                {"rank": 4, "machine": "b", "device": "cuda:2", "precision": "fp8"},
            ],
        }

        plan = build_plan(manifest)

        self.assertEqual(plan.world_size, 5)
        self.assertEqual(
            {stream.precision for stream in plan.precision_streams},
            {"fp32", "bf16", "fp16", "int8", "fp8"},
        )
        self.assertEqual({group.machine for group in plan.machine_groups}, {"a", "b"})
        self.assertIn("initialize NCCL or Gloo", " ".join(plan.stages))

    def test_rejects_non_contiguous_ranks(self):
        manifest = {
            "world_size": 2,
            "ranks": [
                {"rank": 0, "machine": "a", "precision": "fp32"},
                {"rank": 2, "machine": "b", "precision": "fp16"},
            ],
        }

        with self.assertRaises(ManifestError):
            build_plan(manifest)

    def test_rejects_unsupported_precision(self):
        manifest = {
            "world_size": 1,
            "ranks": [
                {"rank": 0, "machine": "a", "precision": "int4"},
            ],
        }

        with self.assertRaises(ManifestError):
            build_plan(manifest)


if __name__ == "__main__":
    unittest.main()
