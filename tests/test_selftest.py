import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precisionflow_lab.selftest import build_native_self_test_processes, render_native_self_test_plan


class SelfTestTests(unittest.TestCase):
    def test_native_self_test_processes_set_rank_environment(self):
        processes = build_native_self_test_processes(nproc_per_node=2, python_executable="python")

        self.assertEqual(len(processes), 2)
        self.assertEqual(processes[0]["environment"]["RANK"], "0")
        self.assertEqual(processes[1]["environment"]["LOCAL_RANK"], "1")
        self.assertEqual(processes[0]["environment"]["WORLD_SIZE"], "2")
        self.assertIn("precisionflow_lab", processes[0]["command"])

    def test_native_self_test_plan_renders_commands(self):
        plan = render_native_self_test_plan(build_native_self_test_processes(nproc_per_node=1, python_executable="python"))

        self.assertIn("PrecisionFlow Native Self-Test Plan", plan)
        self.assertIn("MASTER_ADDR=127.0.0.1", plan)
        self.assertIn("python -m precisionflow_lab connect", plan)


if __name__ == "__main__":
    unittest.main()
