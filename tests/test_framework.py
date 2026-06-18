import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precisionflow_lab.cli import main
from precisionflow_lab.framework import build_framework_plan, render_framework_markdown


MANIFEST = Path(__file__).resolve().parents[1] / "configs" / "multinode_2x4.json"


class FrameworkTests(unittest.TestCase):
    def test_framework_plan_derives_node_launch_commands(self):
        plan = build_framework_plan(
            MANIFEST,
            master_addr="192.0.2.10",
            image="precisionflow-connect:gpu",
            network_interface="ib0",
        )

        self.assertEqual(plan["node_count"], 2)
        self.assertEqual(plan["world_size"], 8)
        self.assertEqual(plan["nodes"][0]["node_rank"], 0)
        self.assertEqual(plan["nodes"][0]["nproc_per_node"], 4)
        self.assertIn("--nnodes=2", plan["nodes"][0]["torchrun_command"])
        self.assertIn("--node_rank=1", plan["nodes"][1]["torchrun_command"])
        self.assertIn("docker run", plan["nodes"][0]["docker_command"])
        self.assertIn("--network host", plan["nodes"][0]["docker_command"])
        self.assertIn("NCCL_SOCKET_IFNAME=ib0", plan["nodes"][0]["docker_command"])
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1,2,3", plan["nodes"][0]["docker_command"])

    def test_framework_markdown_explains_layers(self):
        plan = build_framework_plan(MANIFEST, network_interface="eth0")
        markdown = render_framework_markdown(plan)

        self.assertIn("PrecisionFlow Connect Framework Plan", markdown)
        self.assertIn("container layer", markdown)
        self.assertIn("launcher layer", markdown)
        self.assertIn("Dockerized launch", markdown)
        self.assertIn("Post-Run Closure", markdown)

    def test_framework_cli_json(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(["framework", str(MANIFEST), "--network-interface", "ib0", "--json"])

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["framework"], "PrecisionFlow Connect")
        self.assertEqual(payload["nodes"][0]["node_rank"], 0)
        self.assertIn("docker run", payload["nodes"][0]["docker_command"])

    def test_framework_cli_cpu_only_omits_gpu_flag(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(["framework", str(MANIFEST), "--cpu-only"])

        self.assertEqual(code, 0)
        self.assertNotIn("--gpus all", stdout.getvalue())
        self.assertIn("Dockerized launch", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
