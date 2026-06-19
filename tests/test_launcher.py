import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precisionflow_lab.launcher import build_launch_plan, build_self_test_command, render_launch_markdown


MANIFEST = Path(__file__).resolve().parents[1] / "configs" / "multinode_2x4.json"


class LauncherTests(unittest.TestCase):
    def test_launch_plan_contains_readiness_and_training_commands(self):
        plan = build_launch_plan(
            MANIFEST,
            node_rank=1,
            master_addr="192.0.2.10",
            network_interface="ib0",
            training_script="examples/minimal_ddp_train.py",
            training_args=["--", "--epochs", "2"],
        )

        self.assertEqual(plan["node_rank"], 1)
        self.assertEqual(plan["environment"]["NCCL_SOCKET_IFNAME"], "ib0")
        self.assertIn("connect", plan["readiness_gate"]["command"])
        self.assertIn("examples/minimal_ddp_train.py", plan["training"]["command"])
        self.assertIn("--epochs", plan["training"]["command"])

    def test_render_launch_markdown(self):
        plan = build_launch_plan(
            MANIFEST,
            node_rank=0,
            master_addr="192.0.2.10",
            training_script="examples/minimal_ddp_train.py",
        )
        markdown = render_launch_markdown(plan)

        self.assertIn("Readiness Gate", markdown)
        self.assertIn("Training Handoff", markdown)

    def test_self_test_command_uses_torch_distributed_run(self):
        command = build_self_test_command(nproc_per_node=2, backend="gloo", python_executable="python")

        self.assertEqual(command[:3], ["python", "-m", "torch.distributed.run"])
        self.assertIn("--nnodes=1", command)
        self.assertIn("--node_rank=0", command)
        self.assertIn("--master_addr=127.0.0.1", command)
        self.assertIn("connect", command)


if __name__ == "__main__":
    unittest.main()
