import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precisionflow_lab.report import render_markdown_report
from precisionflow_lab.runtime import (
    build_preflight_report,
    precision_capability_from_compute_capability,
    torchrun_context,
)


class RuntimeTests(unittest.TestCase):
    def test_torchrun_context_parses_rank_values(self):
        context = torchrun_context(
            {
                "MASTER_ADDR": "192.0.2.10",
                "MASTER_PORT": "29500",
                "RANK": "3",
                "LOCAL_RANK": "1",
                "WORLD_SIZE": "8",
                "LOCAL_WORLD_SIZE": "4",
                "NODE_RANK": "0",
            }
        )

        self.assertEqual(context["rank"], 3)
        self.assertEqual(context["local_rank"], 1)
        self.assertEqual(context["world_size"], 8)
        self.assertEqual(context["master_port"], 29500)

    def test_precision_capability_matrix_tracks_modern_gpu_features(self):
        ampere = precision_capability_from_compute_capability(8, 0)
        hopper = precision_capability_from_compute_capability(9, 0)

        self.assertTrue(ampere["tf32"])
        self.assertTrue(ampere["bf16"])
        self.assertFalse(ampere["fp8"])
        self.assertTrue(hopper["fp8"])
        self.assertTrue(hopper["int8"])

    def test_preflight_report_renders_required_sections(self):
        manifest = Path(__file__).resolve().parents[1] / "configs" / "multinode_2x4.json"
        report = build_preflight_report(
            manifest_path=manifest,
            torch_module=object(),
            environ={
                "MASTER_ADDR": "192.0.2.10",
                "MASTER_PORT": "29500",
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "8",
                "LOCAL_WORLD_SIZE": "4",
                "NODE_RANK": "0",
            },
        )
        markdown = render_markdown_report(report)

        self.assertEqual(report["cluster_topology"]["world_size"], 8)
        self.assertIn("## Cluster Topology", markdown)
        self.assertIn("## Backend Status", markdown)
        self.assertIn("## Network Interface Status", markdown)
        self.assertIn("## Precision Capability Table", markdown)
        self.assertIn("## Collective Communication Test Result", markdown)
        self.assertIn("fp8", markdown)

    def test_anonymize_hosts_redacts_manifest_machine_names(self):
        manifest = Path(__file__).resolve().parents[1] / "configs" / "multinode_2x4.json"
        report = build_preflight_report(
            manifest_path=manifest,
            torch_module=object(),
            anonymize_hosts=True,
        )
        rendered = str(report["cluster_topology"])

        self.assertIn("node-0", rendered)
        self.assertIn("node-1", rendered)
        self.assertNotIn("node-a", rendered)
        self.assertNotIn("node-b", rendered)


if __name__ == "__main__":
    unittest.main()
