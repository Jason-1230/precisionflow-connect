import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precisionflow_lab.configure import ConfigureError, build_manifest_from_nodes, parse_node_spec, write_manifest


class ConfigureTests(unittest.TestCase):
    def test_parse_node_spec_with_precision(self):
        node = parse_node_spec("node-a=cuda:0,cuda:1@bf16")

        self.assertEqual(node["machine"], "node-a")
        self.assertEqual(node["devices"], ["cuda:0", "cuda:1"])
        self.assertEqual(node["precision"], "bf16")

    def test_build_manifest_from_nodes_assigns_contiguous_ranks(self):
        manifest = build_manifest_from_nodes(
            [
                "node-a=cuda:0,cuda:1@bf16",
                "node-b=cuda:0@fp16",
            ],
            job_name="demo",
        )

        self.assertEqual(manifest["world_size"], 3)
        self.assertEqual([row["rank"] for row in manifest["ranks"]], [0, 1, 2])
        self.assertEqual(manifest["ranks"][2]["machine"], "node-b")

    def test_unsupported_precision_is_rejected(self):
        with self.assertRaises(ConfigureError):
            build_manifest_from_nodes(["node-a=cuda:0@fp64"])

    def test_write_manifest_creates_parent_directory(self):
        manifest = build_manifest_from_nodes(["node-a=cuda:0@fp32"])
        with tempfile.TemporaryDirectory() as tempdir:
            target = Path(tempdir) / "nested" / "cluster.json"
            write_manifest(manifest, target)

            self.assertTrue(target.exists())


if __name__ == "__main__":
    unittest.main()
