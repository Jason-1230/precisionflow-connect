import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precisionflow_lab.cli import main


MANIFEST = Path(__file__).resolve().parents[1] / "configs" / "multinode_2x4.json"


class CliTests(unittest.TestCase):
    def test_connect_json_preflight_can_anonymize_hosts(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(["connect", "--manifest", str(MANIFEST), "--json", "--anonymize-hostnames"])

        self.assertEqual(code, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["tool"], "PrecisionFlow Connect")
        self.assertEqual(report["network"]["hostname"], "host-local")
        self.assertNotIn("DESKTOP", json.dumps(report))

    def test_connect_writes_report_outputs(self):
        with tempfile.TemporaryDirectory() as tempdir:
            json_path = Path(tempdir) / "connect.json"
            markdown_path = Path(tempdir) / "connect.md"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(
                    [
                        "connect",
                        "--manifest",
                        str(MANIFEST),
                        "--json-output",
                        str(json_path),
                        "--markdown-output",
                        str(markdown_path),
                        "--anonymize-hostnames",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertTrue(json_path.exists())
            self.assertTrue(markdown_path.exists())
            self.assertIn("PrecisionFlow Connect System Report", markdown_path.read_text(encoding="utf-8"))

    def test_inspect_alias_prints_manifest_summary(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(["inspect", str(MANIFEST), "--tensor-elements", "100"])

        self.assertEqual(code, 0)
        self.assertIn("world_size: 8", stdout.getvalue())
        self.assertIn("initialize NCCL or Gloo", stdout.getvalue())

    def test_version_flag(self):
        stdout = io.StringIO()
        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stdout(stdout):
            main(["--version"])

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("precisionflow-connect 0.3.0", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
