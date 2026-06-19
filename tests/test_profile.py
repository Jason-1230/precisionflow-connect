import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precisionflow_lab.cli import main
from precisionflow_lab.profile import build_capability_profile, render_profile_markdown


class ProfileTests(unittest.TestCase):
    def test_profile_includes_deployment_and_readiness_sections(self):
        profile = build_capability_profile()

        targets = {row["target"] for row in profile["deployment_targets"]}
        check_areas = {row["area"] for row in profile["readiness_checks"]}
        frameworks = {row["framework"] for row in profile["framework_handoff"]}

        self.assertIn("Docker runtime", targets)
        self.assertIn("bare-metal torchrun", targets)
        self.assertIn("scheduler handoff", targets)
        self.assertIn("network", check_areas)
        self.assertIn("collectives", check_areas)
        self.assertIn("precision", check_areas)
        self.assertIn("Hugging Face Accelerate", frameworks)
        self.assertIn("DeepSpeed", frameworks)

    def test_profile_markdown_renders_reference_style_tables(self):
        markdown = render_profile_markdown()

        self.assertIn("Capability Profile", markdown)
        self.assertIn("Deployment Targets", markdown)
        self.assertIn("Readiness Checks", markdown)
        self.assertIn("Framework Handoff", markdown)
        self.assertIn("Failure Lifecycle", markdown)

    def test_profile_cli_json(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(["profile", "--json"])

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["name"], "PrecisionFlow Connect")
        self.assertGreaterEqual(len(payload["readiness_checks"]), 6)


if __name__ == "__main__":
    unittest.main()
