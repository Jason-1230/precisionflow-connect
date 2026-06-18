import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precisionflow_lab.diagnose import diagnose_report, load_report, render_diagnosis_markdown


class DiagnoseTests(unittest.TestCase):
    def test_failure_report_produces_actionable_findings(self):
        report_path = Path(__file__).resolve().parents[1] / "examples" / "failure_report.json"
        report = load_report(report_path)
        findings = diagnose_report(report)
        codes = {item.code for item in findings}

        self.assertIn("WORLD_SIZE_MISMATCH", codes)
        self.assertIn("BACKEND_INIT_FAILED", codes)
        self.assertIn("CUDA_DEVICE_NOT_VISIBLE", codes)
        self.assertIn("NO_SOCKET_IFNAME_PINNED", codes)
        self.assertTrue(any(item.severity == "ERROR" for item in findings))

    def test_diagnosis_markdown_has_table(self):
        report_path = Path(__file__).resolve().parents[1] / "examples" / "failure_report.json"
        markdown = render_diagnosis_markdown(load_report(report_path))

        self.assertIn("# PrecisionFlow Connect Diagnosis", markdown)
        self.assertIn("| severity | code | area | evidence | recommendation |", markdown)
        self.assertIn("WORLD_SIZE_MISMATCH", markdown)


if __name__ == "__main__":
    unittest.main()
