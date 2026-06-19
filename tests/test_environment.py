import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precisionflow_lab.environment import build_environment_inventory, render_environment_markdown


class _Props:
    name = "Synthetic GPU"
    total_memory = 40 * 1024**3


class _Cuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 1

    @staticmethod
    def current_device():
        return 0

    @staticmethod
    def get_device_properties(index):
        return _Props()

    @staticmethod
    def get_device_capability(index):
        return (8, 0)


class _Dist:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def is_nccl_available():
        return True

    @staticmethod
    def is_gloo_available():
        return True


class _Version:
    cuda = "12.1"


class _Torch:
    __version__ = "2.4.0"
    version = _Version()
    distributed = _Dist()
    cuda = _Cuda()


class EnvironmentTests(unittest.TestCase):
    def test_environment_inventory_records_backends_and_precision(self):
        inventory = build_environment_inventory(
            torch_module=_Torch(),
            environ={
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29500",
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
            },
            anonymize_hosts=True,
        )

        self.assertTrue(inventory["torch"]["installed"])
        self.assertTrue(inventory["distributed_backends"]["nccl_available"])
        self.assertTrue(inventory["precision_capability_matrix"][0]["bf16"])

    def test_environment_markdown_contains_sections(self):
        inventory = build_environment_inventory(torch_module=_Torch(), environ={}, anonymize_hosts=True)
        markdown = render_environment_markdown(inventory)

        self.assertIn("# PrecisionFlow Connect Environment", markdown)
        self.assertIn("## Distributed Backends", markdown)
        self.assertIn("## Precision Readiness Matrix", markdown)


if __name__ == "__main__":
    unittest.main()
