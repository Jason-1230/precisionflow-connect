from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .framework import build_framework_plan


def _join_command(parts: list[str]) -> str:
    return shlex.join([str(part) for part in parts if str(part) != ""])


def _strip_argument_separator(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def build_self_test_command(
    *,
    nproc_per_node: int = 2,
    backend: str = "gloo",
    report_dir: str = "reports/self-test",
    master_addr: str = "127.0.0.1",
    master_port: int = 29577,
    python_executable: str | None = None,
) -> list[str]:
    executable = python_executable or sys.executable
    report_root = Path(report_dir)
    return [
        executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={nproc_per_node}",
        "--node_rank=0",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        "-m",
        "precisionflow_lab",
        "connect",
        "--live",
        "--backend",
        backend,
        "--anonymize-hostnames",
        "--json-output",
        str(report_root / "connect.json"),
        "--markdown-output",
        str(report_root / "connect.md"),
    ]


def build_training_torchrun_command(
    *,
    nnodes: int,
    nproc_per_node: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
    training_script: str,
    training_args: list[str] | None = None,
) -> list[str]:
    return [
        "torchrun",
        f"--nnodes={nnodes}",
        f"--nproc_per_node={nproc_per_node}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        training_script,
        *_strip_argument_separator(list(training_args or [])),
    ]


def build_launch_plan(
    manifest_path: str | Path,
    *,
    node_rank: int,
    master_addr: str,
    master_port: int = 29500,
    backend: str = "auto",
    network_interface: str | None = None,
    report_dir: str = "reports",
    training_script: str | None = None,
    training_args: list[str] | None = None,
    image: str = "precisionflow-connect:latest",
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    framework = build_framework_plan(
        manifest_path,
        master_addr=master_addr,
        master_port=master_port,
        image=image,
        network_interface=network_interface,
        backend=backend,
        report_dir=report_dir,
        timeout_seconds=timeout_seconds,
    )
    if node_rank < 0 or node_rank >= framework["node_count"]:
        raise ValueError(f"node_rank must be between 0 and {framework['node_count'] - 1}")

    node = framework["nodes"][node_rank]
    env = {"NCCL_DEBUG": "INFO"}
    if node.get("cuda_visible_devices"):
        env["CUDA_VISIBLE_DEVICES"] = node["cuda_visible_devices"]
    if network_interface:
        env["NCCL_SOCKET_IFNAME"] = network_interface
        env["GLOO_SOCKET_IFNAME"] = network_interface

    training_command: list[str] | None = None
    if training_script:
        training_command = build_training_torchrun_command(
            nnodes=framework["node_count"],
            nproc_per_node=node["nproc_per_node"],
            node_rank=node_rank,
            master_addr=master_addr,
            master_port=master_port,
            training_script=training_script,
            training_args=training_args,
        )

    return {
        "job_name": framework["job_name"],
        "node_rank": node_rank,
        "machine": node["machine"],
        "rank_range": node["ranks"],
        "nproc_per_node": node["nproc_per_node"],
        "declared_devices": node["declared_devices"],
        "environment": env,
        "readiness_gate": {
            "description": "Run this command before the training script on every node.",
            "command": node["torchrun_args"],
        },
        "training": {
            "description": "Run this after the readiness gate passes on every node.",
            "command": training_command,
        },
        "post_run": [
            f"precisionflow-connect doctor {report_dir}/connect.json",
            "archive the JSON and Markdown reports with the training run",
        ],
    }


def render_launch_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# PrecisionFlow Launch Plan",
        "",
        f"- Job name: {plan['job_name']}",
        f"- Machine: {plan['machine']}",
        f"- node_rank: {plan['node_rank']}",
        f"- ranks: {plan['rank_range']}",
        f"- nproc_per_node: {plan['nproc_per_node']}",
        f"- declared devices: {plan['declared_devices']}",
        "",
        "## Environment",
        "",
    ]
    for key, value in plan["environment"].items():
        lines.append(f"- {key}={value}")
    lines.extend(
        [
            "",
            "## Readiness Gate",
            "",
            plan["readiness_gate"]["description"],
            "",
            "```bash",
            _join_command(plan["readiness_gate"]["command"]),
            "```",
            "",
            "## Training Handoff",
            "",
            plan["training"]["description"],
            "",
        ]
    )
    if plan["training"]["command"]:
        lines.extend(["```bash", _join_command(plan["training"]["command"]), "```", ""])
    else:
        lines.append("No training script was provided. Add --training-script to generate a handoff command.")
        lines.append("")

    lines.extend(["## Post-Run", ""])
    for item in plan["post_run"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def execute_launch_plan(plan: dict[str, Any]) -> int:
    env = os.environ.copy()
    env.update({key: str(value) for key, value in plan["environment"].items()})

    readiness = plan["readiness_gate"]["command"]
    readiness_code = subprocess.run(readiness, env=env, check=False).returncode
    if readiness_code != 0:
        return readiness_code

    training_command = plan["training"]["command"]
    if training_command:
        return subprocess.run(training_command, env=env, check=False).returncode
    return 0
