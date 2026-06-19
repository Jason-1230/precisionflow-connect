from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _project_src_path() -> str:
    return str(Path(__file__).resolve().parents[1])


def build_native_self_test_processes(
    *,
    nproc_per_node: int = 2,
    backend: str = "gloo",
    report_dir: str = "reports/self-test",
    master_addr: str = "127.0.0.1",
    master_port: int = 29577,
    python_executable: str | None = None,
) -> list[dict[str, Any]]:
    executable = python_executable or sys.executable
    report_root = Path(report_dir)
    processes: list[dict[str, Any]] = []
    for rank in range(nproc_per_node):
        command = [
            executable,
            "-m",
            "precisionflow_lab",
            "connect",
            "--live",
            "--backend",
            backend,
            "--anonymize-hostnames",
        ]
        if rank == 0:
            command.extend(
                [
                    "--json-output",
                    str(report_root / "connect.json"),
                    "--markdown-output",
                    str(report_root / "connect.md"),
                ]
            )
        env = {
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port),
            "WORLD_SIZE": str(nproc_per_node),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "LOCAL_WORLD_SIZE": str(nproc_per_node),
            "NODE_RANK": "0",
            "USE_LIBUV": "0",
        }
        processes.append({"rank": rank, "command": command, "environment": env})
    return processes


def run_native_self_test(
    *,
    nproc_per_node: int = 2,
    backend: str = "gloo",
    report_dir: str = "reports/self-test",
    master_addr: str = "127.0.0.1",
    master_port: int = 29577,
    timeout_seconds: int = 120,
) -> int:
    report_root = Path(report_dir)
    report_root.mkdir(parents=True, exist_ok=True)
    specs = build_native_self_test_processes(
        nproc_per_node=nproc_per_node,
        backend=backend,
        report_dir=report_dir,
        master_addr=master_addr,
        master_port=master_port,
    )

    children: list[tuple[int, subprocess.Popen[str]]] = []
    for spec in specs:
        env = os.environ.copy()
        env.update({key: str(value) for key, value in spec["environment"].items()})
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            _project_src_path()
            if not existing_pythonpath
            else os.pathsep.join([_project_src_path(), existing_pythonpath])
        )
        children.append(
            (
                int(spec["rank"]),
                subprocess.Popen(
                    spec["command"],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ),
            )
        )

    exit_code = 0
    for rank, child in children:
        try:
            stdout, stderr = child.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            child.terminate()
            stdout, stderr = child.communicate(timeout=10)
            print(f"[rank {rank}] timed out after {timeout_seconds} seconds", file=sys.stderr)
            exit_code = 1
        if stdout:
            print(f"[rank {rank} stdout]\n{stdout.rstrip()}")
        if stderr:
            print(f"[rank {rank} stderr]\n{stderr.rstrip()}", file=sys.stderr)
        if child.returncode:
            exit_code = child.returncode if exit_code == 0 else exit_code
    return exit_code


def render_native_self_test_plan(processes: list[dict[str, Any]]) -> str:
    lines = ["# PrecisionFlow Native Self-Test Plan", ""]
    for spec in processes:
        lines.extend(
            [
                f"## Rank {spec['rank']}",
                "",
                "Environment:",
                "",
            ]
        )
        for key, value in spec["environment"].items():
            lines.append(f"- {key}={value}")
        lines.extend(["", "Command:", "", "```bash", " ".join(spec["command"]), "```", ""])
    return "\n".join(lines)
