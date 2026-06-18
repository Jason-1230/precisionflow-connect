from __future__ import annotations

from pathlib import Path
from typing import Any

from .planner import build_plan, load_manifest, validate_manifest


def _device_ids(devices: list[str]) -> str | None:
    ids: list[str] = []
    for device in devices:
        prefix = "cuda:"
        if device.startswith(prefix):
            ids.append(device[len(prefix) :])
    return ",".join(ids) if ids else None


def _join_command(parts: list[str]) -> str:
    return " ".join(str(part) for part in parts if str(part) != "")


def _torchrun_parts(
    *,
    nnodes: int,
    nproc_per_node: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
    backend: str,
    manifest_inside_container: str,
    report_dir: str,
    timeout_seconds: int,
) -> list[str]:
    return [
        "torchrun",
        f"--nnodes={nnodes}",
        f"--nproc_per_node={nproc_per_node}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        "-m",
        "precisionflow_lab",
        "connect",
        "--live",
        "--backend",
        backend,
        "--manifest",
        manifest_inside_container,
        "--anonymize-hostnames",
        "--timeout-seconds",
        str(timeout_seconds),
        "--json-output",
        f"{report_dir}/connect.json",
        "--markdown-output",
        f"{report_dir}/connect.md",
    ]


def _docker_parts(
    *,
    image: str,
    container_workdir: str,
    network_interface: str | None,
    visible_devices: str | None,
    torchrun_parts: list[str],
    use_gpus: bool,
) -> list[str]:
    parts = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "--ipc",
        "host",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
    ]
    if use_gpus:
        parts.extend(["--gpus", "all"])
    parts.extend(["-v", "$PWD:/workspace", "-w", container_workdir])

    env_rows = {"NCCL_DEBUG": "INFO"}
    if network_interface:
        env_rows["NCCL_SOCKET_IFNAME"] = network_interface
        env_rows["GLOO_SOCKET_IFNAME"] = network_interface
    if visible_devices:
        env_rows["CUDA_VISIBLE_DEVICES"] = visible_devices

    for key, value in env_rows.items():
        parts.extend(["-e", f"{key}={value}"])

    parts.append(image)
    parts.extend(torchrun_parts)
    return parts


def build_framework_plan(
    manifest_path: str | Path,
    *,
    master_addr: str = "192.0.2.10",
    master_port: int = 29500,
    image: str = "precisionflow-connect:latest",
    network_interface: str | None = None,
    backend: str = "auto",
    container_workdir: str = "/workspace",
    manifest_inside_container: str | None = None,
    report_dir: str = "reports",
    timeout_seconds: int = 120,
    use_gpus: bool = True,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = load_manifest(manifest_file)
    plan = build_plan(manifest)
    ranks = validate_manifest(manifest)
    manifest_in_container = manifest_inside_container or f"configs/{manifest_file.name}"
    nnodes = len(plan.machine_groups)

    nodes: list[dict[str, Any]] = []
    for node_rank, group in enumerate(plan.machine_groups):
        node_ranks = [rank for rank in ranks if rank.rank in group.ranks]
        devices = [rank.device for rank in node_ranks]
        visible_devices = _device_ids(devices)
        torchrun = _torchrun_parts(
            nnodes=nnodes,
            nproc_per_node=len(group.ranks),
            node_rank=node_rank,
            master_addr=master_addr,
            master_port=master_port,
            backend=backend,
            manifest_inside_container=manifest_in_container,
            report_dir=report_dir,
            timeout_seconds=timeout_seconds,
        )
        docker = _docker_parts(
            image=image,
            container_workdir=container_workdir,
            network_interface=network_interface,
            visible_devices=visible_devices,
            torchrun_parts=torchrun,
            use_gpus=use_gpus,
        )
        nodes.append(
            {
                "machine": group.machine,
                "node_rank": node_rank,
                "ranks": list(group.ranks),
                "nproc_per_node": len(group.ranks),
                "declared_devices": devices,
                "cuda_visible_devices": visible_devices,
                "torchrun_command": _join_command(torchrun),
                "docker_command": _join_command(docker),
            }
        )

    return {
        "framework": "PrecisionFlow Connect",
        "job_name": plan.job_name,
        "world_size": plan.world_size,
        "node_count": nnodes,
        "image": image,
        "master_addr": master_addr,
        "master_port": master_port,
        "backend": backend,
        "network_interface": network_interface,
        "manifest": str(manifest_path),
        "manifest_inside_container": manifest_in_container,
        "layers": [
            {
                "name": "container layer",
                "purpose": "build a reproducible Python/PyTorch/NCCL runtime image for every node",
            },
            {
                "name": "launcher layer",
                "purpose": "derive node_rank, nproc_per_node, rank ranges, device visibility, and torchrun commands",
            },
            {
                "name": "connectivity validation layer",
                "purpose": "initialize NCCL/Gloo, run barrier/all_reduce/all_gather, and collect rank reports",
            },
            {
                "name": "reporting layer",
                "purpose": "emit JSON/Markdown reports and feed failed reports into the doctor command",
            },
        ],
        "nodes": nodes,
        "post_run": [
            f"precisionflow-connect doctor {report_dir}/connect.json",
            "archive reports/connect.md as the reproducibility evidence for the bring-up run",
        ],
    }


def render_framework_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# PrecisionFlow Connect Framework Plan",
        "",
        f"- Job name: {plan['job_name']}",
        f"- World size: {plan['world_size']}",
        f"- Node count: {plan['node_count']}",
        f"- Container image: {plan['image']}",
        f"- Backend: {plan['backend']}",
        f"- Master endpoint: {plan['master_addr']}:{plan['master_port']}",
        f"- Network interface: {plan['network_interface'] or 'auto'}",
        "",
        "## Framework Layers",
        "",
    ]
    for layer in plan["layers"]:
        lines.append(f"- {layer['name']}: {layer['purpose']}")

    lines.extend(["", "## Per-Node Launch Commands", ""])
    for node in plan["nodes"]:
        lines.extend(
            [
                f"### {node['machine']}",
                "",
                f"- node_rank: {node['node_rank']}",
                f"- ranks: {node['ranks']}",
                f"- nproc_per_node: {node['nproc_per_node']}",
                f"- declared devices: {node['declared_devices']}",
                f"- CUDA_VISIBLE_DEVICES: {node['cuda_visible_devices'] or 'not pinned'}",
                "",
                "Bare-metal torchrun:",
                "",
                "```bash",
                node["torchrun_command"],
                "```",
                "",
                "Dockerized launch:",
                "",
                "```bash",
                node["docker_command"],
                "```",
                "",
            ]
        )

    lines.extend(["## Post-Run Closure", ""])
    for item in plan["post_run"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)
