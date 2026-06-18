from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .runtime import PRECISION_COLUMNS


def write_json_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _yes_no(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "n/a"


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def render_markdown_report(report: dict[str, Any]) -> str:
    topology = report.get("cluster_topology", {})
    runtime = report.get("rank_runtime", {})
    network = report.get("network", {})

    lines: list[str] = [
        "# PrecisionFlow Connect System Report",
        "",
        f"- Tool: {report.get('tool', 'PrecisionFlow Connect')}",
        f"- Mode: {report.get('mode', 'unknown')}",
        f"- Created at UTC: {report.get('created_at_utc', 'unknown')}",
        f"- Overall status: {report.get('overall_status', 'unknown')}",
        "",
        "## Cluster Topology",
        "",
    ]

    if topology.get("status") == "PASS":
        lines.extend(
            [
                f"- Job name: {topology.get('job_name')}",
                f"- Declared world size: {topology.get('world_size')}",
                f"- Manifest: {topology.get('manifest')}",
                "",
            ]
        )
    else:
        lines.extend([f"- Status: {topology.get('status')}", f"- Detail: {topology.get('detail')}", ""])

    rank_rows = [
        [row.get("rank"), row.get("machine"), row.get("device"), row.get("precision")]
        for row in topology.get("rank_mapping", [])
    ]
    if rank_rows:
        lines.extend(_table(["rank", "machine", "declared device", "declared precision"], rank_rows))
        lines.append("")

    observed_rows = [
        [
            row.get("rank"),
            row.get("hostname"),
            row.get("local_rank"),
            row.get("device"),
            row.get("backend"),
        ]
        for row in report.get("rank_mapping_observed", [])
    ]
    if observed_rows:
        lines.extend(["## Rank-To-Machine/Device Mapping", ""])
        lines.extend(_table(["rank", "host", "local rank", "runtime device", "backend"], observed_rows))
        lines.append("")
    else:
        lines.extend(
            [
                "## Rank-To-Machine/Device Mapping",
                "",
                f"- Runtime rank: {runtime.get('rank')}",
                f"- Runtime local_rank: {runtime.get('local_rank')}",
                f"- Runtime world_size: {runtime.get('world_size')}",
                f"- Runtime device: {runtime.get('device', 'not assigned in preflight')}",
                "",
            ]
        )

    lines.extend(["## Backend Status", ""])
    lines.extend(
        _table(
            ["backend", "status", "detail"],
            [[row.get("backend"), row.get("status"), row.get("detail")] for row in report.get("backend_status", [])],
        )
    )
    lines.append("")

    endpoint = network.get("master_endpoint", {})
    selected = network.get("selected_interfaces", {})
    interface_names = [item.get("name") for item in network.get("interfaces", [])]
    lines.extend(
        [
            "## Network Interface Status",
            "",
            f"- Hostname: {network.get('hostname')}",
            f"- Host addresses: {', '.join(network.get('host_addresses', [])) or 'n/a'}",
            f"- Interfaces: {', '.join(interface_names) or 'n/a'}",
            f"- NCCL_SOCKET_IFNAME: {selected.get('nccl_socket_ifname') or 'not set'}",
            f"- GLOO_SOCKET_IFNAME: {selected.get('gloo_socket_ifname') or 'not set'}",
            f"- Master endpoint: {endpoint.get('addr')}:{endpoint.get('port')} ({endpoint.get('status')})",
            f"- Master detail: {endpoint.get('detail')}",
            "",
            "## Environment Variable Status",
            "",
        ]
    )
    lines.extend(
        _table(
            ["name", "required", "present", "status", "value"],
            [
                [row.get("name"), row.get("required"), row.get("present"), row.get("status"), row.get("value")]
                for row in report.get("environment", [])
            ],
        )
    )
    lines.append("")

    lines.extend(["## Precision Capability Table", ""])
    precision_rows = []
    for row in report.get("precision_capability_matrix", []):
        precision_rows.append(
            [
                row.get("host"),
                row.get("device"),
                row.get("device_name"),
                row.get("compute_capability"),
                row.get("memory_gib"),
                *[_yes_no(row.get(column)) for column in PRECISION_COLUMNS],
                row.get("warning") or "",
            ]
        )
    lines.extend(
        _table(
            ["host", "device", "name", "cc", "GiB", *PRECISION_COLUMNS, "warning"],
            precision_rows,
        )
    )
    lines.append("")

    lines.extend(["## Collective Communication Test Result", ""])
    lines.extend(
        _table(
            ["test", "status", "detail"],
            [[row.get("name"), row.get("status"), row.get("detail")] for row in report.get("collective_tests", [])],
        )
    )
    lines.append("")

    lines.extend(["## Failure Diagnosis Or Warning", ""])
    for item in report.get("failure_diagnosis_or_warning", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_markdown_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_report(report), encoding="utf-8")
