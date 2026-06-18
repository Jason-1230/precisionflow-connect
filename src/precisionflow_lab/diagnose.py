from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    area: str
    evidence: str
    recommendation: str


def load_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    with report_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _missing_required_env(report: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    for row in report.get("environment", []):
        if row.get("required") and not row.get("present"):
            findings.append(
                Finding(
                    severity="ERROR",
                    code="ENV_REQUIRED_MISSING",
                    area="torchrun environment",
                    evidence=f"{row.get('name')} is required but absent",
                    recommendation="Launch with torchrun or set the complete MASTER_ADDR, MASTER_PORT, RANK, LOCAL_RANK, and WORLD_SIZE environment.",
                )
            )
        elif row.get("required") and row.get("status") == "FAIL":
            detail = row.get("detail") or "invalid value"
            findings.append(
                Finding(
                    severity="ERROR",
                    code="ENV_REQUIRED_INVALID",
                    area="torchrun environment",
                    evidence=f"{row.get('name')}={row.get('value')} ({detail})",
                    recommendation="Use integer values for rank, local rank, world size, and master port. Regenerate the launch command if values differ across nodes.",
                )
            )
    return findings


def _network_findings(report: dict[str, Any]) -> list[Finding]:
    network = report.get("network", {})
    endpoint = network.get("master_endpoint", {})
    findings: list[Finding] = []

    if endpoint.get("status") in {"WARN", "FAIL"}:
        findings.append(
            Finding(
                severity="WARN",
                code="MASTER_ENDPOINT_UNVERIFIED",
                area="network",
                evidence=str(endpoint.get("detail", "master endpoint was not verified")),
                recommendation="Verify that every node uses the same MASTER_ADDR and MASTER_PORT, and that the port is reachable from non-zero ranks.",
            )
        )

    selected = network.get("selected_interfaces", {})
    if not selected.get("nccl_socket_ifname") and not selected.get("gloo_socket_ifname"):
        findings.append(
            Finding(
                severity="INFO",
                code="NO_SOCKET_IFNAME_PINNED",
                area="network",
                evidence="NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME are not set",
                recommendation="Pin the training network interface when the host has multiple NICs, VPN adapters, or container bridges.",
            )
        )
    return findings


def _backend_findings(report: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    for row in report.get("backend_status", []):
        status = row.get("status")
        if status == "FAIL":
            findings.append(
                Finding(
                    severity="ERROR",
                    code="BACKEND_INIT_FAILED",
                    area="backend",
                    evidence=f"{row.get('backend')}: {row.get('detail')}",
                    recommendation="Check PyTorch distributed build support, CUDA visibility for NCCL, matching backend choice across ranks, and network binding.",
                )
            )
        elif status == "NOT_RUN":
            findings.append(
                Finding(
                    severity="INFO",
                    code="BACKEND_NOT_RUN",
                    area="backend",
                    evidence=f"{row.get('backend')}: {row.get('detail')}",
                    recommendation="Run with torchrun and --live to prove actual NCCL/Gloo process-group initialization.",
                )
            )
    return findings


def _collective_findings(report: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    for row in report.get("collective_tests", []):
        status = row.get("status")
        if status == "FAIL":
            findings.append(
                Finding(
                    severity="ERROR",
                    code=f"COLLECTIVE_{str(row.get('name', 'unknown')).upper()}_FAILED",
                    area="collective communication",
                    evidence=str(row.get("detail")),
                    recommendation="Check rank consistency, backend health, GPU visibility, tensor device placement, and whether all ranks enter the same collective sequence.",
                )
            )
        elif status == "NOT_RUN":
            findings.append(
                Finding(
                    severity="INFO",
                    code=f"COLLECTIVE_{str(row.get('name', 'unknown')).upper()}_NOT_RUN",
                    area="collective communication",
                    evidence=str(row.get("detail")),
                    recommendation="Use --live to run barrier, all-reduce, and all-gather smoke tests across the full world size.",
                )
            )
    return findings


def _precision_findings(report: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    rows = report.get("precision_capability_matrix", [])
    if not rows:
        findings.append(
            Finding(
                severity="WARN",
                code="PRECISION_MATRIX_MISSING",
                area="precision capability",
                evidence="no precision capability rows were reported",
                recommendation="Run the probe on every training node with PyTorch installed and visible CUDA devices.",
            )
        )
        return findings

    cpu_only = all(str(row.get("device")) == "cpu" for row in rows)
    if cpu_only:
        findings.append(
            Finding(
                severity="WARN",
                code="CUDA_DEVICE_NOT_VISIBLE",
                area="precision capability",
                evidence="only CPU capability row was reported",
                recommendation="Check CUDA_VISIBLE_DEVICES, installed CUDA runtime, NVIDIA driver, and whether the process is running on a GPU node.",
            )
        )

    fp8_hosts = [row.get("host") for row in rows if row.get("fp8")]
    if rows and not fp8_hosts:
        findings.append(
            Finding(
                severity="INFO",
                code="FP8_NOT_AVAILABLE",
                area="precision capability",
                evidence="no visible device reports fp8 support",
                recommendation="Treat fp8 paths as optional and gate them by device capability; mixed A100/H100 clusters should report this heterogeneity explicitly.",
            )
        )
    return findings


def _manifest_findings(report: dict[str, Any]) -> list[Finding]:
    topology = report.get("cluster_topology", {})
    warnings = report.get("failure_diagnosis_or_warning", [])
    findings: list[Finding] = []
    if topology.get("status") not in {"PASS", None}:
        findings.append(
            Finding(
                severity="ERROR",
                code="MANIFEST_INVALID",
                area="manifest",
                evidence=str(topology.get("detail", topology.get("status"))),
                recommendation="Fix world size, contiguous ranks, machine names, device declarations, and supported precision labels before launching.",
            )
        )
    for warning in warnings:
        if "world_size" in str(warning):
            findings.append(
                Finding(
                    severity="ERROR",
                    code="WORLD_SIZE_MISMATCH",
                    area="rank mapping",
                    evidence=str(warning),
                    recommendation="Make torchrun --nnodes * --nproc_per_node equal the manifest world_size, or update the manifest to match the real launch.",
                )
            )
    return findings


def diagnose_report(report: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    findings.extend(_manifest_findings(report))
    findings.extend(_missing_required_env(report))
    findings.extend(_network_findings(report))
    findings.extend(_backend_findings(report))
    findings.extend(_collective_findings(report))
    findings.extend(_precision_findings(report))

    if not findings and report.get("overall_status") == "PASS":
        findings.append(
            Finding(
                severity="PASS",
                code="CONNECTIVITY_HEALTHY",
                area="summary",
                evidence="backend initialization and collective smoke tests passed",
                recommendation="Archive the JSON/Markdown report with the experiment run for reproducibility.",
            )
        )
    return findings


def findings_as_dicts(findings: list[Finding]) -> list[dict[str, str]]:
    return [asdict(item) for item in findings]


def render_diagnosis_markdown(report: dict[str, Any], findings: list[Finding] | None = None) -> str:
    rows = findings if findings is not None else diagnose_report(report)
    lines = [
        "# PrecisionFlow Connect Diagnosis",
        "",
        f"- Report status: {report.get('overall_status', 'unknown')}",
        f"- Report mode: {report.get('mode', 'unknown')}",
        "",
        "| severity | code | area | evidence | recommendation |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    item.severity,
                    item.code,
                    item.area,
                    item.evidence.replace("|", "/"),
                    item.recommendation.replace("|", "/"),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)
