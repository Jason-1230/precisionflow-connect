from __future__ import annotations

import os
import platform
import sys
from typing import Any

from .runtime import env_status, network_status, probe_precision_capabilities


def _safe_bool_call(owner: Any, name: str) -> bool:
    method = getattr(owner, name, None)
    if method is None:
        return False
    try:
        return bool(method())
    except Exception:
        return False


def _import_torch() -> Any | None:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    return torch


def build_environment_inventory(
    *,
    torch_module: Any | None = None,
    environ: dict[str, str] | None = None,
    anonymize_hosts: bool = False,
) -> dict[str, Any]:
    env = dict(os.environ if environ is None else environ)
    torch = torch_module if torch_module is not None else _import_torch()

    torch_info: dict[str, Any]
    distributed_info: dict[str, Any]
    cuda_info: dict[str, Any]

    if torch is None:
        torch_info = {"installed": False, "version": None, "cuda_build": None}
        distributed_info = {
            "available": False,
            "nccl_available": False,
            "gloo_available": False,
        }
        cuda_info = {
            "available": False,
            "device_count": 0,
            "current_device": None,
        }
    else:
        torch_version = getattr(torch, "__version__", None)
        torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        dist = getattr(torch, "distributed", None)
        cuda = getattr(torch, "cuda", None)
        torch_info = {
            "installed": True,
            "version": torch_version,
            "cuda_build": torch_cuda_version,
        }
        distributed_info = {
            "available": _safe_bool_call(dist, "is_available") if dist is not None else False,
            "nccl_available": _safe_bool_call(dist, "is_nccl_available") if dist is not None else False,
            "gloo_available": _safe_bool_call(dist, "is_gloo_available") if dist is not None else False,
        }
        cuda_available = _safe_bool_call(cuda, "is_available") if cuda is not None else False
        if cuda_available:
            try:
                device_count = int(cuda.device_count())
            except Exception:
                device_count = 0
            try:
                current_device = int(cuda.current_device())
            except Exception:
                current_device = None
        else:
            device_count = 0
            current_device = None
        cuda_info = {
            "available": cuda_available,
            "device_count": device_count,
            "current_device": current_device,
        }

    return {
        "tool": "PrecisionFlow Connect",
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "platform": platform.platform(),
        },
        "torch": torch_info,
        "distributed_backends": distributed_info,
        "cuda": cuda_info,
        "environment": env_status(env),
        "network": network_status(env, anonymize_hosts=anonymize_hosts),
        "precision_capability_matrix": probe_precision_capabilities(torch, anonymize_hosts=anonymize_hosts),
    }


def render_environment_markdown(inventory: dict[str, Any]) -> str:
    lines = [
        "# PrecisionFlow Connect Environment",
        "",
        "## Python",
        "",
        f"- Executable: {inventory['python']['executable']}",
        f"- Version: {inventory['python']['version']}",
        f"- Platform: {inventory['python']['platform']}",
        "",
        "## PyTorch",
        "",
        f"- Installed: {inventory['torch']['installed']}",
        f"- Version: {inventory['torch']['version'] or 'n/a'}",
        f"- CUDA build: {inventory['torch']['cuda_build'] or 'n/a'}",
        "",
        "## Distributed Backends",
        "",
        "| backend | available |",
        "| --- | --- |",
        f"| torch.distributed | {inventory['distributed_backends']['available']} |",
        f"| NCCL | {inventory['distributed_backends']['nccl_available']} |",
        f"| Gloo | {inventory['distributed_backends']['gloo_available']} |",
        "",
        "## CUDA",
        "",
        f"- Available: {inventory['cuda']['available']}",
        f"- Device count: {inventory['cuda']['device_count']}",
        f"- Current device: {inventory['cuda']['current_device']}",
        "",
        "## Runtime Environment",
        "",
        "| variable | required | present | status | value | detail |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in inventory["environment"]:
        lines.append(
            f"| {row['name']} | {row['required']} | {row['present']} | {row['status']} | "
            f"{row.get('value') or ''} | {row.get('detail') or ''} |"
        )

    lines.extend(
        [
            "",
            "## Network",
            "",
            f"- Hostname: {inventory['network']['hostname']}",
            f"- Master endpoint status: {inventory['network']['master_endpoint']['status']}",
            f"- Master endpoint detail: {inventory['network']['master_endpoint']['detail']}",
            f"- NCCL socket interface: {inventory['network']['selected_interfaces']['nccl_socket_ifname'] or 'auto'}",
            f"- Gloo socket interface: {inventory['network']['selected_interfaces']['gloo_socket_ifname'] or 'auto'}",
            "",
            "## Precision Readiness Matrix",
            "",
            "| host | device | name | fp32 | tf32 | fp16 | bf16 | fp8 | int8 | warning |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in inventory["precision_capability_matrix"]:
        lines.append(
            "| {host} | {device} | {name} | {fp32} | {tf32} | {fp16} | {bf16} | {fp8} | {int8} | {warning} |".format(
                host=row.get("host"),
                device=row.get("device"),
                name=row.get("device_name"),
                fp32=row.get("fp32"),
                tf32=row.get("tf32"),
                fp16=row.get("fp16"),
                bf16=row.get("bf16"),
                fp8=row.get("fp8"),
                int8=row.get("int8"),
                warning=row.get("warning") or "",
            )
        )
    lines.append("")
    return "\n".join(lines)
