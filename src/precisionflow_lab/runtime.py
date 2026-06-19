from __future__ import annotations

import copy
import os
import socket
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from .planner import build_plan, load_manifest, validate_manifest


REQUIRED_TORCHRUN_ENV = (
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
)

OPTIONAL_DISTRIBUTED_ENV = (
    "LOCAL_WORLD_SIZE",
    "NODE_RANK",
    "NCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "NCCL_DEBUG",
    "NCCL_IB_DISABLE",
    "NCCL_P2P_DISABLE",
    "CUDA_VISIBLE_DEVICES",
)

INTEGER_ENV = ("MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "NODE_RANK")
PRECISION_COLUMNS = ("fp32", "tf32", "fp16", "bf16", "fp8", "int8")
REPORT_SCHEMA_VERSION = "0.4"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _import_torch() -> Any | None:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    return torch


def env_status(environ: dict[str, str] | None = None) -> list[dict[str, str | bool | None]]:
    env = dict(os.environ if environ is None else environ)
    rows: list[dict[str, str | bool | None]] = []
    for name in REQUIRED_TORCHRUN_ENV + OPTIONAL_DISTRIBUTED_ENV:
        present = name in env and env[name] != ""
        status = "PASS" if present else ("FAIL" if name in REQUIRED_TORCHRUN_ENV else "WARN")
        detail = None
        if present and name in INTEGER_ENV and _parse_int(env.get(name)) is None:
            status = "FAIL"
            detail = "expected integer"
        rows.append(
            {
                "name": name,
                "required": name in REQUIRED_TORCHRUN_ENV,
                "present": present,
                "status": status,
                "value": env.get(name),
                "detail": detail,
            }
        )
    return rows


def torchrun_context(environ: dict[str, str] | None = None) -> dict[str, Any]:
    env = dict(os.environ if environ is None else environ)
    return {
        "master_addr": env.get("MASTER_ADDR"),
        "master_port": _parse_int(env.get("MASTER_PORT")),
        "rank": _parse_int(env.get("RANK")),
        "local_rank": _parse_int(env.get("LOCAL_RANK")),
        "world_size": _parse_int(env.get("WORLD_SIZE")),
        "local_world_size": _parse_int(env.get("LOCAL_WORLD_SIZE")),
        "node_rank": _parse_int(env.get("NODE_RANK")),
    }


def network_status(environ: dict[str, str] | None = None, anonymize_hosts: bool = False) -> dict[str, Any]:
    env = dict(os.environ if environ is None else environ)
    raw_hostname = socket.gethostname()
    try:
        host_addresses = socket.gethostbyname_ex(raw_hostname)[2]
    except OSError as exc:
        host_addresses = []
        hostname_warning = str(exc)
    else:
        hostname_warning = None

    try:
        interfaces = [{"index": index, "name": name} for index, name in socket.if_nameindex()]
    except (AttributeError, OSError):
        interfaces = []

    master_addr = env.get("MASTER_ADDR")
    master_port = env.get("MASTER_PORT")
    if master_addr and master_port:
        try:
            socket.getaddrinfo(master_addr, int(master_port))
            master_status = "PASS"
            master_detail = f"{master_addr}:{master_port} resolved locally"
        except (OSError, ValueError) as exc:
            master_status = "WARN"
            master_detail = f"{master_addr}:{master_port} could not be resolved locally: {exc}"
    else:
        master_status = "WARN"
        master_detail = "MASTER_ADDR or MASTER_PORT is not set in this process"

    selected = {
        "nccl_socket_ifname": env.get("NCCL_SOCKET_IFNAME"),
        "gloo_socket_ifname": env.get("GLOO_SOCKET_IFNAME"),
    }

    warnings = []
    if hostname_warning:
        warnings.append(f"hostname lookup warning: {hostname_warning}")
    if not interfaces:
        warnings.append("no network interface list available from socket.if_nameindex")

    rendered_hostname = "host-local" if anonymize_hosts else raw_hostname
    rendered_addresses = ["redacted"] if anonymize_hosts and host_addresses else host_addresses
    rendered_master_addr = "redacted" if anonymize_hosts and master_addr else master_addr
    rendered_master_detail = master_detail.replace(master_addr, "redacted") if anonymize_hosts and master_addr else master_detail

    return {
        "hostname": rendered_hostname,
        "host_addresses": rendered_addresses,
        "interfaces": interfaces,
        "selected_interfaces": selected,
        "master_endpoint": {
            "addr": rendered_master_addr,
            "port": _parse_int(master_port),
            "status": master_status,
            "detail": rendered_master_detail,
        },
        "warnings": warnings,
    }


def precision_capability_from_compute_capability(major: int | None, minor: int | None = None) -> dict[str, bool]:
    if major is None:
        return {"fp32": True, "tf32": False, "fp16": False, "bf16": False, "fp8": False, "int8": False}
    capability = (major, minor or 0)
    return {
        "fp32": True,
        "tf32": capability >= (8, 0),
        "fp16": capability >= (5, 3),
        "bf16": capability >= (8, 0),
        "fp8": capability >= (9, 0),
        "int8": capability >= (6, 1),
    }


def probe_precision_capabilities(torch_module: Any | None = None, anonymize_hosts: bool = False) -> list[dict[str, Any]]:
    torch = torch_module if torch_module is not None else _import_torch()
    host = "host-local" if anonymize_hosts else socket.gethostname()
    if torch is None:
        row = {
            "host": host,
            "device": "cpu",
            "device_name": "PyTorch not installed",
            "compute_capability": "n/a",
            "memory_gib": None,
            "probe_type": "hardware_estimate",
            "warning": "install PyTorch on the target cluster to probe CUDA devices",
        }
        row.update(precision_capability_from_compute_capability(None))
        return [row]

    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        row = {
            "host": host,
            "device": "cpu",
            "device_name": "CUDA unavailable",
            "compute_capability": "n/a",
            "memory_gib": None,
            "probe_type": "hardware_estimate",
            "warning": "CUDA is unavailable in this process; live NCCL probing requires GPUs",
        }
        row.update(precision_capability_from_compute_capability(None))
        return [row]

    rows: list[dict[str, Any]] = []
    for index in range(cuda.device_count()):
        props = cuda.get_device_properties(index)
        major, minor = cuda.get_device_capability(index)
        row = {
            "host": host,
            "device": f"cuda:{index}",
            "device_name": props.name,
            "compute_capability": f"{major}.{minor}",
            "memory_gib": round(props.total_memory / (1024**3), 2),
            "probe_type": "hardware_estimate",
            "warning": None,
        }
        row.update(precision_capability_from_compute_capability(major, minor))
        rows.append(row)
    return rows


def _anonymize_topology(topology: dict[str, Any]) -> dict[str, Any]:
    machine_names = sorted(
        {
            row.get("machine")
            for row in topology.get("rank_mapping", [])
            if row.get("machine") not in (None, "")
        }
    )
    mapping = {name: f"node-{index}" for index, name in enumerate(machine_names)}

    for row in topology.get("rank_mapping", []):
        if row.get("machine") in mapping:
            row["machine"] = mapping[row["machine"]]
    for group in topology.get("machine_groups", []):
        if group.get("machine") in mapping:
            group["machine"] = mapping[group["machine"]]
    topology["machine_name_anonymized"] = bool(mapping)
    return topology


def manifest_summary(manifest_path: str | Path | None, anonymize_hosts: bool = False) -> dict[str, Any]:
    if manifest_path is None:
        return {
            "status": "NOT_PROVIDED",
            "detail": "no manifest supplied; runtime probe will rely on torchrun environment only",
            "rank_mapping": [],
            "machine_groups": [],
            "precision_streams": [],
        }

    manifest = load_manifest(manifest_path)
    plan = build_plan(manifest)
    ranks = validate_manifest(manifest)
    topology = {
        "status": "PASS",
        "manifest": str(manifest_path),
        "job_name": plan.job_name,
        "world_size": plan.world_size,
        "rank_mapping": [asdict(item) for item in ranks],
        "machine_groups": [asdict(item) for item in plan.machine_groups],
        "precision_streams": [asdict(item) for item in plan.precision_streams],
        "stages": list(plan.stages),
    }
    if anonymize_hosts:
        topology = _anonymize_topology(topology)
    return topology


def _manifest_runtime_warnings(topology: dict[str, Any], context: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if topology.get("status") != "PASS":
        return warnings

    declared_world_size = topology.get("world_size")
    runtime_world_size = context.get("world_size")
    if runtime_world_size is not None and declared_world_size != runtime_world_size:
        warnings.append(f"manifest world_size={declared_world_size} differs from runtime WORLD_SIZE={runtime_world_size}")

    rank = context.get("rank")
    if rank is not None:
        known_ranks = {row["rank"] for row in topology.get("rank_mapping", [])}
        if rank not in known_ranks:
            warnings.append(f"runtime RANK={rank} is not present in manifest rank mapping")
    return warnings


def build_preflight_report(
    manifest_path: str | Path | None = None,
    environ: dict[str, str] | None = None,
    torch_module: Any | None = None,
    anonymize_hosts: bool = False,
) -> dict[str, Any]:
    context = torchrun_context(environ)
    topology = manifest_summary(manifest_path, anonymize_hosts=anonymize_hosts)
    warnings = _manifest_runtime_warnings(topology, context)
    env_rows = env_status(environ)
    network = network_status(environ, anonymize_hosts=anonymize_hosts)

    required_rows = [row for row in env_rows if row["required"]]
    required_present = [row for row in required_rows if row["present"]]
    parse_failures = [row for row in required_rows if row["present"] and row["status"] == "FAIL"]
    all_required_present = all(row["present"] for row in required_rows)
    if topology.get("status") not in ("PASS", "NOT_PROVIDED") or parse_failures:
        overall_status = "FAIL"
    elif all_required_present and not warnings and network.get("master_endpoint", {}).get("status") == "PASS":
        overall_status = "PASS"
    else:
        overall_status = "WARN"

    diagnosis = list(warnings)
    diagnosis.extend(f"{row['name']}: {row['detail']}" for row in parse_failures if row.get("detail"))
    if not diagnosis:
        if all_required_present:
            diagnosis.append("preflight checks passed; live collective tests were not run")
        else:
            diagnosis.append("preflight mode validates manifest and local probes; run under torchrun --live for distributed checks")

    report = {
        "tool": "PrecisionFlow Connect",
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "mode": "preflight",
        "created_at_utc": _utc_now(),
        "overall_status": overall_status,
        "rank_runtime": context,
        "cluster_topology": topology,
        "environment": env_rows,
        "network": network,
        "backend_status": [
            {
                "backend": "nccl/gloo",
                "status": "NOT_RUN",
                "detail": "use --live under torchrun to initialize a real process group",
            }
        ],
        "precision_capability_matrix": probe_precision_capabilities(torch_module, anonymize_hosts=anonymize_hosts),
        "collective_tests": [
            {"name": "barrier", "status": "NOT_RUN", "detail": "requires --live"},
            {"name": "all_reduce", "status": "NOT_RUN", "detail": "requires --live"},
            {"name": "all_gather", "status": "NOT_RUN", "detail": "requires --live"},
        ],
        "failure_diagnosis_or_warning": diagnosis,
    }
    return report


def _required_environment_errors(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for row in report.get("environment", []):
        if not row.get("required"):
            continue
        if not row.get("present"):
            errors.append(f"missing required torchrun environment variable {row.get('name')}")
        elif row.get("status") == "FAIL":
            detail = row.get("detail") or "invalid value"
            errors.append(f"{row.get('name')}: {detail}")
    return errors


def _declared_rank(topology: dict[str, Any], rank: int | None) -> dict[str, Any] | None:
    if rank is None:
        return None
    for row in topology.get("rank_mapping", []):
        if row.get("rank") == rank:
            return row
    return None


def _precision_supported(
    precision_rows: list[dict[str, Any]],
    *,
    device: str,
    precision: str,
) -> bool:
    for row in precision_rows:
        if row.get("device") == device:
            return bool(row.get(precision))
    return False


def _live_topology_validation(
    report: dict[str, Any],
    *,
    runtime_device: str | None,
) -> dict[str, Any]:
    topology = report.get("cluster_topology", {})
    context = report.get("rank_runtime", {})
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    if topology.get("status") == "NOT_PROVIDED":
        warnings.append("no manifest supplied; live mode validated runtime communication only")
        return {"status": "WARN", "errors": errors, "warnings": warnings, "rows": rows}
    if topology.get("status") != "PASS":
        errors.append(f"manifest status is {topology.get('status')}: {topology.get('detail')}")
        return {"status": "FAIL", "errors": errors, "warnings": warnings, "rows": rows}

    declared_world_size = topology.get("world_size")
    runtime_world_size = context.get("world_size")
    if runtime_world_size != declared_world_size:
        errors.append(f"manifest world_size={declared_world_size} differs from runtime WORLD_SIZE={runtime_world_size}")

    rank = context.get("rank")
    declared = _declared_rank(topology, rank)
    if declared is None:
        errors.append(f"runtime RANK={rank} is not present in manifest rank mapping")
    else:
        declared_device = declared.get("device")
        declared_precision = declared.get("precision")
        device_match = runtime_device == declared_device
        precision_match = _precision_supported(
            report.get("precision_capability_matrix", []),
            device=runtime_device or "",
            precision=declared_precision,
        )
        if not device_match:
            errors.append(
                f"rank {rank} declared device {declared_device} but runtime selected {runtime_device}"
            )
        if not precision_match:
            errors.append(
                f"rank {rank} declared precision {declared_precision} is not estimated as supported on {runtime_device}"
            )
        rows.append(
            {
                "rank": rank,
                "declared_machine": declared.get("machine"),
                "declared_device": declared_device,
                "observed_device": runtime_device,
                "device_match": device_match,
                "declared_precision": declared_precision,
                "estimated_precision_supported": precision_match,
            }
        )

    status = "FAIL" if errors else ("WARN" if warnings else "PASS")
    return {"status": status, "errors": errors, "warnings": warnings, "rows": rows}


def _select_backend(torch: Any, requested_backend: str) -> str:
    dist = torch.distributed
    if requested_backend != "auto":
        if requested_backend == "nccl" and not dist.is_nccl_available():
            raise RuntimeError("requested NCCL backend is not available in this PyTorch build")
        if requested_backend == "gloo" and not dist.is_gloo_available():
            raise RuntimeError("requested Gloo backend is not available in this PyTorch build")
        return requested_backend

    if torch.cuda.is_available() and dist.is_nccl_available():
        return "nccl"
    if dist.is_gloo_available():
        return "gloo"
    raise RuntimeError("neither NCCL nor Gloo is available in this PyTorch build")


def _mark_live_failure(report: dict[str, Any], backend: str, exc: Exception) -> dict[str, Any]:
    report["mode"] = "live"
    report["overall_status"] = "FAIL"
    report["backend_status"] = [
        {
            "backend": backend,
            "status": "FAIL",
            "detail": str(exc),
        }
    ]
    report["failure_diagnosis_or_warning"] = [
        "distributed initialization or collective communication failed",
        str(exc),
    ]
    return report


def run_live_report(
    manifest_path: str | Path | None = None,
    backend: str = "auto",
    tensor_elements: int = 8,
    timeout_seconds: int = 120,
    environ: dict[str, str] | None = None,
    anonymize_hosts: bool = False,
) -> dict[str, Any]:
    report = build_preflight_report(manifest_path=manifest_path, environ=environ, anonymize_hosts=anonymize_hosts)
    report["mode"] = "live"
    report["requested_backend"] = backend

    torch = _import_torch()
    if torch is None:
        return _mark_live_failure(report, backend, RuntimeError("PyTorch is not installed"))
    if not torch.distributed.is_available():
        return _mark_live_failure(report, backend, RuntimeError("torch.distributed is not available"))

    selected_backend = backend
    initialized = False
    try:
        selected_backend = _select_backend(torch, backend)
        if selected_backend == "nccl" and not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA devices")
        local_rank = _parse_int((environ or os.environ).get("LOCAL_RANK"))
        if selected_backend == "nccl":
            if local_rank is None:
                raise RuntimeError("LOCAL_RANK is required for NCCL device assignment")
            if local_rank < 0 or local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} is outside the visible CUDA device range 0..{torch.cuda.device_count() - 1}"
                )
            torch.cuda.set_device(local_rank)

        torch.distributed.init_process_group(
            backend=selected_backend,
            timeout=timedelta(seconds=timeout_seconds),
        )
        initialized = True
        dist = torch.distributed
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if local_rank is None:
            local_rank = _parse_int((environ or os.environ).get("LOCAL_RANK")) or 0

        if selected_backend == "nccl":
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")

        dist.barrier()

        values = torch.full((tensor_elements,), float(rank + 1), device=device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        expected = world_size * (world_size + 1) / 2
        all_reduce_passed = bool(torch.allclose(values, torch.full_like(values, expected)))

        rank_tensor = torch.tensor([rank], dtype=torch.int64, device=device)
        gathered_tensors = [torch.zeros_like(rank_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, rank_tensor)
        gathered_ranks = [int(item.item()) for item in gathered_tensors]
        all_gather_passed = gathered_ranks == list(range(world_size))

        local_report = copy.deepcopy(report)
        local_report["overall_status"] = "PASS" if all_reduce_passed and all_gather_passed else "FAIL"
        local_report["rank_runtime"] = {
            **torchrun_context(environ),
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "device": str(device),
        }
        topology_validation = _live_topology_validation(local_report, runtime_device=str(device))
        blocking_errors = _required_environment_errors(local_report) + topology_validation["errors"]
        collective_passed = all_reduce_passed and all_gather_passed
        local_report["overall_status"] = "PASS" if collective_passed and not blocking_errors else "FAIL"
        local_report["backend_status"] = [
            {
                "backend": selected_backend,
                "status": "PASS",
                "detail": f"{selected_backend.upper()} backend initialized successfully",
            }
        ]
        local_report["collective_tests"] = [
            {"name": "barrier", "status": "PASS", "detail": "all ranks entered and exited barrier"},
            {
                "name": "all_reduce",
                "status": "PASS" if all_reduce_passed else "FAIL",
                "detail": f"sum expected {expected:g}, rank {rank} observed {float(values[0].item()):g}",
            },
            {
                "name": "all_gather",
                "status": "PASS" if all_gather_passed else "FAIL",
                "detail": f"gathered ranks {gathered_ranks}",
            },
        ]
        local_report["topology_validation"] = topology_validation
        local_report["rank_probe"] = {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "device": str(device),
            "backend": selected_backend,
            "hostname": f"host-rank-{rank}" if anonymize_hosts else socket.gethostname(),
        }
        diagnosis: list[str] = []
        if collective_passed:
            diagnosis.append("multi-node communication passed")
            diagnosis.append("collective communication smoke test passed")
        else:
            diagnosis.append("collective communication smoke test failed")
        diagnosis.extend(blocking_errors)
        diagnosis.extend(topology_validation["warnings"])
        local_report["failure_diagnosis_or_warning"] = diagnosis

        gathered_reports: list[dict[str, Any] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_reports, local_report)
        dist.barrier()

        if rank != 0:
            return local_report

        rank_reports = [item for item in gathered_reports if item is not None]
        aggregate = copy.deepcopy(local_report)
        aggregate["rank_reports"] = rank_reports
        aggregate["rank_mapping_observed"] = [item["rank_probe"] for item in rank_reports]
        aggregate_errors: list[str] = []
        aggregate_rows: list[dict[str, Any]] = []
        aggregate_warnings: list[str] = []
        for item in rank_reports:
            validation = item.get("topology_validation", {})
            aggregate_errors.extend(validation.get("errors", []))
            aggregate_warnings.extend(validation.get("warnings", []))
            aggregate_rows.extend(validation.get("rows", []))
        aggregate["topology_validation"] = {
            "status": "FAIL" if aggregate_errors else ("WARN" if aggregate_warnings else "PASS"),
            "errors": list(dict.fromkeys(aggregate_errors)),
            "warnings": list(dict.fromkeys(aggregate_warnings)),
            "rows": aggregate_rows,
        }
        aggregate["overall_status"] = "PASS" if not aggregate_errors and collective_passed else "FAIL"
        aggregate["backend_status"] = [
            {
                "backend": selected_backend,
                "status": "PASS",
                "detail": f"{selected_backend.upper()} backend initialized successfully across {world_size} ranks",
            }
        ]
        aggregate["failure_diagnosis_or_warning"] = [
            "multi-node communication passed",
            "collective communication smoke test passed",
            "rank-to-machine/device mapping captured on rank 0",
        ]
        aggregate["failure_diagnosis_or_warning"].extend(dict.fromkeys(aggregate_errors))
        aggregate["failure_diagnosis_or_warning"].extend(dict.fromkeys(aggregate_warnings))
        return aggregate
    except Exception as exc:
        return _mark_live_failure(report, selected_backend, exc)
    finally:
        if initialized and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
