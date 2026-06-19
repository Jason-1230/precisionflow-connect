from __future__ import annotations

from typing import Any


def build_capability_profile() -> dict[str, Any]:
    return {
        "name": "PrecisionFlow Connect",
        "tagline": "Multi-node readiness and heterogeneous precision probing for distributed AI training environments.",
        "purpose": (
            "Check Docker, torchrun, network, backend, rank mapping, collective communication, "
            "and heterogeneous precision readiness before launching a distributed training workload."
        ),
        "deployment_targets": [
            {
                "target": "manifest configuration",
                "entrypoint": "precisionflow-connect configure --node node-a=cuda:0,cuda:1@bf16 --node node-b=cuda:0,cuda:1@fp16",
                "evidence": "validated world size, rank-to-machine mapping, rank-to-device mapping, and declared precision streams",
            },
            {
                "target": "environment inventory",
                "entrypoint": "precisionflow-connect env",
                "evidence": "Python, PyTorch, backend, CUDA, torchrun environment, network, and precision readiness rows",
            },
            {
                "target": "local preflight",
                "entrypoint": "precisionflow-connect connect --manifest configs/multinode_2x4.json",
                "evidence": "manifest validation, local environment rows, network interface list, and precision readiness matrix",
            },
            {
                "target": "bare-metal torchrun",
                "entrypoint": "torchrun ... -m precisionflow_lab connect --live",
                "evidence": "NCCL/Gloo initialization plus barrier, all-reduce, and all-gather smoke tests",
            },
            {
                "target": "readiness-gated training handoff",
                "entrypoint": "precisionflow-connect launch configs/multinode_2x4.json --node-rank 0 --master-addr 192.0.2.10 --training-script examples/minimal_ddp_train.py",
                "evidence": "readiness command, training torchrun command, environment bindings, and post-run report closure",
            },
            {
                "target": "Docker runtime",
                "entrypoint": "precisionflow-connect framework configs/multinode_2x4.json --image precisionflow-connect:gpu",
                "evidence": "per-node Docker commands with device visibility, host networking, and socket interface binding",
            },
            {
                "target": "scheduler handoff",
                "entrypoint": "run the generated torchrun command inside a Slurm, Kubernetes, or cloud job step",
                "evidence": "same report schema across scheduler environments",
            },
        ],
        "architecture_layers": [
            {
                "layer": "cluster manifest",
                "responsibility": "declare world size, rank-to-machine mapping, rank-to-device mapping, and desired precision streams",
            },
            {
                "layer": "environment inventory",
                "responsibility": "record Python, PyTorch, backend, CUDA, torchrun, network, and heterogeneous precision readiness before launch",
            },
            {
                "layer": "launcher planner",
                "responsibility": "derive node_rank, nproc_per_node, CUDA_VISIBLE_DEVICES, torchrun commands, and Docker commands",
            },
            {
                "layer": "runtime probe",
                "responsibility": "inspect torchrun environment, network interfaces, PyTorch distributed support, and CUDA devices",
            },
            {
                "layer": "collective smoke tests",
                "responsibility": "initialize NCCL/Gloo, run barrier, all-reduce, all-gather, and gather per-rank reports",
            },
            {
                "layer": "report and diagnosis",
                "responsibility": "emit JSON/Markdown reports and convert failures into actionable findings",
            },
            {
                "layer": "training handoff",
                "responsibility": "gate a real training script behind the same torchrun topology and archive readiness evidence",
            },
        ],
        "readiness_checks": [
            {
                "area": "launcher",
                "checks": ["MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE"],
                "failure_signal": "missing or invalid torchrun environment variables",
                "next_action": "regenerate the launch command and make nnodes * nproc_per_node match the manifest world size",
            },
            {
                "area": "rank mapping",
                "checks": ["contiguous ranks", "machine groups", "device assignment", "runtime rank presence"],
                "failure_signal": "manifest/runtime world size mismatch or unknown runtime rank",
                "next_action": "fix the manifest or launch arguments before running the real training job",
            },
            {
                "area": "network",
                "checks": ["master endpoint", "host interfaces", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"],
                "failure_signal": "unresolved master endpoint or unpinned multi-NIC environment",
                "next_action": "bind the intended training interface and check cross-node reachability",
            },
            {
                "area": "backend",
                "checks": ["torch.distributed", "NCCL availability", "Gloo availability", "backend selection"],
                "failure_signal": "backend initialization failure",
                "next_action": "check PyTorch build, CUDA visibility, driver/runtime versions, and backend consistency",
            },
            {
                "area": "collectives",
                "checks": ["barrier", "all_reduce", "all_gather", "all_gather_object"],
                "failure_signal": "ranks do not enter or complete the same collective sequence",
                "next_action": "inspect rank consistency, backend health, tensor device placement, and network binding",
            },
            {
                "area": "precision",
                "checks": ["fp32", "tf32", "fp16", "bf16", "fp8", "int8"],
                "failure_signal": "requested precision is unavailable on part of the cluster",
                "next_action": "gate precision paths by device capability and record heterogeneous precision rows in the report",
            },
        ],
        "framework_handoff": [
            {
                "framework": "PyTorch torchrun",
                "use": "run PrecisionFlow Connect with the same nnodes, nproc_per_node, master address, and backend before training",
            },
            {
                "framework": "Hugging Face Accelerate",
                "use": "use the report to confirm mixed-precision support and multi-node communication before accelerate launch",
            },
            {
                "framework": "DeepSpeed",
                "use": "use the report as a pre-run environment record before ZeRO, pipeline, or mixed-precision jobs",
            },
            {
                "framework": "Kubernetes, Slurm, or cloud launchers",
                "use": "run the same generated command inside the scheduler job and archive the report with the experiment output",
            },
        ],
        "failure_lifecycle": [
            "detect launcher, backend, network, collective, and precision failures",
            "classify each finding by severity, code, and area",
            "recommend a concrete next action",
            "rerun the same manifest after fixing the environment",
            "archive the JSON/Markdown report with the training run",
        ],
    }


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def render_profile_markdown(profile: dict[str, Any] | None = None) -> str:
    payload = profile or build_capability_profile()
    lines = [
        "# PrecisionFlow Connect Capability Profile",
        "",
        payload["tagline"],
        "",
        "## Purpose",
        "",
        payload["purpose"],
        "",
        "## Deployment Targets",
        "",
    ]
    lines.extend(
        _table(
            ["target", "entrypoint", "evidence"],
            [[row["target"], row["entrypoint"], row["evidence"]] for row in payload["deployment_targets"]],
        )
    )
    lines.extend(["", "## Architecture Layers", ""])
    lines.extend(
        _table(
            ["layer", "responsibility"],
            [[row["layer"], row["responsibility"]] for row in payload["architecture_layers"]],
        )
    )
    lines.extend(["", "## Readiness Checks", ""])
    lines.extend(
        _table(
            ["area", "checks", "failure signal", "next action"],
            [
                [row["area"], ", ".join(row["checks"]), row["failure_signal"], row["next_action"]]
                for row in payload["readiness_checks"]
            ],
        )
    )
    lines.extend(["", "## Framework Handoff", ""])
    lines.extend(
        _table(
            ["framework", "use"],
            [[row["framework"], row["use"]] for row in payload["framework_handoff"]],
        )
    )
    lines.extend(["", "## Failure Lifecycle", ""])
    for item in payload["failure_lifecycle"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)
