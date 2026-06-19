from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from . import __version__
from .diagnose import diagnose_report, findings_as_dicts, load_report, render_diagnosis_markdown
from .framework import build_framework_plan, render_framework_markdown
from .planner import ManifestError, build_plan, load_manifest
from .profile import build_capability_profile, render_profile_markdown
from .report import render_markdown_report, write_json_report, write_markdown_report
from .runtime import build_preflight_report, run_live_report


def _format_bytes(value: int) -> str:
    mib = value / (1024 * 1024)
    return f"{mib:.2f} MiB"


def render_text(plan, tensor_elements: int) -> str:
    lines = [
        f"job_name: {plan.job_name}",
        f"world_size: {plan.world_size}",
        "machines: " + ", ".join(group.machine for group in plan.machine_groups),
        "precision_streams:",
    ]

    for stream in plan.precision_streams:
        lines.append(
            f"  {stream.precision}: ranks {list(stream.ranks)} -> "
            f"{_format_bytes(stream.estimated_bytes(tensor_elements))} per step"
        )

    lines.append("leader_exchange:")
    for group in plan.machine_groups:
        lines.append(f"  machine {group.machine} leader rank {group.leader_rank}")

    lines.append("stages:")
    for index, stage in enumerate(plan.stages, start=1):
        lines.append(f"  {index}. {stage}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="precisionflow-connect")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="validate a manifest and print a rank, machine, device, and precision summary",
    )
    inspect_parser.add_argument("manifest")
    inspect_parser.add_argument("--tensor-elements", type=int, default=1_000_000)
    inspect_parser.add_argument("--json", action="store_true", help="print machine-readable JSON")

    plan_parser = subparsers.add_parser(
        "plan",
        help="compatibility alias for inspect",
    )
    plan_parser.add_argument("manifest")
    plan_parser.add_argument("--tensor-elements", type=int, default=1_000_000)
    plan_parser.add_argument("--json", action="store_true", help="print machine-readable JSON")

    connect_parser = subparsers.add_parser(
        "connect",
        help="probe torchrun, backend, network, collective communication, and precision capability status",
    )
    connect_parser.add_argument("--manifest", help="optional cluster manifest for declared rank/device mapping")
    connect_parser.add_argument("--live", action="store_true", help="initialize torch.distributed and run collectives")
    connect_parser.add_argument("--backend", choices=("auto", "nccl", "gloo"), default="auto")
    connect_parser.add_argument("--tensor-elements", type=int, default=8, help="tensor size for collective smoke tests")
    connect_parser.add_argument("--timeout-seconds", type=int, default=120)
    connect_parser.add_argument(
        "--anonymize-hostnames",
        action="store_true",
        help="redact hostnames and host addresses in generated reports",
    )
    connect_parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    connect_parser.add_argument("--json-output", help="write report JSON to this path from rank 0")
    connect_parser.add_argument("--markdown-output", help="write report Markdown to this path from rank 0")

    doctor_parser = subparsers.add_parser("doctor", help="diagnose a generated PrecisionFlow Connect JSON report")
    doctor_parser.add_argument("report")
    doctor_parser.add_argument("--json", action="store_true", help="print machine-readable findings JSON")
    doctor_parser.add_argument("--strict-exit", action="store_true", help="return non-zero when ERROR findings are present")

    framework_parser = subparsers.add_parser(
        "framework",
        help="render a Docker and torchrun launch framework plan for a distributed connectivity run",
    )
    framework_parser.add_argument("manifest")
    framework_parser.add_argument("--master-addr", default="192.0.2.10")
    framework_parser.add_argument("--master-port", type=int, default=29500)
    framework_parser.add_argument("--image", default="precisionflow-connect:latest")
    framework_parser.add_argument("--network-interface")
    framework_parser.add_argument("--backend", choices=("auto", "nccl", "gloo"), default="auto")
    framework_parser.add_argument("--container-workdir", default="/workspace")
    framework_parser.add_argument("--manifest-in-container")
    framework_parser.add_argument("--report-dir", default="reports")
    framework_parser.add_argument("--timeout-seconds", type=int, default=120)
    framework_parser.add_argument("--cpu-only", action="store_true", help="omit Docker --gpus all from generated commands")
    framework_parser.add_argument("--json", action="store_true", help="print machine-readable JSON")

    profile_parser = subparsers.add_parser(
        "profile",
        help="print the deployment, readiness-check, and framework-handoff capability profile",
    )
    profile_parser.add_argument("--json", action="store_true", help="print machine-readable JSON")

    args = parser.parse_args(argv)

    try:
        if args.command in ("inspect", "plan"):
            plan = build_plan(load_manifest(args.manifest))
            if args.json:
                print(json.dumps(asdict(plan), indent=2))
            else:
                print(render_text(plan, args.tensor_elements))
            return 0
        if args.command == "connect":
            if args.live:
                report = run_live_report(
                    manifest_path=args.manifest,
                    backend=args.backend,
                    tensor_elements=args.tensor_elements,
                    timeout_seconds=args.timeout_seconds,
                    anonymize_hosts=args.anonymize_hostnames,
                )
            else:
                report = build_preflight_report(
                    manifest_path=args.manifest,
                    anonymize_hosts=args.anonymize_hostnames,
                )

            rank = report.get("rank_runtime", {}).get("rank")
            should_emit = (not args.live) or rank in (None, 0)
            if should_emit and args.json_output:
                write_json_report(report, args.json_output)
            if should_emit and args.markdown_output:
                write_markdown_report(report, args.markdown_output)
            if should_emit:
                if args.json:
                    print(json.dumps(report, indent=2))
                else:
                    print(render_markdown_report(report))
            return 1 if report.get("overall_status") == "FAIL" else 0
        if args.command == "doctor":
            report = load_report(args.report)
            findings = diagnose_report(report)
            if args.json:
                print(json.dumps(findings_as_dicts(findings), indent=2))
            else:
                print(render_diagnosis_markdown(report, findings))
            has_error = any(item.severity == "ERROR" for item in findings)
            return 1 if args.strict_exit and has_error else 0
        if args.command == "framework":
            plan = build_framework_plan(
                args.manifest,
                master_addr=args.master_addr,
                master_port=args.master_port,
                image=args.image,
                network_interface=args.network_interface,
                backend=args.backend,
                container_workdir=args.container_workdir,
                manifest_inside_container=args.manifest_in_container,
                report_dir=args.report_dir,
                timeout_seconds=args.timeout_seconds,
                use_gpus=not args.cpu_only,
            )
            if args.json:
                print(json.dumps(plan, indent=2))
            else:
                print(render_framework_markdown(plan))
            return 0
        if args.command == "profile":
            profile = build_capability_profile()
            if args.json:
                print(json.dumps(profile, indent=2))
            else:
                print(render_profile_markdown(profile))
            return 0
    except ManifestError as exc:
        parser.exit(status=2, message=f"precisionflow-connect: {exc}\n")

    return 1
