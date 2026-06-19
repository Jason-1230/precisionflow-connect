from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

from . import __version__
from .configure import ConfigureError, build_manifest_from_nodes, render_manifest_summary, write_manifest
from .diagnose import diagnose_report, findings_as_dicts, load_report, render_diagnosis_markdown
from .environment import build_environment_inventory, render_environment_markdown
from .framework import build_framework_plan, render_framework_markdown
from .launcher import build_launch_plan, build_self_test_command, execute_launch_plan, render_launch_markdown
from .planner import ManifestError, build_plan, load_manifest
from .profile import build_capability_profile, render_profile_markdown
from .report import render_markdown_report, write_json_report, write_markdown_report
from .runtime import build_preflight_report, run_live_report
from .selftest import build_native_self_test_processes, render_native_self_test_plan, run_native_self_test


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

    configure_parser = subparsers.add_parser(
        "configure",
        help="build a cluster manifest from node/device shorthand",
    )
    configure_parser.add_argument("--job-name", default="precisionflow-job")
    configure_parser.add_argument(
        "--node",
        action="append",
        default=[],
        help="node shorthand such as node-a=cuda:0,cuda:1@bf16; repeat once per machine",
    )
    configure_parser.add_argument("--default-precision", default="fp32")
    configure_parser.add_argument("--output", help="write manifest JSON to this path")
    configure_parser.add_argument("--json", action="store_true", help="print manifest JSON")

    env_parser = subparsers.add_parser(
        "env",
        help="print Python, PyTorch, backend, network, torchrun, and precision readiness inventory",
    )
    env_parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    env_parser.add_argument("--anonymize-hostnames", action="store_true")

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
        help="probe torchrun, backend, network, collective communication, and precision readiness status",
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

    self_test_parser = subparsers.add_parser(
        "self-test",
        help="run or print a local torch.distributed smoke test command",
    )
    self_test_parser.add_argument("--nproc-per-node", type=int, default=2)
    self_test_parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    self_test_parser.add_argument("--report-dir", default="reports/self-test")
    self_test_parser.add_argument("--master-addr", default="127.0.0.1")
    self_test_parser.add_argument("--master-port", type=int, default=29577)
    self_test_parser.add_argument("--timeout-seconds", type=int, default=120)
    self_test_parser.add_argument(
        "--launcher",
        choices=("native", "torchrun"),
        default="native",
        help="native uses local subprocesses; torchrun prints or runs the torch.distributed launcher path",
    )
    self_test_parser.add_argument("--dry-run", action="store_true", help="print the command without executing it")

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

    launch_parser = subparsers.add_parser(
        "launch",
        help="generate or execute a readiness-gated torchrun handoff for a training script",
    )
    launch_parser.add_argument("manifest")
    launch_parser.add_argument("--node-rank", type=int, required=True)
    launch_parser.add_argument("--master-addr", required=True)
    launch_parser.add_argument("--master-port", type=int, default=29500)
    launch_parser.add_argument("--backend", choices=("auto", "nccl", "gloo"), default="auto")
    launch_parser.add_argument("--network-interface")
    launch_parser.add_argument("--report-dir", default="reports")
    launch_parser.add_argument("--training-script")
    launch_parser.add_argument("--image", default="precisionflow-connect:latest")
    launch_parser.add_argument("--timeout-seconds", type=int, default=120)
    launch_parser.add_argument("--execute", action="store_true", help="run readiness gate and training command locally")
    launch_parser.add_argument("--json", action="store_true", help="print machine-readable JSON")

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    launch_training_args: list[str] = []
    if "launch" in raw_argv:
        launch_index = raw_argv.index("launch")
        try:
            separator_index = raw_argv.index("--", launch_index + 1)
        except ValueError:
            pass
        else:
            launch_training_args = raw_argv[separator_index + 1 :]
            raw_argv = raw_argv[:separator_index]

    args = parser.parse_args(raw_argv)
    if args.command == "launch":
        args.training_args = launch_training_args

    try:
        if args.command == "configure":
            manifest = build_manifest_from_nodes(
                args.node,
                job_name=args.job_name,
                default_precision=args.default_precision.lower(),
            )
            if args.output:
                write_manifest(manifest, args.output)
            if args.json:
                print(json.dumps(manifest, indent=2))
            else:
                if args.output:
                    print(f"Wrote manifest to {args.output}")
                    print("")
                print(render_manifest_summary(manifest))
            return 0
        if args.command == "env":
            inventory = build_environment_inventory(anonymize_hosts=args.anonymize_hostnames)
            if args.json:
                print(json.dumps(inventory, indent=2))
            else:
                print(render_environment_markdown(inventory))
            return 0
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
        if args.command == "self-test":
            if args.launcher == "torchrun":
                command = build_self_test_command(
                    nproc_per_node=args.nproc_per_node,
                    backend=args.backend,
                    report_dir=args.report_dir,
                    master_addr=args.master_addr,
                    master_port=args.master_port,
                )
                if args.dry_run:
                    print(" ".join(command))
                    return 0
                import subprocess

                env = os.environ.copy()
                env.setdefault("USE_LIBUV", "0")
                return subprocess.run(command, env=env, check=False).returncode
            processes = build_native_self_test_processes(
                nproc_per_node=args.nproc_per_node,
                backend=args.backend,
                report_dir=args.report_dir,
                master_addr=args.master_addr,
                master_port=args.master_port,
            )
            if args.dry_run:
                print(render_native_self_test_plan(processes))
                return 0
            return run_native_self_test(
                nproc_per_node=args.nproc_per_node,
                backend=args.backend,
                report_dir=args.report_dir,
                master_addr=args.master_addr,
                master_port=args.master_port,
                timeout_seconds=args.timeout_seconds,
            )
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
        if args.command == "launch":
            launch_plan = build_launch_plan(
                args.manifest,
                node_rank=args.node_rank,
                master_addr=args.master_addr,
                master_port=args.master_port,
                backend=args.backend,
                network_interface=args.network_interface,
                report_dir=args.report_dir,
                training_script=args.training_script,
                training_args=args.training_args,
                image=args.image,
                timeout_seconds=args.timeout_seconds,
            )
            if args.json:
                print(json.dumps(launch_plan, indent=2))
            else:
                print(render_launch_markdown(launch_plan))
            if args.execute:
                return execute_launch_plan(launch_plan)
            return 0
    except (ConfigureError, ManifestError, ValueError) as exc:
        parser.exit(status=2, message=f"precisionflow-connect: {exc}\n")

    return 1
