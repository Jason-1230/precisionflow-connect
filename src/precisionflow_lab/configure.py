from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import PRECISION_BYTES
from .planner import validate_manifest


class ConfigureError(ValueError):
    """Raised when a node shorthand cannot be converted into a manifest."""


def parse_node_spec(spec: str, *, default_precision: str = "fp32") -> dict[str, Any]:
    """Parse a CLI node shorthand.

    Format:
        node-a=cuda:0,cuda:1@bf16
        node-b=cuda:0,cuda:1
        cpu-node=cpu@fp32

    The precision suffix is optional and defaults to ``default_precision``.
    """

    if "=" not in spec:
        raise ConfigureError("node specs must use MACHINE=DEVICE[,DEVICE]@PRECISION")

    machine, payload = spec.split("=", 1)
    machine = machine.strip()
    payload = payload.strip()
    if not machine:
        raise ConfigureError("node spec has an empty machine name")
    if not payload:
        raise ConfigureError(f"node spec for {machine!r} has no devices")

    precision = default_precision.lower()
    devices_part = payload
    if "@" in payload:
        devices_part, precision = payload.rsplit("@", 1)
        precision = precision.strip().lower()

    devices = [device.strip() for device in devices_part.split(",") if device.strip()]
    if not devices:
        raise ConfigureError(f"node spec for {machine!r} has no devices")
    if precision not in PRECISION_BYTES:
        supported = ", ".join(sorted(PRECISION_BYTES))
        raise ConfigureError(f"unsupported precision {precision!r}; supported: {supported}")

    return {"machine": machine, "devices": devices, "precision": precision}


def build_manifest_from_nodes(
    node_specs: list[str],
    *,
    job_name: str = "precisionflow-job",
    default_precision: str = "fp32",
) -> dict[str, Any]:
    if not node_specs:
        raise ConfigureError("at least one --node entry is required")
    if default_precision not in PRECISION_BYTES:
        supported = ", ".join(sorted(PRECISION_BYTES))
        raise ConfigureError(f"unsupported default precision {default_precision!r}; supported: {supported}")

    ranks: list[dict[str, Any]] = []
    next_rank = 0
    for spec in node_specs:
        node = parse_node_spec(spec, default_precision=default_precision)
        for device in node["devices"]:
            ranks.append(
                {
                    "rank": next_rank,
                    "machine": node["machine"],
                    "device": device,
                    "precision": node["precision"],
                }
            )
            next_rank += 1

    manifest = {
        "job_name": job_name,
        "world_size": len(ranks),
        "ranks": ranks,
    }
    validate_manifest(manifest)
    return manifest


def write_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target


def render_manifest_summary(manifest: dict[str, Any]) -> str:
    machines = sorted({row["machine"] for row in manifest["ranks"]})
    precisions = sorted({row["precision"] for row in manifest["ranks"]})
    lines = [
        "# PrecisionFlow Manifest",
        "",
        f"- Job name: {manifest['job_name']}",
        f"- World size: {manifest['world_size']}",
        f"- Machines: {', '.join(machines)}",
        f"- Declared precisions: {', '.join(precisions)}",
        "",
        "| rank | machine | device | precision |",
        "| --- | --- | --- | --- |",
    ]
    for row in manifest["ranks"]:
        lines.append(f"| {row['rank']} | {row['machine']} | {row['device']} | {row['precision']} |")
    lines.append("")
    return "\n".join(lines)
