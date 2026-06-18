from __future__ import annotations

import json
from pathlib import Path

from .models import (
    MachineGroup,
    PRECISION_BYTES,
    PrecisionFlowPlan,
    PrecisionStream,
    RankSpec,
)


class ManifestError(ValueError):
    """Raised when a precision-flow manifest is not launcher-safe."""


def load_manifest(path: str | Path) -> dict:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _rank_specs(manifest: dict) -> list[RankSpec]:
    if "ranks" not in manifest:
        raise ManifestError("manifest must include a 'ranks' list")
    return [RankSpec.from_dict(item) for item in manifest["ranks"]]


def validate_manifest(manifest: dict) -> list[RankSpec]:
    ranks = _rank_specs(manifest)
    world_size = int(manifest.get("world_size", -1))
    rank_ids = [item.rank for item in ranks]

    if world_size != len(ranks):
        raise ManifestError(f"world_size={world_size} but {len(ranks)} ranks were declared")
    if sorted(rank_ids) != list(range(world_size)):
        raise ManifestError("ranks must be contiguous from 0 to world_size - 1")

    for item in ranks:
        if item.precision not in PRECISION_BYTES:
            supported = ", ".join(sorted(PRECISION_BYTES))
            raise ManifestError(f"rank {item.rank} uses unsupported precision '{item.precision}'; supported: {supported}")
        if not item.machine:
            raise ManifestError(f"rank {item.rank} has an empty machine name")

    return sorted(ranks, key=lambda item: item.rank)


def build_plan(manifest: dict) -> PrecisionFlowPlan:
    ranks = validate_manifest(manifest)
    world_size = int(manifest["world_size"])
    job_name = str(manifest.get("job_name", "precisionflow-job"))

    precision_streams: list[PrecisionStream] = []
    for precision in sorted(PRECISION_BYTES):
        stream_ranks = tuple(item.rank for item in ranks if item.precision == precision)
        if stream_ranks:
            precision_streams.append(
                PrecisionStream(
                    precision=precision,
                    ranks=stream_ranks,
                    bytes_per_element=PRECISION_BYTES[precision],
                )
            )

    machine_names = sorted({item.machine for item in ranks})
    machine_groups: list[MachineGroup] = []
    for machine in machine_names:
        machine_ranks = tuple(item.rank for item in ranks if item.machine == machine)
        machine_groups.append(
            MachineGroup(
                machine=machine,
                ranks=machine_ranks,
                leader_rank=min(machine_ranks),
            )
        )

    stages = (
        "validate rank, machine, device, and declared precision manifest",
        "check torchrun rank/local_rank/world_size environment consistency",
        "initialize NCCL or Gloo process group across declared ranks",
        "run barrier, all-reduce, and all-gather collective smoke tests",
        "emit topology, backend, network, environment, and precision capability report",
    )

    return PrecisionFlowPlan(
        job_name=job_name,
        world_size=world_size,
        precision_streams=tuple(precision_streams),
        machine_groups=tuple(machine_groups),
        stages=stages,
    )
