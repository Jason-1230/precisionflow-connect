from __future__ import annotations

from dataclasses import dataclass


PRECISION_BYTES = {
    "fp32": 4,
    "tf32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
}


@dataclass(frozen=True)
class RankSpec:
    rank: int
    machine: str
    device: str
    precision: str

    @classmethod
    def from_dict(cls, data: dict) -> "RankSpec":
        return cls(
            rank=int(data["rank"]),
            machine=str(data["machine"]),
            device=str(data.get("device", "cpu")),
            precision=str(data["precision"]).lower(),
        )


@dataclass(frozen=True)
class PrecisionStream:
    precision: str
    ranks: tuple[int, ...]
    bytes_per_element: int

    def estimated_bytes(self, tensor_elements: int) -> int:
        return len(self.ranks) * tensor_elements * self.bytes_per_element


@dataclass(frozen=True)
class MachineGroup:
    machine: str
    ranks: tuple[int, ...]
    leader_rank: int


@dataclass(frozen=True)
class PrecisionFlowPlan:
    job_name: str
    world_size: int
    precision_streams: tuple[PrecisionStream, ...]
    machine_groups: tuple[MachineGroup, ...]
    stages: tuple[str, ...]
