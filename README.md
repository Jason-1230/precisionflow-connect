# PrecisionFlow Connect

PrecisionFlow Connect checks whether a multi-node PyTorch training environment is ready to run. It validates Docker runtime setup, `torchrun` launch arguments, rank and device mapping, network interface binding, NCCL/Gloo backend initialization, collective communication, and per-device precision capability.

```text
cluster manifest
  -> launch planner              Docker + torchrun commands per node
  -> runtime probe               env, network, PyTorch distributed, CUDA devices
  -> collective smoke tests      barrier, all-reduce, all-gather
  -> precision matrix            fp32, tf32, fp16, bf16, fp8, int8 per visible GPU
  -> system report               JSON/Markdown report plus diagnosis findings
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m precisionflow_lab connect --manifest configs/multinode_2x4.json --anonymize-hostnames
python -m precisionflow_lab framework configs/multinode_2x4.json --network-interface ib0
python -m precisionflow_lab profile
python -m unittest discover -s tests
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m precisionflow_lab connect --manifest .\configs\multinode_2x4.json --anonymize-hostnames
python -m precisionflow_lab framework .\configs\multinode_2x4.json --network-interface ib0
python -m precisionflow_lab profile
python -m unittest discover -s tests
```

For live GPU checks, install a PyTorch build that matches the CUDA/NCCL runtime on the target cluster:

```bash
python -m pip install -e ".[torch]"
```

## Deployment Modes

| mode | command | output |
| --- | --- | --- |
| local preflight | `precisionflow-connect connect --manifest configs/multinode_2x4.json` | manifest, local env, network, and precision capability report |
| bare-metal multi-node | `torchrun ... -m precisionflow_lab connect --live` | NCCL/Gloo initialization and collective smoke test evidence |
| Docker runtime | `precisionflow-connect framework configs/multinode_2x4.json --image precisionflow-connect:gpu` | per-node Docker commands with host networking and GPU visibility |
| scheduler handoff | run the generated command inside Slurm, Kubernetes, or a cloud job step | the same JSON/Markdown report schema across launch environments |

## Capability Profile

The `profile` command prints the project-level deployment, readiness, and framework-handoff view:

```bash
precisionflow-connect profile
precisionflow-connect profile --json
```

The profile includes:

- deployment targets for local preflight, Docker runtime, bare-metal `torchrun`, and scheduler handoff;
- architecture layers from manifest parsing to report generation;
- readiness checks for launcher, rank mapping, network, backend, collectives, and precision;
- handoff notes for PyTorch `torchrun`, Hugging Face Accelerate, DeepSpeed, and scheduler-based jobs;
- a failure lifecycle: detect, classify, recommend, rerun, and archive.

## Readiness Checks

| area | checks | typical failure signal | next action |
| --- | --- | --- | --- |
| launcher | `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `LOCAL_RANK`, `WORLD_SIZE` | missing or invalid `torchrun` environment | regenerate launch commands and align `nnodes * nproc_per_node` with manifest `world_size` |
| rank mapping | contiguous ranks, machine groups, device assignment | manifest/runtime world size mismatch | fix the manifest or launch arguments before training |
| network | master endpoint, host interfaces, socket interface binding | unresolved master endpoint or unpinned multi-NIC host | bind `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` to the training network |
| backend | `torch.distributed`, NCCL, Gloo | backend initialization failure | check PyTorch build, CUDA visibility, driver/runtime versions, and backend consistency |
| collectives | barrier, all-reduce, all-gather | ranks hang or return inconsistent tensors | inspect rank consistency, tensor device placement, backend health, and network binding |
| precision | `fp32`, `tf32`, `fp16`, `bf16`, `fp8`, `int8` | requested precision unavailable on part of the cluster | gate precision paths by device capability and record heterogeneous precision rows |

## Docker Runtime

Build a CPU image for local checks:

```bash
docker build -f docker/Dockerfile -t precisionflow-connect:cpu .
```

Build a GPU image from a cluster-approved PyTorch/CUDA base image:

```bash
docker build -f docker/Dockerfile \
  --build-arg BASE_IMAGE=<cluster-approved-pytorch-cuda-image> \
  -t precisionflow-connect:gpu .
```

Run the local two-container Gloo example:

```bash
docker compose -f docker/compose.2node.gloo.yml up --abort-on-container-exit
```

Generate a per-node Docker and `torchrun` plan:

```bash
precisionflow-connect framework configs/multinode_2x4.json \
  --master-addr 192.0.2.10 \
  --master-port 29500 \
  --image precisionflow-connect:gpu \
  --network-interface ib0
```

## Live Multi-Node Run

Run `torchrun` on every node. Replace `192.0.2.10` with the rank-0 host address on the training network.

Rank-0 node:

```bash
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
  --master_addr=192.0.2.10 --master_port=29500 \
  -m precisionflow_lab connect --live --backend auto \
  --manifest configs/multinode_2x4.json \
  --anonymize-hostnames \
  --markdown-output reports/connect.md \
  --json-output reports/connect.json
```

Rank-1 node:

```bash
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \
  --master_addr=192.0.2.10 --master_port=29500 \
  -m precisionflow_lab connect --live --backend auto \
  --manifest configs/multinode_2x4.json \
  --anonymize-hostnames
```

Only rank 0 writes the final report. When a host has multiple NICs, set `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` to the intended training interface.

## Framework Handoff

| framework or launcher | how to use the report |
| --- | --- |
| PyTorch `torchrun` | run the probe with the same `nnodes`, `nproc_per_node`, master address, and backend before training |
| Hugging Face Accelerate | confirm mixed-precision support and multi-node communication before `accelerate launch` |
| DeepSpeed | archive the environment report before ZeRO, pipeline, or mixed-precision runs |
| Slurm, Kubernetes, cloud launchers | run the generated command inside the scheduled job and store the report with experiment artifacts |

## Example Report

```text
# PrecisionFlow Connect System Report

- Mode: live
- Overall status: PASS
- Declared world size: 8
- Backend: nccl
- Master endpoint: 192.0.2.10:29500

## Rank Mapping

| rank | host | local rank | runtime device | backend |
| --- | --- | --- | --- | --- |
| 0 | node-a | 0 | cuda:0 | nccl |
| 1 | node-a | 1 | cuda:1 | nccl |
| 2 | node-a | 2 | cuda:2 | nccl |
| 3 | node-a | 3 | cuda:3 | nccl |
| 4 | node-b | 0 | cuda:0 | nccl |
| 5 | node-b | 1 | cuda:1 | nccl |
| 6 | node-b | 2 | cuda:2 | nccl |
| 7 | node-b | 3 | cuda:3 | nccl |

## Precision Capability

| host | device | name | fp32 | tf32 | fp16 | bf16 | fp8 | int8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| node-a | cuda:0 | NVIDIA A100-SXM4-40GB | yes | yes | yes | yes | no | yes |
| node-b | cuda:0 | NVIDIA H100-SXM5-80GB | yes | yes | yes | yes | yes | yes |

## Collective Tests

| test | status | detail |
| --- | --- | --- |
| barrier | PASS | all ranks entered and exited barrier |
| all_reduce | PASS | sum expected 36, rank 0 observed 36 |
| all_gather | PASS | gathered ranks [0, 1, 2, 3, 4, 5, 6, 7] |
```

## Diagnosis Example

```bash
precisionflow-connect doctor examples/failure_report.json
```

Example output:

```text
| severity | code | area | recommendation |
| --- | --- | --- | --- |
| ERROR | WORLD_SIZE_MISMATCH | rank mapping | Make torchrun --nnodes * --nproc_per_node equal the manifest world_size, or update the manifest. |
| ERROR | BACKEND_INIT_FAILED | backend | Check backend choice, CUDA visibility, PyTorch distributed support, and network binding. |
| WARN | CUDA_DEVICE_NOT_VISIBLE | precision capability | Check CUDA_VISIBLE_DEVICES, driver/runtime installation, and whether the process is on a GPU node. |
```

## Project Layout

```text
configs/                         cluster manifests
docker/                          Dockerfile and Docker Compose templates
examples/                        sample failure report
scripts/                         local helper scripts
src/precisionflow_lab/            CLI, runtime checks, planner, diagnosis, profile, report rendering
tests/                            unit tests
```
