# PrecisionFlow Connect

PrecisionFlow Connect checks whether a multi-node PyTorch training environment is ready to run. It covers the runtime pieces that usually need to be correct before launching a real distributed job: Docker image setup, `torchrun` launch arguments, rank mapping, GPU visibility, backend initialization, network interface binding, collective communication, and per-device precision capability.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

For live GPU checks, install a PyTorch build that matches the CUDA/NCCL runtime on the cluster:

```bash
python -m pip install -e ".[torch]"
```

## Quick Start

Run a local preflight check with the example manifest:

```bash
python -m precisionflow_lab connect --manifest configs/multinode_2x4.json --anonymize-hostnames
```

Generate a Docker and `torchrun` launch plan:

```bash
python -m precisionflow_lab framework configs/multinode_2x4.json \
  --master-addr 192.0.2.10 \
  --master-port 29500 \
  --image precisionflow-connect:gpu \
  --network-interface ib0
```

Diagnose a saved failure report:

```bash
python -m precisionflow_lab doctor examples/failure_report.json
```

Run tests:

```bash
python -m unittest discover -s tests
```

## What The Check Covers

- `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, and `LOCAL_WORLD_SIZE`.
- Rank-to-machine and rank-to-device mapping from a cluster manifest.
- NCCL or Gloo process-group initialization.
- Barrier, all-reduce, and all-gather collective smoke tests.
- Network interface visibility and optional `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` binding.
- GPU capability probing for `fp32`, `tf32`, `fp16`, `bf16`, `fp8`, and `int8`.
- JSON and Markdown reports for later review.

## Docker

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

For GPU clusters, adapt `docker/compose.gpu-template.yml` and the manifest in `configs/`.

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
python -m precisionflow_lab doctor examples/failure_report.json
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
src/precisionflow_lab/            CLI, runtime checks, planner, diagnosis, report rendering
tests/                            unit tests
```
