from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal DDP handoff example for PrecisionFlow Connect.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend)

    rank = dist.get_rank() if dist.is_initialized() else 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    torch.manual_seed(7 + rank)
    model = torch.nn.Linear(16, 4).to(device)
    if dist.is_initialized():
        if device.type == "cuda":
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        else:
            model = torch.nn.parallel.DistributedDataParallel(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    target = torch.zeros(args.batch_size, 4, device=device)
    last_loss = None

    for _ in range(args.epochs):
        inputs = torch.randn(args.batch_size, 16, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), target)
        loss.backward()
        optimizer.step()
        last_loss = loss.detach()

    if last_loss is None:
        last_loss = torch.tensor(0.0, device=device)
    reduced_loss = last_loss.clone()
    if dist.is_initialized():
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
        reduced_loss = reduced_loss / dist.get_world_size()

    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "backend": backend,
                    "world_size": world_size,
                    "epochs": args.epochs,
                    "mean_loss": round(float(reduced_loss.item()), 6),
                },
                indent=2,
            )
        )

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
