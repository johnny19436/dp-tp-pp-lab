#!/usr/bin/env python3
"""Minimal torch.distributed NCCL sanity check on 2 GPUs."""
import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))

    print(
        f"rank {rank} local_rank {local_rank} world_size {world_size} "
        f"-> cuda:{local_rank} ({torch.cuda.get_device_name(local_rank)})",
        flush=True,
    )
    dist.barrier()

    t = torch.ones(1, device=f"cuda:{local_rank}") * (rank + 1)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world_size + 1))
    got = int(t.item())
    print(f"rank {rank} all_reduce sum={got} expected={expected}", flush=True)
    assert got == expected
    dist.barrier()
    if rank == 0:
        print("torch.distributed sanity check PASSED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
