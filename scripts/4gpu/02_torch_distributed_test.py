#!/usr/bin/env python3
"""4-GPU torch.distributed sanity: all_reduce on all ranks."""
import os
import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    t = torch.ones(1024, device="cuda") * (rank + 1)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = world * (world + 1) / 2.0
    ok = bool(torch.allclose(t, torch.full_like(t, expected)))
    print(f"rank={rank}/{world} local={local} ok={ok} sample={t[0].item()} expected={expected}")
    dist.barrier()
    if rank == 0:
        print("torch.distributed 4-GPU all_reduce PASSED" if ok else "FAILED")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
