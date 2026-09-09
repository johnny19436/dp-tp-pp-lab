# Extending this lab to 8×GPU (and beyond)

This repo was measured on **2×A100 NV12**. The same scripts are intended to grow to an **8×GPU** node (or multi-node later) with minimal changes.

## What stays the same

- Model flags in `scripts/common.sh` (`NUM_LAYERS`, `HIDDEN_SIZE`, …)
- Mock-data Megatron entrypoint (`pretrain_gpt.py`)
- Result parsing in `analysis/summarize.py` / `analysis/generate_report.py`

## What you change on 8 GPUs

1. **World size / launch**
   - Today scripts hard-code `nproc_per_node=1` or `2`.
   - Introduce env vars, e.g. `NPROC_PER_NODE=8`, and pass them to `run_megatron`.

2. **Parallel layout examples (single node, 8 GPUs)**

   | Goal | Suggested layout | Notes |
   |------|------------------|-------|
   | Max throughput, model fits | `TP=1 PP=1 DP=8` | Strong/weak DP scaling study |
   | Memory headroom | `TP=2 PP=1 DP=4` or `TP=4 PP=1 DP=2` | More collectives than pure DP |
   | Deep model / capacity | `TP=1 PP=2 DP=4`, `TP=1 PP=4 DP=2`, `TP=2 PP=2 DP=2` | Watch pipeline bubble vs microbatches |
   | Balanced hybrid | `TP=2 PP=2 DP=2` | Common 8-GPU starter |

   Remember: `DP = world_size / (TP × PP)` (ignoring CP/EP).

3. **Topology**
   - Re-run `scripts/00_system_check.sh` and NCCL tests with `-g 8` (or MPI multi-process) on the new node.
   - NVLink domains matter: 8×A100 SXM is usually fully NVLinked within the node; PCIe-only boxes will change DP/TP efficiency a lot.

4. **Batch / memory**
   - Keep `global_batch_size` divisible by `DP × micro_batch_size` (and by PP microbatch count).
   - On 8 GPUs you may raise `global_batch_size` for weak scaling while freezing the model.

5. **Do not enable yet** (same as original plan): multi-node RDMA focus, FSDP/ZeRO, FP8, distributed optimizer overlap experiments — add those as separate study branches.

## Bootstrap on a fresh machine

```bash
git clone <this-repo> dp-tp-pp-lab && cd dp-tp-pp-lab
bash scripts/setup_third_party.sh   # Megatron pin + nccl-tests build
bash scripts/00_system_check.sh
bash scripts/01_nccl_benchmark.sh   # edit -g for GPU count when extending
```
