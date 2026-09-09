# dp-tp-pp-lab

# Single-Node 2×A100 DP / TP / PP Lab

Reproducible Megatron-LM study of **Data Parallel / Tensor Parallel / Pipeline Parallel** on one NVLink node. Measured on 2×A100; designed to extend to **8×GPU** (see [`docs/EXTENDING_8GPU.md`](docs/EXTENDING_8GPU.md)).

Full write-up with plots: [`report.pdf`](report.pdf).

## Repo layout

```
dp-tp-pp-lab/
├── README.md
├── report.pdf                 # detailed report (read this)
├── plan.txt                   # original experiment plan
├── run_all.sh                 # end-to-end runner
├── scripts/                   # system check, NCCL, Megatron launches, nsys
├── analysis/                  # summarize.py, generate_report.py, figures/
├── results/                   # training_results.csv, nccl_results.csv
├── logs/                      # raw training + dmon logs
├── profiles/                  # Nsight .nsys-rep + textual stats
├── docs/EXTENDING_8GPU.md
└── third_party/PINS.txt       # pinned Megatron / nccl-tests revisions
```

Third-party sources (`Megatron-LM`, `nccl-tests`) are **not** vendored (too large / regenerable). Fetch with:

```bash
bash scripts/setup_third_party.sh
```

## Quick start (new machine)

```bash
git clone <this-repo-url> dp-tp-pp-lab && cd dp-tp-pp-lab
bash scripts/setup_third_party.sh
bash scripts/00_system_check.sh
bash scripts/03_megatron_smoke.sh    # tiny 2-GPU smoke
# bash run_all.sh                    # full suite (long)
python3 analysis/generate_report.py
```

## Hardware / software

| Item | Value |
|------|-------|
| GPUs | 2 × NVIDIA A100-SXM4-40GB |
| Topology | GPU0 ↔ GPU1 = **NV12** (12 NVLinks @ 25 GB/s each) |
| Driver | 580.159.03 |
| CUDA toolkit | 13.1 |
| PyTorch | 2.10.0a0+a36e1d39eb.nv26.01 (NGC) |
| Megatron-LM | `be85fc5df550f2236a850ecbf0dd1bf1b4cb4814` |
| Precision | BF16 |
| Data | `--mock-data` (no dataset I/O) |

Confirmed via `logs/00_system_check.log` and `nvidia-smi topo -m`.

## Frozen model (plan §5)

```
num_layers=24, hidden_size=2048, ffn_hidden_size=8192, num_attention_heads=16
seq_length=2048, vocab_size=32000
micro_batch_size=1 (unless noted), global_batch_size=64
warmup=20, measured=100 (train-iters=120)
```

1-GPU peak allocated ≈ **34.7 GB** — fits on one 40 GB A100 (no size reduction needed).

## NCCL NVLink baseline

From `results/nccl_results.csv` (out-of-place, 2 GPUs):

| Collective | 1 GiB busbw (GB/s) | Notes |
|------------|-------------------:|-------|
| AllReduce | ~194 | DP gradient sync |
| AllGather | ~149 | TP-related |
| ReduceScatter | ~164 | TP-related |

Raw logs: `logs/01_nccl_benchmark.log`.

## Main results

Tokens/step = `global_batch_size × seq_length` = 64 × 2048 = **131072** (128 × 2048 for weak scaling).

| Config | GPUs | TP | PP | DP | Tokens/s | Speedup | Tokens/s/GPU | Peak alloc (MB) | Avg SM util |
|--------|-----:|---:|---:|---:|---------:|--------:|-------------:|----------------:|------------:|
| 1GPU | 1 | 1 | 1 | 1 | 16173 | 1.00× | 16173 | 34682 | 96% / — |
| DP2 | 2 | 1 | 1 | 2 | 31666 | **1.96×** | 15833 | 34682 | 94% / 94% |
| TP2 | 2 | 2 | 1 | 1 | 19621 | 1.21× | 9810 | **17884** | 74% / 95% |
| PP2 mbs=1 | 2 | 1 | 2 | 1 | 28453 | 1.76× | 14227 | 19268 | 94% / 88% |
| PP2 mbs=2 | 2 | 1 | 2 | 1 | 32376 | 2.00× | 16188 | 23763 | 93% / 92% |
| PP2 mbs=4 | 2 | 1 | 2 | 1 | 33591 | 2.08× | 16795 | 32371 | 93% / 93% |

CSV: `results/training_results.csv`. Per-run logs + `*.dmon.csv` under `logs/`.

### DP weak scaling

| Config | global_batch | mean step (ms) |
|--------|-------------:|---------------:|
| 1GPU | 64 | 8105 |
| DP2 weak | 128 | 8159 |

Weak-scaling efficiency ≈ `8105 / 8159` ≈ **99.3%** (step time nearly unchanged when work/GPU is held constant).

### Communication pattern (Nsight)

Profiles: `profiles/nsys_{dp2,tp2,pp2}.nsys-rep` (+ `*_stats.txt`).

| Config | Dominant NCCL in kernel summary | Pattern |
|--------|----------------------------------|---------|
| DP2 | AllReduce ~**1.3%** of GPU kernel time | Rare gradient sync after backward |
| TP2 | AllReduce ~**37%** of GPU kernel time | Frequent collectives inside every layer |
| PP2 | SendRecv ~**7%** | P2P activation/gradient handoff + idle/bubble |

| Config | Main communication | Main bottleneck |
|--------|--------------------|-----------------|
| DP2 | gradient AllReduce | sync (small on NV12) |
| TP2 | frequent tensor collectives | communication latency/bandwidth |
| PP2 | activation/gradient P2P | pipeline bubble (improves with more microbatches) |

## Answers to plan §16 questions

1. **How much faster is DP=2 than one GPU?**  
   **1.96×** tokens/s (31666 / 16173).

2. **What is DP scaling efficiency?**  
   `1.96 / 2` = **98%** strong-scaling efficiency. Weak scaling ≈ **99.3%**.

3. **For the same global batch, is TP=2 faster than one GPU?**  
   **Yes, but only modestly** — **1.21×** (19621 vs 16173 tokens/s). Step time drops from ~8.1 s to ~6.7 s.

4. **Is TP=2 more or less efficient than DP=2?**  
   **Less efficient.** TP2 = 1.21× vs DP2 = 1.96×; tokens/s/GPU is 9810 vs 15833.

5. **How much VRAM does TP save?**  
   Peak max-allocated **34682 → 17884 MB** (~**48%** less / ~1.94× reduction per GPU).

6. **How much VRAM does PP save?**  
   With mbs=1: **34682 → 19268 MB** (~**44%** less). Savings shrink as microbatch grows (mbs=4 peaks at ~32 GB) because activations grow with microbatch size.

7. **How does PP throughput change with microbatches?**  
   mbs 1 → 2 → 4: **28453 → 32376 → 33591** tokens/s. Larger microbatches raise GEMM efficiency enough to outweigh a slightly larger bubble fraction on PP=2.

8. **What do DP / TP / PP look like in an Nsight timeline?**  
   - **DP:** long GEMM/FlashAttention stretches; short AllReduce bursts near optimizer/grad sync.  
   - **TP:** GEMM ↔ NCCL AllReduce interleaved densely (NCCL ~37% of kernel time).  
   - **PP:** SendRecv between stages; stages offset in time (pipeline bubble / drain).

9. **Which strategy communicates most frequently?**  
   **TP** (tens of thousands of AllReduce kernel instances in the profile window).

10. **Why is DP normally preferred when the model already fits on one GPU?**  
    Highest aggregate throughput with minimal communication (one AllReduce of grads per step). On this NV12 node, DP reached **98%** scaling efficiency.

11. **Why might TP still be useful even if the model fits?**  
    Cuts per-GPU parameter/activation memory (~half here), enables larger models or batches later, and is a building block for multi-dimensional parallelism — at the cost of much more communication.

12. **Why is PP mainly useful for larger models / larger GPU counts?**  
    It splits layers across stages so a model that does not fit on one GPU can fit across stages. On 2 GPUs the bubble is manageable with many microbatches; with deeper pipelines (more stages) bubble control and scheduling matter more, which is where PP’s value grows.

## How to reproduce

```bash
cd /root/working/dp-tp-pp-lab
bash run_all.sh                 # full suite
# or individually:
bash scripts/00_system_check.sh
bash scripts/01_nccl_benchmark.sh
python3 -m torch.distributed.run --standalone --nproc_per_node=2 scripts/02_torch_distributed_test.py
bash scripts/03_megatron_smoke.sh
bash scripts/10_baseline_1gpu.sh
bash scripts/11_dp2.sh
bash scripts/14_dp_weak_scaling.sh
bash scripts/12_tp2.sh
MICRO_BATCH_SIZE=1 bash scripts/13_pp2.sh
bash scripts/15_pp_microbatch_sweep.sh
bash scripts/profile_nsys.sh dp2   # also tp2 / pp2
python3 analysis/summarize.py
```

## Definition of done checklist

- [x] 2×A100 + NV12 confirmed  
- [x] NCCL AllReduce / AllGather / ReduceScatter  
- [x] 1GPU, DP2, TP2, PP2 training  
- [x] DP strong + weak scaling  
- [x] PP microbatch sweep  
- [x] GPU util + peak VRAM  
- [x] Three Nsight traces  
- [x] `training_results.csv` + `nccl_results.csv`  
- [x] This README with conclusions  

Optional capacity OOM sweep (plan §17) was **not** run, so the main benchmark schedule is not delayed.
