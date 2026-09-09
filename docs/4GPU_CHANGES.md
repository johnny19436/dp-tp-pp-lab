# 4-GPU Experiment: Reuse / Generalize / New Files

## Reused unchanged (2-GPU artifacts preserved)

- `results/training_results.csv`, `results/nccl_results.csv`
- `logs/*` (2-GPU training / system / NCCL logs)
- `profiles/nsys_*` (2-GPU Nsight traces)
- `report.pdf`, `analysis/generate_report.py`, `analysis/summarize.py`
- Original scripts: `scripts/10_baseline_1gpu.sh` … `scripts/15_pp_microbatch_sweep.sh`, `scripts/profile_nsys.sh`, `run_all.sh`
- `plan.txt`, `README.md` (historical 2-GPU docs)

## Generalized

- `scripts/common.sh` — optional `EXPERIMENT=4gpu` namespace for log/result/profile dirs; `apply_model_preset` for `original` vs `large` model; dmon still works for 1–4 GPUs via existing helpers used by new scripts
- `scripts/setup_third_party.sh` — fixed `third_party` path to repo root (was incorrectly under `scripts/`)

## New (4-GPU namespace)

| Path | Role |
|------|------|
| `plan_4gpu.txt` | Experiment plan |
| `scripts/4gpu/*.sh` | 4-GPU system/NCCL/training/profile launches |
| `scripts/common.sh` (extensions) | Model presets + experiment dirs |
| `run_all_4gpu.sh` | End-to-end 4-GPU runner |
| `analysis/4gpu/summarize_4gpu.py` | Parse logs → CSVs |
| `analysis/4gpu/generate_report_4gpu.py` | Plots + `report_4gpu.pdf` |
| `results/4gpu/` | NCCL / training / capacity CSVs |
| `logs/4gpu/` | Raw logs + dmon |
| `profiles/4gpu/` | Nsight traces |
| `reports/4gpu/report_4gpu.pdf` | Final report |
| `docs/4GPU_CHANGES.md` | This note |

## Frozen large model (after capacity validation)

See `results/4gpu/large_model_config.txt` once Phase 4–5 complete.
