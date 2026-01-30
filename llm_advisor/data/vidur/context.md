## Vidur Simulator Benchmark Context

This document describes how the numbers in `output/summary.csv` were generated,
based on the Vidur simulator code and the provided `run.sh`. It does not claim
any real hardware measurements beyond what the simulator’s profiling inputs
encode.

### How the numbers were generated
- Entry point: `python -m vidur.main` (see `/mnt/projects/vidur/run.sh`).
- The simulator builds a `SimulationConfig` from CLI args, constructs a
  simulated cluster, request generator, and scheduler, then runs a discrete
  event simulation (`vidur/simulator.py`).
- Metrics are computed from per‑request logs and summarized by
  `/mnt/projects/vidur/summarize_sim.py` into `output/summary.csv`.

### Simulator technique (high level)
- Execution time is **predicted** from profiling datasets (CSV files) rather
  than measured on live hardware.
- Profiling inputs used by the execution‑time predictor are pulled from:
  - `data/profiling/compute/{DEVICE}/{MODEL}/mlp.csv`
  - `data/profiling/compute/{DEVICE}/{MODEL}/attention.csv`
  - `data/profiling/network/{NETWORK_DEVICE}/all_reduce.csv`
  - `data/profiling/network/{NETWORK_DEVICE}/send_recv.csv`
  - `data/profiling/cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads.csv`
  (see `vidur/config/config.py` and `docs/profiling.md`).

### Workload and configuration used in `run.sh`
From `/mnt/projects/vidur/run.sh`:
- Model: `meta-llama/Llama-2-70b-hf`
- Device: `a100`
- Network device: `a100_dgx`
- Replicas: `1`
- Tensor parallel: `tp=4` and `tp=8`
- Pipeline stages: default `1` (not overridden in `run.sh`)
- Requests: synthetic
  - `num_requests=512`
  - Lengths: normal distribution with
    - avg prefill tokens: 1024 / 2048 / 4096 / 8192 / 16384
    - avg decode tokens: 512 / 1024 / 1024 / 2048 / 4096
    - std prefill/decode tokens: 0
  - Arrival process: Poisson with `qps=10`
- Scheduler: `sarathi`

### What the outputs represent
`output/summary.csv` aggregates `request_metrics.csv` for each run directory:
- Throughput (input/output/total tokens per second, requests per second)
- Latency distributions (E2E, TTFT, TPOT) at P50/P90/P99
- Total tokens processed and simulated duration

These are **simulated** metrics based on the profiling data and scheduler model,
not measurements from a running cluster.

### Hardware / network context used by the simulator
The simulator maps `device` and `network_device` to profiling datasets:
- `device=a100` → A100 compute profiling data
- `network_device=a100_dgx` → network profiling data labeled for A100 DGX

Per `docs/profiling.md`, `a100_dgx` refers to an A100 DGX node with 8 A100 GPUs
and full NVLink connectivity. This is a **profiling dataset label**, not a live
cluster used in these runs.

### Reliability / limitations (from code + docs)
- Fidelity depends on the quality of profiling datasets and execution‑time
  predictors (see `vidur/execution_time_predictor/*`).
- It does **not** measure real cluster behavior (no driver/CUDA/NCCL versions,
  no runtime kernel behavior, no live networking).
- The simulator is suitable for comparative analysis and capacity planning, but
  results should be validated on real deployments for production decisions.

