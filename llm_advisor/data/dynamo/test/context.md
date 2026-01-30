# Benchmark Methodology (H200_TP1P_TP1D)

This file explains how the benchmark numbers were obtained, using only verified
repo logic and documented facts. It avoids speculation about deployment or
cluster details that are not present in this directory.

## Documented Run Metadata
From `tests/planner/README.md`, this directory is described as:
- Model: `nvidia/Llama-3.1-8B-Instruct-FP8`
- Hardware: H200
- Max context length: 16384
- Parallelization: TP1 Prefill, TP1 Decode
- Example slice: ISL/OSL 3000/150, TTFT ~80 ms, ITL ~10 ms

## Measurement Workflow (code-verified)

## Prefill-Only vs Decode-Only (what it means)
The profiling pipeline removes the other worker and runs a single worker type:
- **Prefill-only**: a single worker runs prefill measurements; no separate decode worker is deployed.
- **Decode-only**: a single decode worker is deployed; the prefill worker is removed.

This is enforced by config modifiers that delete the other service and adjust
worker arguments (e.g., enable prefix/radix caching or KV block reuse).

### Prefill Sweep
For a set of ISL values between 100 and `max_context_length`, the profiler:
1) Sends synthetic requests to the model endpoint via AIPerf.
2) Extracts TTFT.
3) Computes throughput per GPU:
   `throughput = ISL / TTFT / num_gpus * 1000 * attention_dp_size`

Implementation:
- `benchmarks/profiler/utils/profile_prefill.py`

### Decode Sweep
For a grid over ISL and concurrency:
1) Computes `max_concurrency = max_kv_tokens / (ISL + OSL)`.
2) Sweeps `num_request` values derived from `max_concurrency`.
3) Uses AIPerf to measure ITL and throughput per GPU at each point.
4) Stores:
   - `x_kv_usage = (ISL + OSL/2) * num_request / max_kv_tokens`
   - `y_context_length = ISL + OSL/2`

Implementation:
- `benchmarks/profiler/utils/profile_decode.py`

### Where KV Cache Comes From in Decode-Only
Decode-only does **not** mean “no prefill.” It means there is **no separate prefill service**.
Each request still includes an input prompt (ISL), and the decode worker itself
performs the prefill step locally to build KV. With prefix/radix caching or KV
block reuse enabled, the decode worker can reuse KV across requests with shared
prefixes, reducing prefill cost but not eliminating it.

### AIPerf Invocation
The profiler builds `aiperf profile` CLI commands with synthetic ISL/OSL,
concurrency, and request counts. The command construction is in:
- `benchmarks/profiler/utils/aiperf.py`

## Outputs Produced
The sweep outputs are recorded in:
- `selected_prefill_interpolation/raw_data.npz`
- `selected_decode_interpolation/raw_data.npz`

These arrays are directly exported to CSV without modification.

## Missing Scientific Provenance (not present here)
The following details are required for full reproducibility but are not captured
in this directory:
- Deployment YAML/config used for profiling
- Container image tags/digests and runtime versions
- GPU driver, CUDA, NCCL versions
- Cluster topology and network fabric
- Environment variables and engine runtime flags
- AIPerf artifacts/logs and version info

