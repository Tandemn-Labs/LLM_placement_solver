# Pre-Swept Results Methodology

This document explains how numbers in `pre_swept_results/` are produced, based
on verified repo logic. It does not infer environment or deployment details that
are not captured here.

## What `pre_swept_results` Contains
`pre_swept_results/` stores aggregated NPZ files organized by:
`{gpu_type}/{framework}/{model}/{mode}.npz` where `mode` is `prefill` or `decode`.

Each NPZ can contain **multiple configurations** (rows) with associated metrics.

## How NPZs Are Built (code-verified)
Raw benchmark outputs are merged into the pre-swept NPZs using:
- `components/src/dynamo/planner/utils/pre_swept_results_utils.py`

Key steps:
1) A raw NPZ (from a profiling run) is validated to include required config keys.
2) It is merged into an existing `pre_swept_results/.../{mode}.npz` or copied
   as the initial file if none exists.

Relevant code:
- `merge_raw_data(...)` in `pre_swept_results_utils.py`
- `PrefillNpz` and `DecodeNpz` enforce required config keys.

## Required Config Keys
Every row in a pre-swept NPZ carries the following config fields:
- `gpu_type`
- `model`
- `framework`
- `framework_version`
- `tp`
- `dp`
- `pp`
- `block_size`
- `max_batch_size`
- `gpu_count`

## Metrics Stored
Prefill NPZ:
- `prefill_isl`, `prefill_ttft`, `prefill_thpt_per_gpu`

Decode NPZ:
- `x_kv_usage`, `y_context_length`, `z_itl`, `z_thpt_per_gpu`, `max_kv_tokens`

## What Is Not Captured Here
`pre_swept_results` does **not** store:
- Deployment YAML or container images
- Driver/CUDA/NCCL versions
- Cluster topology or network fabric
- Environment variables
- Benchmark tool versions/logs

If you need full reproducibility, those must be provided separately.

