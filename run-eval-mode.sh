#!/bin/bash

set -euo pipefail

# Sweep throughput-only evaluation mode for different instance families, TP, PP, and IO lengths.
# This script DOES NOT run the solver optimization; it only uses --evaluate-throughput.
#
# New directory structure:
# - Config: Uses base config directories (e.g., config/large/config.csv)
# - Results: Go to eval_results/ directory (separate from solver optimization results)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLVER="${SCRIPT_DIR}/solver.py"
CONFIG_ROOT="${SCRIPT_DIR}/config"

# Config directories (must contain config.csv)
config_dir_list=(
  "${CONFIG_ROOT}/large"
)

# Workload phases
workload_phase_list=("aggregated")

# Input/output token length pairs: "input output"
# io_length_pairs=("1024 256" "2048 512" "4096 1024" "8192 2048" "16384 4096")
io_length_pairs=("2048 512")

# Batch size pairs: "min max" (use min=max for fixed batch)
batch_size_pairs=("32 32")

# Optional fixed TP/PP lists (leave empty to auto-limit by instance)
tp_degree_list=()
pp_stage_list=()
# tp_degree_list=(1 2 4 8)
# pp_stage_list=(1 2 4 8)

run_on_background=true
cloud_provider="AWS"
eval_output_root="${SCRIPT_DIR}/eval_results"

timestamp=$(date +%Y%m%d_%H%M%S)

# Build list of instance families from gpu_pool.csv + cloud_instances_specs.csv
instance_info=$(
python3 - <<'PY' "${CONFIG_ROOT}/gpu_pool.csv" "${CONFIG_ROOT}/cloud_instances_specs.csv"
import csv
import os
import re
import sys

pool_path = sys.argv[1]
specs_path = sys.argv[2]

specs = {}
if os.path.exists(specs_path):
    with open(specs_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("Instance Name") or row.get("instance_name") or "").strip()
            if not name:
                continue
            num_gpus = row.get("num_gpus")
            if num_gpus in (None, "", "nan"):
                gpu_count_str = str(row.get("GPU Count", ""))
                nums = re.findall(r"\d+", gpu_count_str)
                num_gpus = nums[0] if nums else None
            try:
                specs[name] = int(float(num_gpus))
            except Exception:
                continue

pool_counts = {}
with open(pool_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = (row.get("instance_name") or "").strip()
        if not name:
            continue
        pool_counts[name] = pool_counts.get(name, 0) + int(float(row.get("count", 0)))

for name in sorted(pool_counts.keys()):
    num_gpus = specs.get(name)
    if num_gpus is None:
        continue
    print(f"{name},{pool_counts[name]},{num_gpus}")
PY
)

while IFS=',' read -r instance_family instance_count gpus_per_instance; do
  if [ -z "${instance_family}" ]; then
    continue
  fi

  # Auto TP degrees unless overridden
  if [ "${#tp_degree_list[@]}" -gt 0 ]; then
    tp_degrees=("${tp_degree_list[@]}")
  else
    tp_degrees=()
    # for d in 1 2 4 8 16; do
    for d in 1 2 4 8; do
    # for d in 4; do
      if [ "${d}" -le "${gpus_per_instance}" ]; then
        tp_degrees+=("${d}")
      fi
    done
  fi

  # Auto PP stages unless overridden
  if [ "${#pp_stage_list[@]}" -gt 0 ]; then
    pp_stages_list=("${pp_stage_list[@]}")
  else
    pp_stages_list=()
    # for p in 1 2 4 8 16; do
    for p in 1 2 4; do
    # for p in 1 2; do
      if [ "${p}" -le "${instance_count}" ]; then
        pp_stages_list+=("${p}")
      fi
    done
  fi

  for config_dir in "${config_dir_list[@]}"; do
    for workload_phase in "${workload_phase_list[@]}"; do
      for io_pair in "${io_length_pairs[@]}"; do
        input_token_length=$(echo "${io_pair}" | awk '{print $1}')
        output_token_length=$(echo "${io_pair}" | awk '{print $2}')
        for batch_pair in "${batch_size_pairs[@]}"; do
          min_batch_size=$(echo "${batch_pair}" | awk '{print $1}')
          max_batch_size=$(echo "${batch_pair}" | awk '{print $2}')
          for tp_degree in "${tp_degrees[@]}"; do
            for pp_stages in "${pp_stages_list[@]}"; do
              output_log_dir="${eval_output_root}/${config_dir##*/}/in${input_token_length}-out${output_token_length}-bs${min_batch_size}/eval_family-${instance_family}-pp${pp_stages}-tp${tp_degree}-${timestamp}"
              mkdir -p "${output_log_dir}"
              output_log_path="${output_log_dir}/output.txt"
              echo "** Starting eval: family=${instance_family} pp=${pp_stages} tp=${tp_degree} in=${input_token_length} out=${output_token_length} bs=${min_batch_size}-${max_batch_size}"
              if [ "${run_on_background}" = true ]; then
                python3 "${SOLVER}" \
                  --config-dir "${config_dir}" \
                  --output-dir "${output_log_dir}" \
                  --cloud-provider "${cloud_provider}" \
                  --sequence-length "${input_token_length}" \
                  --workload-phase "${workload_phase}" \
                  --output-length "${output_token_length}" \
                  --min-batch-size "${min_batch_size}" \
                  --max-batch-size "${max_batch_size}" \
                  --evaluate-throughput \
                  --eval-instance-family "${instance_family}" \
                  --eval-tp-degree "${tp_degree}" \
                  --eval-pp-stages "${pp_stages}" \
                  &> "${output_log_path}" &
              else
                python3 "${SOLVER}" \
                  --config-dir "${config_dir}" \
                  --output-dir "${output_log_dir}" \
                  --cloud-provider "${cloud_provider}" \
                  --sequence-length "${input_token_length}" \
                  --workload-phase "${workload_phase}" \
                  --output-length "${output_token_length}" \
                  --min-batch-size "${min_batch_size}" \
                  --max-batch-size "${max_batch_size}" \
                  --evaluate-throughput \
                  --eval-instance-family "${instance_family}" \
                  --eval-tp-degree "${tp_degree}" \
                  --eval-pp-stages "${pp_stages}" \
                  &> "${output_log_path}"
              fi
            done
          done
        done
      done
    done
  done
done <<< "${instance_info}"

