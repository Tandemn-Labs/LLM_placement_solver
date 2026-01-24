#!/usr/bin/env python3
"""
Generate network_bandwidth.csv from instance specs and gpu_pool.csv.

Fail-fast behavior:
- Missing instance in cloud_instances_specs.csv
- Missing/invalid num_gpus, interconnect, or external bandwidth
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd


def parse_first_number(value: str) -> float | None:
    numbers = re.findall(r"[\d.]+", str(value))
    return float(numbers[0]) if numbers else None


def parse_internal_bw_gbps(value: str) -> float:
    """
    Parse internal GPU interconnect bandwidth.
    Returns GB/s (not Gbps). Requires a concrete numeric value.
    """
    value = str(value)
    if not value or value == "nan":
        raise ValueError("Missing Internal GPU Interconnect Bandwidth")

    # Examples:
    # "600 GB/s NVLink per GPU | 600 GB/s NVSwitch total"
    # "32 GB/s PCIe 4.0 x16"
    # "900 GB/s NVLink per GPU | 3.6 TB/s NVSwitch total"
    match = re.search(r"([\d.]+)\s*(GB/s|TB/s)", value)
    if not match:
        raise ValueError(f"Unrecognized internal bandwidth format: {value}")

    bw = float(match.group(1))
    unit = match.group(2)
    if unit == "TB/s":
        bw *= 1000.0
    return bw


def parse_external_bw_gbps(value: str) -> float:
    """
    Parse external network bandwidth.
    Returns Gbps. Requires a concrete numeric value.
    """
    value = str(value)
    if not value or value == "nan":
        raise ValueError("Missing External Network Bandwidth")

    match = re.search(r"([\d.]+)\s*Gbps", value)
    if not match:
        raise ValueError(f"Unrecognized external bandwidth format: {value}")
    return float(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate network_bandwidth.csv from instance specs.")
    parser.add_argument("--gpu-pool", required=True, help="Path to gpu_pool.csv (instance_name,count)")
    parser.add_argument("--cloud-specs", required=True, help="Path to cloud_instances_specs.csv")
    parser.add_argument("--output", required=True, help="Output path for network_bandwidth.csv")
    parser.add_argument("--cloud-provider", default=None, help="Filter by cloud provider (e.g., AWS)")
    args = parser.parse_args()

    gpu_pool_path = Path(args.gpu_pool)
    cloud_specs_path = Path(args.cloud_specs)
    output_path = Path(args.output)

    if not gpu_pool_path.exists():
        raise FileNotFoundError(f"gpu_pool.csv not found: {gpu_pool_path}")
    if not cloud_specs_path.exists():
        raise FileNotFoundError(f"cloud_instances_specs.csv not found: {cloud_specs_path}")

    pool_df = pd.read_csv(gpu_pool_path)
    if "instance_name" not in pool_df.columns or "count" not in pool_df.columns:
        raise ValueError("gpu_pool.csv must have columns: instance_name,count")

    specs_df = pd.read_csv(cloud_specs_path)
    if args.cloud_provider:
        specs_df = specs_df[specs_df["Cloud Provider"] == args.cloud_provider]
        if specs_df.empty:
            raise ValueError(f"No entries for cloud provider: {args.cloud_provider}")

    # Build lookup by instance name
    specs_by_instance = {}
    for _, row in specs_df.iterrows():
        instance_name = row["Instance Name"]
        specs_by_instance[instance_name] = row

    # Expand GPUs
    gpu_nodes = []
    for _, row in pool_df.iterrows():
        instance_name = row["instance_name"]
        instance_count = int(row["count"])
        if instance_name not in specs_by_instance:
            raise ValueError(f"Instance not found in cloud specs: {instance_name}")

        spec = specs_by_instance[instance_name]
        num_gpus = parse_first_number(spec.get("GPU Count", ""))
        if not num_gpus or num_gpus < 1:
            raise ValueError(f"Invalid GPU Count for {instance_name}: {spec.get('GPU Count')}")

        internal_bw_gbps = parse_internal_bw_gbps(spec.get("Internal GPU Interconnect Bandwidth", ""))
        external_bw_gbps = parse_external_bw_gbps(spec.get("External Network Bandwidth", ""))

        for inst_idx in range(instance_count):
            for gpu_idx in range(int(num_gpus)):
                gpu_nodes.append(
                    {
                        "instance_name": instance_name,
                        "instance_idx": inst_idx,
                        "gpu_idx": gpu_idx,
                        "internal_bw_gbps": internal_bw_gbps,
                        "external_bw_gbps": external_bw_gbps,
                        "num_gpus": int(num_gpus),
                    }
                )

    total_gpus = len(gpu_nodes)
    if total_gpus == 0:
        raise ValueError("No GPUs found in gpu_pool.csv")

    # Build bandwidth matrix (GB/s)
    matrix = [[0.0] * total_gpus for _ in range(total_gpus)]
    for i, src in enumerate(gpu_nodes):
        for j, dst in enumerate(gpu_nodes):
            if i == j:
                matrix[i][j] = 10000.0
                continue
            same_instance = (
                src["instance_name"] == dst["instance_name"]
                and src["instance_idx"] == dst["instance_idx"]
            )
            if same_instance:
                # Intra-node GPU-GPU bandwidth
                matrix[i][j] = src["internal_bw_gbps"]
            else:
                # Inter-node bandwidth (convert Gbps to GB/s and split across GPUs per node)
                per_gpu_bw_src = src["external_bw_gbps"] / src["num_gpus"] / 8.0
                per_gpu_bw_dst = dst["external_bw_gbps"] / dst["num_gpus"] / 8.0
                matrix[i][j] = min(per_gpu_bw_src, per_gpu_bw_dst)

    # Write CSV
    gpu_ids = [f"gpu_{i}" for i in range(total_gpus)]
    df_out = pd.DataFrame(matrix, index=gpu_ids, columns=gpu_ids)
    df_out.to_csv(output_path)
    print(f"Wrote {total_gpus}x{total_gpus} network matrix to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

