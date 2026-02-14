#!/usr/bin/env python3
"""
Merge performance data from three sources:
1. Solver (analytical model)
2. Benchmark (real AWS measurements)
3. Simulator (Vidur)
"""

import pandas as pd
import os

# File paths
SOLVER_FILE = "/mnt/projects/LLM_placement_solver/solution_summaries_merged.csv"
BENCHMARK_FILE = "/mnt/projects/tandemn-profiling/roofline/results/benchmark_results_merged_concise.csv"
SIMULATOR_FILE = "/mnt/projects/vidur/output/summary.csv"
OUTPUT_FILE = "/mnt/projects/LLM_placement_solver/unified_performance_summary.csv"

def load_solver_data():
    """Load and preprocess solver data."""
    df = pd.read_csv(SOLVER_FILE)
    df['source'] = 'solver'
    df['gpu_type'] = df['device_type'].apply(map_aws_to_gpu)
    # Note: solver already has 'pp' column, 'pipeline_stages' is separate info
    return df

def map_aws_to_gpu(device_type):
    """Map AWS instance type to GPU type. Returns all unique GPU types for mixed configs."""
    if pd.isna(device_type):
        return None
    device_str = str(device_type).lower()

    # Collect all GPU types present (order matters for consistent output)
    gpu_types = []

    # Check in specific order to avoid substring conflicts (g6e before g6, p4de before p4d)
    if 'p4de' in device_str or 'p4d' in device_str:
        gpu_types.append('A100')
    if 'p3dn' in device_str or 'p3' in device_str:
        gpu_types.append('V100')
    if 'g6e' in device_str:
        gpu_types.append('L40S')
    elif 'g6' in device_str:  # only if g6e not present
        gpu_types.append('L4')
    if 'g5' in device_str:
        gpu_types.append('A10G')

    if not gpu_types:
        return None

    # Return comma-separated list of unique GPU types
    return ','.join(gpu_types)

def load_benchmark_data():
    """Load and preprocess benchmark data."""
    df = pd.read_csv(BENCHMARK_FILE)
    df['source'] = 'benchmark'
    df['gpu_type'] = df['device_type'].apply(map_aws_to_gpu)
    return df

def load_simulator_data():
    """Load and preprocess simulator data."""
    df = pd.read_csv(SIMULATOR_FILE)
    df['source'] = 'simulator'
    # Rename columns to unified names
    df = df.rename(columns={
        'device': 'device_type',
        'tp_size': 'tp',
        'pp_stages': 'pp',
        'avg_input_tokens': 'max_input_length',
        'avg_output_tokens': 'max_output_length',
        'total_tps': 'total_tokens_per_sec',
        'input_tps': 'input_tokens_per_sec',
        'output_tps': 'output_tokens_per_sec',
    })
    # For simulator, device_type is already the GPU type - uppercase it
    df['gpu_type'] = df['device_type'].str.upper()
    return df

def merge_dataframes(dfs):
    """Merge dataframes, keeping all columns."""
    # Concatenate all dataframes
    merged = pd.concat(dfs, ignore_index=True, sort=False)

    # Get all columns
    all_columns = list(merged.columns)

    # Define preferred column order (common/important columns first)
    priority_columns = [
        'source',
        'model_name',
        'device_type',
        'gpu_type',
        'tp',
        'pp',
        'pipeline_stages',
        'tp_per_stage',
        'num_gpus',
        'num_replicas',
        'max_input_length',
        'max_output_length',
        'batch_size',
        'total_tokens_per_sec',
        'input_tokens_per_sec',
        'output_tokens_per_sec',
        'cost_per_hour',
        'dollar_per_million_token',
        'total_cost',
        'mem_per_gpu_gb',
    ]

    # Build final column order
    final_columns = []
    remaining = set(all_columns)
    for col in priority_columns:
        if col in remaining:
            final_columns.append(col)
            remaining.remove(col)
    # Add remaining columns alphabetically
    final_columns.extend(sorted(remaining))

    # Reorder columns
    merged = merged[final_columns]

    return merged

def main():
    print("Loading solver data...")
    solver_df = load_solver_data()
    print(f"  Loaded {len(solver_df)} rows, {len(solver_df.columns)} columns")

    print("Loading benchmark data...")
    benchmark_df = load_benchmark_data()
    print(f"  Loaded {len(benchmark_df)} rows, {len(benchmark_df.columns)} columns")

    print("Loading simulator data...")
    simulator_df = load_simulator_data()
    print(f"  Loaded {len(simulator_df)} rows, {len(simulator_df.columns)} columns")

    print("\nMerging dataframes...")
    merged_df = merge_dataframes([solver_df, benchmark_df, simulator_df])
    print(f"  Total: {len(merged_df)} rows, {len(merged_df.columns)} columns")

    print(f"\nSaving to {OUTPUT_FILE}...")
    merged_df.to_csv(OUTPUT_FILE, index=False)
    print("Done!")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nRows per source:")
    print(merged_df['source'].value_counts().to_string())

    print(f"\nAll columns in unified file:")
    for i, col in enumerate(merged_df.columns):
        print(f"  {i+1:2d}. {col}")

    print(f"\nColumn coverage by source:")
    for source in ['solver', 'benchmark', 'simulator']:
        source_df = merged_df[merged_df['source'] == source]
        non_null_cols = source_df.columns[source_df.notna().any()].tolist()
        print(f"\n  {source}: {len(non_null_cols)} columns with data")

if __name__ == "__main__":
    main()
