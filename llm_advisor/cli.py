#!/usr/bin/env python3
"""
Simple CLI for GPU configuration recommendations.

Usage:
    python -m llm_advisor.cli --model llama-70b --gpu-pool config/gpu_pool.csv --input-len 2048 --output-len 512 --provider openai --llm-model gpt-4o-mini --api-key sk-...
"""

import argparse
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_advisor import create_advisor, GPUPool, WorkloadSpec

# AWS instance -> (GPU type, GPUs per instance)
INSTANCE_TO_GPU = {
    "g5.4xlarge": ("A10G", 1), "g5.12xlarge": ("A10G", 4), "g5.24xlarge": ("A10G", 4), "g5.48xlarge": ("A10G", 8),
    "g6.12xlarge": ("L4", 4), "g6.24xlarge": ("L4", 4), "g6.48xlarge": ("L4", 8),
    "g6e.4xlarge": ("L40S", 1), "g6e.12xlarge": ("L40S", 4), "g6e.24xlarge": ("L40S", 4), "g6e.48xlarge": ("L40S", 8),
    "p3dn.24xlarge": ("V100", 8), "p4d.24xlarge": ("A100", 8), "p4de.24xlarge": ("A100", 8), "p5.48xlarge": ("H100", 8),
}


def load_gpu_pool(csv_path: str) -> GPUPool:
    """Load GPU pool from csv with columns: instance_name, count"""
    df = pd.read_csv(csv_path)
    resources = {}
    for _, row in df.iterrows():
        instance = row["instance_name"].strip()
        count = int(row["count"])
        if instance in INSTANCE_TO_GPU:
            gpu_type, gpus_per = INSTANCE_TO_GPU[instance]
            resources[gpu_type] = resources.get(gpu_type, 0) + count * gpus_per
        else:
            print(f"Warning: Unknown instance {instance}")
    return GPUPool(resources=resources)


def main():
    parser = argparse.ArgumentParser(description="Get GPU config recommendations for LLM deployment")
    parser.add_argument("--model", "-m", required=True, help="Model name to deploy (e.g., llama-70b)")
    parser.add_argument("--gpu-pool", "-g", required=True, help="Path to gpu_pool.csv")
    parser.add_argument("--input-len", "-i", type=int, required=True, help="Input length in tokens")
    parser.add_argument("--output-len", "-o", type=int, required=True, help="Output length in tokens")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--provider",choices=["anthropic", "openai"],default="anthropic",help="Advisor LLM provider (default: anthropic)")
    parser.add_argument("--llm-model", help="Advisor LLM model name (default depends on provider)")
    parser.add_argument("--api-key",help="LLM API key (defaults to provider env var: ANTHROPIC_API_KEY or OPENAI_API_KEY)")
    parser.add_argument("--prompt-only", action="store_true", help="Show prompt only, don't call LLM")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    gpu_pool = load_gpu_pool(args.gpu_pool)
    workload = WorkloadSpec(input_length=args.input_len, output_length=args.output_len, batch_size=args.batch_size)
    advisor = create_advisor(api_key=args.api_key, provider=args.provider, llm_model=args.llm_model)

    print(f"Model: {args.model}")
    print(f"GPU Pool: {gpu_pool.to_string()}")
    print(f"Workload: {args.input_len} in, {args.output_len} out, batch={args.batch_size}")
    print()

    if args.prompt_only:
        print(advisor.get_prompt_only(args.model, gpu_pool, workload))
        return

    rec = advisor.get_recommendation(args.model, gpu_pool, workload)

    if args.json:
        import json
        print(json.dumps(rec.to_dict(), indent=2))
    else:
        print("=" * 60)
        print(f"RECOMMENDATION: {rec.gpu_type}, TP={rec.tp}, PP={rec.pp}, {rec.num_gpus} GPUs")
        print(f"Confidence: {rec.confidence}")
        if rec.predicted_throughput:
            print(f"Predicted: {rec.predicted_throughput} tok/s")
        if rec.warnings:
            print(f"Warnings: {rec.warnings}")
        print("=" * 60)
        print(rec.reasoning)


if __name__ == "__main__":
    main()
