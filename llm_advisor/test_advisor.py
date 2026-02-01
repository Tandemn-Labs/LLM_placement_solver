#!/usr/bin/env python3
"""
Test script for the LLM Advisor.

Run with: python -m llm_advisor.test_advisor
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_advisor import (
    LLMAdvisor,
    GPUPool,
    WorkloadSpec,
    PerfDataLoader,
    create_advisor,
)


def test_data_loading():
    """Test that performance data loads correctly."""
    print("=" * 60)
    print("TEST: Data Loading")
    print("=" * 60)

    loader = PerfDataLoader()
    summary = loader.get_summary()

    print(f"\nTotal entries: {summary['total_entries']}")
    print(f"By source: {summary['by_source']}")
    print(f"GPU types: {summary['gpu_types']}")
    print(f"Models: {summary['models'][:5]}...")  # First 5

    # Test querying
    l40s_entries = loader.find_by_gpu_type("L40S")
    print(f"\nL40S entries: {len(l40s_entries)}")

    a100_entries = loader.find_by_gpu_type("A100")
    print(f"A100 entries: {len(a100_entries)}")

    # Test finding relevant entries
    relevant = loader.find_relevant_entries(
        model_name="70b",
        gpu_type="L40S",
        input_length_range=(1024, 4096),
        output_length_range=(256, 1024),
        max_results=5,
    )
    print(f"\nRelevant entries for 70B on L40S: {len(relevant)}")
    for e in relevant[:3]:
        print(f"  - {e.source}: TP={e.tp}, PP={e.pp}, {e.total_tokens_per_sec:.1f} tok/s")

    return True


def test_prompt_generation():
    """Test that prompt generation works correctly."""
    print("\n" + "=" * 60)
    print("TEST: Prompt Generation")
    print("=" * 60)

    advisor = create_advisor()

    # Define test scenario
    model_name = "llama-70b"
    gpu_pool = GPUPool(resources={"L40S": 8, "A10G": 4})
    workload = WorkloadSpec(
        input_length=2048,
        output_length=512,
        batch_size=8,
    )

    # Get prompt
    prompt = advisor.get_prompt_only(model_name, gpu_pool, workload)

    print(f"\nGenerated prompt length: {len(prompt)} characters")
    print("\n--- PROMPT PREVIEW (first 3000 chars) ---\n")
    print(prompt[:3000])
    print("\n--- END PREVIEW ---\n")

    # Save full prompt to file
    prompt_file = Path(__file__).parent.parent / "test_prompt.txt"
    with open(prompt_file, "w") as f:
        f.write(prompt)
    print(f"Full prompt saved to: {prompt_file}")

    return True


def test_recommendation_mock():
    """Test recommendation with mock response (no API key needed)."""
    print("\n" + "=" * 60)
    print("TEST: Recommendation (Mock)")
    print("=" * 60)

    # Create advisor without API key for mock mode
    advisor = LLMAdvisor(api_key=None)

    model_name = "deepseek-70b"
    gpu_pool = GPUPool(resources={"L40S": 8})
    workload = WorkloadSpec(
        input_length=2048,
        output_length=512,
        batch_size=1,
    )

    rec = advisor.get_recommendation(model_name, gpu_pool, workload)

    print(f"\nRecommendation:")
    print(f"  GPU: {rec.gpu_type}")
    print(f"  TP: {rec.tp}, PP: {rec.pp}")
    print(f"  Total GPUs: {rec.num_gpus}")
    print(f"  Replicas: {rec.replicas}")
    print(f"  Confidence: {rec.confidence}")
    print(f"  Warnings: {rec.warnings}")

    return True


def test_recommendation_real():
    """Test recommendation with Anthropic API (requires ANTHROPIC_API_KEY)."""
    print("\n" + "=" * 60)
    print("TEST: Recommendation (Real API)")
    print("=" * 60)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set. Skipping real API test.")
        return True

    advisor = create_advisor(api_key=api_key)

    model_name = "llama-70b"
    gpu_pool = GPUPool(resources={"L40S": 8, "A100": 4})
    workload = WorkloadSpec(
        input_length=2048,
        output_length=512,
        batch_size=8,
        target_throughput=1000,  # 1000 tok/s target
    )

    print(f"\nQuerying LLM for recommendation...")
    print(f"  Model: {model_name}")
    print(f"  GPU Pool: {gpu_pool.to_string()}")
    print(f"  Workload: {workload.input_length}in/{workload.output_length}out, batch={workload.batch_size}")

    rec = advisor.get_recommendation(model_name, gpu_pool, workload)

    print(f"\n--- RECOMMENDATION ---")
    print(f"  GPU: {rec.gpu_type}")
    print(f"  TP: {rec.tp}, PP: {rec.pp}")
    print(f"  Total GPUs: {rec.num_gpus}")
    print(f"  Replicas: {rec.replicas}")
    print(f"  Confidence: {rec.confidence}")
    print(f"  Predicted throughput: {rec.predicted_throughput} tok/s")
    print(f"  Warnings: {rec.warnings}")
    print(f"\n--- REASONING ---")
    print(rec.reasoning[:1500] if len(rec.reasoning) > 1500 else rec.reasoning)

    return True


# def test_recommendation_openai_real():
#     """Test recommendation with OpenAI API (requires OPENAI_API_KEY)."""
#     print("\n" + "=" * 60)
#     print("TEST: Recommendation (OpenAI API)")
#     print("=" * 60)

#     api_key = os.environ.get("OPENAI_API_KEY")
#     if not api_key:
#         print("OPENAI_API_KEY not set. Skipping OpenAI API test.")
#         return True

#     advisor = create_advisor(api_key=api_key, provider="openai")

#     model_name = "llama-70b"
#     gpu_pool = GPUPool(resources={"L40S": 8, "A100": 4})
#     workload = WorkloadSpec(
#         input_length=2048,
#         output_length=512,
#         batch_size=8,
#         target_throughput=1000,  # 1000 tok/s target
#     )

#     print(f"\nQuerying LLM for recommendation...")
#     print(f"  Model: {model_name}")
#     print(f"  GPU Pool: {gpu_pool.to_string()}")
#     print(f"  Workload: {workload.input_length}in/{workload.output_length}out, batch={workload.batch_size}")

#     rec = advisor.get_recommendation(model_name, gpu_pool, workload)

#     print(f"\n--- RECOMMENDATION ---")
#     print(f"  GPU: {rec.gpu_type}")
#     print(f"  TP: {rec.tp}, PP={rec.pp}")
#     print(f"  Total GPUs: {rec.num_gpus}")
#     print(f"  Replicas: {rec.replicas}")
#     print(f"  Confidence: {rec.confidence}")
#     print(f"  Predicted throughput: {rec.predicted_throughput} tok/s")
#     print(f"  Warnings: {rec.warnings}")
#     print(f"\n--- REASONING ---")
#     print(rec.reasoning[:1500] if len(rec.reasoning) > 1500 else rec.reasoning)

#     return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("LLM ADVISOR TEST SUITE")
    print("=" * 60)

    tests = [
        ("Data Loading", test_data_loading),
        ("Prompt Generation", test_prompt_generation),
        ("Recommendation (Mock)", test_recommendation_mock),
        ("Recommendation (Anthropic API)", test_recommendation_real),
        ("Recommendation (OpenAI API)", test_recommendation_openai_real),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "ERROR"))

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    for name, status in results:
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
