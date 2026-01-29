"""
LLM Advisor - AI-powered GPU configuration selection for LLM inference.

Uses sparse performance data from multiple sources (benchmarks, simulator, solver)
combined with GPU specifications to make informed recommendations.
"""

from .gpu_specs import (
    GPU_SPECS,
    MODEL_REQUIREMENTS,
    get_gpu_spec,
    estimate_model_size,
    format_gpu_specs_for_prompt,
)

from .perf_data import (
    PerfDataLoader,
    PerfEntry,
    SOURCE_TRUST,
    load_perf_data,
    format_entries_for_prompt,
)

from .advisor import (
    LLMAdvisor,
    GPUPool,
    WorkloadSpec,
    ConfigRecommendation,
    create_advisor,
    quick_recommend,
)

__all__ = [
    # GPU specs
    "GPU_SPECS",
    "MODEL_REQUIREMENTS",
    "get_gpu_spec",
    "estimate_model_size",
    "format_gpu_specs_for_prompt",
    # Performance data
    "PerfDataLoader",
    "PerfEntry",
    "SOURCE_TRUST",
    "load_perf_data",
    "format_entries_for_prompt",
    # Advisor
    "LLMAdvisor",
    "GPUPool",
    "WorkloadSpec",
    "ConfigRecommendation",
    "create_advisor",
    "quick_recommend",
]

__version__ = "0.1.0"
