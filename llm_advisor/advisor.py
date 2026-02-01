"""
LLM-based GPU configuration advisor.
Uses sparse performance data + GPU specs to recommend optimal configurations.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json
import os

DEFAULT_LLM_PROVIDER = "anthropic"
DEFAULT_LLM_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o-mini",
}
PROVIDER_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _normalize_provider(provider: Optional[str]) -> str:
    return (provider or DEFAULT_LLM_PROVIDER).strip().lower()

from .gpu_specs import GPU_SPECS, format_gpu_specs_for_prompt, estimate_model_size
from .perf_data import (
    PerfDataLoader, PerfEntry, InfeasibleEntry, DataSourceContext,
    format_entries_for_prompt, format_infeasible_for_prompt, SOURCE_TRUST
)
from .model_arch import (
    get_architecture_for_model, format_architecture_context, compare_models
)


@dataclass
class GPUPool:
    """Represents available GPU resources."""
    resources: Dict[str, int]  # gpu_type -> count

    def to_string(self) -> str:
        """Format as readable string."""
        parts = []
        for gpu_type, count in sorted(self.resources.items()):
            parts.append(f"{count}x {gpu_type}")
        return ", ".join(parts)

    def get_gpu_types(self) -> List[str]:
        """Get list of available GPU types."""
        return list(self.resources.keys())


@dataclass
class WorkloadSpec:
    """Specification of the workload."""
    input_length: int
    output_length: int
    batch_size: int = 1
    num_requests: Optional[int] = None  # Total requests to process
    target_throughput: Optional[float] = None  # Desired tok/s
    slo_seconds: Optional[float] = None  # Max latency SLO


@dataclass
class ConfigRecommendation:
    """A configuration recommendation from the advisor."""
    gpu_type: str
    tp: int
    pp: int
    num_gpus: int
    replicas: int
    confidence: str  # HIGH, MEDIUM, LOW
    reasoning: str
    predicted_throughput: Optional[float] = None
    predicted_cost: Optional[float] = None
    warnings: List[str] = None

    def to_dict(self) -> dict:
        return {
            "gpu_type": self.gpu_type,
            "tp": self.tp,
            "pp": self.pp,
            "num_gpus": self.num_gpus,
            "replicas": self.replicas,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "predicted_throughput": self.predicted_throughput,
            "predicted_cost": self.predicted_cost,
            "warnings": self.warnings or [],
        }


class LLMAdvisor:
    """
    LLM-based advisor for GPU configuration selection.

    Uses sparse performance data from multiple sources (benchmarks, simulator, solver)
    combined with GPU specifications to make informed recommendations.
    """

    def __init__(
        self,
        perf_data: PerfDataLoader = None,
        api_key: str = None,
        provider: str = DEFAULT_LLM_PROVIDER,
        model: Optional[str] = None,
    ):
        """
        Initialize the advisor.

        Args:
            perf_data: Performance data loader (will create default if None)
            api_key: API key for LLM (defaults to provider env var)
            provider: LLM provider ("anthropic" or "openai")
            model: LLM model to use (provider-specific default if None)
        """
        self.perf_data = perf_data or PerfDataLoader()
        self.provider = _normalize_provider(provider)
        env_var = PROVIDER_API_KEY_ENV[self.provider]
        self.api_key = api_key or os.environ.get(env_var)
        self.model = model or DEFAULT_LLM_MODELS.get(self.provider, DEFAULT_LLM_MODELS[DEFAULT_LLM_PROVIDER])

        if not self.api_key:
            print(f"Warning: No API key provided. Set {env_var} or pass api_key.")

    def build_context(
        self,
        model_name: str,
        gpu_pool: GPUPool,
        workload: WorkloadSpec,
    ) -> Dict[str, Any]:
        """
        Build context for the LLM by gathering relevant information.

        Returns a structured context with:
        - GPU specifications
        - Relevant performance data
        - Model requirements
        - Workload details
        """
        context = {}

        # 1. Get GPU specs for available GPUs
        available_gpu_types = gpu_pool.get_gpu_types()
        context["gpu_specs"] = {
            gpu: GPU_SPECS.get(gpu, {}) for gpu in available_gpu_types
        }

        # 2. Find relevant performance data
        all_relevant = []
        for gpu_type in available_gpu_types:
            entries = self.perf_data.find_relevant_entries(
                model_name=model_name,
                gpu_type=gpu_type,
                input_length_range=(workload.input_length * 0.5, workload.input_length * 2),
                output_length_range=(workload.output_length * 0.5, workload.output_length * 2),
                max_results=15,
            )
            all_relevant.extend(entries)

        # Also get entries for the model regardless of workload match
        model_entries = self.perf_data.find_relevant_entries(
            model_name=model_name,
            max_results=20,
        )
        for e in model_entries:
            if e not in all_relevant:
                all_relevant.append(e)

        # Sort by trust and relevance
        all_relevant.sort(key=lambda e: (-e.trust_score, -e.total_tokens_per_sec))
        context["perf_data"] = all_relevant[:30]

        # 3. Get model size estimate
        context["model_info"] = estimate_model_size(model_name)

        # 4. Get available configs from data
        context["available_configs"] = {}
        for gpu_type in available_gpu_types:
            context["available_configs"][gpu_type] = self.perf_data.get_available_configs_for_gpu(gpu_type)

        # 5. Data coverage info
        context["data_summary"] = self.perf_data.get_summary()

        # 6. Find infeasible configurations (these are important constraints!)
        infeasible = []
        for gpu_type in available_gpu_types:
            entries = self.perf_data.find_infeasible_entries(
                model_name=model_name,
                gpu_type=gpu_type,
                input_length_range=(workload.input_length * 0.5, workload.input_length * 2),
                max_results=10,
            )
            infeasible.extend(entries)
        context["infeasible_configs"] = infeasible[:20]

        # 7. Get source contexts (methodology, limitations) - CRITICAL for skeptical reasoning
        context["source_contexts"] = self.perf_data.get_all_contexts()

        # 8. Get model architecture info from HuggingFace
        # Get target model architecture first
        target_arch = get_architecture_for_model(model_name)
        if target_arch:
            context["target_architecture"] = target_arch

        # Collect unique model names from relevant entries
        models_to_check = []
        if target_arch:
            models_to_check.append(target_arch.model_id)  # Use normalized model ID

        for entry in all_relevant[:15]:
            if entry.model_name and entry.model_name not in models_to_check:
                models_to_check.append(entry.model_name)

        # Also check vidur/solver model names which might be different
        unique_models = set()
        for entry in self.perf_data.entries:
            if entry.model_name and "70b" in entry.model_name.lower():
                unique_models.add(entry.model_name)
        for m in list(unique_models)[:5]:  # Add up to 5 more unique models
            if m not in models_to_check:
                models_to_check.append(m)

        context["model_architectures"] = format_architecture_context(models_to_check[:8])  # Limit to 8 models

        return context

    def build_prompt(
        self,
        model_name: str,
        gpu_pool: GPUPool,
        workload: WorkloadSpec,
        context: Dict[str, Any],
    ) -> str:
        """
        Build a comprehensive prompt for the LLM.

        This is critical - the prompt must provide:
        1. Clear task description
        2. Available data with trust levels
        3. Hardware specs
        4. Constraints and requirements
        5. Output format
        """
        prompt_parts = []

        # System context
        prompt_parts.append("""# GPU Configuration Advisor - Critical Analysis Required

You are an expert system for selecting optimal GPU configurations for LLM inference.
Your task is to recommend the best (gpu_type, tensor_parallelism, pipeline_parallelism, replicas) configuration.

## CRITICAL: Be Skeptical About Data Applicability

The performance data you have access to is SPARSE and comes from DIFFERENT contexts. You MUST think critically about whether each data point actually applies to the user's scenario.

### Model Equivalence Notes

**When models ARE performance-equivalent:**
- `DeepSeek-R1-Distill-Llama-70B` uses the SAME Llama-3 architecture as base `llama-3-70b`
- Distillation affects weights/output quality, NOT compute characteristics or memory requirements
- Benchmarks on distilled variants SHOULD transfer to base models of same architecture

**When models are NOT equivalent:**
- `Llama-2-70b` vs `Llama-3-70b` have different architectures (different layer dims, attention heads)
- `FP8` vs `FP16` precision dramatically affects memory (~50% less for FP8) and throughput
- Different serving engines (vLLM, TGI, TensorRT-LLM) have very different performance profiles
- Different model families (Llama vs Mistral vs Qwen) even at same param count

## Key Principles

1. **ONLY claim HIGH confidence if you have an EXACT match**: same model, same GPU, same workload, real benchmark
2. **Memory constraints are hard**: Model must fit in GPU memory with KV cache. Trust INFEASIBLE results.
3. **Tensor Parallelism (TP)**: Higher TP = more inter-GPU communication. Without NVLink, TP>4 often degrades.
4. **Pipeline Parallelism (PP)**: Has bubble overhead. Higher PP = more latency, but enables larger models.
5. **Extrapolation is RISKY**: If you're extrapolating from different models/GPUs/workloads, say so explicitly.
6. **Quantify uncertainty**: Give ranges, not point estimates, when data is sparse.
7. **Think critically**: Don't just pattern match - reason about WHY a data point does or doesn't apply.
""")

        # Add model architecture context (from HuggingFace)
        model_arch_context = context.get("model_architectures", "")
        if model_arch_context:
            prompt_parts.append(model_arch_context)

        # Add dynamically loaded source contexts
        source_contexts = context.get("source_contexts", {})
        if source_contexts:
            prompt_parts.append(self.perf_data.format_contexts_for_prompt())

        # Query section
        prompt_parts.append(f"""
## Your Task

Recommend the optimal GPU configuration for:

**Model**: {model_name}
**Available GPUs**: {gpu_pool.to_string()}
**Workload**:
  - Input length: {workload.input_length} tokens
  - Output length: {workload.output_length} tokens
  - Batch size: {workload.batch_size}
""")

        if workload.target_throughput:
            prompt_parts.append(f"  - Target throughput: {workload.target_throughput} tok/s")
        if workload.slo_seconds:
            prompt_parts.append(f"  - Latency SLO: {workload.slo_seconds}s")
        if workload.num_requests:
            prompt_parts.append(f"  - Total requests: {workload.num_requests}")

        # GPU Specs
        prompt_parts.append("\n" + format_gpu_specs_for_prompt(gpu_pool.get_gpu_types()))

        # Model info - use real architecture if available
        target_arch = context.get("target_architecture")
        model_info = context.get("model_info", {})

        if target_arch:
            prompt_parts.append(f"""
## Target Model Size (from HuggingFace config)

- Parameters: ~{target_arch.params_billions:.1f}B (computed from architecture)
- FP16 model size: ~{target_arch.fp16_memory_gb:.1f} GB
- Layers: {target_arch.num_hidden_layers}
- Hidden size: {target_arch.hidden_size}
- Attention heads: {target_arch.num_attention_heads} (KV heads: {target_arch.num_key_value_heads or target_arch.num_attention_heads})
- Max context: {target_arch.max_position_embeddings} tokens
""")
        elif model_info:
            prompt_parts.append(f"""
## Model Size Estimates (heuristic - no HuggingFace config found)

- Parameters: ~{model_info.get('params_billions', 'unknown')}B
- FP16 model size: ~{model_info.get('fp16_size_gb', 'unknown')} GB
- Minimum VRAM (with KV cache): ~{model_info.get('min_vram_inference_gb', 'unknown')} GB
- Recommended TP: {model_info.get('recommended_tp', 'varies')}
""")

        # Performance data
        perf_entries = context.get("perf_data", [])
        prompt_parts.append("\n" + format_entries_for_prompt(perf_entries, max_entries=25))

        # Available configs summary
        prompt_parts.append("\n## Available Configurations in Data\n")
        for gpu_type, configs in context.get("available_configs", {}).items():
            if configs:
                prompt_parts.append(f"\n**{gpu_type}**:")
                for cfg in configs[:10]:
                    sources = ", ".join(cfg["sources"])
                    prompt_parts.append(f"  - TP={cfg['tp']}, PP={cfg['pp']}: {cfg['data_points']} data points ({sources})")

        # Infeasible configurations (important constraints!)
        infeasible = context.get("infeasible_configs", [])
        if infeasible:
            prompt_parts.append(format_infeasible_for_prompt(infeasible, max_entries=10))

        # Output format
        prompt_parts.append("""

## Required Output Format

Provide your recommendation in the following JSON format, followed by your CRITICAL ANALYSIS:

```json
{
  "recommendation": {
    "gpu_type": "<GPU type>",
    "tp": <tensor parallelism>,
    "pp": <pipeline parallelism>,
    "num_gpus": <total GPUs needed = tp * pp * replicas>,
    "replicas": <number of model replicas>
  },
  "confidence": "<HIGH|MEDIUM|LOW>",
  "confidence_reasoning": "<why this confidence level - be honest about data gaps>",
  "predicted_throughput_range": {
    "low": <pessimistic estimate>,
    "mid": <expected>,
    "high": <optimistic estimate>
  },
  "key_assumptions": ["<list assumptions you're making that could be wrong>"],
  "warnings": ["<caveats, risks, things that could go wrong>"],
  "what_would_reduce_uncertainty": ["<what benchmarks or data would help>"]
}
```

Then provide CRITICAL ANALYSIS covering:

1. **Data Applicability Assessment**
   - For each data point you used: Is this actually applicable? Why or why not?
   - What's the model mismatch? (architecture, precision, version)
   - What's the setup mismatch? (serving engine, config, hardware)

2. **Reasoning Chain**
   - Why this GPU type? What are you assuming about performance scaling?
   - Why this TP/PP? What communication/memory tradeoffs are you considering?
   - What could go wrong with this recommendation?

3. **Uncertainty Quantification**
   - Where is your estimate most uncertain?
   - What's the worst case if your assumptions are wrong?

4. **Alternative Configurations**
   - What else should the user consider?
   - Under what conditions would a different config be better?

**CONFIDENCE GUIDELINES:**
- HIGH: ONLY if you have exact match (same model, same GPU, same workload) from real benchmark
- MEDIUM: Similar model/GPU with real data, or exact match from simulator
- LOW: Extrapolating from different models/GPUs, or using solver-only data

Be honest about what you don't know. A thoughtful "I'm uncertain because X" is more valuable than a confident wrong answer.
""")

        return "\n".join(prompt_parts)

    def get_recommendation(
        self,
        model_name: str,
        gpu_pool: GPUPool,
        workload: WorkloadSpec,
    ) -> ConfigRecommendation:
        """
        Get a configuration recommendation from the LLM.

        Args:
            model_name: Name of the LLM to deploy (e.g., "llama-70b")
            gpu_pool: Available GPU resources
            workload: Workload specification

        Returns:
            ConfigRecommendation with the suggested configuration
        """
        # Build context
        context = self.build_context(model_name, gpu_pool, workload)

        # Build prompt
        prompt = self.build_prompt(model_name, gpu_pool, workload, context)

        # Call LLM
        response_text = self._call_llm(prompt)

        # Parse response
        recommendation = self._parse_response(response_text)

        return recommendation

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        if not self.api_key:
            return self._mock_response(prompt)

        if self.provider == "anthropic":
            try:
                import anthropic

                client = anthropic.Anthropic(api_key=self.api_key)

                message = client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                return message.content[0].text

            except ImportError:
                print("Warning: anthropic package not installed. Using mock response.")
                return self._mock_response(prompt)
            except Exception as e:
                print(f"Error calling Anthropic: {e}")
                return self._mock_response(prompt)

        if self.provider == "openai":
            try:
                from openai import OpenAI

                client = OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=3000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                )
                return response.choices[0].message.content

            except ImportError:
                print("Warning: openai package not installed. Using mock response.")
                return self._mock_response(prompt)
            except Exception as e:
                print(f"Error calling OpenAI: {e}")
                return self._mock_response(prompt)

        print(f"Warning: Unknown provider '{self.provider}'. Using mock response.")
        return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> str:
        """Generate a mock response for testing without API."""
        api_hint = PROVIDER_API_KEY_ENV.get(self.provider, "ANTHROPIC_API_KEY")
        return """```json
{
  "recommendation": {
    "gpu_type": "L40S",
    "tp": 4,
    "pp": 1,
    "num_gpus": 4,
    "replicas": 1
  },
  "confidence": "MEDIUM",
  "predicted_throughput_tok_s": 500,
  "warnings": ["No API key - this is a mock response for testing"]
}
```

**Reasoning (MOCK)**:
This is a placeholder response. To get real recommendations, please set your """ + api_hint + "."

    def _parse_response(self, response_text: str) -> ConfigRecommendation:
        """Parse LLM response into ConfigRecommendation."""
        # Extract JSON block
        json_str = None
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        elif "{" in response_text:
            # Try to find JSON object
            start = response_text.find("{")
            # Find matching closing brace
            depth = 0
            for i, c in enumerate(response_text[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        json_str = response_text[start:i+1]
                        break

        if json_str:
            try:
                data = json.loads(json_str)
                rec = data.get("recommendation", data)

                # Extract reasoning (everything after JSON)
                reasoning_start = response_text.find("```", response_text.find("```json") + 7) + 3
                reasoning = response_text[reasoning_start:].strip() if reasoning_start > 3 else ""

                return ConfigRecommendation(
                    gpu_type=rec.get("gpu_type", "unknown"),
                    tp=rec.get("tp", 1),
                    pp=rec.get("pp", 1),
                    num_gpus=rec.get("num_gpus", 1),
                    replicas=rec.get("replicas", 1),
                    confidence=data.get("confidence", "LOW"),
                    reasoning=reasoning,
                    predicted_throughput=data.get("predicted_throughput_tok_s"),
                    warnings=data.get("warnings", []),
                )
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")

        # Fallback
        return ConfigRecommendation(
            gpu_type="unknown",
            tp=1,
            pp=1,
            num_gpus=1,
            replicas=1,
            confidence="LOW",
            reasoning=f"Failed to parse response:\n{response_text[:500]}",
            warnings=["Failed to parse LLM response"],
        )

    def get_prompt_only(
        self,
        model_name: str,
        gpu_pool: GPUPool,
        workload: WorkloadSpec,
    ) -> str:
        """
        Get just the prompt (for debugging or using with other LLMs).
        """
        context = self.build_context(model_name, gpu_pool, workload)
        return self.build_prompt(model_name, gpu_pool, workload, context)


# Convenience functions
def create_advisor(
    csv_path: str = None,
    api_key: str = None,
    provider: str = DEFAULT_LLM_PROVIDER,
    llm_model: Optional[str] = None,
) -> LLMAdvisor:
    """Create an LLM advisor with optional custom data path."""
    perf_data = PerfDataLoader(csv_path) if csv_path else PerfDataLoader()
    return LLMAdvisor(perf_data=perf_data, api_key=api_key, provider=provider, model=llm_model)


def quick_recommend(
    model_name: str,
    gpu_pool: Dict[str, int],
    input_length: int,
    output_length: int,
    batch_size: int = 1,
    api_key: Optional[str] = None,
    provider: str = DEFAULT_LLM_PROVIDER,
    llm_model: Optional[str] = None,
) -> ConfigRecommendation:
    """
    Quick recommendation function.

    Example:
        result = quick_recommend(
            model_name="llama-70b",
            gpu_pool={"L40S": 8, "A10G": 4},
            input_length=2048,
            output_length=512,
        )
    """
    advisor = create_advisor(api_key=api_key, provider=provider, llm_model=llm_model)
    return advisor.get_recommendation(
        model_name=model_name,
        gpu_pool=GPUPool(resources=gpu_pool),
        workload=WorkloadSpec(
            input_length=input_length,
            output_length=output_length,
            batch_size=batch_size,
        ),
    )
