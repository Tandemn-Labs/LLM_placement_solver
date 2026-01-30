"""
Performance data loader with source trust levels and methodology context.
Loads sparse benchmark/simulator/solver data from organized subdirectories.
Each data source includes a context.md or script.py explaining how the data was generated.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import re


@dataclass
class DataSourceContext:
    """
    Rich context about a data source's methodology and limitations.
    Parsed from context.md files that accompany each data source.
    """
    source_name: str

    # Core methodology
    methodology_summary: str  # 1-2 sentence summary
    measurement_type: str  # "real_hardware", "simulation", "analytical_model"

    # Configuration details
    hardware_used: List[str] = field(default_factory=list)  # e.g., ["A100", "H200"]
    serving_engine: Optional[str] = None  # e.g., "vLLM", "TGI"
    engine_version: Optional[str] = None
    precision: Optional[str] = None  # e.g., "FP8", "FP16"
    models_tested: List[str] = field(default_factory=list)

    # Provenance
    workload_description: str = ""
    how_numbers_generated: str = ""  # Key methodology details

    # Limitations - CRITICAL for skeptical reasoning
    limitations: List[str] = field(default_factory=list)
    what_is_not_captured: List[str] = field(default_factory=list)
    reliability_notes: str = ""

    # Raw context for LLM (trimmed)
    raw_context: str = ""  # First ~2000 chars of context.md for LLM reference

    def format_for_prompt(self, max_length: int = 1500) -> str:
        """Format context for inclusion in LLM prompt."""
        lines = [f"### {self.source_name} Data Source Context"]
        lines.append(f"**Type**: {self.measurement_type}")
        lines.append(f"**Summary**: {self.methodology_summary}")

        if self.serving_engine:
            engine_str = self.serving_engine
            if self.engine_version:
                engine_str += f" {self.engine_version}"
            lines.append(f"**Engine**: {engine_str}")

        if self.precision:
            lines.append(f"**Precision**: {self.precision}")

        if self.hardware_used:
            lines.append(f"**Hardware**: {', '.join(self.hardware_used)}")

        if self.models_tested:
            lines.append(f"**Models**: {', '.join(self.models_tested[:3])}{'...' if len(self.models_tested) > 3 else ''}")

        if self.how_numbers_generated:
            lines.append(f"\n**How Generated**: {self.how_numbers_generated[:500]}")

        if self.limitations:
            lines.append("\n**LIMITATIONS (Read Carefully)**:")
            for lim in self.limitations[:5]:
                lines.append(f"  - {lim}")

        if self.what_is_not_captured:
            lines.append("\n**What Is NOT Captured**:")
            for item in self.what_is_not_captured[:3]:
                lines.append(f"  - {item}")

        if self.reliability_notes:
            lines.append(f"\n**Reliability Notes**: {self.reliability_notes[:300]}")

        result = "\n".join(lines)
        return result[:max_length] if len(result) > max_length else result


# Trust levels for different data sources
SOURCE_TRUST = {
    "benchmark": {
        "level": "HIGH",
        "score": 1.0,
        "measurement_type": "real_hardware",
        "description": "Real measurements from actual hardware (AWS). Most reliable for EXACT matches.",
    },
    "dynamo_swept": {
        "level": "HIGH",
        "score": 0.9,
        "measurement_type": "real_hardware",
        "description": "Pre-swept real measurements from Dynamo profiler. Narrow applicability (specific engine/model/GPU).",
    },
    "dynamo_test": {
        "level": "HIGH",
        "score": 0.9,
        "measurement_type": "real_hardware",
        "description": "Real hardware profiling from Dynamo test suite. Model-specific.",
    },
    "vidur": {
        "level": "MEDIUM",
        "score": 0.7,
        "measurement_type": "simulation",
        "description": "Vidur simulator predictions. Validated to ~9% error on some workloads.",
    },
    "solver": {
        "level": "LOW",
        "score": 0.4,
        "measurement_type": "analytical_model",
        "description": "Analytical roofline model. Approximation, may miss real-world effects. Trust INFEASIBLE results.",
    },
}


# Data source directory mapping
DATA_SOURCES = {
    "benchmark": {
        "subdir": "our_own_experiment",
        "data_file": "data.csv",
        "context_file": "script.py",  # Uses script instead of context.md
    },
    "solver": {
        "subdir": "solver_based",
        "data_file": "data.csv",
        "context_file": "context.md",
    },
    "vidur": {
        "subdir": "vidur",
        "data_file": "data.csv",
        "context_file": "context.md",
    },
    "dynamo_swept": {
        "subdir": "dynamo/swept",
        "data_file": "data.csv",
        "context_file": "context.md",
    },
    "dynamo_test": {
        "subdir": "dynamo/test",
        "data_file": None,  # Multiple CSVs
        "context_file": "context.md",
    },
}


@dataclass
class PerfEntry:
    """A single performance data entry."""
    source: str
    model_name: str
    gpu_type: str
    device_type: str  # Original device (e.g., g6e.48xlarge)
    tp: int
    pp: int
    input_length: float
    output_length: float
    total_tokens_per_sec: float
    input_tokens_per_sec: float
    output_tokens_per_sec: float
    batch_size: Optional[float] = None
    num_gpus: Optional[int] = None
    cost_per_hour: Optional[float] = None
    dollar_per_million_token: Optional[float] = None
    trust_level: str = "UNKNOWN"
    trust_score: float = 0.0
    # Context about the measurement setup
    serving_engine: Optional[str] = None  # e.g., "vLLM", "TGI", "TensorRT-LLM"
    engine_version: Optional[str] = None
    precision: Optional[str] = None  # e.g., "FP16", "FP8", "INT8"
    notes: Optional[str] = None  # Any caveats about this data point

    def __post_init__(self):
        """Set trust level based on source."""
        if self.source in SOURCE_TRUST:
            self.trust_level = SOURCE_TRUST[self.source]["level"]
            self.trust_score = SOURCE_TRUST[self.source]["score"]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "model_name": self.model_name,
            "gpu_type": self.gpu_type,
            "device_type": self.device_type,
            "tp": self.tp,
            "pp": self.pp,
            "input_length": self.input_length,
            "output_length": self.output_length,
            "total_tokens_per_sec": self.total_tokens_per_sec,
            "input_tokens_per_sec": self.input_tokens_per_sec,
            "output_tokens_per_sec": self.output_tokens_per_sec,
            "batch_size": self.batch_size,
            "num_gpus": self.num_gpus,
            "cost_per_hour": self.cost_per_hour,
            "dollar_per_million_token": self.dollar_per_million_token,
            "trust_level": self.trust_level,
            "trust_score": self.trust_score,
            "serving_engine": self.serving_engine,
            "engine_version": self.engine_version,
            "precision": self.precision,
            "notes": self.notes,
        }


@dataclass
class InfeasibleEntry:
    """An infeasible configuration with failure reason."""
    source: str
    model_name: str
    gpu_type: str
    instance_family: str
    tp: int
    pp: int
    input_length: float
    output_length: float
    batch_size: Optional[float]
    num_gpus: Optional[int]
    failure_reason: str  # Detailed reason why this config is infeasible

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "model_name": self.model_name,
            "gpu_type": self.gpu_type,
            "instance_family": self.instance_family,
            "tp": self.tp,
            "pp": self.pp,
            "input_length": self.input_length,
            "output_length": self.output_length,
            "batch_size": self.batch_size,
            "num_gpus": self.num_gpus,
            "failure_reason": self.failure_reason,
        }


class PerfDataLoader:
    """Load and query sparse performance data from organized subdirectories."""

    def __init__(self, data_dir: str = None):
        """Initialize with path to data directory."""
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"

        self.data_dir = Path(data_dir)
        self.entries: List[PerfEntry] = []
        self.infeasible_entries: List[InfeasibleEntry] = []
        self.source_contexts: Dict[str, DataSourceContext] = {}
        self._load_all_sources()

    def _load_all_sources(self):
        """Load data and context from all source subdirectories."""
        for source, config in DATA_SOURCES.items():
            source_dir = self.data_dir / config["subdir"]
            if not source_dir.exists():
                print(f"Warning: Source directory not found: {source_dir}")
                continue

            # Load context first
            context_file = config.get("context_file")
            if context_file:
                context_path = source_dir / context_file
                if context_path.exists():
                    self.source_contexts[source] = self._parse_context(source, context_path)

            # Load data
            if config["data_file"]:
                data_path = source_dir / config["data_file"]
                if data_path.exists():
                    self._load_source(source, data_path)
            else:
                # Multiple CSVs (e.g., dynamo_test)
                for csv_file in source_dir.glob("*.csv"):
                    self._load_source(source, csv_file)

        print(f"Loaded {len(self.entries)} performance entries, {len(self.infeasible_entries)} infeasible entries")
        print(f"Loaded context for {len(self.source_contexts)} sources: {list(self.source_contexts.keys())}")

    def _parse_context(self, source: str, context_path: Path) -> DataSourceContext:
        """Parse context.md or script.py to extract methodology and limitations."""
        try:
            content = context_path.read_text()
        except Exception as e:
            print(f"Warning: Could not read context file {context_path}: {e}")
            return self._default_context(source)

        # Different parsing for script.py vs context.md
        if context_path.suffix == ".py":
            return self._parse_script_context(source, content)
        else:
            return self._parse_markdown_context(source, content)

    def _parse_markdown_context(self, source: str, content: str) -> DataSourceContext:
        """Parse context.md file to extract structured information."""
        ctx = DataSourceContext(
            source_name=source,
            methodology_summary="",
            measurement_type=SOURCE_TRUST.get(source, {}).get("measurement_type", "unknown"),
            raw_context=content[:2500],  # Keep first 2500 chars for LLM reference
        )

        # Extract sections
        content_lower = content.lower()

        # Try to extract methodology summary from first paragraph
        lines = content.split('\n')
        for line in lines[1:10]:  # Skip title, look at first few lines
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('```'):
                ctx.methodology_summary = line[:200]
                break

        # Extract "How the numbers were generated" or "How NPZs Are Built" section
        how_patterns = [
            r'### How (?:the )?[Nn]umbers (?:[Aa]re |[Ww]ere )?[Gg]enerated(.*?)(?=###|\Z)',
            r'### How NPZs [Aa]re [Bb]uilt(.*?)(?=###|\Z)',
            r'## (?:Measurement|Profiling) [Ww]orkflow(.*?)(?=##|\Z)',
        ]
        for pattern in how_patterns:
            how_match = re.search(pattern, content, re.DOTALL)
            if how_match:
                how_text = how_match.group(1).strip()
                # Clean up code blocks
                how_text = re.sub(r'```[\s\S]*?```', '[code snippet]', how_text)
                ctx.how_numbers_generated = how_text[:800]
                break

        # Extract hardware/configuration
        if 'h100' in content_lower:
            ctx.hardware_used.append('H100')
        if 'h200' in content_lower:
            ctx.hardware_used.append('H200')
        if 'a100' in content_lower:
            ctx.hardware_used.append('A100')
        if 'a40' in content_lower or 'a10g' in content_lower:
            ctx.hardware_used.append('A10G/A40')
        if 'l40s' in content_lower:
            ctx.hardware_used.append('L40S')

        # Extract serving engine
        if 'vllm' in content_lower:
            ctx.serving_engine = 'vLLM'
            # Try to find version
            version_match = re.search(r'vllm[^\d]*(\d+\.\d+\.\d+(?:\.\d+)?)', content_lower)
            if version_match:
                ctx.engine_version = version_match.group(1)
        elif 'tgi' in content_lower or 'text-generation-inference' in content_lower:
            ctx.serving_engine = 'TGI'
        elif 'tensorrt' in content_lower:
            ctx.serving_engine = 'TensorRT-LLM'

        # Extract precision
        if 'fp8' in content_lower:
            ctx.precision = 'FP8'
        elif 'fp16' in content_lower or 'float16' in content_lower:
            ctx.precision = 'FP16'
        elif 'int8' in content_lower:
            ctx.precision = 'INT8'

        # Extract model names
        model_patterns = [
            r'llama[- ]?\d+[- ]?\d*b',
            r'Llama-\d+-\d+b-hf',
            r'deepseek[- ]\S+',
            r'nvidia/\S+',
            r'meta-llama/\S+',
        ]
        for pattern in model_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            ctx.models_tested.extend(matches[:3])
        ctx.models_tested = list(set(ctx.models_tested))[:5]

        # Extract limitations - try multiple section header patterns
        limitations_patterns = [
            r'### (?:Reliability|Limitations).*?(?=###|\Z)',
            r'## (?:Reliability|Limitations).*?(?=##|\Z)',
            r'### What [Ii]s [Nn]ot [Cc]aptured.*?(?=###|\Z)',
        ]
        for pattern in limitations_patterns:
            limitations_match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if limitations_match:
                lim_text = limitations_match.group(0)
                # Extract bullet points
                bullets = re.findall(r'[-*]\s*(.+?)(?=\n[-*]|\n\n|\Z)', lim_text, re.DOTALL)
                if bullets:
                    ctx.limitations = [b.strip()[:200] for b in bullets[:5]]
                    break

        # Extract "What Is NOT Captured" or "Missing" section
        not_captured_patterns = [
            r'### (?:What[^#]+NOT|Missing|What Is Not Captured).*?(?=###|\Z)',
            r'## (?:What[^#]+NOT|Missing).*?(?=##|\Z)',
        ]
        for pattern in not_captured_patterns:
            not_captured_match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if not_captured_match:
                not_text = not_captured_match.group(0)
                bullets = re.findall(r'[-*]\s*(.+?)(?=\n[-*]|\n\n|\Z)', not_text, re.DOTALL)
                if bullets:
                    ctx.what_is_not_captured = [b.strip()[:200] for b in bullets[:5]]
                    break

        # Extract reliability notes
        reliability_match = re.search(r'(?:reliability|fidelity|accuracy)[^.]*\.', content, re.IGNORECASE)
        if reliability_match:
            ctx.reliability_notes = reliability_match.group(0).strip()[:300]

        # Add source-specific default limitations if none found
        if not ctx.limitations:
            if "dynamo" in source.lower():
                ctx.limitations = [
                    "Pre-profiled results from specific engine version (check framework_version field)",
                    "FP8 quantized models only - NOT equivalent to FP16",
                    "Specific to H100/H200 hardware",
                ]
            elif "solver" in source.lower():
                ctx.limitations = [
                    "Analytical roofline model - pure math, no real execution",
                    "May overestimate throughput, especially for communication-bound configs",
                    "Does not capture real-world effects like memory fragmentation",
                ]
            elif "vidur" in source.lower():
                ctx.limitations = [
                    "Discrete event simulation, not real measurements",
                    "Accuracy depends on profiling dataset quality",
                    "Validated to ~9% error on some workloads, may be worse on edge cases",
                ]

        return ctx

    def _parse_script_context(self, source: str, content: str) -> DataSourceContext:
        """Parse script.py to extract context from code structure."""
        ctx = DataSourceContext(
            source_name=source,
            methodology_summary="Real hardware benchmarks collected via automated script on AWS instances.",
            measurement_type="real_hardware",
            raw_context=content[:2500],
        )

        # Extract GPU configs from script
        gpu_match = re.search(r'GPU_CONFIGS\s*=\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}', content, re.DOTALL)
        if gpu_match:
            gpu_text = gpu_match.group(0)
            if 'L40S' in gpu_text:
                ctx.hardware_used.append('L40S')
            if 'A100' in gpu_text:
                ctx.hardware_used.append('A100')
            if 'A10G' in gpu_text:
                ctx.hardware_used.append('A10G')
            if 'H100' in gpu_text:
                ctx.hardware_used.append('H100')

        ctx.how_numbers_generated = (
            "Benchmarks are collected by deploying models on AWS EC2 instances "
            "(g6e for L40S, g5 for A10G, p4d for A100, p5 for H100), "
            "running throughput measurements with controlled batch sizes and sequence lengths, "
            "and recording tokens/sec with cost information."
        )

        ctx.limitations = [
            "Real measurements but specific to AWS instance types and network topology",
            "Model deployment uses specific serving engine configuration (check data for details)",
            "Cost calculated based on AWS on-demand pricing, may not reflect spot/reserved pricing",
            "Benchmark conditions may differ from production workloads",
        ]

        ctx.what_is_not_captured = [
            "Specific serving engine version used for each benchmark",
            "Container/runtime environment details",
            "Network conditions between GPUs (intra-node vs inter-node)",
        ]

        return ctx

    def _default_context(self, source: str) -> DataSourceContext:
        """Create default context when no context file is available."""
        trust_info = SOURCE_TRUST.get(source, {})
        return DataSourceContext(
            source_name=source,
            methodology_summary=trust_info.get("description", "Unknown methodology"),
            measurement_type=trust_info.get("measurement_type", "unknown"),
            limitations=["No context file available - methodology unknown"],
        )

    def _load_source(self, source: str, filepath: Path):
        """Load entries from a specific source file."""
        try:
            df = pd.read_csv(filepath)
            count = 0
            infeasible_count = 0

            if source == "dynamo_swept":
                count = self._load_dynamo_swept(df)
            elif source == "dynamo_test":
                count = self._load_dynamo_test(df, filepath.name)
            elif source == "solver":
                count, infeasible_count = self._load_solver(df)
            elif source == "vidur":
                count = self._load_vidur(df)
            elif source == "benchmark":
                count = self._load_benchmark(df)

            msg = f"  Loaded {count} entries from {source} ({filepath.name})"
            if infeasible_count > 0:
                msg += f" + {infeasible_count} infeasible"
            print(msg)
        except Exception as e:
            print(f"Error loading {source} from {filepath}: {e}")

    def _load_benchmark(self, df: pd.DataFrame) -> int:
        """Load our_own_experiment benchmark data."""
        count = 0
        ctx = self.source_contexts.get("benchmark")

        for _, row in df.iterrows():
            total_tps = row.get("tokens_per_sec")
            if pd.isna(total_tps) or total_tps == 0:
                continue

            # Infer GPU type from device
            device = str(row.get("device_type", "")).lower()
            gpu_type = self._infer_gpu_type(device)

            entry = PerfEntry(
                source="benchmark",
                model_name=str(row.get("model_name", "")),
                gpu_type=gpu_type,
                device_type=str(row.get("device_type", "")),
                tp=int(row["tp"]) if pd.notna(row.get("tp")) else 1,
                pp=int(row.get("pp", 1)) if pd.notna(row.get("pp")) else 1,
                input_length=float(row.get("max_input_length", 0)),
                output_length=float(row.get("max_output_length", 0)),
                total_tokens_per_sec=float(total_tps),
                input_tokens_per_sec=float(row.get("input_tokens_per_sec", 0)) if pd.notna(row.get("input_tokens_per_sec")) else 0,
                output_tokens_per_sec=float(row.get("output_tokens_per_sec", 0)) if pd.notna(row.get("output_tokens_per_sec")) else 0,
                batch_size=float(row["batch_size"]) if pd.notna(row.get("batch_size")) else None,
                cost_per_hour=float(row["total_cost"]) if pd.notna(row.get("total_cost")) else None,
                dollar_per_million_token=float(row["dollar_per_million_token"]) if pd.notna(row.get("dollar_per_million_token")) else None,
                precision="FP16",  # Assuming FP16 for benchmarks unless noted
                notes="Real hardware benchmark - verify model/config match your scenario",
            )
            self.entries.append(entry)
            count += 1

        return count

    def _load_solver(self, df: pd.DataFrame) -> tuple:
        """Load solver-based evaluation results."""
        success_count = 0
        infeasible_count = 0
        ctx = self.source_contexts.get("solver")

        for _, row in df.iterrows():
            status = str(row.get("status", "SUCCESS")).upper()

            # Parse GPU type from device_type column
            device_type_str = str(row.get("device_type", ""))
            gpu_types = []
            for gpu in ["A100", "V100", "H100", "L40S", "L4", "A10G"]:
                if gpu in device_type_str.upper():
                    gpu_types.append(gpu)
            gpu_type = ",".join(gpu_types) if gpu_types else "UNKNOWN"

            model_name = str(row.get("model_name", ""))
            input_length = float(row.get("max_input_length", 0)) if pd.notna(row.get("max_input_length")) else 0
            output_length = float(row.get("max_output_length", 0)) if pd.notna(row.get("max_output_length")) else 0
            batch_size = float(row.get("batch_size")) if pd.notna(row.get("batch_size")) else None
            num_gpus = int(row.get("num_gpus")) if pd.notna(row.get("num_gpus")) else None

            # Get TP and PP
            tp = int(row.get("tp", 1)) if pd.notna(row.get("tp")) else 1
            pp = int(row.get("pipeline_stages", row.get("pp", 1))) if pd.notna(row.get("pipeline_stages", row.get("pp"))) else 1

            if status == "SUCCESS":
                total_tps = row.get("total_tokens_per_sec")
                if pd.isna(total_tps) or total_tps == 0:
                    continue

                entry = PerfEntry(
                    source="solver",
                    model_name=model_name,
                    gpu_type=gpu_type,
                    device_type=device_type_str,
                    tp=tp,
                    pp=pp,
                    input_length=input_length,
                    output_length=output_length,
                    total_tokens_per_sec=float(total_tps),
                    input_tokens_per_sec=float(row.get("input_tokens_per_sec", 0)) if pd.notna(row.get("input_tokens_per_sec")) else 0,
                    output_tokens_per_sec=float(row.get("output_tokens_per_sec", 0)) if pd.notna(row.get("output_tokens_per_sec")) else 0,
                    batch_size=batch_size,
                    num_gpus=num_gpus,
                    cost_per_hour=float(row["cost_per_hour"]) if pd.notna(row.get("cost_per_hour")) else None,
                    dollar_per_million_token=float(row["dollar_per_million_token"]) if pd.notna(row.get("dollar_per_million_token")) else None,
                    precision="FP16",
                    notes="Analytical roofline model estimate - NOT a real measurement. May significantly overestimate throughput.",
                )
                self.entries.append(entry)
                success_count += 1

            elif status == "INFEASIBLE":
                error_msg = str(row.get("error", "Unknown reason"))
                infeasible = InfeasibleEntry(
                    source="solver",
                    model_name=model_name,
                    gpu_type=gpu_type,
                    instance_family=device_type_str,
                    tp=tp,
                    pp=pp,
                    input_length=input_length,
                    output_length=output_length,
                    batch_size=batch_size,
                    num_gpus=num_gpus,
                    failure_reason=error_msg,
                )
                self.infeasible_entries.append(infeasible)
                infeasible_count += 1

        return success_count, infeasible_count

    def _load_vidur(self, df: pd.DataFrame) -> int:
        """Load Vidur simulator data."""
        count = 0
        ctx = self.source_contexts.get("vidur")

        for _, row in df.iterrows():
            total_tps = row.get("total_tokens_per_sec")
            if pd.isna(total_tps) or total_tps == 0:
                continue

            gpu_type = str(row.get("gpu_type", "")).upper()
            device_type = str(row.get("device_type", ""))

            entry = PerfEntry(
                source="vidur",
                model_name=str(row.get("model_name", "")),
                gpu_type=gpu_type,
                device_type=device_type,
                tp=int(row["tp"]) if pd.notna(row.get("tp")) else 1,
                pp=int(row.get("pp", row.get("pipeline_stages", 1))) if pd.notna(row.get("pp", row.get("pipeline_stages"))) else 1,
                input_length=float(row.get("max_input_length", 0)),
                output_length=float(row.get("max_output_length", 0)),
                total_tokens_per_sec=float(total_tps),
                input_tokens_per_sec=float(row.get("input_tokens_per_sec", 0)) if pd.notna(row.get("input_tokens_per_sec")) else 0,
                output_tokens_per_sec=float(row.get("output_tokens_per_sec", 0)) if pd.notna(row.get("output_tokens_per_sec")) else 0,
                batch_size=float(row["batch_size"]) if pd.notna(row.get("batch_size")) else None,
                num_gpus=int(row.get("num_gpus", row.get("num_replicas"))) if pd.notna(row.get("num_gpus", row.get("num_replicas"))) else None,
                precision="FP16",
                notes="Vidur simulator prediction - validated to ~9% error on some workloads. NOT a real measurement.",
            )
            self.entries.append(entry)
            count += 1

        return count

    def _load_dynamo_swept(self, df: pd.DataFrame) -> int:
        """Load Dynamo pre-swept profiling data."""
        count = 0
        ctx = self.source_contexts.get("dynamo_swept")

        for _, row in df.iterrows():
            tps_per_gpu = row.get("throughput_per_gpu")
            if pd.isna(tps_per_gpu) or tps_per_gpu == 0:
                continue

            gpu_count = int(row.get("gpu_count", 8)) if pd.notna(row.get("gpu_count")) else 8
            total_tps = float(tps_per_gpu) * gpu_count

            gpu_type = str(row.get("gpu_type", "")).upper()
            if "h100" in gpu_type.lower():
                gpu_type = "H100"
            elif "h200" in gpu_type.lower():
                gpu_type = "H200"

            model_name = str(row.get("model", ""))
            precision = "FP8" if "FP8" in model_name else "FP16"

            framework = str(row.get("framework", "")) if pd.notna(row.get("framework")) else None
            framework_version = str(row.get("framework_version", "")) if pd.notna(row.get("framework_version")) else None

            # Determine data type (prefill vs decode)
            data_type = str(row.get("data_type", ""))

            entry = PerfEntry(
                source="dynamo_swept",
                model_name=model_name,
                gpu_type=gpu_type,
                device_type=str(row.get("gpu_type", "")),
                tp=int(row["tp"]) if pd.notna(row.get("tp")) else 1,
                pp=int(row["pp"]) if pd.notna(row.get("pp")) else 1,
                input_length=float(row.get("context_length", row.get("input_sequence_length", 0))) if pd.notna(row.get("context_length", row.get("input_sequence_length"))) else 0,
                output_length=0,  # Dynamo focuses on decode
                total_tokens_per_sec=total_tps,
                input_tokens_per_sec=0,
                output_tokens_per_sec=total_tps if data_type == "decode" else 0,
                batch_size=float(row["max_batch_size"]) if pd.notna(row.get("max_batch_size")) else None,
                num_gpus=gpu_count,
                serving_engine=framework,
                engine_version=framework_version,
                precision=precision,
                notes=f"Dynamo pre-swept profiling. {precision} precision, {framework} {framework_version}. Narrow applicability.",
            )
            self.entries.append(entry)
            count += 1

        return count

    def _load_dynamo_test(self, df: pd.DataFrame, filename: str) -> int:
        """Load Dynamo test profiling data (prefill/decode interpolation)."""
        count = 0
        ctx = self.source_contexts.get("dynamo_test")

        is_prefill = "prefill" in filename.lower()

        for _, row in df.iterrows():
            if is_prefill:
                # Prefill data
                tps_per_gpu = row.get("prefill_thpt_per_gpu")
                if pd.isna(tps_per_gpu) or tps_per_gpu == 0:
                    continue

                input_length = float(row.get("prefill_isl", 0))
                tp = int(row.get("tp_prefill", 1)) if pd.notna(row.get("tp_prefill")) else 1
            else:
                # Decode data
                tps_per_gpu = row.get("z_thpt_per_gpu")
                if pd.isna(tps_per_gpu) or tps_per_gpu == 0:
                    continue

                input_length = float(row.get("y_context_length", 0))
                tp = int(row.get("tp_decode", 1)) if pd.notna(row.get("tp_decode")) else 1

            gpu_type = str(row.get("gpu_type", "")).upper()
            if "h200" in gpu_type.lower():
                gpu_type = "H200"
            elif "h100" in gpu_type.lower():
                gpu_type = "H100"

            model_name = str(row.get("model", ""))
            precision = "FP8" if "FP8" in model_name else "FP16"

            entry = PerfEntry(
                source="dynamo_test",
                model_name=model_name,
                gpu_type=gpu_type,
                device_type=str(row.get("gpu_type", "")),
                tp=tp,
                pp=1,  # Dynamo test data is typically TP-only
                input_length=input_length,
                output_length=0,
                total_tokens_per_sec=float(tps_per_gpu),  # Per-GPU throughput
                input_tokens_per_sec=float(tps_per_gpu) if is_prefill else 0,
                output_tokens_per_sec=0 if is_prefill else float(tps_per_gpu),
                num_gpus=1,  # Per-GPU metrics
                precision=precision,
                notes=f"Dynamo {'prefill' if is_prefill else 'decode'} profiling. {precision} precision. Per-GPU throughput.",
            )
            self.entries.append(entry)
            count += 1

        return count

    def _infer_gpu_type(self, device: str) -> str:
        """Infer GPU type from device string."""
        device = device.lower()
        if "p4de" in device or "p4d" in device:
            return "A100"
        elif "p3dn" in device or "p3" in device:
            return "V100"
        elif "g6e" in device:
            return "L40S"
        elif "g6" in device:
            return "L4"
        elif "g5" in device:
            return "A10G"
        elif "p5" in device:
            return "H100"
        elif "a100" in device:
            return "A100"
        elif "a40" in device:
            return "A40"
        elif "h100" in device:
            return "H100"
        elif "h200" in device:
            return "H200"
        return "UNKNOWN"

    def get_source_context(self, source: str) -> Optional[DataSourceContext]:
        """Get the methodology context for a data source."""
        return self.source_contexts.get(source)

    def get_all_contexts(self) -> Dict[str, DataSourceContext]:
        """Get all source contexts."""
        return self.source_contexts

    def format_contexts_for_prompt(self) -> str:
        """Format all source contexts for inclusion in LLM prompt."""
        if not self.source_contexts:
            return "No source context available."

        lines = ["## Data Source Methodology and Limitations\n"]
        lines.append("**READ THESE CAREFULLY** - Understanding data provenance is critical for making good recommendations.\n")

        for source, ctx in sorted(self.source_contexts.items()):
            lines.append(ctx.format_for_prompt())
            lines.append("")

        return "\n".join(lines)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the data."""
        source_counts = {}
        gpu_types = set()
        models = set()

        for e in self.entries:
            source_counts[e.source] = source_counts.get(e.source, 0) + 1
            if e.gpu_type and e.gpu_type != "nan" and e.gpu_type != "UNKNOWN":
                gpu_types.add(e.gpu_type)
            models.add(e.model_name)

        infeasible_by_gpu = {}
        for e in self.infeasible_entries:
            infeasible_by_gpu[e.gpu_type] = infeasible_by_gpu.get(e.gpu_type, 0) + 1

        return {
            "total_entries": len(self.entries),
            "total_infeasible": len(self.infeasible_entries),
            "by_source": source_counts,
            "infeasible_by_gpu": infeasible_by_gpu,
            "gpu_types": sorted(gpu_types),
            "models": sorted(models),
            "sources_with_context": list(self.source_contexts.keys()),
        }

    def find_infeasible_entries(
        self,
        model_name: Optional[str] = None,
        gpu_type: Optional[str] = None,
        input_length_range: Optional[tuple] = None,
        max_results: int = 20,
    ) -> List[InfeasibleEntry]:
        """Find infeasible configurations matching criteria."""
        results = []

        for entry in self.infeasible_entries:
            if model_name:
                if not self._model_matches(model_name.lower(), entry.model_name.lower()):
                    continue

            if gpu_type:
                if gpu_type.upper() not in entry.gpu_type.upper():
                    continue

            if input_length_range:
                if entry.input_length < input_length_range[0] * 0.5:
                    continue
                if entry.input_length > input_length_range[1] * 2:
                    continue

            results.append(entry)

        return results[:max_results]

    def find_relevant_entries(
        self,
        model_name: Optional[str] = None,
        gpu_type: Optional[str] = None,
        tp: Optional[int] = None,
        pp: Optional[int] = None,
        input_length_range: Optional[tuple] = None,
        output_length_range: Optional[tuple] = None,
        max_results: int = 50,
        sort_by_trust: bool = True,
    ) -> List[PerfEntry]:
        """
        Find entries matching the query criteria.
        Returns entries sorted by relevance (trust level, then closeness to query).
        """
        results = []

        for entry in self.entries:
            if model_name:
                model_lower = model_name.lower()
                entry_model_lower = entry.model_name.lower()
                if not self._model_matches(model_lower, entry_model_lower):
                    continue

            if gpu_type:
                if gpu_type.upper() not in entry.gpu_type.upper():
                    continue

            if tp is not None and entry.tp != tp:
                continue

            if pp is not None and entry.pp != pp:
                continue

            if input_length_range:
                if entry.input_length < input_length_range[0] * 0.5:
                    continue
                if entry.input_length > input_length_range[1] * 2:
                    continue

            if output_length_range:
                if entry.output_length < output_length_range[0] * 0.5:
                    continue
                if entry.output_length > output_length_range[1] * 2:
                    continue

            results.append(entry)

        if sort_by_trust:
            results.sort(key=lambda e: (-e.trust_score, -e.total_tokens_per_sec))

        return results[:max_results]

    def _model_matches(self, query: str, entry_model: str) -> bool:
        """Check if model names match (fuzzy)."""
        query_size = self._extract_model_size(query)
        entry_size = self._extract_model_size(entry_model)

        if query_size and entry_size:
            return query_size == entry_size

        return query in entry_model or entry_model in query

    def _extract_model_size(self, model_name: str) -> Optional[str]:
        """Extract model size from name (e.g., '70b', '7b')."""
        match = re.search(r'(\d+)b', model_name.lower())
        if match:
            return match.group(1) + "b"
        return None

    def find_by_gpu_type(self, gpu_type: str) -> List[PerfEntry]:
        """Find all entries for a specific GPU type."""
        return [e for e in self.entries if gpu_type.upper() in e.gpu_type.upper()]

    def find_exact_match(
        self,
        model_name: str,
        gpu_type: str,
        tp: int,
        pp: int,
    ) -> Optional[PerfEntry]:
        """Find exact match (if exists)."""
        for entry in self.entries:
            if (
                self._model_matches(model_name.lower(), entry.model_name.lower())
                and gpu_type.upper() in entry.gpu_type.upper()
                and entry.tp == tp
                and entry.pp == pp
            ):
                return entry
        return None

    def get_available_configs_for_gpu(self, gpu_type: str) -> List[Dict]:
        """Get all available (tp, pp) configurations for a GPU type."""
        configs = {}
        for entry in self.entries:
            if gpu_type.upper() in entry.gpu_type.upper():
                key = (entry.tp, entry.pp)
                if key not in configs:
                    configs[key] = {
                        "tp": entry.tp,
                        "pp": entry.pp,
                        "count": 0,
                        "sources": set(),
                    }
                configs[key]["count"] += 1
                configs[key]["sources"].add(entry.source)

        result = []
        for (tp, pp), info in configs.items():
            result.append({
                "tp": tp,
                "pp": pp,
                "data_points": info["count"],
                "sources": list(info["sources"]),
            })

        return sorted(result, key=lambda x: (x["tp"], x["pp"]))


def format_entries_for_prompt(entries: List[PerfEntry], max_entries: int = 20) -> str:
    """Format performance entries as a readable string for LLM prompt."""
    if not entries:
        return "No relevant performance data found."

    lines = [f"## Relevant Performance Data ({len(entries)} entries, showing top {min(len(entries), max_entries)})\n"]
    lines.append("**CRITICAL**: Evaluate each entry for applicability to your target scenario!\n")

    for i, entry in enumerate(entries[:max_entries]):
        lines.append(f"### Entry {i+1} [{entry.trust_level} trust, source: {entry.source}]")
        lines.append(f"- Model: {entry.model_name}")

        context_parts = []
        if entry.precision:
            context_parts.append(f"Precision: {entry.precision}")
        if entry.serving_engine:
            engine_str = entry.serving_engine
            if entry.engine_version:
                engine_str += f" {entry.engine_version}"
            context_parts.append(f"Engine: {engine_str}")
        if context_parts:
            lines.append(f"- Setup: {', '.join(context_parts)}")

        lines.append(f"- GPU: {entry.gpu_type} ({entry.device_type})")
        lines.append(f"- Parallelism: TP={entry.tp}, PP={entry.pp}")
        lines.append(f"- Workload: input={int(entry.input_length)} tokens, output={int(entry.output_length)} tokens")
        lines.append(f"- Throughput: {entry.total_tokens_per_sec:.1f} total tok/s")
        if entry.input_tokens_per_sec > 0:
            lines.append(f"  - Input: {entry.input_tokens_per_sec:.1f} tok/s, Output: {entry.output_tokens_per_sec:.1f} tok/s")
        if entry.dollar_per_million_token:
            lines.append(f"- Cost: ${entry.dollar_per_million_token:.2f} per million tokens")

        if entry.notes:
            lines.append(f"- **Note**: {entry.notes}")

        lines.append("")

    return "\n".join(lines)


def format_infeasible_for_prompt(entries: List[InfeasibleEntry], max_entries: int = 10) -> str:
    """Format infeasible entries as a readable string for LLM prompt."""
    if not entries:
        return ""

    lines = [f"\n## Known Infeasible Configurations ({len(entries)} entries, showing top {min(len(entries), max_entries)})\n"]
    lines.append("These configurations are KNOWN to fail - avoid recommending them:\n")

    for i, entry in enumerate(entries[:max_entries]):
        lines.append(f"### Infeasible {i+1}: {entry.gpu_type} on {entry.instance_family}, TP={entry.tp}, PP={entry.pp}")
        lines.append(f"- Model: {entry.model_name}")
        lines.append(f"- Workload: input={int(entry.input_length)}, output={int(entry.output_length)}, batch={entry.batch_size}")
        reason = entry.failure_reason
        if len(reason) > 200:
            if "memory overflow" in reason.lower():
                lines.append(f"- Reason: Memory overflow - GPU memory insufficient")
            elif "underutilized" in reason.lower():
                lines.append(f"- Reason: GPU underutilized - too few layers per stage")
            else:
                lines.append(f"- Reason: {reason[:200]}...")
        else:
            lines.append(f"- Reason: {reason}")
        lines.append("")

    return "\n".join(lines)


# Convenience function
def load_perf_data(data_dir: str = None) -> PerfDataLoader:
    """Load performance data from organized subdirectories."""
    return PerfDataLoader(data_dir)
