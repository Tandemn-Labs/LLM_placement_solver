"""
Performance data loader with source trust levels.
Loads sparse benchmark/simulator/solver data from individual files.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path


# Trust levels for different data sources
SOURCE_TRUST = {
    "benchmark": {
        "level": "HIGH",
        "score": 1.0,
        "description": "Real measurements from actual hardware (AWS). Most reliable.",
    },
    "dynamo": {
        "level": "HIGH",
        "score": 0.9,
        "description": "Pre-swept results from Dynamo profiler on H100. Real measurements.",
    },
    "simulator": {
        "level": "MEDIUM",
        "score": 0.7,
        "description": "Vidur simulator predictions. Validated to ~9% error.",
    },
    "solver_eval": {
        "level": "LOW",
        "score": 0.5,
        "description": "Solver-based throughput/cost evaluation. Analytical model with memory constraints.",
    },
    "solver": {
        "level": "LOW",
        "score": 0.4,
        "description": "Analytical roofline model. Approximation, may miss real-world effects.",
    },
}

# Data files mapping (in data/ directory)
DATA_FILES = {
    "benchmark": "our_own_benchmark.csv",
    "solver": "solver_based_number.csv",
    "simulator": "vidur_based_simulator_number.csv",
    "dynamo": "dynamo_number.csv",
}

# Additional data files (relative to project root)
EXTRA_DATA_FILES = {
    "solver_eval": "eval_results/eval_results.csv",
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
    """Load and query sparse performance data from individual files."""

    def __init__(self, data_dir: str = None, project_root: str = None):
        """Initialize with path to data directory."""
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        if project_root is None:
            project_root = Path(__file__).parent.parent

        self.data_dir = Path(data_dir)
        self.project_root = Path(project_root)
        self.entries: List[PerfEntry] = []
        self.infeasible_entries: List[InfeasibleEntry] = []
        self._load_all_sources()

    def _load_all_sources(self):
        """Load data from all source files."""
        # Load from data/ directory
        for source, filename in DATA_FILES.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                self._load_source(source, filepath)
            else:
                print(f"Warning: Data file not found: {filepath}")

        # Load from extra locations (relative to project root)
        for source, relpath in EXTRA_DATA_FILES.items():
            filepath = self.project_root / relpath
            if filepath.exists():
                self._load_source(source, filepath)
            else:
                print(f"Warning: Data file not found: {filepath}")

        print(f"Loaded {len(self.entries)} performance entries, {len(self.infeasible_entries)} infeasible entries")

    def _load_source(self, source: str, filepath: Path):
        """Load entries from a specific source file."""
        try:
            df = pd.read_csv(filepath)
            count = 0
            infeasible_count = 0

            if source == "dynamo":
                count = self._load_dynamo(df)
            elif source == "solver_eval":
                count, infeasible_count = self._load_eval_results(df)
            elif source in ["benchmark", "solver", "simulator"]:
                count = self._load_standard(source, df)

            msg = f"  Loaded {count} entries from {source} ({filepath.name})"
            if infeasible_count > 0:
                msg += f" + {infeasible_count} infeasible"
            print(msg)
        except Exception as e:
            print(f"Error loading {source} from {filepath}: {e}")

    def _load_standard(self, source: str, df: pd.DataFrame) -> int:
        """Load standard format (benchmark, solver, simulator).

        Supports both unified schema (total_tokens_per_sec) and
        legacy/compact schema (tokens_per_sec).
        """
        count = 0
        for _, row in df.iterrows():
            # Skip rows with missing throughput
            total_tps = row.get("total_tokens_per_sec")
            if pd.isna(total_tps) or total_tps == 0:
                total_tps = row.get("tokens_per_sec")
            if pd.isna(total_tps) or total_tps == 0:
                continue

            # Get GPU type - handle different formats
            gpu_type = str(row.get("gpu_type", ""))
            if gpu_type in ["nan", ""]:
                # Try to infer from device_type
                device = str(row.get("device_type", "")).lower()
                gpu_type = self._infer_gpu_type(device)

            # Normalize GPU type
            gpu_type = gpu_type.upper() if gpu_type else "UNKNOWN"

            cost_per_hour = row.get("cost_per_hour")
            if pd.isna(cost_per_hour) or cost_per_hour == 0:
                # Some compact files store cost in total_cost instead.
                cost_per_hour = row.get("total_cost")

            entry = PerfEntry(
                source=source,
                model_name=str(row.get("model_name", "")),
                gpu_type=gpu_type,
                device_type=str(row.get("device_type", "")),
                tp=int(row["tp"]) if pd.notna(row.get("tp")) else 1,
                pp=int(row.get("pp", row.get("pipeline_stages", 1))) if pd.notna(row.get("pp", row.get("pipeline_stages"))) else 1,
                input_length=float(row.get("max_input_length", 0)),
                output_length=float(row.get("max_output_length", 0)),
                total_tokens_per_sec=float(total_tps),
                input_tokens_per_sec=float(row.get("input_tokens_per_sec", 0)) if pd.notna(row.get("input_tokens_per_sec")) else 0,
                output_tokens_per_sec=float(row.get("output_tokens_per_sec", 0)) if pd.notna(row.get("output_tokens_per_sec")) else 0,
                batch_size=float(row["batch_size"]) if pd.notna(row.get("batch_size")) else None,
                num_gpus=int(row["num_gpus"]) if pd.notna(row.get("num_gpus")) else None,
                cost_per_hour=float(cost_per_hour) if pd.notna(cost_per_hour) else None,
                dollar_per_million_token=float(row["dollar_per_million_token"]) if pd.notna(row.get("dollar_per_million_token")) else None,
            )
            self.entries.append(entry)
            count += 1

        return count

    def _load_dynamo(self, df: pd.DataFrame) -> int:
        """Load Dynamo format data (different schema)."""
        count = 0
        for _, row in df.iterrows():
            # Dynamo has throughput_per_gpu - multiply by gpu_count for total
            tps_per_gpu = row.get("throughput_per_gpu")
            if pd.isna(tps_per_gpu) or tps_per_gpu == 0:
                continue

            gpu_count = int(row.get("gpu_count", 8)) if pd.notna(row.get("gpu_count")) else 8
            total_tps = float(tps_per_gpu) * gpu_count

            # Map gpu_type
            gpu_type = str(row.get("gpu_type", "")).upper()
            if "h100" in gpu_type.lower():
                gpu_type = "H100"
            elif "a100" in gpu_type.lower():
                gpu_type = "A100"

            # Extract model name
            model_name = str(row.get("model", ""))

            entry = PerfEntry(
                source="dynamo",
                model_name=model_name,
                gpu_type=gpu_type,
                device_type=str(row.get("gpu_type", "")),
                tp=int(row["tp"]) if pd.notna(row.get("tp")) else 1,
                pp=int(row["pp"]) if pd.notna(row.get("pp")) else 1,
                input_length=float(row.get("context_length", row.get("input_sequence_length", 0))) if pd.notna(row.get("context_length", row.get("input_sequence_length"))) else 0,
                output_length=0,  # Dynamo focuses on decode, output length not directly stored
                total_tokens_per_sec=total_tps,
                input_tokens_per_sec=0,
                output_tokens_per_sec=total_tps,  # Decode throughput
                batch_size=float(row["max_batch_size"]) if pd.notna(row.get("max_batch_size")) else None,
                num_gpus=gpu_count,
            )
            self.entries.append(entry)
            count += 1

        return count

    def _load_eval_results(self, df: pd.DataFrame) -> tuple:
        """Load eval_results format (solver-based throughput/cost evaluation).

        Returns tuple of (success_count, infeasible_count).
        """
        success_count = 0
        infeasible_count = 0

        for _, row in df.iterrows():
            status = str(row.get("status", "")).upper()

            # Extract common fields
            gpu_type = str(row.get("device_type", ""))
            # Normalize GPU type (e.g., "NVIDIA A10G" -> "A10G")
            if "A10G" in gpu_type:
                gpu_type = "A10G"
            elif "L40S" in gpu_type:
                gpu_type = "L40S"
            elif "L4" in gpu_type and "L40" not in gpu_type:
                gpu_type = "L4"
            elif "A100" in gpu_type:
                gpu_type = "A100"
            elif "V100" in gpu_type:
                gpu_type = "V100"
            elif "H100" in gpu_type:
                gpu_type = "H100"

            instance_family = str(row.get("instance_family", ""))
            model_name = str(row.get("model_name", ""))
            input_length = float(row.get("input_length", 0)) if pd.notna(row.get("input_length")) else 0
            output_length = float(row.get("output_length", 0)) if pd.notna(row.get("output_length")) else 0
            batch_size = float(row.get("batch_size")) if pd.notna(row.get("batch_size")) else None
            num_gpus = int(row.get("num_gpus")) if pd.notna(row.get("num_gpus")) else None

            # Get TP and PP - handle multiple column names
            tp = row.get("tp_degree", row.get("tp_degrees"))
            tp = int(tp) if pd.notna(tp) else 1
            pp = row.get("pp_stages", row.get("num_pipeline_stages"))
            pp = int(pp) if pd.notna(pp) else 1

            if status == "SUCCESS":
                # Load as performance entry
                total_tps = row.get("throughput_tokens_per_sec")
                if pd.isna(total_tps) or total_tps == 0:
                    continue

                entry = PerfEntry(
                    source="solver_eval",
                    model_name=model_name,
                    gpu_type=gpu_type,
                    device_type=instance_family,
                    tp=tp,
                    pp=pp,
                    input_length=input_length,
                    output_length=output_length,
                    total_tokens_per_sec=float(total_tps),
                    input_tokens_per_sec=0,
                    output_tokens_per_sec=float(total_tps),  # Assume decode-focused
                    batch_size=batch_size,
                    num_gpus=num_gpus,
                    cost_per_hour=float(row["cost_per_hour"]) if pd.notna(row.get("cost_per_hour")) else None,
                    dollar_per_million_token=float(row["dollar_per_million_token"]) if pd.notna(row.get("dollar_per_million_token")) else None,
                )
                self.entries.append(entry)
                success_count += 1

            elif status == "INFEASIBLE":
                # Load as infeasible entry with failure reason
                error_msg = str(row.get("error", "Unknown reason"))

                infeasible = InfeasibleEntry(
                    source="solver_eval",
                    model_name=model_name,
                    gpu_type=gpu_type,
                    instance_family=instance_family,
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

            # Skip ERROR status entries (incomplete data)

        return success_count, infeasible_count

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
        return ""

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

        # Count infeasible by GPU type
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
            # Model filter
            if model_name:
                if not self._model_matches(model_name.lower(), entry.model_name.lower()):
                    continue

            # GPU type filter
            if gpu_type:
                if gpu_type.upper() not in entry.gpu_type.upper():
                    continue

            # Input length filter
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
            # Model filter (fuzzy match)
            if model_name:
                model_lower = model_name.lower()
                entry_model_lower = entry.model_name.lower()
                # Check for common patterns
                if not self._model_matches(model_lower, entry_model_lower):
                    continue

            # GPU type filter
            if gpu_type:
                if gpu_type.upper() not in entry.gpu_type.upper():
                    continue

            # TP filter
            if tp is not None and entry.tp != tp:
                continue

            # PP filter
            if pp is not None and entry.pp != pp:
                continue

            # Input length filter
            if input_length_range:
                if entry.input_length < input_length_range[0] * 0.5:
                    continue
                if entry.input_length > input_length_range[1] * 2:
                    continue

            # Output length filter
            if output_length_range:
                if entry.output_length < output_length_range[0] * 0.5:
                    continue
                if entry.output_length > output_length_range[1] * 2:
                    continue

            results.append(entry)

        # Sort by trust score (higher first)
        if sort_by_trust:
            results.sort(key=lambda e: (-e.trust_score, -e.total_tokens_per_sec))

        return results[:max_results]

    def _model_matches(self, query: str, entry_model: str) -> bool:
        """Check if model names match (fuzzy)."""
        # Extract size indicators
        query_size = self._extract_model_size(query)
        entry_size = self._extract_model_size(entry_model)

        if query_size and entry_size:
            return query_size == entry_size

        # Fallback to substring match
        return query in entry_model or entry_model in query

    def _extract_model_size(self, model_name: str) -> Optional[str]:
        """Extract model size from name (e.g., '70b', '7b')."""
        import re
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

        # Convert to list
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
    lines.append("Trust levels: HIGH = real benchmarks, MEDIUM = simulator, LOW = analytical model\n")

    for i, entry in enumerate(entries[:max_entries]):
        lines.append(f"### Entry {i+1} [{entry.trust_level} trust, source: {entry.source}]")
        lines.append(f"- Model: {entry.model_name}")
        lines.append(f"- GPU: {entry.gpu_type} ({entry.device_type})")
        lines.append(f"- Parallelism: TP={entry.tp}, PP={entry.pp}")
        lines.append(f"- Workload: input={int(entry.input_length)} tokens, output={int(entry.output_length)} tokens")
        lines.append(f"- Throughput: {entry.total_tokens_per_sec:.1f} total tok/s")
        if entry.input_tokens_per_sec > 0:
            lines.append(f"  - Input: {entry.input_tokens_per_sec:.1f} tok/s, Output: {entry.output_tokens_per_sec:.1f} tok/s")
        if entry.dollar_per_million_token:
            lines.append(f"- Cost: ${entry.dollar_per_million_token:.2f} per million tokens")
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
        # Summarize failure reason (truncate if too long)
        reason = entry.failure_reason
        if len(reason) > 200:
            # Extract key info from the reason
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
    """Load performance data from individual files."""
    return PerfDataLoader(data_dir)
