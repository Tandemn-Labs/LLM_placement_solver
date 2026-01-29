"""
Performance data loader with source trust levels.
Loads sparse benchmark/simulator/solver data and provides query interface.
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
    "simulator": {
        "level": "MEDIUM",
        "score": 0.7,
        "description": "Vidur simulator predictions. Validated to ~9% error.",
    },
    "solver": {
        "level": "LOW",
        "score": 0.4,
        "description": "Analytical roofline model. Approximation, may miss real-world effects.",
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


class PerfDataLoader:
    """Load and query sparse performance data."""

    def __init__(self, csv_path: str = None):
        """Initialize with path to unified CSV."""
        if csv_path is None:
            csv_path = Path(__file__).parent.parent / "unified_performance_summary.csv"

        self.csv_path = Path(csv_path)
        self.df = None
        self.entries: List[PerfEntry] = []
        self._load_data()

    def _load_data(self):
        """Load CSV and convert to PerfEntry objects."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Performance data not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        for _, row in self.df.iterrows():
            # Skip rows with missing critical data
            if pd.isna(row.get("total_tokens_per_sec")):
                continue

            entry = PerfEntry(
                source=row.get("source", "unknown"),
                model_name=str(row.get("model_name", "")),
                gpu_type=str(row.get("gpu_type", "")),
                device_type=str(row.get("device_type", "")),
                tp=int(row["tp"]) if pd.notna(row.get("tp")) else 1,
                pp=int(row["pp"]) if pd.notna(row.get("pp")) else 1,
                input_length=float(row.get("max_input_length", 0)),
                output_length=float(row.get("max_output_length", 0)),
                total_tokens_per_sec=float(row.get("total_tokens_per_sec", 0)),
                input_tokens_per_sec=float(row.get("input_tokens_per_sec", 0)) if pd.notna(row.get("input_tokens_per_sec")) else 0,
                output_tokens_per_sec=float(row.get("output_tokens_per_sec", 0)) if pd.notna(row.get("output_tokens_per_sec")) else 0,
                batch_size=float(row["batch_size"]) if pd.notna(row.get("batch_size")) else None,
                num_gpus=int(row["num_gpus"]) if pd.notna(row.get("num_gpus")) else None,
                cost_per_hour=float(row["cost_per_hour"]) if pd.notna(row.get("cost_per_hour")) else None,
                dollar_per_million_token=float(row["dollar_per_million_token"]) if pd.notna(row.get("dollar_per_million_token")) else None,
            )
            self.entries.append(entry)

        print(f"Loaded {len(self.entries)} performance entries from {self.csv_path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the data."""
        source_counts = {}
        gpu_types = set()
        models = set()

        for e in self.entries:
            source_counts[e.source] = source_counts.get(e.source, 0) + 1
            if e.gpu_type and e.gpu_type != "nan":
                gpu_types.add(e.gpu_type)
            models.add(e.model_name)

        return {
            "total_entries": len(self.entries),
            "by_source": source_counts,
            "gpu_types": sorted(gpu_types),
            "models": sorted(models),
        }

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


# Convenience function
def load_perf_data(csv_path: str = None) -> PerfDataLoader:
    """Load performance data from unified CSV."""
    return PerfDataLoader(csv_path)
