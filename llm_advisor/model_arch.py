"""
Model architecture fetcher and comparator.
Fetches actual model configs from Hugging Face to provide precise architecture context.
"""

import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import requests

# Cache directory for model configs
CACHE_DIR = Path(__file__).parent / ".model_cache"


@dataclass
class ModelArchitecture:
    """Precise model architecture information from config.json."""
    model_id: str
    architectures: List[str] = field(default_factory=list)  # e.g., ["LlamaForCausalLM"]
    model_type: str = ""  # e.g., "llama", "mistral", "qwen2"

    # Core dimensions
    num_hidden_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: Optional[int] = None  # For GQA models
    intermediate_size: int = 0
    vocab_size: int = 0

    # Context length
    max_position_embeddings: int = 0
    rope_theta: Optional[float] = None

    # Computed properties
    params_billions: float = 0.0
    fp16_memory_gb: float = 0.0

    # Raw config for reference
    raw_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute derived values."""
        if self.num_hidden_layers > 0 and self.hidden_size > 0:
            self._estimate_params()

    def _estimate_params(self):
        """Estimate parameter count and memory requirements."""
        # Approximate parameter count for transformer
        # Embedding: vocab_size * hidden_size
        # Each layer: ~12 * hidden_size^2 (attention + FFN)
        # Output: hidden_size * vocab_size

        embed_params = self.vocab_size * self.hidden_size * 2  # input + output

        # Attention: Q, K, V, O projections
        # For GQA: K,V are smaller
        kv_heads = self.num_key_value_heads or self.num_attention_heads
        head_dim = self.hidden_size // self.num_attention_heads

        qo_params = 2 * self.hidden_size * self.hidden_size  # Q and O
        kv_params = 2 * self.hidden_size * (kv_heads * head_dim)  # K and V
        attn_params = qo_params + kv_params

        # FFN: gate, up, down projections (for LLaMA-style)
        ffn_params = 3 * self.hidden_size * self.intermediate_size

        # Layer norms
        ln_params = 2 * self.hidden_size * 2  # 2 per layer

        layer_params = attn_params + ffn_params + ln_params
        total_params = embed_params + self.num_hidden_layers * layer_params

        self.params_billions = total_params / 1e9
        self.fp16_memory_gb = (total_params * 2) / (1024**3)  # 2 bytes per param for FP16

    def architecture_signature(self) -> str:
        """Generate a signature for architecture comparison.

        Includes vocab_size because it affects memory and embedding computation.
        """
        key_attrs = [
            self.model_type,
            self.num_hidden_layers,
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.intermediate_size,
            self.vocab_size,  # Include vocab_size - affects memory significantly
        ]
        return ":".join(str(x) for x in key_attrs)

    def is_same_architecture(self, other: "ModelArchitecture") -> bool:
        """Check if two models have identical compute architecture."""
        return self.architecture_signature() == other.architecture_signature()

    def compare_to(self, other: "ModelArchitecture") -> Dict[str, Any]:
        """Compare architecture to another model."""
        comparison = {
            "same_architecture": self.is_same_architecture(other),
            "differences": [],
            "similarities": [],
        }

        attrs = [
            ("model_type", self.model_type, other.model_type),
            ("num_hidden_layers", self.num_hidden_layers, other.num_hidden_layers),
            ("hidden_size", self.hidden_size, other.hidden_size),
            ("num_attention_heads", self.num_attention_heads, other.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads, other.num_key_value_heads),
            ("intermediate_size", self.intermediate_size, other.intermediate_size),
            ("vocab_size", self.vocab_size, other.vocab_size),
        ]

        for name, val1, val2 in attrs:
            if val1 == val2:
                comparison["similarities"].append(f"{name}={val1}")
            else:
                comparison["differences"].append(f"{name}: {val1} vs {val2}")

        return comparison

    def format_for_prompt(self) -> str:
        """Format architecture info for LLM prompt."""
        lines = [
            f"**{self.model_id}** Architecture:",
            f"  - Type: {self.model_type} ({', '.join(self.architectures)})",
            f"  - Layers: {self.num_hidden_layers}",
            f"  - Hidden size: {self.hidden_size}",
            f"  - Attention heads: {self.num_attention_heads} (KV heads: {self.num_key_value_heads or self.num_attention_heads})",
            f"  - FFN intermediate: {self.intermediate_size}",
            f"  - Vocab size: {self.vocab_size}",
            f"  - Max context: {self.max_position_embeddings}",
            f"  - Estimated params: {self.params_billions:.1f}B",
            f"  - FP16 memory: {self.fp16_memory_gb:.1f} GB",
        ]
        return "\n".join(lines)


def fetch_model_config(model_id: str, use_cache: bool = True) -> Optional[Dict]:
    """Fetch model config.json from Hugging Face."""
    CACHE_DIR.mkdir(exist_ok=True)

    # Check cache first
    cache_key = hashlib.md5(model_id.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if use_cache and cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except:
            pass

    # Try to fetch from HuggingFace
    urls = [
        f"https://huggingface.co/{model_id}/raw/main/config.json",
        f"https://huggingface.co/{model_id}/resolve/main/config.json",
    ]

    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                config = resp.json()
                # Cache it
                cache_file.write_text(json.dumps(config, indent=2))
                return config
        except Exception:
            continue

    return None


def get_model_architecture(model_id: str) -> Optional[ModelArchitecture]:
    """Get model architecture from Hugging Face config."""
    config = fetch_model_config(model_id)
    if not config:
        return None

    arch = ModelArchitecture(
        model_id=model_id,
        architectures=config.get("architectures", []),
        model_type=config.get("model_type", ""),
        num_hidden_layers=config.get("num_hidden_layers", 0),
        hidden_size=config.get("hidden_size", 0),
        num_attention_heads=config.get("num_attention_heads", 0),
        num_key_value_heads=config.get("num_key_value_heads"),
        intermediate_size=config.get("intermediate_size", 0),
        vocab_size=config.get("vocab_size", 0),
        max_position_embeddings=config.get("max_position_embeddings", 0),
        rope_theta=config.get("rope_theta"),
        raw_config=config,
    )

    return arch


# Pre-defined architecture signatures for common models (fallback when HF fetch fails)
KNOWN_ARCHITECTURES = {
    # Llama 3 70B family - all have identical compute architecture
    "llama3-70b": ModelArchitecture(
        model_id="llama3-70b",
        architectures=["LlamaForCausalLM"],
        model_type="llama",
        num_hidden_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        num_key_value_heads=8,
        intermediate_size=28672,
        vocab_size=128256,
        max_position_embeddings=131072,
    ),
    # Llama 2 70B family
    "llama2-70b": ModelArchitecture(
        model_id="llama2-70b",
        architectures=["LlamaForCausalLM"],
        model_type="llama",
        num_hidden_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        num_key_value_heads=8,  # GQA
        intermediate_size=28672,
        vocab_size=32000,
        max_position_embeddings=4096,
    ),
    # Llama 3 8B family
    "llama3-8b": ModelArchitecture(
        model_id="llama3-8b",
        architectures=["LlamaForCausalLM"],
        model_type="llama",
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        vocab_size=128256,
        max_position_embeddings=131072,
    ),
}


def normalize_model_id(model_name: str) -> str:
    """Normalize model name for lookup."""
    name = model_name.lower()

    # Map common patterns to canonical IDs
    if "deepseek" in name and "distill" in name and "llama" in name:
        if "70b" in name:
            return "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        elif "8b" in name:
            return "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    if "nvidia" in name and "llama" in name:
        if "70b" in name:
            return "nvidia/Llama-3.3-70B-Instruct-FP8"
        elif "8b" in name:
            return "nvidia/Llama-3.1-8B-Instruct-FP8"

    # Handle generic "llama-70b" or "llama3-70b" queries - map to known model
    if ("llama-3" in name or "llama3" in name or "llama-3.1" in name or "llama-3.3" in name) and "70b" in name:
        return "nvidia/Llama-3.3-70B-Instruct-FP8"  # Use nvidia since it's publicly accessible
    if ("llama-3" in name or "llama3" in name) and "8b" in name:
        return "nvidia/Llama-3.1-8B-Instruct-FP8"

    # Generic "llama-70b" without version - assume Llama 3 (most common)
    if "llama" in name and "70b" in name and "2" not in name:
        return "nvidia/Llama-3.3-70B-Instruct-FP8"
    if "llama" in name and ("8b" in name or "7b" in name) and "2" not in name:
        return "nvidia/Llama-3.1-8B-Instruct-FP8"

    # Llama 2 variants
    if "llama-2" in name or "llama2" in name:
        if "70b" in name:
            return "meta-llama/Llama-2-70b-hf"
        elif "7b" in name:
            return "meta-llama/Llama-2-7b-hf"

    return model_name


def get_architecture_for_model(model_name: str) -> Optional[ModelArchitecture]:
    """Get architecture for a model by name, trying HuggingFace first then fallbacks."""
    # Normalize the model name
    model_id = normalize_model_id(model_name)

    # Try to fetch from HuggingFace
    arch = get_model_architecture(model_id)
    if arch:
        return arch

    # Try known architectures
    name_lower = model_name.lower()
    if "70b" in name_lower:
        if "llama-3" in name_lower or "llama3" in name_lower or "distill" in name_lower:
            return KNOWN_ARCHITECTURES.get("llama3-70b")
        elif "llama-2" in name_lower or "llama2" in name_lower:
            return KNOWN_ARCHITECTURES.get("llama2-70b")
    elif "8b" in name_lower:
        if "llama-3" in name_lower or "llama3" in name_lower:
            return KNOWN_ARCHITECTURES.get("llama3-8b")

    return None


def compare_models(model1: str, model2: str) -> Dict[str, Any]:
    """Compare architectures of two models."""
    arch1 = get_architecture_for_model(model1)
    arch2 = get_architecture_for_model(model2)

    if not arch1 or not arch2:
        return {
            "comparable": False,
            "reason": f"Could not fetch architecture for {'both' if not arch1 and not arch2 else model1 if not arch1 else model2}",
        }

    comparison = arch1.compare_to(arch2)
    comparison["model1"] = arch1.format_for_prompt()
    comparison["model2"] = arch2.format_for_prompt()

    return comparison


def format_architecture_context(models: List[str]) -> str:
    """Format architecture context for multiple models for LLM prompt."""
    lines = ["## Model Architecture Details (from HuggingFace configs)\n"]

    # Deduplicate by architecture signature
    architectures = {}
    seen_signatures = {}

    for model in models:
        arch = get_architecture_for_model(model)
        if arch:
            sig = arch.architecture_signature()
            if sig not in seen_signatures:
                # First time seeing this architecture
                architectures[arch.model_id] = arch
                seen_signatures[sig] = arch.model_id
                lines.append(arch.format_for_prompt())
                lines.append("")
            else:
                # This model has same architecture as another
                # Track the equivalence but don't print duplicate
                existing_id = seen_signatures[sig]
                if arch.model_id != existing_id:
                    architectures[arch.model_id] = arch

    # Find equivalent architectures
    if len(architectures) > 1:
        lines.append("### Architecture Equivalence Analysis:")

        # Group by signature
        sig_to_models = {}
        for model_id, arch in architectures.items():
            sig = arch.architecture_signature()
            if sig not in sig_to_models:
                sig_to_models[sig] = []
            sig_to_models[sig].append(model_id)

        # Report equivalent groups
        for sig, model_ids in sig_to_models.items():
            if len(model_ids) > 1:
                arch = architectures[model_ids[0]]
                lines.append(f"  **IDENTICAL architecture** (benchmarks SHOULD transfer):")
                for mid in model_ids:
                    lines.append(f"    - {mid}")
                lines.append(f"    (same: layers={arch.num_hidden_layers}, hidden={arch.hidden_size}, heads={arch.num_attention_heads}, vocab={arch.vocab_size})")
                lines.append("")

        # Report different architectures
        sigs = list(sig_to_models.keys())
        if len(sigs) > 1:
            lines.append(f"  **DIFFERENT architectures** (benchmarks may NOT transfer accurately):")
            for sig in sigs:
                model_id = sig_to_models[sig][0]
                arch = architectures[model_id]
                lines.append(f"    - {model_id}: layers={arch.num_hidden_layers}, hidden={arch.hidden_size}, vocab={arch.vocab_size}, mem={arch.fp16_memory_gb:.0f}GB")
            lines.append("")

    return "\n".join(lines)
