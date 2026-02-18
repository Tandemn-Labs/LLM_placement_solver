"""
Build a canonical CSV from 6 disparate profiling data sources.

Reads data from:
  1. dynamo/swept/data.csv         → data_source="dynamo_swept"
  2. dynamo/test/*.csv             → data_source="dynamo_test"
  3. solver_based/data.csv         → data_source="solver"
  4. vidur/data.csv                → data_source="vidur"
  5. our_own_experiment/data.csv   → data_source="our_experiment"
  6. our_own_experiment/perfdb_l40s_llama70b.csv → data_source="our_experiment_perfdb"
  7. splitwise/data.csv            → data_source="splitwise"

Writes: canonical/data.csv

NOTE on p90 vs p95:
  Vidur and solver report p90 latencies. These are mapped into the p95 columns
  because the sources do not report a true p95. The column names say "p95" but
  for vidur/solver rows the actual percentile is p90.

Usage:
    python build_canonical.py [--validate]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent  # llm_advisor/data/
SCRIPT_DIR = Path(__file__).resolve().parent        # llm_advisor/data/canonical/

# Add llm_advisor package to path so we can import helpers
LLM_ADVISOR_DIR = DATA_DIR.parent  # llm_advisor/
if str(LLM_ADVISOR_DIR.parent) not in sys.path:
    sys.path.insert(0, str(LLM_ADVISOR_DIR.parent))

from llm_advisor.model_arch import fetch_model_config, get_model_architecture
from llm_advisor.gpu_specs import GPU_SPECS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL_COLUMNS = [
    # --- identifiers ---
    "data_source",
    "data_source_type",
    "model_name",
    "model_architecture",
    "precision",
    "params_billion",
    # --- parallelism ---
    "tp",
    "pp",
    "dp",
    # --- hardware ---
    "gpu_model",
    "gpu_count_total",
    "gpu_mem_gb",
    "num_nodes",
    "gpus_per_node",
    "interconnect",
    # --- cloud ---
    "cloud",
    "region",
    "instance_type",
    "price_per_instance_hour_usd",
    # --- runtime ---
    "runtime_stack",
    # --- workload ---
    "task_type",
    "request_pattern",
    "num_requests",
    "max_num_seqs",
    # --- input/output lengths ---
    "input_len_tokens_min",
    "input_len_tokens_max",
    "input_len_tokens_avg",
    "input_len_tokens_fixed",
    "output_len_tokens_min",
    "output_len_tokens_max",
    "output_len_tokens_avg",
    "output_len_tokens_fixed",
    # --- throughput ---
    "tokens_per_sec_total",
    "tokens_per_sec_per_gpu",
    "tokens_per_sec_prefill",
    "tokens_per_sec_decode",
    # --- latency ---
    "ttft_ms_p50",
    "ttft_ms_p95",
    "ttft_ms_p99",
    "tpot_ms_p50",
    "tpot_ms_p95",
    "tpot_ms_p99",
    "e2e_ms_p50",
    "e2e_ms_p95",
    "e2e_ms_p99",
    # --- cost ---
    "total_cost_usd",
    "cost_per_1m_tokens_total_usd",
    "cost_per_1m_tokens_prefill_usd",
    "cost_per_1m_tokens_decode_usd",
    # --- feature flags (string: "None" = explicitly disabled, NaN = unknown) ---
    "is_lmcache",
    "is_continuous_batching",
    "kv_offload_target",
    "cuda_graphs",
    "spec_decode",
    # --- derived (existing) ---
    "prefill_decode_ratio",
    # --- batch size (where applicable) ---
    "batch_size",
    # --- raw HuggingFace config.json (JSON string) ---
    "model_config_json",
    # --- derived (model structure) ---
    "is_moe",
    "num_experts_active",
    "vocab_size",
    "attention_heads_per_kv_head",
    # --- derived (sizing) ---
    "model_size_gb",
    "params_per_gpu",
    "model_fits_single_gpu",
    "vram_headroom_gb",
    # --- derived (hardware) ---
    "gpu_bandwidth_gbps",
    "gpu_tflops_fp16",
    "gpu_generation",
    # --- derived (efficiency ratios) ---
    "bandwidth_per_param",
    "flops_per_param",
    "kv_heads_per_tp",
    # --- derived (topology) ---
    "crosses_node_boundary",
    # --- derived (cost) ---
    "price_per_gpu_hour_usd",
]


# Instance pricing (on-demand, us-east-1)
INSTANCE_PRICING = {
    # L40S (g6e)
    "g6e.2xlarge": 0.99,
    "g6e.12xlarge": 4.68,
    "g6e.48xlarge": 13.35,
    # A10G (g5)
    "g5.2xlarge": 1.006,
    "g5.12xlarge": 4.096,
    "g5.48xlarge": 16.384,
    # L4 (g6)
    "g6.2xlarge": 0.526,
    "g6.12xlarge": 0.752,
    "g6.48xlarge": 1.204,
}

# Map GPU model key -> vram_gb from gpu_specs.py (H100_SXM, H200, H200_SXM now in GPU_SPECS)
GPU_MEM_MAP = {k: v["vram_gb"] for k, v in GPU_SPECS.items()}

# Instance type → GPUs per instance (AWS on-demand)
INSTANCE_GPUS = {
    "g5.2xlarge": 1, "g5.12xlarge": 4, "g5.48xlarge": 8,
    "g6.2xlarge": 1, "g6.12xlarge": 4, "g6.48xlarge": 8,
    "g6e.2xlarge": 1, "g6e.12xlarge": 4, "g6e.48xlarge": 8,
    "p4d.24xlarge": 8, "p4de.24xlarge": 8, "p5.48xlarge": 8,
}

DGX_GPUS_PER_NODE = 8

# Short model name → HuggingFace canonical ID
MODEL_NAME_MAP = {
    "llama3-70b-decode": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama3-70b-prefill": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama2-70b": "meta-llama/Llama-2-70b-hf",
    "bloom-176b": "bigscience/bloom",
}

# Splitwise hardware name → (gpu_model, gpu_mem_gb)
SPLITWISE_HW_MAP = {
    "a100-80gb": ("A100", 80),
    "h100-80gb": ("H100", 80),
    "h100-80gb-pcap": ("H100", 80),  # power-capped H100
}

# Fallback architecture info when HF API is unavailable
# Tuple: (architecture_class, params_billion, raw_config_dict_or_None)
# params_billion values are exact counts from safetensors headers
FALLBACK_ARCHITECTURES = {
    "meta-llama/Meta-Llama-3-70B-Instruct": ("LlamaForCausalLM", 70.553706, None),
    "meta-llama/Llama-2-70b-hf": ("LlamaForCausalLM", 68.976653, None),
    "nvidia/Llama-3.3-70B-Instruct-FP8": ("LlamaForCausalLM", 70.553706, None),
    "nvidia/Llama-3.1-8B-Instruct-FP8": ("LlamaForCausalLM", 8.030261, None),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": ("LlamaForCausalLM", 70.553706, None),
    "bigscience/bloom": ("BloomForCausalLM", 176.247271, None),
}

# HuggingFace token for gated model access (set via HF_TOKEN env var or --hf-token)
HF_TOKEN: str | None = os.environ.get("HF_TOKEN")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_row() -> dict:
    """Return a dict with all canonical columns set to NaN."""
    return {col: np.nan for col in CANONICAL_COLUMNS}


# Cache for HF lookups within a single run
# Maps model_id -> (architecture_str, params_billion, raw_config_dict_or_None)
_hf_cache: dict = {}

# Disk cache for config.json (shared with model_arch.py)
_CONFIG_CACHE_DIR = LLM_ADVISOR_DIR / ".model_cache"


def _fetch_config_with_auth(model_id: str) -> dict | None:
    """Fetch config.json from HuggingFace, using HF_TOKEN for gated models."""
    import hashlib

    _CONFIG_CACHE_DIR.mkdir(exist_ok=True)
    cache_key = hashlib.md5(model_id.encode()).hexdigest()
    cache_file = _CONFIG_CACHE_DIR / f"{cache_key}.json"

    # Check disk cache first
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass

    # Build headers
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    urls = [
        f"https://huggingface.co/{model_id}/raw/main/config.json",
        f"https://huggingface.co/{model_id}/resolve/main/config.json",
    ]
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                config = resp.json()
                cache_file.write_text(json.dumps(config, indent=2))
                return config
        except Exception:
            continue

    return None


def _fetch_exact_params(model_id: str) -> float:
    """Fetch exact parameter count from HF Model Info API (safetensors.total).

    Returns params in billions, or NaN if unavailable.
    """
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    try:
        resp = requests.get(
            f"https://huggingface.co/api/models/{model_id}",
            headers=headers,
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            safetensors = data.get("safetensors")
            if safetensors:
                total = safetensors.get("total")
                if total and total > 0:
                    return total / 1e9
    except Exception:
        pass
    return np.nan


def _hf_lookup(model_id: str):
    """Return (architecture_str, params_billion, raw_config_json_str) for a model."""
    if model_id in _hf_cache:
        return _hf_cache[model_id]

    # Try fetch with auth (handles both cached + API)
    config = _fetch_config_with_auth(model_id)

    # Get exact param count from safetensors metadata (ground truth)
    params_b = _fetch_exact_params(model_id)

    # If API didn't provide exact count, check FALLBACK_ARCHITECTURES
    if pd.isna(params_b) and model_id in FALLBACK_ARCHITECTURES:
        params_b = FALLBACK_ARCHITECTURES[model_id][1]

    if config:
        arch = get_model_architecture(model_id)
        if arch and arch.architectures:
            arch_str = arch.architectures[0]
            # Fall back to estimated count only if neither API nor fallback provided exact
            if pd.isna(params_b):
                params_b = arch.params_billions
                print(f"    WARN: using estimated params for {model_id}: {params_b:.3f}B")
        else:
            archs = config.get("architectures", [])
            arch_str = archs[0] if archs else np.nan
        result = (arch_str, params_b, config)
    elif model_id in FALLBACK_ARCHITECTURES:
        fb = FALLBACK_ARCHITECTURES[model_id]
        result = (fb[0], params_b if pd.notna(params_b) else fb[1], fb[2])
    else:
        result = (np.nan, params_b, None)

    _hf_cache[model_id] = result
    return result


def _normalize_gpu_type(raw: str) -> str:
    """Normalize GPU type string to canonical form."""
    raw_lower = raw.lower().strip()
    if raw_lower in ("h100_sxm", "h100-sxm"):
        return "H100_SXM"
    if raw_lower in ("h200_sxm", "h200-sxm"):
        return "H200_SXM"
    if raw_lower == "h200":
        return "H200"
    if raw_lower in ("h100", "h100_pcie"):
        return "H100"
    if raw_lower == "a100":
        return "A100"
    if raw_lower == "a10g":
        return "A10G"
    if raw_lower == "l40s":
        return "L40S"
    if raw_lower == "l4":
        return "L4"
    return raw.upper()


def _gpu_mem(gpu_model: str) -> float:
    """Look up GPU memory in GB."""
    return GPU_MEM_MAP.get(gpu_model, np.nan)


def _interconnect_for_gpu(gpu_model: str) -> str:
    """Return interconnect type. NVLink for SXM/A100/H100, PCIe otherwise."""
    nvlink_gpus = {"H100_SXM", "H200_SXM", "H200", "A100", "H100", "V100", "A40"}
    if gpu_model in nvlink_gpus:
        return "NVLink"
    return "PCIe"


def _infer_gpu_type(device: str) -> str:
    """Infer GPU type from AWS instance type string."""
    device = device.lower()
    if "g6e" in device:
        return "L40S"
    if "g6" in device:
        return "L4"
    if "g5" in device:
        return "A10G"
    if "p4de" in device:
        return "A100"
    if "p4d" in device:
        return "A100"
    if "p5" in device:
        return "H100"
    if "p3" in device:
        return "V100"
    return "UNKNOWN"


def _parse_device_type(device_str: str):
    """
    Parse device_type strings like '3x g5.12xlarge', 'g6e.48xlarge',
    'g5.12xlarge#0,g5.12xlarge#1,g6e.4xlarge#1' etc.

    Returns (num_nodes, base_instance_type, full_device_str).
    """
    device_str = str(device_str).strip()

    # Pattern: "3x g5.12xlarge"
    multi_match = re.match(r"(\d+)x\s+(.+)", device_str)
    if multi_match:
        num_nodes = int(multi_match.group(1))
        base_instance = multi_match.group(2).strip()
        return num_nodes, base_instance, device_str

    # Pattern: "g5.12xlarge#0,g5.12xlarge#1,g6e.4xlarge#1" (solver heterogeneous)
    if "#" in device_str:
        return 1, device_str, device_str

    # Simple: "g6e.48xlarge"
    return 1, device_str, device_str


def _instance_price(device_str: str) -> float:
    """Compute hourly price for a device_type string."""
    num_nodes, base, _ = _parse_device_type(device_str)
    base_clean = re.sub(r"#\d+", "", base).strip()
    price = INSTANCE_PRICING.get(base_clean, np.nan)
    if pd.notna(price):
        return price * num_nodes
    return np.nan


def _set_input_len_single(row: dict, val):
    """Set all 4 input_len fields to a single value."""
    for suffix in ("min", "max", "avg", "fixed"):
        row[f"input_len_tokens_{suffix}"] = val


def _set_output_len_single(row: dict, val):
    """Set all 4 output_len fields to a single value."""
    for suffix in ("min", "max", "avg", "fixed"):
        row[f"output_len_tokens_{suffix}"] = val


def _gpu_spec(gpu_model, field: str):
    """Lookup a single GPU spec field. Returns NaN for unknown/heterogeneous GPUs."""
    if pd.isna(gpu_model) or not isinstance(gpu_model, str):
        return np.nan
    # Heterogeneous configs contain commas (e.g. "L40S,A10G")
    if "," in gpu_model:
        return np.nan
    spec = GPU_SPECS.get(gpu_model)
    if spec is None:
        return np.nan
    return spec.get(field, np.nan)


def _parse_model_config(raw_json_str):
    """Parse model_config_json string → dict. Returns None on failure."""
    if pd.isna(raw_json_str) or not isinstance(raw_json_str, str):
        return None
    try:
        return json.loads(raw_json_str)
    except (json.JSONDecodeError, TypeError):
        return None


def _infer_gpus_per_node(instance_type):
    """Infer GPUs per node from instance type string.

    Handles: "3x g6e.48xlarge", "DGX-A100", simple instance names.
    Returns NaN for heterogeneous solver configs with '#' notation.
    """
    if pd.isna(instance_type) or not isinstance(instance_type, str):
        return np.nan
    instance_type = instance_type.strip()
    if not instance_type:
        return np.nan

    # Heterogeneous solver configs
    if "#" in instance_type:
        return np.nan

    # DGX systems
    if instance_type.upper().startswith("DGX"):
        return DGX_GPUS_PER_NODE

    # Multi-node: "3x g6e.48xlarge"
    multi_match = re.match(r"\d+x\s+(.+)", instance_type)
    base = multi_match.group(1).strip() if multi_match else instance_type

    return INSTANCE_GPUS.get(base, np.nan)


# ---------------------------------------------------------------------------
# Loader: dynamo/swept
# ---------------------------------------------------------------------------

def load_dynamo_swept() -> pd.DataFrame:
    """Load dynamo/swept/data.csv → canonical rows."""
    path = DATA_DIR / "dynamo" / "swept" / "data.csv"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    df = pd.read_csv(path)
    rows = []

    for _, src in df.iterrows():
        r = _empty_row()
        r["data_source"] = "dynamo_swept"
        r["data_source_type"] = "measured"

        model_name = str(src.get("model", ""))
        if model_name == "nan" or not model_name:
            continue
        r["model_name"] = model_name

        arch_str, params_b, raw_config = _hf_lookup(model_name)
        r["model_architecture"] = arch_str
        r["params_billion"] = params_b
        r["model_config_json"] = json.dumps(raw_config, separators=(",", ":")) if raw_config else np.nan

        r["precision"] = "fp8" if "FP8" in model_name else "fp16"

        r["tp"] = int(src["tp"]) if pd.notna(src.get("tp")) else np.nan
        r["pp"] = int(src["pp"]) if pd.notna(src.get("pp")) else np.nan
        r["dp"] = int(src["dp"]) if pd.notna(src.get("dp")) else np.nan

        gpu_raw = str(src.get("gpu_type", ""))
        gpu_model = _normalize_gpu_type(gpu_raw)
        r["gpu_model"] = gpu_model
        gpu_count = int(src["gpu_count"]) if pd.notna(src.get("gpu_count")) else np.nan
        r["gpu_count_total"] = gpu_count
        r["gpu_mem_gb"] = _gpu_mem(gpu_model)
        r["interconnect"] = _interconnect_for_gpu(gpu_model)

        fw = src.get("framework", "")
        fw_ver = src.get("framework_version", "")
        if pd.notna(fw) and pd.notna(fw_ver):
            r["runtime_stack"] = f"{fw} {fw_ver}"

        r["max_num_seqs"] = src.get("max_batch_size") if pd.notna(src.get("max_batch_size")) else np.nan

        tps_per_gpu = float(src["throughput_per_gpu"]) if pd.notna(src.get("throughput_per_gpu")) else np.nan
        if pd.isna(tps_per_gpu) or tps_per_gpu == 0:
            continue

        r["tokens_per_sec_per_gpu"] = tps_per_gpu
        if pd.notna(gpu_count):
            r["tokens_per_sec_total"] = tps_per_gpu * gpu_count

        data_type = str(src.get("data_type", ""))

        if data_type == "prefill":
            isl = src.get("input_sequence_length")
            if pd.notna(isl):
                _set_input_len_single(r, float(isl))
            r["tokens_per_sec_prefill"] = r["tokens_per_sec_total"]
            ttft = src.get("ttft_ms")
            if pd.notna(ttft):
                r["ttft_ms_p50"] = float(ttft)
        elif data_type == "decode":
            ctx_len = src.get("context_length")
            if pd.notna(ctx_len):
                _set_input_len_single(r, float(ctx_len))
            r["tokens_per_sec_decode"] = r["tokens_per_sec_total"]
            itl = src.get("itl_ms")
            if pd.notna(itl):
                r["tpot_ms_p50"] = float(itl)

        r["task_type"] = data_type

        rows.append(r)

    result = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    print(f"  dynamo_swept: {len(result)} rows")
    return result


# ---------------------------------------------------------------------------
# Loader: dynamo/test
# ---------------------------------------------------------------------------

def load_dynamo_test() -> pd.DataFrame:
    """Load dynamo/test/*.csv → canonical rows."""
    test_dir = DATA_DIR / "dynamo" / "test"
    if not test_dir.exists():
        print(f"  SKIP: {test_dir} not found")
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    rows = []

    for csv_path in sorted(test_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        is_prefill = "prefill" in csv_path.name.lower()

        for _, src in df.iterrows():
            r = _empty_row()
            r["data_source"] = "dynamo_test"
            r["data_source_type"] = "measured"

            model_name = str(src.get("model", ""))
            if model_name == "nan" or not model_name:
                continue
            r["model_name"] = model_name

            arch_str, params_b, raw_config = _hf_lookup(model_name)
            r["model_architecture"] = arch_str
            r["params_billion"] = params_b
            r["model_config_json"] = json.dumps(raw_config, separators=(",", ":")) if raw_config else np.nan

            r["precision"] = "fp8" if "FP8" in model_name else "fp16"

            gpu_raw = str(src.get("gpu_type", ""))
            gpu_model = _normalize_gpu_type(gpu_raw)
            r["gpu_model"] = gpu_model
            r["gpu_mem_gb"] = _gpu_mem(gpu_model)
            r["interconnect"] = _interconnect_for_gpu(gpu_model)

            if is_prefill:
                tp = int(src["tp_prefill"]) if pd.notna(src.get("tp_prefill")) else 1
                r["tp"] = tp
                r["pp"] = 1
                r["gpu_count_total"] = tp

                isl = src.get("prefill_isl")
                if pd.notna(isl):
                    _set_input_len_single(r, float(isl))

                tps = float(src["prefill_thpt_per_gpu"]) if pd.notna(src.get("prefill_thpt_per_gpu")) else np.nan
                if pd.isna(tps) or tps == 0:
                    continue
                r["tokens_per_sec_per_gpu"] = tps
                r["tokens_per_sec_total"] = tps * tp
                r["tokens_per_sec_prefill"] = tps * tp

                ttft = src.get("prefill_ttft")
                if pd.notna(ttft):
                    r["ttft_ms_p50"] = float(ttft)

                r["task_type"] = "prefill"
            else:
                tp = int(src["tp_decode"]) if pd.notna(src.get("tp_decode")) else 1
                r["tp"] = tp
                r["pp"] = 1
                r["gpu_count_total"] = tp

                ctx_len = src.get("y_context_length")
                if pd.notna(ctx_len):
                    _set_input_len_single(r, float(ctx_len))

                tps = float(src["z_thpt_per_gpu"]) if pd.notna(src.get("z_thpt_per_gpu")) else np.nan
                if pd.isna(tps) or tps == 0:
                    continue
                r["tokens_per_sec_per_gpu"] = tps
                r["tokens_per_sec_total"] = tps * tp
                r["tokens_per_sec_decode"] = tps * tp

                itl = src.get("z_itl")
                if pd.notna(itl):
                    r["tpot_ms_p50"] = float(itl)

                r["task_type"] = "decode"

            rows.append(r)

    result = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    print(f"  dynamo_test: {len(result)} rows")
    return result


# ---------------------------------------------------------------------------
# Loader: solver_based
# ---------------------------------------------------------------------------

def _parse_tp_per_stage(tp_per_stage_str: str) -> int:
    """Parse tp_per_stage dict string and return the max TP value."""
    if pd.isna(tp_per_stage_str) or not tp_per_stage_str:
        return 1
    # Format: "{PP_1:2, PP_2:2, PP_3:1, PP_4:2, PP_5:2}"
    matches = re.findall(r":(\d+)", str(tp_per_stage_str))
    if matches:
        return max(int(m) for m in matches)
    return 1


def load_solver() -> pd.DataFrame:
    """Load solver_based/data.csv → canonical rows (SUCCESS only)."""
    path = DATA_DIR / "solver_based" / "data.csv"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    df = pd.read_csv(path)
    rows = []

    for _, src in df.iterrows():
        status = str(src.get("status", "")).upper()
        if status != "SUCCESS":
            continue

        r = _empty_row()
        r["data_source"] = "solver"
        r["data_source_type"] = "analytical"

        raw_model = str(src.get("model_name", ""))
        model_name = MODEL_NAME_MAP.get(raw_model, raw_model)
        r["model_name"] = model_name

        arch_str, params_b, raw_config = _hf_lookup(model_name)
        r["model_architecture"] = arch_str
        r["params_billion"] = params_b
        r["model_config_json"] = json.dumps(raw_config, separators=(",", ":")) if raw_config else np.nan

        r["precision"] = "fp16"

        # TP: column is all NaN, parse from tp_per_stage
        tp_val = src.get("tp")
        if pd.notna(tp_val):
            r["tp"] = int(tp_val)
        else:
            r["tp"] = _parse_tp_per_stage(src.get("tp_per_stage", ""))

        pp_val = src.get("pipeline_stages", src.get("pp"))
        r["pp"] = int(pp_val) if pd.notna(pp_val) else 1

        r["instance_type"] = str(src.get("device_type", ""))
        gpu_type_raw = str(src.get("gpu_type", ""))
        r["gpu_model"] = gpu_type_raw if gpu_type_raw and gpu_type_raw != "nan" else "UNKNOWN"

        num_gpus = src.get("num_gpus")
        r["gpu_count_total"] = int(num_gpus) if pd.notna(num_gpus) else np.nan

        # mem_per_gpu_gb is a dict string for solver; skip parsing, just store raw
        # We can't easily get a single value from heterogeneous configs
        r["gpu_mem_gb"] = np.nan

        total_tps = src.get("total_tokens_per_sec")
        if pd.isna(total_tps) or total_tps == 0:
            continue
        r["tokens_per_sec_total"] = float(total_tps)
        if pd.notna(num_gpus) and num_gpus > 0:
            r["tokens_per_sec_per_gpu"] = float(total_tps) / float(num_gpus)

        r["tokens_per_sec_prefill"] = float(src["input_tokens_per_sec"]) if pd.notna(src.get("input_tokens_per_sec")) else np.nan
        r["tokens_per_sec_decode"] = float(src["output_tokens_per_sec"]) if pd.notna(src.get("output_tokens_per_sec")) else np.nan

        # Latency: seconds → ms, p90 → p95 slot
        for canon, solver_col in [
            ("ttft_ms_p50", "ttft_p50_s"),
            ("ttft_ms_p95", "ttft_p90_s"),
            ("ttft_ms_p99", "ttft_p99_s"),
            ("tpot_ms_p50", "tpot_p50_s"),
            ("tpot_ms_p95", "tpot_p90_s"),
            ("tpot_ms_p99", "tpot_p99_s"),
            ("e2e_ms_p50", "e2e_p50_s"),
            ("e2e_ms_p95", "e2e_p90_s"),
            ("e2e_ms_p99", "e2e_p99_s"),
        ]:
            val = src.get(solver_col)
            if pd.notna(val):
                r[canon] = float(val) * 1000.0

        r["price_per_instance_hour_usd"] = float(src["cost_per_hour"]) if pd.notna(src.get("cost_per_hour")) else np.nan
        r["cost_per_1m_tokens_total_usd"] = float(src["dollar_per_million_token"]) if pd.notna(src.get("dollar_per_million_token")) else np.nan
        r["total_cost_usd"] = float(src["total_cost"]) if pd.notna(src.get("total_cost")) else np.nan

        r["cloud"] = "aws"
        r["interconnect"] = "PCIe"

        il = src.get("max_input_length")
        if pd.notna(il):
            _set_input_len_single(r, float(il))
        ol = src.get("max_output_length")
        if pd.notna(ol):
            _set_output_len_single(r, float(ol))

        bs = src.get("batch_size")
        if pd.notna(bs):
            r["batch_size"] = float(bs)
            r["max_num_seqs"] = float(bs)

        nr = src.get("num_requests")
        if pd.notna(nr):
            r["num_requests"] = float(nr)

        rows.append(r)

    result = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    print(f"  solver: {len(result)} rows")
    return result


# ---------------------------------------------------------------------------
# Loader: vidur
# ---------------------------------------------------------------------------

def load_vidur() -> pd.DataFrame:
    """Load vidur/data.csv → canonical rows."""
    path = DATA_DIR / "vidur" / "data.csv"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    df = pd.read_csv(path)
    rows = []

    for _, src in df.iterrows():
        total_tps = src.get("total_tokens_per_sec")
        if pd.isna(total_tps) or total_tps == 0:
            continue

        r = _empty_row()
        r["data_source"] = "vidur"
        r["data_source_type"] = "simulated"

        model_name = str(src.get("model_name", ""))
        r["model_name"] = model_name

        arch_str, params_b, raw_config = _hf_lookup(model_name)
        r["model_architecture"] = arch_str
        r["params_billion"] = params_b
        r["model_config_json"] = json.dumps(raw_config, separators=(",", ":")) if raw_config else np.nan

        r["precision"] = "fp16"

        tp = int(src["tp"]) if pd.notna(src.get("tp")) else 1
        pp_val = src.get("pp", src.get("pipeline_stages", 1))
        pp = int(pp_val) if pd.notna(pp_val) else 1
        r["tp"] = tp
        r["pp"] = pp

        r["gpu_model"] = "A100"
        r["gpu_mem_gb"] = 80
        r["interconnect"] = "NVLink"

        num_gpus = src.get("num_gpus", src.get("num_replicas"))
        if pd.notna(num_gpus):
            r["gpu_count_total"] = int(num_gpus)

        r["tokens_per_sec_total"] = float(total_tps)
        if pd.notna(num_gpus) and num_gpus > 0:
            r["tokens_per_sec_per_gpu"] = float(total_tps) / float(num_gpus)

        r["tokens_per_sec_prefill"] = float(src["input_tokens_per_sec"]) if pd.notna(src.get("input_tokens_per_sec")) else np.nan
        r["tokens_per_sec_decode"] = float(src["output_tokens_per_sec"]) if pd.notna(src.get("output_tokens_per_sec")) else np.nan

        # Latency: seconds → ms, p90 → p95 slot
        for canon, vidur_col in [
            ("ttft_ms_p50", "ttft_p50_s"),
            ("ttft_ms_p95", "ttft_p90_s"),
            ("ttft_ms_p99", "ttft_p99_s"),
            ("tpot_ms_p50", "tpot_p50_s"),
            ("tpot_ms_p95", "tpot_p90_s"),
            ("tpot_ms_p99", "tpot_p99_s"),
            ("e2e_ms_p50", "e2e_p50_s"),
            ("e2e_ms_p95", "e2e_p90_s"),
            ("e2e_ms_p99", "e2e_p99_s"),
        ]:
            val = src.get(vidur_col)
            if pd.notna(val):
                r[canon] = float(val) * 1000.0

        r["runtime_stack"] = "vidur (sarathi scheduler)"
        r["task_type"] = "online"
        r["request_pattern"] = "queries_per_second"

        nr = src.get("num_requests")
        if pd.notna(nr):
            r["num_requests"] = float(nr)

        rps = src.get("rps")
        if pd.notna(rps):
            r["batch_size"] = float(rps)  # store rps as reference

        il = src.get("max_input_length")
        if pd.notna(il):
            _set_input_len_single(r, float(il))
        ol = src.get("max_output_length")
        if pd.notna(ol):
            _set_output_len_single(r, float(ol))

        r["instance_type"] = str(src.get("device_type", ""))

        rows.append(r)

    result = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    print(f"  vidur: {len(result)} rows")
    return result


# ---------------------------------------------------------------------------
# Loader: our_own_experiment
# ---------------------------------------------------------------------------

def load_our_experiment() -> pd.DataFrame:
    """Load our_own_experiment/data.csv → canonical rows."""
    path = DATA_DIR / "our_own_experiment" / "data.csv"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    df = pd.read_csv(path)
    rows = []

    for _, src in df.iterrows():
        total_tps = src.get("tokens_per_sec")
        if pd.isna(total_tps) or total_tps == 0:
            continue

        r = _empty_row()
        r["data_source"] = "our_experiment"
        r["data_source_type"] = "measured"

        model_name = str(src.get("model_name", ""))
        r["model_name"] = model_name

        arch_str, params_b, raw_config = _hf_lookup(model_name)
        r["model_architecture"] = arch_str
        r["params_billion"] = params_b
        r["model_config_json"] = json.dumps(raw_config, separators=(",", ":")) if raw_config else np.nan

        r["precision"] = "fp16"

        # Feature flags: explicitly disabled
        r["is_lmcache"] = "None"
        r["is_continuous_batching"] = "None"
        r["kv_offload_target"] = "None"
        r["cuda_graphs"] = "None"
        r["spec_decode"] = "None"

        tp = int(src["tp"]) if pd.notna(src.get("tp")) else 1
        pp = int(src.get("pp", 1)) if pd.notna(src.get("pp")) else 1
        r["tp"] = tp
        r["pp"] = pp
        r["gpu_count_total"] = tp * pp

        device_str = str(src.get("device_type", ""))
        r["instance_type"] = device_str

        num_nodes, base_instance, _ = _parse_device_type(device_str)
        r["num_nodes"] = num_nodes
        r["gpus_per_node"] = tp  # SkyPilot allocates tp GPUs per node

        gpu_model = _infer_gpu_type(device_str)
        r["gpu_model"] = gpu_model
        r["gpu_mem_gb"] = _gpu_mem(gpu_model)
        r["interconnect"] = "PCIe"

        r["cloud"] = "aws"
        r["region"] = "us-east-1"

        # Pricing: look up base instance price × num_nodes
        r["price_per_instance_hour_usd"] = _instance_price(device_str)

        r["tokens_per_sec_total"] = float(total_tps)
        r["tokens_per_sec_per_gpu"] = float(total_tps) / (tp * pp)

        r["tokens_per_sec_prefill"] = float(src["input_tokens_per_sec"]) if pd.notna(src.get("input_tokens_per_sec")) else np.nan
        r["tokens_per_sec_decode"] = float(src["output_tokens_per_sec"]) if pd.notna(src.get("output_tokens_per_sec")) else np.nan

        r["total_cost_usd"] = float(src["total_cost"]) if pd.notna(src.get("total_cost")) else np.nan
        r["cost_per_1m_tokens_total_usd"] = float(src["dollar_per_million_token"]) if pd.notna(src.get("dollar_per_million_token")) else np.nan

        r["runtime_stack"] = "vllm 0.10.0"
        r["task_type"] = "batched"
        r["request_pattern"] = "offline_batch"
        r["num_requests"] = 30

        il = src.get("max_input_length")
        if pd.notna(il):
            _set_input_len_single(r, float(il))
        ol = src.get("max_output_length")
        if pd.notna(ol):
            _set_output_len_single(r, float(ol))

        rows.append(r)

    result = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    print(f"  our_experiment: {len(result)} rows")
    return result


# ---------------------------------------------------------------------------
# Loader: our_own_experiment/perfdb
# ---------------------------------------------------------------------------

def load_our_experiment_perfdb() -> pd.DataFrame:
    """Load our_own_experiment/perfdb_l40s_llama70b.csv → canonical rows."""
    path = DATA_DIR / "our_own_experiment" / "perfdb_l40s_llama70b.csv"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    df = pd.read_csv(path)
    rows = []

    for _, src in df.iterrows():
        total_tps = src.get("Total Tokens Per Second")
        if pd.isna(total_tps) or total_tps == 0:
            continue

        r = _empty_row()
        r["data_source"] = "our_experiment_perfdb"
        r["data_source_type"] = "measured"

        # Strip _nolmcache suffix
        raw_name = str(src.get("Model Name", ""))
        model_name = re.sub(r"_nolmcache$", "", raw_name)
        r["model_name"] = model_name

        arch_str, params_b, raw_config = _hf_lookup(model_name)
        r["model_architecture"] = arch_str
        r["params_billion"] = params_b
        r["model_config_json"] = json.dumps(raw_config, separators=(",", ":")) if raw_config else np.nan

        r["precision"] = "fp16"

        # Feature flags: explicitly disabled
        r["is_lmcache"] = "None"
        r["is_continuous_batching"] = "None"
        r["kv_offload_target"] = "None"
        r["cuda_graphs"] = "None"
        r["spec_decode"] = "None"

        r["cloud"] = "aws"
        r["region"] = "us-east-1"

        tp = int(src["TP"]) if pd.notna(src.get("TP")) else 1
        pp = int(src["PP"]) if pd.notna(src.get("PP")) else 1
        r["tp"] = tp
        r["pp"] = pp
        r["gpu_count_total"] = tp * pp

        r["gpu_model"] = "L40S"
        r["gpu_mem_gb"] = float(src["Mem Per GPU GB"]) if pd.notna(src.get("Mem Per GPU GB")) else 48
        r["interconnect"] = "PCIe"

        r["tokens_per_sec_total"] = float(total_tps)
        r["tokens_per_sec_per_gpu"] = float(total_tps) / (tp * pp)

        il = src.get("Max Input Length")
        if pd.notna(il):
            _set_input_len_single(r, float(il))
        ol = src.get("Max Output Length")
        if pd.notna(ol):
            _set_output_len_single(r, float(ol))

        r["runtime_stack"] = "vllm 0.10.0"
        r["task_type"] = "batched"
        r["request_pattern"] = "offline_batch"

        rows.append(r)

    result = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    print(f"  our_experiment_perfdb: {len(result)} rows")
    return result


# ---------------------------------------------------------------------------
# Loader: splitwise
# ---------------------------------------------------------------------------

def load_splitwise() -> pd.DataFrame:
    """Load splitwise/data.csv → canonical rows.

    Source: SplitwiseSim profiling data from DGX-A100 and DGX-H100 machines.
    Times in the source CSV are in milliseconds.
    Multiple repetitions per (model, hardware, prompt_size, batch_size, token_size) config.
    """
    path = DATA_DIR / "splitwise" / "data.csv"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    df = pd.read_csv(path)
    rows = []

    for _, src in df.iterrows():
        r = _empty_row()
        r["data_source"] = "splitwise"
        r["data_source_type"] = "measured"

        # Model name normalization
        raw_model = str(src.get("model", ""))
        model_name = MODEL_NAME_MAP.get(raw_model, raw_model)
        r["model_name"] = model_name

        arch_str, params_b, raw_config = _hf_lookup(model_name)
        r["model_architecture"] = arch_str
        r["params_billion"] = params_b
        r["model_config_json"] = json.dumps(raw_config, separators=(",", ":")) if raw_config else np.nan

        r["precision"] = "fp16"

        # Hardware mapping
        hw_raw = str(src.get("hardware", ""))
        hw_info = SPLITWISE_HW_MAP.get(hw_raw, (hw_raw.upper(), np.nan))
        gpu_model, gpu_mem = hw_info
        r["gpu_model"] = gpu_model
        r["gpu_mem_gb"] = gpu_mem
        r["interconnect"] = "NVLink"  # DGX systems use NVLink

        tp = int(src["tensor_parallel"]) if pd.notna(src.get("tensor_parallel")) else 1
        r["tp"] = tp
        r["pp"] = 1  # splitwise data is TP-only
        r["gpu_count_total"] = tp

        # Instance type is DGX, not cloud
        r["instance_type"] = f"DGX-{gpu_model}"

        # Input/output lengths
        prompt_size = src.get("prompt_size")
        if pd.notna(prompt_size):
            _set_input_len_single(r, float(prompt_size))

        token_size = src.get("token_size")
        if pd.notna(token_size):
            _set_output_len_single(r, float(token_size))

        batch_size = src.get("batch_size")
        if pd.notna(batch_size):
            r["batch_size"] = float(batch_size)
            r["max_num_seqs"] = float(batch_size)

        # Latency (already in ms)
        prompt_time = src.get("prompt_time")
        token_time = src.get("token_time")
        e2e_time = src.get("e2e_time")

        if pd.notna(prompt_time):
            r["ttft_ms_p50"] = float(prompt_time)
        if pd.notna(token_time):
            r["tpot_ms_p50"] = float(token_time)
        if pd.notna(e2e_time):
            r["e2e_ms_p50"] = float(e2e_time)

        # Derive throughput from timing
        bs = float(batch_size) if pd.notna(batch_size) else 1.0
        ps = float(prompt_size) if pd.notna(prompt_size) else 0.0
        ts = float(token_size) if pd.notna(token_size) else 0.0

        if pd.notna(prompt_time) and prompt_time > 0:
            r["tokens_per_sec_prefill"] = (ps * bs) / (prompt_time / 1000.0)

        if pd.notna(token_time) and token_time > 0:
            r["tokens_per_sec_decode"] = bs / (token_time / 1000.0)

        if pd.notna(e2e_time) and e2e_time > 0:
            total_tokens = (ps + ts) * bs
            r["tokens_per_sec_total"] = total_tokens / (e2e_time / 1000.0)
            r["tokens_per_sec_per_gpu"] = r["tokens_per_sec_total"] / tp

        r["runtime_stack"] = "splitwise-sim profiling"
        r["task_type"] = "batched"

        rows.append(r)

    result = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    print(f"  splitwise: {len(result)} rows")
    return result


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------

def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived columns after concatenation."""
    # prefill_decode_ratio
    mask = df["input_len_tokens_avg"].notna() & df["output_len_tokens_avg"].notna()
    mask &= df["output_len_tokens_avg"] > 0
    df.loc[mask, "prefill_decode_ratio"] = (
        df.loc[mask, "input_len_tokens_avg"] / df.loc[mask, "output_len_tokens_avg"]
    )

    # Fill tokens_per_sec_per_gpu where missing
    missing_per_gpu = df["tokens_per_sec_per_gpu"].isna() & df["tokens_per_sec_total"].notna() & (df["gpu_count_total"] > 0)
    df.loc[missing_per_gpu, "tokens_per_sec_per_gpu"] = (
        df.loc[missing_per_gpu, "tokens_per_sec_total"] / df.loc[missing_per_gpu, "gpu_count_total"]
    )

    # cost_per_1m_tokens_prefill_usd
    mask_prefill = (
        df["price_per_instance_hour_usd"].notna()
        & df["tokens_per_sec_prefill"].notna()
        & (df["tokens_per_sec_prefill"] > 0)
    )
    df.loc[mask_prefill, "cost_per_1m_tokens_prefill_usd"] = (
        df.loc[mask_prefill, "price_per_instance_hour_usd"]
        / (df.loc[mask_prefill, "tokens_per_sec_prefill"] * 3.6)
    )

    # cost_per_1m_tokens_decode_usd
    mask_decode = (
        df["price_per_instance_hour_usd"].notna()
        & df["tokens_per_sec_decode"].notna()
        & (df["tokens_per_sec_decode"] > 0)
    )
    df.loc[mask_decode, "cost_per_1m_tokens_decode_usd"] = (
        df.loc[mask_decode, "price_per_instance_hour_usd"]
        / (df.loc[mask_decode, "tokens_per_sec_decode"] * 3.6)
    )

    # -----------------------------------------------------------------------
    # Phase 1: Parse HF configs → model structure columns
    # -----------------------------------------------------------------------
    configs = df["model_config_json"].apply(_parse_model_config)

    df["is_moe"] = configs.apply(
        lambda c: bool(c.get("num_local_experts", 0) > 0) if c else np.nan
    )
    df["num_experts_active"] = configs.apply(
        lambda c: c.get("num_experts_per_tok", np.nan) if c else np.nan
    )
    df["vocab_size"] = configs.apply(
        lambda c: c.get("vocab_size", np.nan) if c else np.nan
    )

    def _gqa_ratio(c):
        if c is None:
            return np.nan
        n_heads = c.get("num_attention_heads")
        n_kv = c.get("num_key_value_heads")
        if n_heads and n_kv and n_kv > 0:
            return n_heads / n_kv
        return np.nan

    df["attention_heads_per_kv_head"] = configs.apply(_gqa_ratio)

    def _num_kv_heads(c):
        if c is None:
            return np.nan
        return c.get("num_key_value_heads", np.nan)

    num_kv_heads = configs.apply(_num_kv_heads)
    tp_valid = df["tp"].where(df["tp"] > 0)
    df["kv_heads_per_tp"] = num_kv_heads / tp_valid

    # -----------------------------------------------------------------------
    # Phase 2: Sizing
    # -----------------------------------------------------------------------
    bytes_per_param = df["precision"].map({"fp8": 1, "fp16": 2, "bf16": 2}).fillna(2)
    df["model_size_gb"] = df["params_billion"] * bytes_per_param

    gpu_count_valid = df["gpu_count_total"].where(df["gpu_count_total"] > 0)
    df["params_per_gpu"] = df["params_billion"] / gpu_count_valid

    df["model_fits_single_gpu"] = np.where(
        df["model_size_gb"].notna() & df["gpu_mem_gb"].notna(),
        df["model_size_gb"] <= df["gpu_mem_gb"],
        np.nan,
    )

    df["vram_headroom_gb"] = np.where(
        df["gpu_mem_gb"].notna() & df["gpu_count_total"].notna() & df["model_size_gb"].notna(),
        (df["gpu_mem_gb"] * df["gpu_count_total"]) - df["model_size_gb"],
        np.nan,
    )

    # -----------------------------------------------------------------------
    # Phase 3: GPU hardware lookups
    # -----------------------------------------------------------------------
    df["gpu_bandwidth_gbps"] = df["gpu_model"].apply(
        lambda g: _gpu_spec(g, "memory_bandwidth_gbps")
    )
    df["gpu_tflops_fp16"] = df["gpu_model"].apply(
        lambda g: _gpu_spec(g, "fp16_tflops")
    )
    df["gpu_generation"] = df["gpu_model"].apply(
        lambda g: _gpu_spec(g, "architecture")
    )

    # -----------------------------------------------------------------------
    # Phase 4: Efficiency ratios
    # -----------------------------------------------------------------------
    params_valid = df["params_billion"].where(df["params_billion"] > 0)
    df["bandwidth_per_param"] = (
        df["gpu_bandwidth_gbps"] * df["tp"] / params_valid
    )
    df["flops_per_param"] = (
        df["gpu_tflops_fp16"] * df["tp"] / params_valid
    )

    # -----------------------------------------------------------------------
    # Phase 5: Topology inference — fill gpus_per_node and num_nodes
    # -----------------------------------------------------------------------
    for idx, row in df.iterrows():
        src = row["data_source"]
        gpn = row["gpus_per_node"]
        nn = row["num_nodes"]

        if pd.notna(gpn) and pd.notna(nn):
            continue  # already populated

        if src == "dynamo_swept":
            # DGX H100/H200, 8 GPUs/node, single node
            df.at[idx, "gpus_per_node"] = DGX_GPUS_PER_NODE
            df.at[idx, "num_nodes"] = 1

        elif src == "dynamo_test":
            # tp GPUs per node, single node
            df.at[idx, "gpus_per_node"] = row["tp"] if pd.notna(row["tp"]) else np.nan
            df.at[idx, "num_nodes"] = 1

        elif src == "solver":
            inst = row.get("instance_type") if pd.notna(row.get("instance_type")) else None
            if inst:
                inferred_gpn = _infer_gpus_per_node(inst)
                df.at[idx, "gpus_per_node"] = inferred_gpn
                gc = row["gpu_count_total"]
                if pd.notna(inferred_gpn) and inferred_gpn > 0 and pd.notna(gc):
                    _, base, _ = _parse_device_type(inst)
                    multi_match = re.match(r"(\d+)x\s+", inst)
                    if multi_match:
                        df.at[idx, "num_nodes"] = int(multi_match.group(1))
                    else:
                        import math
                        df.at[idx, "num_nodes"] = math.ceil(gc / inferred_gpn)

        elif src == "vidur":
            # A100 DGX-style, infer from gpu_count
            gc = row["gpu_count_total"]
            if pd.notna(gc):
                df.at[idx, "gpus_per_node"] = min(int(gc), DGX_GPUS_PER_NODE)
                import math
                df.at[idx, "num_nodes"] = math.ceil(gc / DGX_GPUS_PER_NODE)
            else:
                # Fallback: infer gpu_count from tp * pp
                tp_val = row["tp"] if pd.notna(row["tp"]) else 1
                pp_val = row["pp"] if pd.notna(row["pp"]) else 1
                gc = tp_val * pp_val
                df.at[idx, "gpu_count_total"] = gc
                df.at[idx, "gpus_per_node"] = min(int(gc), DGX_GPUS_PER_NODE)
                import math
                df.at[idx, "num_nodes"] = math.ceil(gc / DGX_GPUS_PER_NODE)

        elif src == "our_experiment":
            # Already populated by loader — skip
            pass

        elif src == "our_experiment_perfdb":
            # L40S on g6e, max 8/node
            gc = row["gpu_count_total"]
            if pd.notna(gc):
                df.at[idx, "gpus_per_node"] = min(int(gc), 8)
                import math
                df.at[idx, "num_nodes"] = math.ceil(gc / 8)

        elif src == "splitwise":
            # DGX, 8 GPUs/node
            df.at[idx, "gpus_per_node"] = DGX_GPUS_PER_NODE
            df.at[idx, "num_nodes"] = 1

    # crosses_node_boundary
    df["crosses_node_boundary"] = np.where(
        df["num_nodes"].notna(),
        df["num_nodes"] > 1,
        np.nan,
    )

    # Recompute gpu_count-dependent sizing columns (Phase 5 may have filled gpu_count_total)
    gpu_count_valid = df["gpu_count_total"].where(df["gpu_count_total"] > 0)
    df["params_per_gpu"] = df["params_billion"] / gpu_count_valid
    df["vram_headroom_gb"] = np.where(
        df["gpu_mem_gb"].notna() & df["gpu_count_total"].notna() & df["model_size_gb"].notna(),
        (df["gpu_mem_gb"] * df["gpu_count_total"]) - df["model_size_gb"],
        np.nan,
    )

    # -----------------------------------------------------------------------
    # Phase 6: Cost
    # -----------------------------------------------------------------------
    mask_price = df["price_per_instance_hour_usd"].notna() & (df["gpu_count_total"] > 0)
    df.loc[mask_price, "price_per_gpu_hour_usd"] = (
        df.loc[mask_price, "price_per_instance_hour_usd"]
        / df.loc[mask_price, "gpu_count_total"]
    )

    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> bool:
    """Run sanity checks. Returns True if all pass."""
    ok = True
    print("\n=== Validation ===")

    # 1. No negative throughput
    for col in ["tokens_per_sec_total", "tokens_per_sec_per_gpu", "tokens_per_sec_prefill", "tokens_per_sec_decode"]:
        neg = df[col].dropna() < 0
        if neg.any():
            print(f"  FAIL: {neg.sum()} negative values in {col}")
            ok = False

    # 2. gpu_count_total > 0
    bad_gpu = df["gpu_count_total"].dropna() <= 0
    if bad_gpu.any():
        print(f"  FAIL: {bad_gpu.sum()} rows with gpu_count_total <= 0")
        ok = False

    # 3. Latency sanity (warn if > 1M ms)
    for col in ["ttft_ms_p50", "tpot_ms_p50", "e2e_ms_p50"]:
        huge = df[col].dropna() > 1_000_000
        if huge.any():
            print(f"  WARN: {huge.sum()} rows in {col} > 1M ms (possible unconverted seconds?)")

    # 4. model_name format
    no_slash = df["model_name"].dropna().apply(lambda x: "/" not in str(x))
    if no_slash.any():
        print(f"  WARN: {no_slash.sum()} model_name values without '/' (may not be HF IDs)")

    # 5. model_size_gb > 0 where populated
    bad_size = df["model_size_gb"].dropna() <= 0
    if bad_size.any():
        models = df.loc[bad_size[bad_size].index, "model_name"].unique().tolist()
        print(f"  WARN: {bad_size.sum()} rows with model_size_gb <= 0 (params_billion=0 for: {models})")

    # 6. vram_headroom_gb not extremely negative
    extreme_neg = df["vram_headroom_gb"].dropna() < -100
    if extreme_neg.any():
        print(f"  WARN: {extreme_neg.sum()} rows with vram_headroom_gb < -100 GB")

    # 7. crosses_node_boundary not all NaN
    if df["crosses_node_boundary"].isna().all():
        print("  FAIL: crosses_node_boundary is all NaN")
        ok = False

    # 8. kv_heads_per_tp < 1 (info — KV head replication scenarios)
    kv_rep = df["kv_heads_per_tp"].dropna() < 1
    if kv_rep.any():
        print(f"  INFO: {kv_rep.sum()} rows with kv_heads_per_tp < 1 (KV head replication)")

    # 9. Summary
    print(f"\n  Total rows: {len(df)}")
    print(f"\n  Rows per source:")
    for src, cnt in df["data_source"].value_counts().items():
        print(f"    {src}: {cnt}")

    print(f"\n  NaN coverage per column:")
    for col in CANONICAL_COLUMNS:
        nan_pct = df[col].isna().mean() * 100
        if nan_pct > 0:
            print(f"    {col}: {nan_pct:.1f}% NaN")

    if ok:
        print("\n  All checks PASSED")
    else:
        print("\n  Some checks FAILED")

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build canonical profiling CSV")
    parser.add_argument("--validate", action="store_true", help="Run validation checks")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: canonical/data.csv)")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace API token (overrides HF_TOKEN env var)")
    args = parser.parse_args()

    # Set HF token from CLI if provided
    global HF_TOKEN
    if args.hf_token:
        HF_TOKEN = args.hf_token

    print("Loading data sources...")

    frames = [
        load_dynamo_swept(),
        load_dynamo_test(),
        load_solver(),
        load_vidur(),
        load_our_experiment(),
        load_our_experiment_perfdb(),
        load_splitwise(),
    ]

    df = pd.concat(frames, ignore_index=True)
    print(f"\nConcatenated: {len(df)} rows")

    df = compute_derived(df)

    # Ensure column order
    df = df[CANONICAL_COLUMNS]

    out_path = Path(args.output) if args.output else SCRIPT_DIR / "data.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")

    if args.validate:
        validate(df)


if __name__ == "__main__":
    main()
