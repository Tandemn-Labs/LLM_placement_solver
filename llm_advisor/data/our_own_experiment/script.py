import csv
import json
from dataclasses import dataclass
import subprocess
from collections import defaultdict
import sys
import argparse
from pathlib import Path
import requests
import signal
import atexit
import logging
from datetime import datetime

# GPU type configurations
# Each GPU type maps to: available GPU counts per instance, instance family, and pricing
GPU_CONFIGS = {
    "L40S": {
        "available_gpus": [1, 4, 8],
        "instance_family": "g6e",
        "pricing": {
            1: {"instance_type": "g6e.2xlarge", "price_per_hour": 0.99},
            4: {"instance_type": "g6e.12xlarge", "price_per_hour": 4.68},
            8: {"instance_type": "g6e.48xlarge", "price_per_hour": 13.35},
        }
    },
    "L4": {
        "available_gpus": [1, 4, 8],
        "instance_family": "g6",
        "pricing": {
            1: {"instance_type": "g6.2xlarge", "price_per_hour": 0.526},
            4: {"instance_type": "g6.12xlarge", "price_per_hour": 0.752},
            8: {"instance_type": "g6.48xlarge", "price_per_hour": 1.204},
        }
    },
    "A10G": {
        "available_gpus": [1, 4, 8],
        "instance_family": "g5",
        "pricing": {
            1: {"instance_type": "g5.2xlarge", "price_per_hour": 1.006},
            4: {"instance_type": "g5.12xlarge", "price_per_hour": 4.096},
            8: {"instance_type": "g5.48xlarge", "price_per_hour": 16.384},
        }
    },
    "A100_40gb": {
        "available_gpus": [1, 4, 8],
        "instance_family": "p4d",
        "pricing": {
            1: {"instance_type": "p4d.24xlarge", "price_per_hour": 32.77},  # 8 GPUs, but can use 1
            4: {"instance_type": "p4d.24xlarge", "price_per_hour": 32.77},  # 8 GPUs, but can use 4
            8: {"instance_type": "p4d.24xlarge", "price_per_hour": 32.77},  # 8 GPUs (40GB per GPU)
        }
    },
    "A100_80gb": {
        "available_gpus": [1, 4, 8],
        "instance_family": "p4de",
        "pricing": {
            1: {"instance_type": "p4de.24xlarge", "price_per_hour": 40.96},  # 8 GPUs, but can use 1
            4: {"instance_type": "p4de.24xlarge", "price_per_hour": 40.96},  # 8 GPUs, but can use 4
            8: {"instance_type": "p4de.24xlarge", "price_per_hour": 40.96},  # 8 GPUs (80GB per GPU)
        }
    },
    "H100": {
        "available_gpus": [1, 4, 8],
        "instance_family": "p5",
        "pricing": {
            1: {"instance_type": "p5.48xlarge", "price_per_hour": 98.32},  # 8 GPUs, but can use 1
            4: {"instance_type": "p5.48xlarge", "price_per_hour": 98.32},  # 8 GPUs, but can use 4
            8: {"instance_type": "p5.48xlarge", "price_per_hour": 98.32},  # 8 GPUs
        }
    },
}

# Default GPU type (for backward compatibility)
DEFAULT_GPU_TYPE = "L40S"

# VM selection strategies for mapping (TP, PP) -> (gpus_per_node, num_nodes)
# - prefer_single_node: Strategy 1 then fallback to Strategy 2 (current/default behavior)
# - fit_tp_then_scale: Always use Strategy 2 (fit TP within a node, then scale nodes)
VM_SELECTION_STRATEGIES = {
    "prefer_single_node": "Prefer single-node when TP×PP fits; otherwise fit TP per node and scale.",
    "fit_tp_then_scale": "Always fit TP per node (smallest instance >= TP) and scale nodes.",
}

# Global logger instance
logger = logging.getLogger("benchmark")

def setup_logger(log_file_path: Path):
    """Set up logger to write to both console and file."""
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler - prints to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)

    # File handler - writes to log file
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Track active cluster for cleanup on unexpected exit
_active_cluster = None
_active_cluster_file = Path("/tmp/.active_benchmark_cluster")

def set_active_cluster(cluster_name):
    """Set active cluster and persist to file for crash recovery."""
    global _active_cluster
    _active_cluster = cluster_name
    if cluster_name:
        _active_cluster_file.write_text(cluster_name)
    elif _active_cluster_file.exists():
        _active_cluster_file.unlink()

def get_active_cluster():
    """Get active cluster from memory or file."""
    global _active_cluster
    if _active_cluster:
        return _active_cluster
    if _active_cluster_file.exists():
        return _active_cluster_file.read_text().strip()
    return None

def cleanup_on_exit():
    """Cleanup handler for unexpected exits."""
    cluster = get_active_cluster()
    if cluster:
        print(f"\n⚠️  Cleanup triggered. Tearing down cluster: {cluster}")
        try:
            subprocess.run(["sky", "down", "-y", cluster], timeout=300)
            print(f"✅ Cluster {cluster} terminated.")
        except Exception as e:
            print(f"❌ Failed to terminate cluster {cluster}: {e}")
            print(f"   Run manually: sky down -y {cluster}")
        finally:
            # Clear the file
            if _active_cluster_file.exists():
                _active_cluster_file.unlink()

def signal_handler(signum, frame):
    """Handle Ctrl+C and other signals."""
    print(f"\n🛑 Received signal {signum}. Cleaning up...")
    cleanup_on_exit()
    sys.exit(1)

def check_orphaned_cluster():
    """Check if there's an orphaned cluster from a previous crashed run."""
    if _active_cluster_file.exists():
        cluster = _active_cluster_file.read_text().strip()
        print(f"\n⚠️  Found orphaned cluster from previous run: {cluster}")
        print(f"   This cluster may still be running and costing money!")
        response = input(f"   Terminate it now? [Y/n]: ").strip().lower()
        if response != 'n':
            print(f"   Terminating {cluster}...")
            subprocess.run(["sky", "down", "-y", cluster])
            _active_cluster_file.unlink()
            print(f"   ✅ Done.")
        else:
            print(f"   Skipped. Run 'sky down -y {cluster}' manually when ready.")
            _active_cluster_file.unlink()

# Register cleanup handlers
atexit.register(cleanup_on_exit)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def get_cluster_config(tp, pp, gpu_type=DEFAULT_GPU_TYPE, vm_strategy="prefer_single_node"):
    """
    Calculate the best instance configuration for given TP/PP and GPU type.

    Returns (gpus_per_node, num_nodes) that can accommodate TP×PP GPUs.

    Strategy:
    1. Prefer single-node configurations when possible (better for PP communication)
    2. Otherwise, use smallest instance that fits TP GPUs per node

    Examples:
        TP=4, PP=2 → 8 GPUs/node × 1 node = 8 GPUs (single node preferred, avoids inter-node latency)
        TP=2, PP=4 → 8 GPUs/node × 1 node = 8 GPUs (single node preferred if available)
        TP=8, PP=1 → 8 GPUs/node × 1 node = 8 GPUs
        TP=4, PP=4 → 4 GPUs/node × 4 nodes = 16 GPUs (no single instance fits 16 GPUs, falls back to multi-node)
    """
    if pp < 1:
        raise ValueError(f"Invalid PP={pp}. Must be >= 1")
    if tp < 1:
        raise ValueError(f"Invalid TP={tp}. Must be >= 1")
    if gpu_type not in GPU_CONFIGS:
        raise ValueError(f"Invalid GPU type: {gpu_type}. Must be one of {list(GPU_CONFIGS.keys())}")

    if vm_strategy not in VM_SELECTION_STRATEGIES:
        raise ValueError(f"Invalid vm_strategy: {vm_strategy}. Must be one of {list(VM_SELECTION_STRATEGIES.keys())}")

    total_gpus = tp * pp
    available_gpus = GPU_CONFIGS[gpu_type]["available_gpus"]

    # Strategy 1: Prefer single-node configuration if possible (better for PP communication)
    # Only used when vm_strategy == "prefer_single_node".
    if vm_strategy == "prefer_single_node":
        # Check if there's an instance type that can fit all GPUs in one node
        single_node_gpus = None
        for gpu_count in available_gpus:
            if gpu_count >= total_gpus:
                single_node_gpus = gpu_count
                break

        if single_node_gpus is not None:
            # Single node is possible and preferred (avoids inter-node network latency)
            return single_node_gpus, 1

    # Strategy 2: Multi-node fallback - find smallest instance that fits TP GPUs
    # This ensures TP communication stays within a node (fast)
    gpus_per_node = None
    for gpu_count in available_gpus:
        if gpu_count >= tp:
            gpus_per_node = gpu_count
            break

    if gpus_per_node is None:
        raise ValueError(f"TP={tp} exceeds max GPUs per instance ({max(available_gpus)}) for {gpu_type}")

    # Calculate how many nodes we need
    num_nodes = (total_gpus + gpus_per_node - 1) // gpus_per_node  # Ceiling division

    # Make sure we have enough total GPUs
    actual_gpus = gpus_per_node * num_nodes
    if actual_gpus < total_gpus:
        raise ValueError(f"Cannot fit TP={tp}, PP={pp} ({total_gpus} GPUs) into available {gpu_type} instances")

    return gpus_per_node, num_nodes


def load_experiments(csv_path):
    experiments = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            experiments.append({
                'tp': int(row['tensor_degree']),
                'pp': int(row['pipeline_degree']),
                'max_input_length': int(row['max_input_length']),
                'max_output_length': int(row['max_output_length']),
                'model': row['model']
            })
    return experiments

def group_by_cluster(experiments, gpu_type=DEFAULT_GPU_TYPE, vm_strategy="prefer_single_node"):
    groups = defaultdict(list)
    for exp in experiments:
        gpus_per_node, num_nodes = get_cluster_config(exp['tp'], exp['pp'], gpu_type, vm_strategy=vm_strategy)
        key = (gpus_per_node, num_nodes)
        groups[key].append(exp)
    return dict(groups)

def group_by_input_output_then_cluster(experiments, gpu_type=DEFAULT_GPU_TYPE, vm_strategy="prefer_single_node"):
    """Group experiments first by input/output length, then by cluster config."""
    # First level: group by input/output length
    io_groups = defaultdict(list)
    for exp in experiments:
        io_key = (exp['max_input_length'], exp['max_output_length'])
        io_groups[io_key].append(exp)
    
    # Second level: within each IO group, group by cluster config
    result = {}
    for io_key, io_exps in io_groups.items():
        cluster_groups = defaultdict(list)
        for exp in io_exps:
            gpus_per_node, num_nodes = get_cluster_config(exp['tp'], exp['pp'], gpu_type, vm_strategy=vm_strategy)
            cluster_key = (gpus_per_node, num_nodes)
            cluster_groups[cluster_key].append(exp)
        result[io_key] = dict(cluster_groups)
    
    return result

def cleanup_old_benchmark_files(work_dir=None):
    """
    Remove old benchmark files that don't have GPU type in their names.
    These files are from the old naming scheme and are now obsolete.
    
    Old naming pattern: roofline-tp{TP}-pp{PP}[-{input}in-{output}out].yaml
    New naming pattern: roofline-tp{TP}-pp{PP}-{input}in-{output}out-{GPU_TYPE}.yaml
    
    This function identifies old files by checking if they don't end with any known GPU type.
    """
    if work_dir is None:
        work_dir = Path("roofline_benchmarks")
    
    if not work_dir.exists():
        print("ℹ️  roofline_benchmarks directory doesn't exist")
        return 0
    
    # List of known GPU types to check against
    known_gpu_types = list(GPU_CONFIGS.keys())
    gpu_type_suffixes = [gpu.replace("_", "-").replace("/", "-") for gpu in known_gpu_types]
    
    removed_count = 0
    removed_files = []
    
    # Check YAML files
    for file_path in work_dir.glob("roofline-*.yaml"):
        # Check if filename doesn't end with any GPU type suffix
        filename = file_path.stem  # without .yaml extension
        has_gpu_type = any(filename.endswith(f"-{suffix}") for suffix in gpu_type_suffixes)
        
        if not has_gpu_type:
            removed_files.append(file_path.name)
            file_path.unlink()
            removed_count += 1
    
    # Check Python benchmark scripts
    for file_path in work_dir.glob("benchmark_roofline-*.py"):
        # Check if filename doesn't end with any GPU type suffix
        filename = file_path.stem.replace("benchmark_roofline-", "")  # remove prefix
        has_gpu_type = any(filename.endswith(f"-{suffix}") for suffix in gpu_type_suffixes)
        
        if not has_gpu_type:
            removed_files.append(file_path.name)
            file_path.unlink()
            removed_count += 1
    
    if removed_count > 0:
        print(f"🗑️  Removed {removed_count} old benchmark files:")
        for fname in sorted(removed_files):
            print(f"   - {fname}")
        print(f"✅ Cleaned up {removed_count} old benchmark files")
    else:
        print("ℹ️  No old benchmark files to clean up")
    
    return removed_count

def generate_yaml(gpus_per_node, num_nodes, cluster_name, experiments, gpu_type=DEFAULT_GPU_TYPE):
    lmcache_exports = """
  # Ensure CUDA libraries are in LD_LIBRARY_PATH for PyTorch
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
  
  # LMCache environment variables (must be set before Ray workers start)
  export LMCACHE_USE_EXPERIMENTAL="True"
  export LMCACHE_CHUNK_SIZE="256"
  export LMCACHE_LOCAL_CPU="True"
  export LMCACHE_MAX_LOCAL_CPU_SIZE="40.0"
  export LMCACHE_SAVE_UNFULL_CHUNK="True"
  # export VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE="shm"
  # export VLLM_USE_RAY_WRAPPED_PP_COMM=1
  export LMCACHE_ENABLE_ASYNC_LOADING="False"
  export LMCACHE_REMOTE_SERDE="cachegen"
  export LMCACHE_USE_LAYERWISE="True"
  export LMCACHE_ENABLE_LAZY_MEMORY_ALLOCATOR="True"
  export NCCL_P2P_DISABLE=0
  # export NCCL_P2P_LEVEL=SYS
  # export NCCL_ALGO=Tree
  # export NCCL_MIN_NCHANNELS=4
  export NCCL_TIMEOUT=3600
  export TORCH_NCCL_BLOCKING_WAIT=1
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export TORCH_NCCL_TRACE_BUFFER_SIZE=10000
  export TORCH_DISTRIBUTED_DEBUG=DETAIL
  export NCCL_DEBUG=INFO
  # export LMCACHE_LOOKUP_TIMEOUT_MS="12000"
  # export LMCACHE_LOCAL_DISK="/tmp/lmcache_disk"
  # export LMCACHE_MAX_LOCAL_DISK_SIZE="100"
  # export LMCACHE_DISK_PERSISTENCE="True"
  # export LMCACHE_LOG_LEVEL="INFO"
"""

    # Check if any experiment needs Ray (PP > 1)
    needs_ray = any(exp['pp'] > 1 for exp in experiments)

    ray_run = f"""{lmcache_exports}
  python roofline_benchmarks/benchmark_{cluster_name}.py
    """
    if num_nodes > 1:
        # Multi-node: use Sky's Ray cluster startup script
        ray_run = f"""{lmcache_exports}
  # Start Ray cluster across all nodes
  export RAY_CMD="uv run ray"
  export RAY_DASHBOARD_HOST=0.0.0.0
  export RAY_HEAD_PORT=6379
  export RAY_DASHBOARD_PORT=8265
  export RAY_CGRAPH_get_timeout=1800
  export RAY_CGRAPH_submit_timeout=180
  ~/sky_templates/ray/start_cluster

  if [ "${{SKYPILOT_NODE_RANK}}" = "0" ]; then
      echo "Ray cluster started on head node"
      ray status --address="127.0.0.1:${{RAY_HEAD_PORT}}" || true

      # Run benchmark here while Ray is still up
      PYTHONHASHSEED=0 python roofline_benchmarks/benchmark_{cluster_name}.py
  fi
        """
    elif needs_ray:
        # Single-node but needs Ray for pipeline parallelism
        ray_run = f"""{lmcache_exports}
  # Start Ray on single node for pipeline parallelism
  export RAY_DASHBOARD_HOST=0.0.0.0
  export RAY_HEAD_PORT=6379
  export RAY_DASHBOARD_PORT=8265

  echo "Starting Ray for single-node pipeline parallelism..."
  uv run ray start --head --port=${{RAY_HEAD_PORT}} --dashboard-host=${{RAY_DASHBOARD_HOST}} --dashboard-port=${{RAY_DASHBOARD_PORT}} --num-cpus=0

  sleep 5
  uv run ray status --address="127.0.0.1:${{RAY_HEAD_PORT}}"

  # Run benchmark
  PYTHONHASHSEED=0 python roofline_benchmarks/benchmark_{cluster_name}.py

  # Cleanup Ray (non-fatal - benchmark may have already cleaned up vLLM workers)
  # vLLM's Ray workers for pipeline parallelism may still be active, causing Ray to crash during shutdown.
  # This is a known issue: Ray workers holding GPU resources can't be cleanly shut down.
  # Since the benchmark completed successfully, we don't fail the job if Ray cleanup has issues.
  echo "Cleaning up Ray (this may show warnings if vLLM workers are still active)..."
  sleep 2  # Give vLLM workers a moment to finish cleanup
  uv run ray stop || echo "⚠️  Ray stop completed with warnings (this is OK - benchmark succeeded)"
        """
    # Handle A100 variants: Only specify accelerators, let SkyPilot choose instance
    # SkyPilot will automatically select the right instance type and install drivers
    # DON'T specify instance_type as it can cause issues with driver installation
    accelerator_spec = ""
    instance_type_constraint = ""
    if gpu_type in ["A100_40gb", "A100-40gb"]:
        # Use "A100" without memory suffix - SkyPilot will select p4d.24xlarge (40GB)
        accelerator_name = "A100"
        accelerator_spec = f"  accelerators: {accelerator_name}:{gpus_per_node}\n"
    elif gpu_type in ["A100_80gb", "A100-80gb"]:
        # Use "A100-80GB" with memory suffix - SkyPilot will select p4de.24xlarge (80GB)
        accelerator_name = "A100-80GB"
        accelerator_spec = f"  accelerators: {accelerator_name}:{gpus_per_node}\n"
    else:
        # For other GPU types, specify accelerators normally
        accelerator_spec = f"  accelerators: {gpu_type}:{gpus_per_node}\n"
    
    # Determine if this is an A100 GPU type
    is_a100 = gpu_type.upper().startswith("A100")

    return f"""
name: {cluster_name}
resources:
  cloud: aws
{accelerator_spec}{instance_type_constraint}  use_spot: false
  disk_size: 500GB
  memory: "64GB+"
  # No region constraint - SkyPilot will try all available AWS regions
  # This helps find capacity when us-east-1 is full (especially during US business hours)
num_nodes: {num_nodes}
workdir: .
setup: |
  export PYTHONHASHSEED=0
  set -euxo pipefail

  # GPU type for conditional setup (A100 requires special handling)
  GPU_TYPE="{gpu_type}"
  IS_A100={"true" if is_a100 else "false"}
  echo "GPU Type: $GPU_TYPE, Is A100: $IS_A100"

  # Diagnostics: Check NVIDIA driver and CUDA availability BEFORE any setup
  echo "=== Initial System Diagnostics ==="
  echo "Checking for NVIDIA driver..."
  nvidia-smi || echo "WARNING: nvidia-smi not found - drivers may need installation!"
  echo "Checking CUDA libraries..."
  ldconfig -p | grep cuda || echo "INFO: No CUDA libraries in ldconfig yet"
  echo "CUDA toolkit path:"
  ls -la /usr/local/cuda* 2>/dev/null || echo "INFO: No /usr/local/cuda* found yet"
  echo "==================================="

  # Check if nvidia-smi exists; if not, we might need to install drivers
  # However, on AWS GPU instances, drivers should already be installed via the AMI
  # SkyPilot should handle this automatically when accelerators are specified
  if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  NVIDIA drivers not found! This should not happen with GPU instances."
    echo "⚠️  SkyPilot should have installed drivers. Checking if this is p4de/p4d instance..."
    # On AWS p4d/p4de, drivers are usually pre-installed in the AMI
    # If they're missing, SkyPilot setup might have failed
  fi

  # ========================================================================
  # NVIDIA Fabric Manager - ONLY for A100-SXM4 (NVSwitch) systems
  # ========================================================================
  if [ "$IS_A100" = "true" ]; then
  # On p4d/p4de instances with A100-SXM4 GPUs connected via NVSwitch:
  #   - CUDA Error 802 ("system not yet initialized") occurs without Fabric Manager
  #   - nvidia-smi shows GPUs but CUDA runtime cannot access them
  #   - NVLink topology shows only PCIe (PHB/NODE/SYS) instead of NV#
  #
  # The Fabric Manager service is REQUIRED to:
  #   - Initialize the NVSwitch fabric topology
  #   - Enable peer-to-peer GPU communication via NVLink
  #   - Allow CUDA runtime to properly enumerate and access GPUs
  # ========================================================================
  echo "=== Installing NVIDIA Fabric Manager for A100-SXM4 NVSwitch support ==="

  # Get the FULL driver version (e.g., 535.216.01) to install EXACT matching Fabric Manager
  # The Fabric Manager MUST match the driver version EXACTLY or it will fail to start
  # Note: Use --id=0 to query single GPU instead of piping to head (avoids SIGPIPE with set -e)
  DRIVER_VERSION_FULL=$(nvidia-smi --id=0 --query-gpu=driver_version --format=csv,noheader)
  DRIVER_VERSION_MAJOR=$(echo "$DRIVER_VERSION_FULL" | cut -d. -f1)
  echo "Detected NVIDIA driver version: $DRIVER_VERSION_FULL (major: $DRIVER_VERSION_MAJOR)"

  # Check if Fabric Manager is already installed and running
  if systemctl is-active --quiet nvidia-fabricmanager 2>/dev/null; then
    echo "✅ NVIDIA Fabric Manager is already running"
  else
    echo "Installing NVIDIA Fabric Manager..."
    sudo apt-get update

    # Check available Fabric Manager versions
    echo "Available Fabric Manager versions:"
    apt-cache madison nvidia-fabricmanager-${{DRIVER_VERSION_MAJOR}} 2>/dev/null | awk 'NR<=5' || true

    # Try to install the EXACT version matching the driver
    # Format: nvidia-fabricmanager-535=535.216.01-1
    FM_INSTALLED=false
    echo "Attempting to install exact version: nvidia-fabricmanager-${{DRIVER_VERSION_MAJOR}}=${{DRIVER_VERSION_FULL}}-1"
    if sudo apt-get install -y "nvidia-fabricmanager-${{DRIVER_VERSION_MAJOR}}=${{DRIVER_VERSION_FULL}}-1" 2>/dev/null; then
      echo "✅ Installed exact Fabric Manager version ${{DRIVER_VERSION_FULL}}"
      FM_INSTALLED=true
    fi

    # If exact version failed, we need to UPDATE the driver to match available Fabric Manager
    if [ "$FM_INSTALLED" = "false" ]; then
      echo "⚠️  Exact Fabric Manager version ${{DRIVER_VERSION_FULL}} not available in apt repository"
      echo "The AWS AMI has driver ${{DRIVER_VERSION_FULL}} but NVIDIA repo has newer Fabric Manager"
      echo ""
      echo "Solution: Update NVIDIA driver to match the available Fabric Manager version"

      # Get the latest available Fabric Manager version for this major
      # Note: Use awk 'NR==1' instead of head -1 to avoid SIGPIPE with set -e
      FM_LATEST=$(apt-cache madison nvidia-fabricmanager-${{DRIVER_VERSION_MAJOR}} 2>/dev/null | awk 'NR==1 {{print $3}}' | sed 's/-1$//')
      echo "Latest available Fabric Manager: $FM_LATEST"

      if [ -n "$FM_LATEST" ]; then
        echo "Updating NVIDIA driver to version $FM_LATEST to match Fabric Manager..."

        # Install matching driver version
        # The driver package is nvidia-driver-535 or similar
        sudo apt-get install -y --allow-downgrades \
          nvidia-driver-${{DRIVER_VERSION_MAJOR}}=${{FM_LATEST}}-1 \
          nvidia-dkms-${{DRIVER_VERSION_MAJOR}}=${{FM_LATEST}}-1 \
          nvidia-kernel-source-${{DRIVER_VERSION_MAJOR}}=${{FM_LATEST}}-1 \
          2>/dev/null || {{
            echo "Could not update driver, trying alternative approach..."
            # Try installing just the Fabric Manager - sometimes the versions are close enough
            sudo apt-get install -y nvidia-fabricmanager-${{DRIVER_VERSION_MAJOR}} || true
          }}

        # Now install Fabric Manager
        sudo apt-get install -y nvidia-fabricmanager-${{DRIVER_VERSION_MAJOR}} || true
        FM_INSTALLED=true
      fi
    fi

    # Start and enable the Fabric Manager service
    echo "Starting NVIDIA Fabric Manager service..."
    sudo systemctl start nvidia-fabricmanager || echo "Warning: Failed to start Fabric Manager"
    sudo systemctl enable nvidia-fabricmanager || echo "Warning: Failed to enable Fabric Manager"

    # Give Fabric Manager time to initialize the NVSwitch fabric
    echo "Waiting for Fabric Manager to initialize NVSwitch fabric..."
    sleep 5

    # Verify Fabric Manager is running
    if systemctl is-active --quiet nvidia-fabricmanager; then
      echo "✅ NVIDIA Fabric Manager started successfully"
      # Verify new driver version (use --id=0 to avoid SIGPIPE)
      nvidia-smi --id=0 --query-gpu=driver_version --format=csv,noheader
    else
      echo "⚠️  NVIDIA Fabric Manager failed to start"
      echo "This usually means version mismatch between driver and Fabric Manager"
      echo "Driver version: $(nvidia-smi --id=0 --query-gpu=driver_version --format=csv,noheader)"
      echo "Installed Fabric Manager:"
      dpkg -l | grep nvidia-fabricmanager || true
      systemctl status nvidia-fabricmanager 2>&1 || true
      echo ""
      echo "⚠️  CUDA will NOT work on this A100-SXM4 instance without Fabric Manager!"
      echo "⚠️  Consider using a different AMI with matching driver/Fabric Manager versions"
    fi
  fi

  # Verify NVLink topology after Fabric Manager
  echo "=== Verifying NVLink topology ==="
  nvidia-smi topo -m 2>&1 | awk 'NR<=20' || echo "NVLink topology check failed"
  echo "================================="
  fi  # End of A100-specific Fabric Manager section

  # Install numactl for NUMA diagnostics
  sudo apt-get install -y numactl 2>/dev/null || echo "numactl installation skipped"

  python3 -m pip install -U pip
  python3 -m pip install -U uv

  uv venv --python 3.12 --seed
  source .venv/bin/activate

  # Install dependencies
  uv pip install "datasets" "requests" "pynvml"

  # ========================================================================
  # vLLM + PyTorch Installation - GPU-specific versions
  # ========================================================================
  # A100 (p4d/p4de): driver 535.x only supports CUDA 12.1, needs vLLM 0.7.3 + torch 2.5.1
  # L40S/L4/others: driver 550.x+ supports CUDA 12.4, can use vLLM 0.10.0 + torch 2.7.1
  # ========================================================================

  if [ "$IS_A100" = "true" ]; then
    # A100-specific: older driver requires cu121 and older vLLM
    echo "=== Installing PyTorch 2.5.1 with CUDA 12.1 (for A100 driver 535.x) ==="
    uv pip install --index-url https://download.pytorch.org/whl/cu121 \
      "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1"

    echo "=== Installing vLLM 0.7.3 (for A100 with torch 2.5.x) ==="
    uv pip install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu121 "vllm==0.7.3"
  else
    # L40S/L4/others: use latest stable vLLM with default PyTorch
    echo "=== Installing vLLM 0.10.0 (latest stable for L40S/L4/other GPUs) ==="
    uv pip install "vllm==0.10.0"
  fi

  # Verify PyTorch has CUDA support
  echo "=== PyTorch CUDA Check ==="
  python3 -c "import torch; print('torch.cuda.is_available():', torch.cuda.is_available()); print('torch.version.cuda:', torch.version.cuda); print('torch.__version__:', torch.__version__); print('torch.cuda.device_count():', torch.cuda.device_count() if torch.cuda.is_available() else 'N/A')" || echo "PyTorch CUDA check failed!"
  echo "==========================="
  
  # ========================================================================
  # LMCACHE DISABLED FOR DEBUGGING CUDA ISSUES
  # LMCache may be overwriting torch with CPU-only version causing CUDA errors
  # Re-enable once CUDA is working on A100
  # ========================================================================
  # # Install LMCache v0.2.1 (compatible with vLLM 0.7.x)
  # # See: https://docs.lmcache.ai/getting_started/installation.html
  # # Note: v0.2.2 doesn't exist! Available v0.2.x tags: v0.2.0, v0.2.1
  # # Using --no-build-isolation to ensure torch version compatibility
  # rm -rf lmcache
  # git clone https://github.com/lmcache/lmcache.git
  # cd lmcache
  # git checkout v0.2.1
  # # Set CUDA architecture explicitly to avoid auto-detection (which may fail during setup)
  # # A100 = 8.0, L40S = 8.9, L4 = 7.5, H100 = 9.0
  # export TORCH_CUDA_ARCH_LIST="8.0;8.9;7.5;9.0"
  # export FORCE_CUDA="1"
  # uv pip install . --no-build-isolation || {{
  #   # Fallback: try with just A100 arch (most common)
  #   export TORCH_CUDA_ARCH_LIST="8.0"
  #   uv pip install . --no-build-isolation
  # }}
  # cd ..

  # # ========================================================================
  # # CRITICAL FIX: LMCache dependencies overwrite torch with CPU-only version!
  # # The log showed: - torch==2.5.1+cu121  ->  + torch==2.7.0 (CPU-only)
  # # We MUST force reinstall CUDA-enabled torch AFTER LMCache installation
  # # Using cu121 for A100 driver 535.x compatibility
  # # ========================================================================
  # echo "=== CRITICAL: Reinstalling PyTorch with CUDA 12.1 (after LMCache) ==="
  # uv pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu121 \
  #   "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1"

  # # Verify torch is now CUDA-enabled
  # echo "=== Verifying torch CUDA after reinstall ==="
  # python3 -c "import torch; print('torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

  echo "=== LMCACHE DISABLED - Skipping LMCache installation ==="
  mkdir -p /tmp/lmcache_disk

run: |
  set -euxo pipefail

  # CRITICAL: Set LD_LIBRARY_PATH for CUDA libraries FIRST
  # This must be done before any Python imports that use CUDA
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${{LD_LIBRARY_PATH:-}}"
  export CUDA_HOME="${{CUDA_HOME:-/usr/local/cuda}}"

  # Diagnostics before running benchmark
  echo "=== Pre-run Diagnostics ==="
  nvidia-smi || echo "WARNING: nvidia-smi not available in run phase!"

  # A100-SXM4 specific diagnostics - check for Fabric Manager
  echo "--- A100 SXM4 Diagnostics ---"
  echo "Checking NVIDIA Fabric Manager (required for NVSwitch on A100-SXM4)..."
  systemctl status nvidia-fabricmanager 2>&1 || echo "Fabric Manager status check failed or not present"

  echo "Checking for NVLink topology..."
  nvidia-smi topo -m 2>&1 || echo "NVLink topology check failed"

  echo "--- NUMA Topology and GPU-NUMA Mapping ---"
  echo "NUMA hardware layout:"
  numactl --hardware 2>&1 || echo "numactl not available"
  echo ""
  echo "GPU PCIe Bus IDs and NUMA node assignments:"
  gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 || echo "0")
  if [ "$gpu_count" -gt 0 ] 2>/dev/null; then
    for gpu in $(seq 0 $((gpu_count - 1))); do
      bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i $gpu 2>/dev/null)
      if [ -n "$bus" ]; then
        # Convert to lowercase and proper sysfs path format (0000:XX:YY.Z)
        bus_lower=$(echo "$bus" | tr '[:upper:]' '[:lower:]')
        numa_node=$(cat /sys/bus/pci/devices/$bus_lower/numa_node 2>/dev/null || echo "N/A")
        local_cpus=$(cat /sys/bus/pci/devices/$bus_lower/local_cpulist 2>/dev/null || echo "N/A")
        echo "GPU$gpu: PCIe=$bus, NUMA_node=$numa_node, local_cpus=$local_cpus"
      else
        echo "GPU$gpu: Could not get PCIe bus ID"
      fi
    done
  else
    echo "Could not detect GPU count"
  fi
  echo "-------------------------------------------"

  echo "Checking for GPU processes and errors..."
  nvidia-smi -q -d ERRORS 2>&1 | awk 'NR<=50' || echo "Error check failed"

  echo "--- CUDA Device Query ---"
  # Try direct CUDA device query through Python (single line to avoid YAML issues)
  python3 -c "import ctypes; cudart = ctypes.CDLL('libcudart.so.12'); count = ctypes.c_int(); result = cudart.cudaGetDeviceCount(ctypes.byref(count)); print(f'cudaGetDeviceCount result: {{result}} (0=success), device count: {{count.value}}')" 2>&1 || echo "Direct CUDA query script failed"
  echo "==========================="

  source .venv/bin/activate

  # Critical: Detailed PyTorch CUDA diagnostics before running benchmark
  echo "=== DETAILED PyTorch CUDA Diagnostics (in run phase) ==="
  echo "--- Environment ---"
  echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH" || true
  echo "CUDA_HOME: $CUDA_HOME" || true
  which python3
  echo "--- Torch info ---"
  python3 -c "import torch; print('torch.__version__:', torch.__version__); print('torch.__file__:', torch.__file__); print('torch.version.cuda:', torch.version.cuda); print('torch.backends.cuda.is_built():', torch.backends.cuda.is_built())"
  echo "--- CUDA check ---"
  python3 -c "import torch; print('torch.cuda.is_available():', torch.cuda.is_available()); print('torch.cuda.device_count():', torch.cuda.device_count() if torch.cuda.is_available() else 'N/A - checking why...')"
  echo "--- Loading CUDA libs directly ---"
  python3 -c "import ctypes; libs=['libcuda.so.1','libcudart.so.12','libnvidia-ml.so.1']; [print(l+': OK') if ctypes.CDLL(l) else None for l in libs]" 2>&1 || echo "Some libs failed"
  echo "--- Manual CUDA init attempt ---"
  python3 -c "import torch; torch.cuda.init(); print('torch.cuda.init() OK, device_count:', torch.cuda.device_count())" 2>&1 || echo "CUDA init failed"
  echo "--- Installed torch packages ---"
  pip list | grep -i -E "torch|nvidia|cuda|triton"
  echo "========================================="
{ray_run}

  echo "Cluster ready for benchmarking"
"""

DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1463127774414770227/PX_asJ7IXfvQrff6vCW-yGL9k7pBk1v2ZMmRyzUdMNS3mht3pDVO8ligaclUAAFItbvz"

def send_discord_message(message):
    payload = {"content": message}
    requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)

def generate_benchmark_script(experiments, gpus_per_node, num_nodes, gpu_type=DEFAULT_GPU_TYPE):
    """Generate Python benchmark script for a set of experiments."""
    exp_list = json.dumps(experiments, indent=2)

    # Calculate cluster pricing info
    gpu_config = GPU_CONFIGS[gpu_type]
    price_per_node = gpu_config["pricing"][gpus_per_node]["price_per_hour"]
    total_price_per_hour = price_per_node * num_nodes
    instance_type = gpu_config["pricing"][gpus_per_node]["instance_type"]
    if num_nodes > 1:
        cluster_instance_type = f"{num_nodes}x {instance_type}"
    else:
        cluster_instance_type = instance_type
    
    return f'''#!/usr/bin/env python3
import os
os.environ["RAY_ADDRESS"] = "127.0.0.1:6379"
os.environ["LMCACHE_LOG_LEVEL"] = "INFO"

import requests
import json
import time
import gc

# DO NOT import torch or call any CUDA functions before vLLM!
# vLLM 0.10.0's V1 engine uses multiprocessing with fork.
# If CUDA is initialized before fork, workers fail with:
#   "Cannot re-initialize CUDA in forked subprocess"
# Let vLLM handle CUDA initialization internally.

import ray
from datasets import load_dataset
from vllm import LLM, SamplingParams

# Import torch AFTER vLLM to avoid "Cannot re-initialize CUDA in forked subprocess" error
# vLLM 0.10.0's V1 engine uses multiprocessing with fork, and CUDA must not be initialized before fork
import torch

# Handle API differences between vLLM versions
try:
    from vllm.config import KVTransferConfig
except ImportError:
    KVTransferConfig = None

try:
    from vllm.distributed import cleanup_dist_env_and_memory
except ImportError:
    # Fallback for older vLLM versions
    def cleanup_dist_env_and_memory(shutdown_ray=True):
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        if shutdown_ray:
            ray.shutdown()

import shutil
import subprocess
import threading
from collections import defaultdict

# ============================================================================
# Distributed GPU Monitoring with Ray Actors
# ============================================================================

@ray.remote
class GPUMonitorActor:
    """Ray actor for GPU monitoring on a single node."""

    def __init__(self, node_id: str, sample_interval: float = 0.5):
        self.node_id = node_id
        self.sample_interval = sample_interval
        self.timeseries = []
        self._start_time = None
        self._stop_event = None
        self._pynvml_available = False
        self._pynvml = None
        self._device_count = 0

        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml_available = True
            self._pynvml = pynvml
            self._device_count = pynvml.nvmlDeviceGetCount()
            print(f"[{{self.node_id}}] GPU monitor initialized with {{self._device_count}} GPUs")
        except Exception as e:
            print(f"[{{self.node_id}}] pynvml not available: {{e}}")
            import traceback
            traceback.print_exc()
            self._device_count = 0

    def start(self):
        """Start collecting samples."""
        if not self._pynvml_available:
            print(f"[{{self.node_id}}] Cannot start monitoring: pynvml not available")
            return
        try:
            self.timeseries = []
            self._start_time = time.time()
            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            print(f"[{{self.node_id}}] Monitoring thread started")
        except Exception as e:
            print(f"[{{self.node_id}}] Failed to start monitoring thread: {{e}}")
            import traceback
            traceback.print_exc()

    def stop(self):
        """Stop collecting samples."""
        if hasattr(self, '_stop_event') and self._stop_event is not None:
            self._stop_event.set()
        if hasattr(self, '_thread') and self._thread is not None:
            self._thread.join(timeout=2.0)

    def _monitor_loop(self):
        pynvml = self._pynvml
        while not self._stop_event.is_set():
            timestamp = time.time()
            relative_time = timestamp - self._start_time
            sample = {{'t': round(relative_time, 3)}}

            for i in range(self._device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                    # Memory usage
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used_gb = mem_info.used / (1024**3)
                    mem_util_pct = (mem_info.used / mem_info.total) * 100

                    # GPU utilization (SM utilization)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util_pct = util.gpu
                    mem_bw_util_pct = util.memory

                    # Use node-prefixed keys
                    sample[f'{{self.node_id}}_gpu{{i}}_mem_gb'] = round(mem_used_gb, 2)
                    sample[f'{{self.node_id}}_gpu{{i}}_mem_pct'] = round(mem_util_pct, 1)
                    sample[f'{{self.node_id}}_gpu{{i}}_sm_pct'] = gpu_util_pct
                    sample[f'{{self.node_id}}_gpu{{i}}_membw_pct'] = mem_bw_util_pct
                except Exception:
                    pass

            if len(sample) > 1:
                self.timeseries.append(sample)

            self._stop_event.wait(self.sample_interval)

    def get_timeseries(self):
        """Return collected time-series data."""
        return self.timeseries

    def get_node_id(self):
        """Return this actor's node ID."""
        return self.node_id
    
    def get_ray_node_id(self):
        """Return the Ray node ID where this actor is running."""
        try:
            import ray
            # Get the current node's resource ID
            node_id = ray.get_runtime_context().get_node_id()
            return node_id
        except:
            return None

    def get_device_count(self):
        """Return number of GPUs on this node."""
        return self._device_count
    
    def health_check(self):
        """Simple health check to verify actor is responsive."""
        return {{"status": "ok", "node_id": self.node_id, "gpus": self._device_count}}


class DistributedGPUMonitor:
    """Manage GPU monitor actors across all Ray nodes."""

    def __init__(self, sample_interval: float = 0.5):
        self.sample_interval = sample_interval
        self.actors = []
        self.node_map = {{}}  # node_id -> actor

    def start(self):
        """Deploy actors on all Ray nodes and start monitoring."""
        # Get all Ray nodes
        nodes = ray.nodes()
        alive_nodes = [n for n in nodes if n.get('Alive', False)]

        print(f"📡 Found {{len(alive_nodes)}} Ray nodes for GPU monitoring")

        # Try using placement group to ensure one actor per node
        # If that fails, fall back to creating actors without pinning
        try:
            from ray.util.placement_group import placement_group, remove_placement_group
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
            
            # Create a placement group with one bundle per node
            bundles = [{{"CPU": 0.1}} for _ in alive_nodes]
            pg = placement_group(bundles, strategy="STRICT_SPREAD")
            ray.get(pg.ready(), timeout=60.0)  # Increased timeout for heavy Ray load
            print(f"   Created placement group with {{len(bundles)}} bundles")
            
            # Create actors using placement group
            for idx, node in enumerate(alive_nodes):
                node_id = f"node{{idx}}"
                node_ip = node.get('NodeManagerAddress', 'unknown')
                
                try:
                    actor = GPUMonitorActor.options(
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=idx
                        ),
                        num_cpus=0.1,
                    ).remote(node_id, self.sample_interval)
                    
                    self.actors.append(actor)
                    self.node_map[node_id] = actor
                    print(f"  ✓ Deployed monitor on {{node_id}} ({{node_ip}}) using placement group")
                except Exception as e:
                    print(f"  ✗ Failed to deploy on {{node_id}}: {{e}}")
                    import traceback
                    traceback.print_exc()
        except Exception as pg_error:
            print(f"   Placement group approach failed: {{pg_error}}")
            print(f"   Falling back to simple actor creation (no pinning)...")
            
            # Fallback: create actors without pinning, let Ray schedule them
            for idx, node in enumerate(alive_nodes):
                node_id = f"node{{idx}}"
                node_ip = node.get('NodeManagerAddress', 'unknown')
                
                try:
                    actor = GPUMonitorActor.options(
                        num_cpus=0.1,
                    ).remote(node_id, self.sample_interval)
                    
                    self.actors.append(actor)
                    self.node_map[node_id] = actor
                    print(f"  ✓ Deployed monitor actor (will query node location)")
                    
                    # Try to get the Ray node where the actor is actually running
                    try:
                        ray_node_id = ray.get(actor.get_ray_node_id.remote(), timeout=15.0)  # Increased timeout
                        print(f"     Actor running on Ray node: {{ray_node_id}}")
                    except Exception as node_check_err:
                        print(f"     Could not determine actor node: {{node_check_err}}")
                except Exception as e:
                    print(f"  ✗ Failed to deploy on {{node_id}}: {{e}}")
                    import traceback
                    traceback.print_exc()

        # Start all actors with individual timeouts to avoid hanging
        # Use longer timeouts when Ray is under heavy load (e.g., from vLLM)
        print(f"   Starting monitoring on {{len(self.actors)}} nodes...")
        started_count = 0
        for idx, actor in enumerate(self.actors):
            try:
                # Start each actor individually with timeout
                # Increased timeout to handle Ray being busy with vLLM operations
                print(f"   Starting monitor on node{{idx}}...")
                future = actor.start.remote()
                ray.get(future, timeout=30.0)  # Increased from 10s to 30s for heavy Ray load
                started_count += 1
                print(f"   ✓ Started monitor on node{{idx}}")
            except Exception as e:
                print(f"   ✗ Failed to start monitor on node{{idx}}: {{e}}")
                import traceback
                traceback.print_exc()
                # Continue with other actors
        
        if started_count > 0:
            print(f"📊 GPU monitoring started on {{started_count}}/{{len(self.actors)}} nodes")
            return True  # Success
        else:
            print(f"⚠️  Warning: No actors started successfully. Falling back to local monitoring.")
            # Clear actors so fallback works
            self.actors = []
            self.node_map = {{}}  # Empty dict - escaped for f-string
            return False  # Failed - should fall back to local

    def stop(self):
        """Stop all monitor actors."""
        if self.actors:
            ray.get([actor.stop.remote() for actor in self.actors])

    def get_timeseries(self):
        """Collect and merge time-series from all nodes."""
        if not self.actors:
            return []

        # Collect from all actors
        all_series = ray.get([actor.get_timeseries.remote() for actor in self.actors])

        # Merge time-series by timestamp
        merged = {{}}
        for series in all_series:
            for sample in series:
                t = sample['t']
                if t not in merged:
                    merged[t] = {{'t': t}}
                for key, value in sample.items():
                    if key != 't':
                        merged[t][key] = value

        # Sort by timestamp and return as list
        return [merged[t] for t in sorted(merged.keys())]

    def get_summary(self):
        """Return summary statistics across all nodes."""
        timeseries = self.get_timeseries()
        if not timeseries:
            return {{}}

        summary = {{}}
        metrics = defaultdict(list)

        for sample in timeseries:
            for key, value in sample.items():
                if key != 't':
                    metrics[key].append(value)

        # Per-GPU summaries
        for key, values in metrics.items():
            if values:
                summary[f'{{key}}_avg'] = round(sum(values) / len(values), 2)
                summary[f'{{key}}_max'] = round(max(values), 2)
                summary[f'{{key}}_min'] = round(min(values), 2)

        # Aggregate across all GPUs
        all_sm_util = []
        all_mem_bw_util = []
        all_mem_util = []
        for key, values in metrics.items():
            if '_sm_pct' in key:
                all_sm_util.extend(values)
            elif '_membw_pct' in key:
                all_mem_bw_util.extend(values)
            elif '_mem_pct' in key:
                all_mem_util.extend(values)

        if all_sm_util:
            summary['avg_sm_util_pct'] = round(sum(all_sm_util) / len(all_sm_util), 2)
            summary['max_sm_util_pct'] = round(max(all_sm_util), 2)
        if all_mem_bw_util:
            summary['avg_mem_bw_util_pct'] = round(sum(all_mem_bw_util) / len(all_mem_bw_util), 2)
            summary['max_mem_bw_util_pct'] = round(max(all_mem_bw_util), 2)
        if all_mem_util:
            summary['avg_mem_util_pct'] = round(sum(all_mem_util) / len(all_mem_util), 2)
            summary['max_mem_util_pct'] = round(max(all_mem_util), 2)

        summary['gpu_samples'] = len(timeseries)
        summary['num_nodes_monitored'] = len(self.actors)
        return summary


# ============================================================================
# Local GPU Monitoring (fallback for single-node)
# ============================================================================

# GPU Monitoring class using pynvml
class GPUMonitor:
    """Background GPU metrics collector using pynvml."""

    def __init__(self, sample_interval=0.5):
        self.sample_interval = sample_interval
        self.timeseries = []  # List of {{timestamp, gpu0_*, gpu1_*, ...}}
        self._start_time = None
        self._stop_event = threading.Event()
        self._thread = None
        self._pynvml_available = False

        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml_available = True
            self._pynvml = pynvml
        except Exception as e:
            print(f"⚠️  pynvml not available, GPU monitoring disabled: {{e}}")

    def start(self):
        if not self._pynvml_available:
            return
        self._stop_event.clear()
        self.timeseries = []
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def _monitor_loop(self):
        pynvml = self._pynvml
        device_count = pynvml.nvmlDeviceGetCount()

        while not self._stop_event.is_set():
            timestamp = time.time()
            relative_time = timestamp - self._start_time
            sample = {{'t': round(relative_time, 3)}}

            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                    # Memory usage
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used_gb = mem_info.used / (1024**3)
                    mem_util_pct = (mem_info.used / mem_info.total) * 100

                    # GPU utilization (SM utilization)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util_pct = util.gpu  # SM utilization
                    mem_bw_util_pct = util.memory  # Memory bandwidth utilization

                    sample[f'gpu{{i}}_mem_gb'] = round(mem_used_gb, 2)
                    sample[f'gpu{{i}}_mem_pct'] = round(mem_util_pct, 1)
                    sample[f'gpu{{i}}_sm_pct'] = gpu_util_pct
                    sample[f'gpu{{i}}_membw_pct'] = mem_bw_util_pct

                except Exception:
                    pass

            if len(sample) > 1:  # More than just timestamp
                self.timeseries.append(sample)

            self._stop_event.wait(self.sample_interval)

    def get_timeseries(self):
        """Return raw time-series data for plotting."""
        return self.timeseries

    def get_summary(self):
        """Return summary statistics of GPU metrics."""
        if not self.timeseries:
            return {{}}

        summary = {{}}

        # Collect values per metric
        metrics = defaultdict(list)
        for sample in self.timeseries:
            for key, value in sample.items():
                if key != 't':
                    metrics[key].append(value)

        # Per-GPU summaries
        for key, values in metrics.items():
            if values:
                summary[f'{{key}}_avg'] = round(sum(values) / len(values), 2)
                summary[f'{{key}}_max'] = round(max(values), 2)
                summary[f'{{key}}_min'] = round(min(values), 2)

        # Aggregate across all GPUs
        all_sm_util = []
        all_mem_bw_util = []
        all_mem_util = []
        for key, values in metrics.items():
            if '_sm_pct' in key:
                all_sm_util.extend(values)
            elif '_membw_pct' in key:
                all_mem_bw_util.extend(values)
            elif '_mem_pct' in key:
                all_mem_util.extend(values)

        if all_sm_util:
            summary['avg_sm_util_pct'] = round(sum(all_sm_util) / len(all_sm_util), 2)
            summary['max_sm_util_pct'] = round(max(all_sm_util), 2)
        if all_mem_bw_util:
            summary['avg_mem_bw_util_pct'] = round(sum(all_mem_bw_util) / len(all_mem_bw_util), 2)
            summary['max_mem_bw_util_pct'] = round(max(all_mem_bw_util), 2)
        if all_mem_util:
            summary['avg_mem_util_pct'] = round(sum(all_mem_util) / len(all_mem_util), 2)
            summary['max_mem_util_pct'] = round(max(all_mem_util), 2)

        summary['gpu_samples'] = len(self.timeseries)
        return summary


class SchedulerMonitor:
    """Monitor vLLM scheduler queue depths and KV cache during generation."""

    def __init__(self, llm, sample_interval=0.25):
        self.llm = llm
        self.sample_interval = sample_interval
        self.timeseries = []
        self._start_time = None
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._stop_event.clear()
        self.timeseries = []
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def _get_scheduler_metrics(self, engine, sample):
        """Extract scheduler queue depths from various vLLM versions."""
        scheduler = getattr(engine, 'scheduler', None)
        if scheduler is None:
            return

        # vLLM v0.10+ uses a list of schedulers (one per virtual engine)
        if isinstance(scheduler, list):
            total_running = 0
            total_waiting = 0
            total_swapped = 0
            for sched in scheduler:
                if hasattr(sched, 'running'):
                    total_running += len(sched.running)
                if hasattr(sched, 'waiting'):
                    total_waiting += len(sched.waiting)
                if hasattr(sched, 'swapped'):
                    total_swapped += len(sched.swapped)
            sample['running'] = total_running
            sample['waiting'] = total_waiting
            sample['swapped'] = total_swapped
        # Single scheduler (older vLLM versions)
        elif hasattr(scheduler, 'running'):
            sample['running'] = len(scheduler.running)
            if hasattr(scheduler, 'waiting'):
                sample['waiting'] = len(scheduler.waiting)
            if hasattr(scheduler, 'swapped'):
                sample['swapped'] = len(scheduler.swapped)

    def _get_kv_cache_metrics(self, engine, sample):
        """Extract KV cache utilization from vLLM."""
        # Try to access cache engine / block manager
        # vLLM stores KV cache info in different places depending on version

        # Method 1: scheduler[0].block_manager (vLLM v0.10+)
        scheduler = getattr(engine, 'scheduler', None)
        if scheduler is not None:
            scheds = scheduler if isinstance(scheduler, list) else [scheduler]
            for sched in scheds:
                block_mgr = getattr(sched, 'block_manager', None)
                if block_mgr is not None:
                    # Try to get GPU block usage
                    if hasattr(block_mgr, 'get_num_free_gpu_blocks'):
                        free_gpu = block_mgr.get_num_free_gpu_blocks()
                        total_gpu = getattr(block_mgr, 'num_total_gpu_blocks',
                                          getattr(block_mgr, 'num_gpu_blocks', None))
                        if total_gpu is not None:
                            used_gpu = total_gpu - free_gpu
                            sample['kv_cache_used_blocks'] = used_gpu
                            sample['kv_cache_total_blocks'] = total_gpu
                            sample['kv_cache_util_pct'] = round(100 * used_gpu / total_gpu, 1) if total_gpu > 0 else 0
                        break
                    # Alternative: gpu_allocator
                    gpu_alloc = getattr(block_mgr, 'gpu_allocator', None)
                    if gpu_alloc is not None:
                        if hasattr(gpu_alloc, 'get_num_free_blocks'):
                            free_gpu = gpu_alloc.get_num_free_blocks()
                            total_gpu = getattr(gpu_alloc, 'num_blocks', None)
                            if total_gpu is not None:
                                used_gpu = total_gpu - free_gpu
                                sample['kv_cache_used_blocks'] = used_gpu
                                sample['kv_cache_total_blocks'] = total_gpu
                                sample['kv_cache_util_pct'] = round(100 * used_gpu / total_gpu, 1) if total_gpu > 0 else 0
                            break

        # Method 2: cache_engine (some versions)
        cache_engine = getattr(engine, 'cache_engine', None)
        if cache_engine is not None and 'kv_cache_used_blocks' not in sample:
            # Try various attributes
            for attr in ['gpu_cache', 'cache']:
                cache = getattr(cache_engine, attr, None)
                if cache is not None and hasattr(cache, '__len__'):
                    sample['kv_cache_layers'] = len(cache)
                    break

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            try:
                timestamp = time.time()
                relative_time = timestamp - self._start_time
                sample = {{'t': round(relative_time, 3)}}

                engine = self.llm.llm_engine

                # Get scheduler queue depths
                self._get_scheduler_metrics(engine, sample)

                # Get KV cache utilization
                self._get_kv_cache_metrics(engine, sample)

                # Try to get number of unfinished requests
                if hasattr(engine, '_request_tracker'):
                    tracker = engine._request_tracker
                    if hasattr(tracker, 'get_num_unfinished_requests'):
                        sample['unfinished'] = tracker.get_num_unfinished_requests()

                if len(sample) > 1:  # More than just timestamp
                    self.timeseries.append(sample)

            except Exception as e:
                # Log first error for debugging
                if not self.timeseries:
                    print(f"⚠️  SchedulerMonitor error: {{e}}")

            self._stop_event.wait(self.sample_interval)

    def get_timeseries(self):
        """Return raw time-series data for plotting."""
        return self.timeseries

    def get_summary(self):
        """Return summary statistics of scheduler queue depths."""
        if not self.timeseries:
            return {{}}

        summary = {{}}

        # Collect all unique keys
        all_keys = set()
        for sample in self.timeseries:
            all_keys.update(sample.keys())
        all_keys.discard('t')

        for key in all_keys:
            values = [s.get(key, 0) for s in self.timeseries if key in s]
            if values:
                summary[f'{{key}}_avg'] = round(sum(values) / len(values), 2)
                summary[f'{{key}}_max'] = max(values)

        summary['scheduler_samples'] = len(self.timeseries)
        return summary

DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1463127774414770227/PX_asJ7IXfvQrff6vCW-yGL9k7pBk1v2ZMmRyzUdMNS3mht3pDVO8ligaclUAAFItbvz"

def send_discord_message(message):
    payload = {{"content": message}}
    requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)

# Clean up existing cache directory
cache_dir = "/tmp/lmcache_disk"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

os.makedirs(cache_dir, exist_ok=True)

EXPERIMENTS = {exp_list}
RESULTS_FILE = "/tmp/benchmark_results.json"
NUM_SAMPLES = 50

# Cluster pricing information
INSTANCE_TYPE = "{cluster_instance_type}"
PRICE_PER_HOUR = {total_price_per_hour}
NUM_NODES = {num_nodes}
GPUS_PER_NODE = {gpus_per_node}


def get_vllm_config_info(llm):
    """Extract all configuration info from vLLM engine."""
    config_info = {{}}
    try:
        # Access the LLM engine's configuration
        engine = llm.llm_engine

        # Cache config (KV cache settings)
        if hasattr(engine, 'cache_config'):
            cache_cfg = engine.cache_config
            config_info['block_size'] = getattr(cache_cfg, 'block_size', None)
            config_info['gpu_memory_utilization'] = getattr(cache_cfg, 'gpu_memory_utilization', None)
            config_info['num_gpu_blocks'] = getattr(cache_cfg, 'num_gpu_blocks', None)
            config_info['num_cpu_blocks'] = getattr(cache_cfg, 'num_cpu_blocks', None)
            config_info['swap_space'] = getattr(cache_cfg, 'swap_space', None)
            config_info['cache_dtype'] = getattr(cache_cfg, 'cache_dtype', None)
            config_info['kv_cache_dtype'] = getattr(cache_cfg, 'kv_cache_dtype', None)

        # Scheduler config (concurrency settings)
        if hasattr(engine, 'scheduler_config'):
            sched_cfg = engine.scheduler_config
            # Extract scheduler config attributes
            config_info['max_num_seqs'] = getattr(sched_cfg, 'max_num_seqs', None)
            config_info['max_num_batched_tokens'] = getattr(sched_cfg, 'max_num_batched_tokens', None)
            config_info['max_model_len'] = getattr(sched_cfg, 'max_model_len', None)
            config_info['chunked_prefill_enabled'] = getattr(sched_cfg, 'chunked_prefill_enabled', None)
            config_info['max_paddings'] = getattr(sched_cfg, 'max_paddings', None)
            config_info['delay_factor'] = getattr(sched_cfg, 'delay_factor', None)
            config_info['chunked_prefill_size'] = getattr(sched_cfg, 'chunked_prefill_size', None)
            config_info['max_seq_len'] = getattr(sched_cfg, 'max_seq_len', None)
            config_info['policy'] = getattr(sched_cfg, 'policy', None)
            config_info['long_prefill_token_threshold'] = getattr(sched_cfg, 'long_prefill_token_threshold', None)
            config_info['max_long_partial_prefills'] = getattr(sched_cfg, 'max_long_partial_prefills', None)
        
        # Also check scheduler instance (not just config) for runtime values
        if hasattr(engine, 'scheduler'):
            scheduler = engine.scheduler
            # Try to get runtime scheduler values if config doesn't have them
            if config_info.get('max_num_seqs') is None:
                config_info['max_num_seqs'] = getattr(scheduler, 'max_num_seqs', None)
            if config_info.get('max_num_batched_tokens') is None:
                config_info['max_num_batched_tokens'] = getattr(scheduler, 'max_num_batched_tokens', None)
        
        # Ensure these keys exist even if scheduler_config doesn't exist or attributes are missing
        if 'max_num_seqs' not in config_info:
            config_info['max_num_seqs'] = None
        if 'max_num_batched_tokens' not in config_info:
            config_info['max_num_batched_tokens'] = None

        # Model config
        if hasattr(engine, 'model_config'):
            model_cfg = engine.model_config
            config_info['dtype'] = str(getattr(model_cfg, 'dtype', None))
            config_info['max_model_len'] = getattr(model_cfg, 'max_model_len', None)
            config_info['quantization'] = getattr(model_cfg, 'quantization', None)
            config_info['quantization_param_path'] = getattr(model_cfg, 'quantization_param_path', None)
            config_info['enforce_eager'] = getattr(model_cfg, 'enforce_eager', None)
            config_info['max_seq_len_to_capture'] = getattr(model_cfg, 'max_seq_len_to_capture', None)

        # Parallel config
        if hasattr(engine, 'parallel_config'):
            parallel_cfg = engine.parallel_config
            config_info['tensor_parallel_size'] = getattr(parallel_cfg, 'tensor_parallel_size', None)
            config_info['pipeline_parallel_size'] = getattr(parallel_cfg, 'pipeline_parallel_size', None)
            config_info['world_size'] = getattr(parallel_cfg, 'world_size', None)
            config_info['rank'] = getattr(parallel_cfg, 'rank', None)

        # Decoding config
        if hasattr(engine, 'decoding_config'):
            decoding_cfg = engine.decoding_config
            config_info['use_chunked_prefill'] = getattr(decoding_cfg, 'use_chunked_prefill', None)
            # Prefer decoding_config value, but keep scheduler_config value if decoding_config doesn't have it
            decoding_max_batched = getattr(decoding_cfg, 'max_num_batched_tokens', None)
            if decoding_max_batched is not None:
                config_info['max_num_batched_tokens'] = decoding_max_batched
            # If not set yet, ensure it's at least None
            if 'max_num_batched_tokens' not in config_info:
                config_info['max_num_batched_tokens'] = None

        # Check for LMCache configuration
        try:
            from lmcache.integration.vllm.utils import ENGINE_NAME
            from lmcache.v1.cache_engine import LMCacheEngineBuilder
            if hasattr(LMCacheEngineBuilder, 'get_engine'):
                lmcache_engine = LMCacheEngineBuilder.get_engine(ENGINE_NAME)
                if lmcache_engine:
                    config_info['lmcache_enabled'] = True
                    # Try to get LMCache config
                    if hasattr(lmcache_engine, 'config'):
                        lmcache_cfg = lmcache_engine.config
                        config_info['lmcache_chunk_size'] = getattr(lmcache_cfg, 'chunk_size', None)
                        config_info['lmcache_local_cpu'] = getattr(lmcache_cfg, 'local_cpu', None)
                        config_info['lmcache_max_local_cpu_size'] = getattr(lmcache_cfg, 'max_local_cpu_size', None)
                        config_info['lmcache_use_layerwise'] = getattr(lmcache_cfg, 'use_layerwise', None)
                        config_info['lmcache_enable_lazy_memory_allocator'] = getattr(lmcache_cfg, 'enable_lazy_memory_allocator', None)
                    # Check environment variables
                    import os
                    config_info['lmcache_use_experimental'] = os.environ.get('LMCACHE_USE_EXPERIMENTAL', None)
                    config_info['lmcache_enable_async_loading'] = os.environ.get('LMCACHE_ENABLE_ASYNC_LOADING', None)
                    config_info['lmcache_remote_serde'] = os.environ.get('LMCACHE_REMOTE_SERDE', None)
                else:
                    config_info['lmcache_enabled'] = False
            else:
                config_info['lmcache_enabled'] = False
        except Exception:
            config_info['lmcache_enabled'] = False

        # Check for KV transfer config (LMCache connector)
        if hasattr(llm, 'llm_engine') and hasattr(engine, 'cache_config'):
            cache_cfg = engine.cache_config
            if hasattr(cache_cfg, 'kv_transfer_config') and cache_cfg.kv_transfer_config:
                ktc = cache_cfg.kv_transfer_config
                config_info['kv_transfer_config_enabled'] = True
                config_info['kv_connector'] = getattr(ktc, 'kv_connector', None)
                config_info['kv_role'] = getattr(ktc, 'kv_role', None)
                config_info['kv_buffer_device'] = getattr(ktc, 'kv_buffer_device', None)
                if hasattr(ktc, 'kv_connector_extra_config'):
                    config_info['kv_connector_extra_config'] = ktc.kv_connector_extra_config
            else:
                config_info['kv_transfer_config_enabled'] = False

        # LLM initialization parameters (from the LLM object)
        config_info['enable_prefix_caching'] = getattr(llm, 'enable_prefix_caching', None)
        config_info['enable_chunked_prefill'] = getattr(llm, 'enable_chunked_prefill', None)
        config_info['trust_remote_code'] = getattr(llm, 'trust_remote_code', None)

    except Exception as e:
        print(f"⚠️  Could not extract vLLM config: {{e}}")
        import traceback
        traceback.print_exc()

    return config_info

def run_benchmark(exp):
    """Run single benchmark experiment."""
    print(f"\\n{{'='*70}}")
    print(f"Running: TP={{exp['tp']}}, PP={{exp['pp']}}, "
          f"input={{exp['max_input_length']}}, output={{exp['max_output_length']}}")
    print("="*70)

    # Determine if multi-node (using Ray)
    backend = "ray" if (exp['pp'] > 1) else None
    gpu_monitor = None  # Will be initialized after LLM creation

    try:
        # if backend == "ray":
        #     restart_ray_cluster()

        # # Enable LMCache for KV cache management
        ktc = KVTransferConfig(
            kv_connector="LMCacheConnectorV1",
            kv_role="kv_both",
            kv_buffer_device="cpu"
        )
        # ktc = KVTransferConfig(
        #     kv_connector="OffloadingConnector",
        #     kv_role="kv_both",
        #     kv_connector_extra_config={{"block_size":64,"num_cpu_blocks":1000}}
        # )

        llm = LLM(
            model=exp['model'],
            tensor_parallel_size=exp['tp'],
            pipeline_parallel_size=exp['pp'],
            max_model_len=min(exp['max_input_length'] + exp['max_output_length'] -1, 32768), # -1 because of the eos token removing
            trust_remote_code=True,
            distributed_executor_backend=backend,
            gpu_memory_utilization=0.85,
            # quantization="awq",
            enforce_eager=True,
            # kv_transfer_config=ktc,
            enable_chunked_prefill=False,
            # truncate_prompt_tokens=exp['max_input_length'],
            enable_prefix_caching=False
        )

        # Initialize GPU monitor AFTER LLM creation (Ray is now initialized)
        # For multi-node PP > 1, we should use distributed monitoring. vLLM initializes Ray internally,
        # but we need to explicitly connect to it using RAY_ADDRESS.
        # For single-node setups (even with Ray), use local monitoring to avoid Ray timeout issues.
        use_distributed = False
        if backend == "ray":
            print("📡 Checking Ray cluster for GPU monitoring...")
            try:
                # vLLM initializes Ray internally, but we need to connect to it explicitly
                # Get Ray address from environment (set at the top of the script)
                ray_address = os.environ.get("RAY_ADDRESS", "127.0.0.1:6379")
                
                # Try to connect to Ray cluster explicitly
                ray_available = False
                try:
                    # If Ray is not initialized, try to connect to the existing cluster
                    if not ray.is_initialized():
                        print(f"   Connecting to Ray cluster at {{ray_address}}...")
                        ray.init(address=ray_address, ignore_reinit_error=True, log_to_driver=False)
                        print(f"   ✅ Connected to Ray cluster")
                    
                    # Now check if we can access nodes
                    nodes = ray.nodes()
                    alive_nodes = [n for n in nodes if n.get('Alive', False)]
                    num_nodes = len(alive_nodes)
                    ray_available = num_nodes > 0
                    print(f"   Ray cluster detected: {{num_nodes}} alive node(s)")
                    for idx, node in enumerate(alive_nodes):
                        node_id = node.get('NodeID', 'unknown')
                        print(f"      Node {{idx}}: {{node_id}}")
                except Exception as ray_check_error:
                    print(f"   Could not access Ray cluster: {{ray_check_error}}")
                    print(f"   Ray.is_initialized(): {{ray.is_initialized() if hasattr(ray, 'is_initialized') else 'N/A'}}")
                    print(f"   RAY_ADDRESS: {{ray_address}}")
                    num_nodes = 0
                
                if ray_available:
                    # Only use distributed monitoring for multi-node setups
                    # Single-node Ray clusters should use local monitoring to avoid timeout issues
                    if num_nodes > 1:
                        print(f"📡 Using distributed GPU monitoring ({{num_nodes}} nodes)")
                        # Try to initialize distributed monitor
                        gpu_monitor = DistributedGPUMonitor(sample_interval=0.5)
                        use_distributed = True
                        print("✅ Distributed GPU monitoring initialized successfully")
                    else:
                        print(f"📊 Using local GPU monitoring (single-node Ray cluster)")
                        gpu_monitor = GPUMonitor(sample_interval=0.5)
                        use_distributed = False
                else:
                    raise Exception("Ray cluster not accessible")
            except Exception as e:
                print(f"⚠️  Failed to initialize distributed GPU monitor: {{e}}")
                import traceback
                traceback.print_exc()
                print("📊 Falling back to local GPU monitoring")
                gpu_monitor = GPUMonitor(sample_interval=0.5)
                use_distributed = False
        else:
            print("📊 Using local GPU monitoring (single node, no Ray)")
            gpu_monitor = GPUMonitor(sample_interval=0.5)

        # Extract vLLM configuration info (KV cache, scheduler settings)
        vllm_config = get_vllm_config_info(llm)
        print(f"📊 vLLM Config: {{vllm_config}}")

        tokenizer = llm.get_tokenizer()
        dataset = load_dataset("emozilla/pg19-test", split="test")

        # Prepare prompts
        prompts = []
        for i in range(min(NUM_SAMPLES, len(dataset))):
            text = "Please summarize the following text: " + dataset[i]["text"]
            tokens = tokenizer.encode(text, add_special_tokens=False)[:exp['max_input_length']]
            prompts.append(tokenizer.decode(tokens))

        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=exp['max_output_length'],
            min_tokens=exp['max_output_length'],
            ignore_eos=True,
        )

        # Warmup run to trigger kernel compilation and initialization
        print(f"🔥 Running warmup with {{min(5, len(prompts))}} samples...")
        _ = llm.generate(prompts[:min(5, len(prompts))], sampling_params)
        torch.cuda.synchronize()  # Wait for warmup to complete
        print(f"✅ Warmup complete, starting actual measurement")

        # Start GPU monitoring
        # If distributed monitoring fails, it will fall back gracefully
        if use_distributed:
            try:
                success = gpu_monitor.start()
                if not success or (hasattr(gpu_monitor, 'actors') and len(gpu_monitor.actors) == 0):
                    print("⚠️  Distributed monitoring failed, falling back to local")
                    raise Exception("Distributed monitoring not available")
            except Exception as monitor_error:
                print(f"⚠️  Failed to start distributed GPU monitoring: {{monitor_error}}")
                print("📊 Falling back to local GPU monitoring")
                gpu_monitor = GPUMonitor(sample_interval=0.5)
                use_distributed = False
                gpu_monitor.start()
        else:
            gpu_monitor.start()

        # Start scheduler monitoring (queue depths)
        scheduler_monitor = SchedulerMonitor(llm, sample_interval=0.25)
        scheduler_monitor.start()

        # Run and measure
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()  # Wait for all GPU work to complete
        elapsed = time.perf_counter() - start

        # Stop monitoring and get summaries
        gpu_monitor.stop()
        scheduler_monitor.stop()
        gpu_metrics = gpu_monitor.get_summary()
        scheduler_metrics = scheduler_monitor.get_summary()
        
        # Log monitoring info for debugging
        if use_distributed and hasattr(gpu_monitor, 'actors'):
            print(f"📊 GPU monitoring: {{len(gpu_monitor.actors)}} nodes monitored")
            if 'num_nodes_monitored' in gpu_metrics:
                print(f"📊 GPU monitoring: {{gpu_metrics['num_nodes_monitored']}} nodes reported in summary")
        else:
            print(f"📊 GPU monitoring: local (single node)")

        # Count tokens (vLLM native way)
        total_prompt_tokens = sum(len(o.prompt_token_ids or []) for o in outputs)
        total_output_tokens = sum(
            sum(len(c.token_ids) for c in o.outputs) for o in outputs
        )

        # Generate experiment ID for timeseries file
        exp_id = f"tp{{exp['tp']}}_pp{{exp['pp']}}_in{{exp['max_input_length']}}_out{{exp['max_output_length']}}"
        timeseries_file = f"/tmp/timeseries_{{exp_id}}.json"

        # Save time-series data to separate file
        timeseries_data = {{
            'exp_id': exp_id,
            'config': exp,
            'elapsed_time': elapsed,
            'gpu_timeseries': gpu_monitor.get_timeseries(),
            'scheduler_timeseries': scheduler_monitor.get_timeseries(),
        }}
        with open(timeseries_file, 'w') as f:
            json.dump(timeseries_data, f)
        print(f"📈 Time-series saved to {{timeseries_file}}")

        time.sleep(30)
        # Cleanup BEFORE creating result
        cleanup_dist_env_and_memory(shutdown_ray=False) #DO  NOT SHUT DOWN RAY
        # Clean up lmcache backend
        try:
            from lmcache.integration.vllm.utils import ENGINE_NAME
            from lmcache.v1.cache_engine import LMCacheEngineBuilder
            LMCacheEngineBuilder.destroy(ENGINE_NAME)
        except:
            # if backend == "ray":
            #     restart_ray_cluster()
            pass

        # Force GPU cleanup
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        send_discord_message(f"✅ Results saved to {{exp['tp']}}-{{exp['pp']}}")

        # Calculate cost efficiency metrics
        total_tokens = total_prompt_tokens + total_output_tokens
        elapsed_hours = elapsed / 3600
        cost_for_run = PRICE_PER_HOUR * elapsed_hours
        tokens_per_dollar = round(total_tokens / cost_for_run, 2) if cost_for_run > 0 else 0
        input_tokens_per_dollar = round(total_prompt_tokens / cost_for_run, 2) if cost_for_run > 0 else 0
        output_tokens_per_dollar = round(total_output_tokens / cost_for_run, 2) if cost_for_run > 0 else 0

        # Collect all configuration
        benchmark_config = {{
            # LLM initialization parameters
            'llm_model': exp['model'],
            'llm_tensor_parallel_size': exp['tp'],
            'llm_pipeline_parallel_size': exp['pp'],
            'llm_max_model_len': min(exp['max_input_length'] + exp['max_output_length'] - 1, 32768),
            'llm_trust_remote_code': True,
            'llm_distributed_executor_backend': backend,
            'llm_gpu_memory_utilization': 0.85,
            'llm_enforce_eager': True,
            'llm_enable_chunked_prefill': False,
            'llm_enable_prefix_caching': False,
            'llm_kv_transfer_config': None,  # Currently commented out
            # Sampling parameters
            'sampling_temperature': 0.8,
            'sampling_max_tokens': exp['max_output_length'],
            'sampling_min_tokens': exp['max_output_length'],
            'sampling_ignore_eos': True,
            # Benchmark configuration
            'benchmark_num_samples': NUM_SAMPLES,
            'benchmark_dataset': 'emozilla/pg19-test',
            'benchmark_prompt_prefix': 'Please summarize the following text: ',
            'benchmark_warmup_samples': min(5, NUM_SAMPLES),
            # GPU monitoring configuration
            'gpu_monitor_sample_interval': 0.5,
            'gpu_monitor_type': 'distributed' if use_distributed else 'local',
            # Scheduler monitoring configuration
            'scheduler_monitor_sample_interval': 0.25,
        }}

        # Build result with all metrics (summary only, timeseries in separate file)
        result = {{
            **exp,
            'exp_id': exp_id,
            'elapsed_time': elapsed,
            'total_prompt_tokens': total_prompt_tokens,
            'total_output_tokens': total_output_tokens,
            'requests_per_sec': round(len(outputs) / elapsed, 3),
            'input_tokens_per_sec': round(total_prompt_tokens / elapsed, 2),
            'output_tokens_per_sec': round(total_output_tokens / elapsed, 2),
            'total_tokens_per_sec': round((total_prompt_tokens + total_output_tokens) / elapsed, 2),
            'status': 'success',
            'timeseries_file': timeseries_file,
            # Infrastructure info
            'instance_type': INSTANCE_TYPE,
            'price_per_hour': PRICE_PER_HOUR,
            'num_nodes': NUM_NODES,
            'gpus_per_node': GPUS_PER_NODE,
            'total_gpus': NUM_NODES * GPUS_PER_NODE,
            # Cost efficiency metrics
            'cost_for_run_usd': round(cost_for_run, 4),
            'tokens_per_dollar': tokens_per_dollar,
            'input_tokens_per_dollar': input_tokens_per_dollar,
            'output_tokens_per_dollar': output_tokens_per_dollar,
            # Benchmark configuration
            **benchmark_config,
            # GPU monitoring details (added after metrics are collected)
            'gpu_monitor_num_nodes': len(gpu_monitor.actors) if (use_distributed and hasattr(gpu_monitor, 'actors')) else 1,
            'gpu_monitor_num_gpus_monitored': len([k for k in gpu_metrics.keys() if '_sm_pct_avg' in k or (k.endswith('_sm_pct_avg') and not k.startswith('avg_'))]),
            'gpu_monitor_num_nodes_reported': gpu_metrics.get('num_nodes_monitored', 1),
            # vLLM config info (extracted from engine)
            **{{f'config_{{k}}': v for k, v in vllm_config.items()}},
            # GPU utilization metrics (summary)
            **gpu_metrics,
            # Scheduler queue metrics (summary)
            **scheduler_metrics,
        }}
        
    except Exception as e:
        # Stop GPU monitor on error (if it was initialized)
        if gpu_monitor is not None:
            gpu_monitor.stop()

        import traceback
        error_msg = str(e)
        if "CUDA out of memory" in error_msg or "OutOfMemoryError" in error_msg:
            error_msg = f"OOM: {{error_msg[:200]}}"
        send_discord_message(f"❌ Cluster {{exp['tp']}}-{{exp['pp']}} FAILED: {{error_msg}}")
        # if backend == "ray":
        #     restart_ray_cluster()
        
        # Collect configuration even for error case
        benchmark_config = {{
            # LLM initialization parameters
            'llm_model': exp['model'],
            'llm_tensor_parallel_size': exp['tp'],
            'llm_pipeline_parallel_size': exp['pp'],
            'llm_max_model_len': min(exp['max_input_length'] + exp['max_output_length'] - 1, 32768),
            'llm_trust_remote_code': True,
            'llm_distributed_executor_backend': backend,
            'llm_gpu_memory_utilization': 0.85,
            'llm_enforce_eager': True,
            'llm_enable_chunked_prefill': False,
            'llm_enable_prefix_caching': False,
            'llm_kv_transfer_config': None,  # Currently commented out
            # Sampling parameters
            'sampling_temperature': 0.8,
            'sampling_max_tokens': exp['max_output_length'],
            'sampling_min_tokens': exp['max_output_length'],
            'sampling_ignore_eos': True,
            # Benchmark configuration
            'benchmark_num_samples': NUM_SAMPLES,
            'benchmark_dataset': 'emozilla/pg19-test',
            'benchmark_prompt_prefix': 'Please summarize the following text: ',
            'benchmark_warmup_samples': min(5, NUM_SAMPLES),
            # GPU monitoring configuration
            'gpu_monitor_sample_interval': 0.5,
            'gpu_monitor_type': 'distributed' if backend == "ray" else 'local',
            # Scheduler monitoring configuration
            'scheduler_monitor_sample_interval': 0.25,
        }}
        
        result = {{
            **exp,
            'status': 'error',
            'error': error_msg,
            # Infrastructure info
            'instance_type': INSTANCE_TYPE,
            'price_per_hour': PRICE_PER_HOUR,
            'num_nodes': NUM_NODES,
            'gpus_per_node': GPUS_PER_NODE,
            'total_gpus': NUM_NODES * GPUS_PER_NODE,
            # Benchmark configuration
            **benchmark_config,
        }}
        
        # Force GPU cleanup on error
        try:
            cleanup_dist_env_and_memory(shutdown_ray=False) #DO  NOT SHUT DOWN RAY
            # if 'llm' in locals():
                # del llm
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except:
            pass
    return result

def main():
    results = []
    for i, exp in enumerate(EXPERIMENTS):
        # if i in [0,1,2,3]:
        #     continue
        result = run_benchmark(exp)
        results.append(result)
        print(f"Result: {{result}}")
        # Save incrementally
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"\\n✅ All done! Results in {{RESULTS_FILE}}")

if __name__ == "__main__":
    main()
'''

def run_cluster_benchmarks(cluster_config, experiments, parent_dir=None, dry_run=True, gpu_type=DEFAULT_GPU_TYPE):
    gpus_per_node, num_nodes = cluster_config
    # Use TP/PP from the first experiment for naming (all experiments in a group have compatible TP/PP)
    tp = experiments[0]['tp']
    pp = experiments[0]['pp']
    # Include input/output length in cluster name to avoid conflicts across IO groups
    input_len = experiments[0]['max_input_length']
    output_len = experiments[0]['max_output_length']
    # Include GPU type in cluster name to avoid conflicts when using different GPU types
    # Normalize GPU type for use in filenames (replace special chars)
    gpu_type_safe = gpu_type.replace("_", "-").replace("/", "-")
    cluster_name = f"roofline-tp{tp}-pp{pp}-{input_len}in-{output_len}out-{gpu_type_safe}"

    # Create consolidated result directory with datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subdir_name = f"tp{tp}-pp{pp}-{timestamp}"
    
    # Get instance family and GPU name for directory organization
    gpu_config = GPU_CONFIGS[gpu_type]
    instance_family = gpu_config["instance_family"]
    # Use gpu_type as GPU name (e.g., "L4", "L40S", "A100-40gb", "A100-80gb")
    gpu_name = gpu_type
    instance_dir = f"aws-{instance_family}-{gpu_name}"
    
    # If parent_dir is provided, create subdirectory structure: parent_dir/aws-{instance_family}-{gpu_name}/tp{tp}-pp{pp}-{timestamp}
    if parent_dir:
        parent_path = Path(parent_dir)
        parent_path.mkdir(exist_ok=True)
        instance_path = parent_path / instance_dir
        instance_path.mkdir(exist_ok=True)
        result_dir = instance_path / subdir_name
    else:
        # If no parent_dir, create: aws-{instance_family}-{gpu_name}/tp{tp}-pp{pp}-{timestamp}
        instance_path = Path(instance_dir)
        instance_path.mkdir(exist_ok=True)
        result_dir = instance_path / subdir_name

    print(f"\n{'='*70}")
    print(f"Cluster: {cluster_name}")
    print(f"Config: {gpus_per_node} GPUs/node × {num_nodes} nodes")
    print(f"Experiments: {len(experiments)}")
    print("="*70)

    if dry_run:
        print("[DRY RUN] Would launch and run:")
        for exp in experiments:
            print(f"  - TP={exp['tp']}, PP={exp['pp']}, "
                  f"in={exp['max_input_length']}, out={exp['max_output_length']}")
        if len(experiments) > 3:
            print(f"  ... and {len(experiments)-3} more")
        return []

    # Create result directory and set up logging
    result_dir.mkdir(exist_ok=True)
    log_file = result_dir / "benchmark.log"
    setup_logger(log_file)

    logger.info(f"Results will be saved to: {result_dir}")

    # Define paths within result directory
    local_results = result_dir / "results.json"

    # 1. Write YAML
    work_dir = Path("roofline_benchmarks")
    work_dir.mkdir(exist_ok=True)
    yaml_path = work_dir / f"{cluster_name}.yaml"
    script_path = work_dir / f"benchmark_{cluster_name}.py"
    
    # Clean up old files without GPU type in name (if they match this config)
    # This prevents confusion from old files that might have different GPU types
    old_yaml_pattern = f"roofline-tp{tp}-pp{pp}-{input_len}in-{output_len}out.yaml"
    old_script_pattern = f"benchmark_roofline-tp{tp}-pp{pp}-{input_len}in-{output_len}out.py"
    old_yaml_path = work_dir / old_yaml_pattern
    old_script_path = work_dir / old_script_pattern
    
    # Only remove if they exist and don't match the new naming (i.e., no GPU type suffix)
    if old_yaml_path.exists() and old_yaml_path != yaml_path:
        logger.info(f"🗑️  Removing old YAML file: {old_yaml_path.name} (replaced by GPU-type-specific file)")
        old_yaml_path.unlink()
    if old_script_path.exists() and old_script_path != script_path:
        logger.info(f"🗑️  Removing old script file: {old_script_path.name} (replaced by GPU-type-specific file)")
        old_script_path.unlink()
    
    yaml_content = generate_yaml(gpus_per_node, num_nodes, cluster_name, experiments, gpu_type)
    yaml_path.write_text(yaml_content)

    # 2. Write benchmark script
    script_content = generate_benchmark_script(experiments, gpus_per_node, num_nodes, gpu_type)
    script_path.write_text(script_content)

    # Track active cluster for cleanup on unexpected exit
    set_active_cluster(cluster_name)

    try:
        # 3. Launch cluster and capture output
        logger.info(f"🚀 Launching {cluster_name}...")
        logger.info(f"YAML to run: {yaml_path}")
        logger.info(f"Running on cluster: {cluster_name}")

        # Run sky launch and capture output (with retry-until-up to handle capacity issues)
        process = subprocess.Popen(
            ["sky", "launch", "-y", "--retry-until-up", "-c", cluster_name, str(yaml_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Stream output to both console and log file
        for line in process.stdout:
            line = line.rstrip()
            logger.info(line)

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, "sky launch")

        # 5. Fetch results
        logger.info(f"📥 Fetching results...")
        subprocess.run([
            "scp",
            f"{cluster_name}:/tmp/benchmark_results.json",
            str(local_results)
        ], check=True)

        # Fetch timeseries files directly into result directory
        logger.info(f"📈 Fetching timeseries data...")
        subprocess.run([
            "scp",
            f"{cluster_name}:/tmp/timeseries_*.json",
            str(result_dir)  # Path object converts to string correctly
        ], check=False)  # Don't fail if no timeseries files

        # Update results with local timeseries paths
        with open(local_results) as f:
            results = json.load(f)

        # Check if all experiments succeeded or any failed
        all_succeeded = all(r.get('status') == 'success' for r in results)
        status_suffix = 'success' if all_succeeded else 'fail'
        
        # Rename directory to include status suffix
        new_subdir_name = f"tp{tp}-pp{pp}-{timestamp}-{status_suffix}"
        new_result_dir = instance_path / new_subdir_name
        
        if result_dir != new_result_dir:
            logger.info(f"📁 Renaming directory: {result_dir.name} -> {new_result_dir.name}")
            result_dir.rename(new_result_dir)
            result_dir = new_result_dir
            # Update local_results path (files inside directory move automatically with rename)
            local_results = result_dir / "results.json"

        for result in results:
            if 'timeseries_file' in result and result.get('status') == 'success':
                # Update path to local location
                remote_filename = Path(result['timeseries_file']).name
                local_ts_path = result_dir / remote_filename
                result['timeseries_file'] = str(local_ts_path)

        # Save updated results
        with open(local_results, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV in the same result directory with timestamp
        csv_filename = f"benchmark_results-{timestamp}.csv"
        csv_path = result_dir / csv_filename
        save_results_csv(results, str(csv_path))

        # Automatically plot timeseries for successful experiments
        try:
            successful_ts_files = [
                Path(r['timeseries_file'])
                for r in results
                if r.get('status') == 'success' and r.get('timeseries_file')
            ]
            if successful_ts_files:
                logger.info(f"📈 Plotting timeseries for {len(successful_ts_files)} successful experiment(s)...")
                # For now, call plot_timeseries.py once on the result directory;
                # the script will discover all timeseries_*.json files inside.
                subprocess.run(
                    [
                        "python",
                        "plot_timeseries.py",
                        str(result_dir),
                        "--output",
                        str(result_dir / "timeseries_plots.pdf"),
                    ],
                    check=False,
                )
        except Exception as plot_err:
            logger.warning(f"⚠️  Failed to generate timeseries plots: {plot_err}")
        
        logger.info(f"✅ Results saved to {result_dir}/")
        send_discord_message(f"✅ Results saved to {result_dir}")
        return results

    except Exception as e:
        # Cluster error occurred, but benchmark might have completed successfully
        logger.error(f"❌ Cluster error: {e}")
        logger.info(f"📥 Attempting to fetch results (benchmark may have completed)...")
        
        fetched_results = None
        try:
            logger.info(f"📥 Fetching results.json from cluster...")
            subprocess.run([
                "scp",
                f"{cluster_name}:/tmp/benchmark_results.json",
                str(local_results)
            ], check=True)
            
            # Best-effort fetch of timeseries files even when sky launch failed
            logger.info(f"📈 Attempting to fetch timeseries data from cluster (best-effort)...")
            subprocess.run([
                "scp",
                f"{cluster_name}:/tmp/timeseries_*.json",
                str(result_dir)
            ], check=False)  # Don't fail the cleanup path if timeseries are missing
            
            # Check if we successfully fetched valid results
            if local_results.exists():
                with open(local_results) as f:
                    fetched_results = json.load(f)
                    # Check if any results have 'success' status
                    has_success = any(r.get('status') == 'success' for r in fetched_results)
                    if has_success:
                        logger.info(f"✅ Found successful benchmark results despite cluster error!")
                        logger.info(f"   This is likely a cleanup error, not a benchmark failure.")
                        send_discord_message(f"✅ Cluster {cluster_name} completed benchmarks but had a cleanup error: {str(e)[:80]}")
                    else:
                        send_discord_message(f"❌ Cluster {cluster_name} FAILED: {str(e)[:100]}")
        except Exception as fetch_error:
            logger.warning(f"⚠️  Could not fetch results: {fetch_error}")
            send_discord_message(f"❌ Cluster {cluster_name} FAILED and results could not be fetched: {str(e)[:60]} | fetch error: {str(fetch_error)[:60]}")
        
        # Use fetched results if they contain successful benchmarks, otherwise mark as failed
        if fetched_results and any(r.get('status') == 'success' for r in fetched_results):
            results = fetched_results
            # Determine status suffix based on whether all succeeded
            all_succeeded = all(r.get('status') == 'success' for r in results)
            status_suffix = 'success' if all_succeeded else 'fail'
        else:
            # No valid results fetched, mark all as failed
            results = [{**exp, 'status': 'cluster_error', 'error': str(e)} for exp in experiments]
            status_suffix = 'fail'
        
        # Rename directory to include status suffix
        new_subdir_name = f"tp{tp}-pp{pp}-{timestamp}-{status_suffix}"
        new_result_dir = instance_path / new_subdir_name
        
        if result_dir != new_result_dir:
            logger.info(f"📁 Renaming directory: {result_dir.name} -> {new_result_dir.name}")
            result_dir.rename(new_result_dir)
            result_dir = new_result_dir
            # Update local_results path
            local_results = result_dir / "results.json"
        
        # If we have successful results and timeseries files, rewrite their paths
        if results and any(r.get('status') == 'success' for r in results):
            for result in results:
                if 'timeseries_file' in result and result.get('status') == 'success':
                    remote_filename = Path(result['timeseries_file']).name
                    local_ts_path = result_dir / remote_filename
                    result['timeseries_file'] = str(local_ts_path)
        
        # Save results (either fetched successful ones or failed markers)
        with open(local_results, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV in the same result directory
        csv_filename = f"benchmark_results-{timestamp}.csv"
        csv_path = result_dir / csv_filename
        save_results_csv(results, str(csv_path))

        # If we have successful results, also attempt to plot timeseries
        if results and any(r.get('status') == 'success' for r in results):
            try:
                logger.info(f"📈 Plotting timeseries for successful experiment(s) after cluster error...")
                subprocess.run(
                    [
                        "python",
                        "plot_timeseries.py",
                        str(result_dir),
                        "--output",
                        str(result_dir / "timeseries_plots.pdf"),
                    ],
                    check=False,
                )
            except Exception as plot_err:
                logger.warning(f"⚠️  Failed to generate timeseries plots after cluster error: {plot_err}")
        
        return results

    finally:
        # 6. Teardown
        logger.info(f"🗑️  Tearing down {cluster_name}...")
        subprocess.run(["sky", "down", "-y", cluster_name])

        # Clear active cluster after successful teardown
        set_active_cluster(None)




def save_results_csv(all_results, output_path="benchmark_results.csv"):
    """Save all results to CSV."""
    if not all_results:
        return
    # Collect all unique fieldnames from all results
    fieldnames = []
    for result in all_results:
        for key in result.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"📊 Saved {len(all_results)} results to {output_path}")


def main():
    # Check for orphaned clusters from previous crashed runs
    check_orphaned_cluster()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Run benchmark experiments on AWS GPU instances',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
Available GPU types: {', '.join(GPU_CONFIGS.keys())}
Default: {DEFAULT_GPU_TYPE}

Examples:
  python automatic_launch_1.py --gpu L40S
  python automatic_launch_1.py experiment.csv --gpu A100 --run
  python automatic_launch_1.py --gpu H100 --run
        '''
    )
    parser.add_argument('csv_file', nargs='?', default='experiment_L40_llama.csv',
                       help='CSV file with experiment configurations (default: experiment_L40_llama.csv)')
    parser.add_argument('--gpu', '--gpu-type', dest='gpu_type', 
                       choices=list(GPU_CONFIGS.keys()),
                       default=DEFAULT_GPU_TYPE,
                       help=f'GPU type to use (default: {DEFAULT_GPU_TYPE})')
    parser.add_argument('--vm-strategy', dest='vm_strategy',
                       choices=list(VM_SELECTION_STRATEGIES.keys()),
                       default='prefer_single_node',
                       help='VM selection strategy for TP/PP -> cluster sizing')
    parser.add_argument('--run', action='store_true',
                       help='Actually launch clusters (default: dry run)')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up old benchmark files without GPU type in their names')
    
    args = parser.parse_args()
    
    # Handle cleanup option
    if args.cleanup:
        print("🧹 Cleaning up old benchmark files...")
        cleanup_old_benchmark_files()
        if not args.run and args.csv_file == parser.get_default('csv_file'):
            # If only cleanup was requested (no other args), exit after cleanup
            return
    
    csv_path = args.csv_file
    gpu_type = args.gpu_type
    vm_strategy = args.vm_strategy
    dry_run = not args.run

    print(f"🔧 Using GPU type: {gpu_type}")
    print(f"📊 Loading experiments from: {csv_path}")
    if dry_run:
        print("💡 DRY RUN mode - add --run to actually launch clusters")

    experiments = load_experiments(csv_path)
    io_groups = group_by_input_output_then_cluster(experiments, gpu_type, vm_strategy=vm_strategy)

    print(f"📊 Loaded {len(experiments)} experiments")
    print(f"📦 Grouped by input/output length, then by cluster config:")
    for (input_len, output_len), cluster_groups in sorted(io_groups.items()):
        print(f"   {input_len}in/{output_len}out: {len(cluster_groups)} cluster configs")
        for (gpus, nodes), exps in sorted(cluster_groups.items()):
            print(f"      {gpus} GPU/node × {nodes} nodes: {len(exps)} experiments")

    if dry_run:
        print("\n💡 This is a DRY RUN. Add --run to actually launch clusters.")

    # Run each group and collect all results
    all_results = []

    def cluster_sort_key(item):
        (gpus_per_node, num_nodes), _ = item
        # Cost/spot-friendly: TP asc (1,4,8), then PP asc (1..4)
        return (gpus_per_node, num_nodes)

    try:
        for (input_len, output_len), cluster_groups in sorted(io_groups.items()):
            # Create parent directory for this input/output length
            parent_dir = f"result-{input_len}in_{output_len}out"
            
            # Collect results for this IO group
            io_group_results = []
            
            for cluster_config, exps in sorted(cluster_groups.items(), key=cluster_sort_key):
                results = run_cluster_benchmarks(cluster_config, exps, parent_dir=parent_dir, dry_run=dry_run, gpu_type=gpu_type)
                if results:
                    io_group_results.extend(results)
                    all_results.extend(results)
            
            # Note: per-cluster CSVs are already saved inside each cluster's result directory
            # by run_cluster_benchmarks(). We intentionally do NOT write an extra aggregated
            # CSV at the IO-group level here to avoid duplicate benchmark_results-*.csv files.
    except Exception as e:
        print(f"\n❌ Unexpected error in main: {e}")
        print("   Cleanup will be triggered automatically...")
        raise  # Re-raise to trigger atexit handler


if __name__ == "__main__":
    main()


    # some notes
    # without lmcache, vllm v0.10.2 worked for tp4,pp3 (almost?)
    # now testing with vllm v0.11.0 and no lmcache (this also worked)
    # have to test with vllm v0.11.0 and lmcache v0.3.9 (trying this now) (didnt work, one exp worked, rest didnt, mostly due to ray dag)
    # have to test with vllm v0.11.0 and lmcache v0.3.10 (same issue)
    # have to test with vllm v0.13.0 and lmcache v0.3.10 (same issue) (this issue is interesting - layers not found? wtf?)
    # there is something called as a lmcache mp connector, will check into that as well
    # need to check if ray restarts help withn vllmv0.11 and v0.3.9 (does not help)
    # my suspicion is that i had not put the time between two llm establishments and cleanup_dist_env_and_memory was not used (this HELPS!)
    # export VLLM_USE_RAY_WRAPPED_PP_COMM=0 maybe trying this helps? (same as all)
    # have to test with vllm v0.13.0 and lmcache v0.3.11 (latest versions, and building from source????)
    # this seems like another flag to try enable_async_loading: True (didnt do anything significiant)
    # vllm with v0.10.2 is NOT WORKING for pp=4 tp=4, maybe thats also an issue, 
    # will try offloading connector as well

    # TP=4/PP=4 worked with vllm v0.10.0 and max_gpu_util = 0.85 and 
    # export NCCL_P2P_DISABLE=1
    # export NCCL_TIMEOUT=1800
    # export TORCH_NCCL_BLOCKING_WAIT=1
    # export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    # export TORCH_NCCL_TRACE_BUFFER_SIZE=10000
    # export TORCH_DISTRIBUTED_DEBUG=DETAIL  
    # export NCCL_DEBUG=INFO 

    # trying lmcache w it (didnt work tbh)