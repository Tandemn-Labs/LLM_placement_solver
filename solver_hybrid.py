#!/usr/bin/env python3
"""
LLM Model Parallelism Placement Solver - HYBRID PARALLELISM EXTENSION
Extends the constrained solver to support multi-GPU segments for data parallelism
within pipeline stages, enabling optimal hybrid parallelism strategies.
"""

import os
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import argparse
import json
import logging
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GPUType:
    """GPU type specification"""
    name: str
    count: int
    memory_gb: float
    global_ids: List[int]  # Global GPU IDs for this type

@dataclass
class Config:
    """Runtime configuration"""
    sequence_length: int
    batch_size: int
    model_name: str
    num_decoder_layers: int
    d_model: int
    d_hidden: int
    vocab_size: int
    num_attention_heads: int
    layer_weight_memory_gb: float
    time_limit_seconds: float
    optimality_gap: float
    bytes_per_element: int = 2  # FP16 by default, can be 4 for FP32

    # NEW: Hybrid parallelism parameters
    enable_data_parallelism: bool = True
    max_gpus_per_segment: int = 8  # Limit to prevent explosion
    communication_efficiency: float = 0.85  # All-reduce efficiency factor

@dataclass
class Segment:
    """Simplified segment representation for hybrid parallelism"""
    start_layer: int
    segment_size: int

    def __hash__(self):
        return hash((self.start_layer, self.segment_size))

    def __eq__(self, other):
        return (self.start_layer, self.segment_size) == (other.start_layer, other.segment_size)

    @property
    def end_layer(self):
        return self.start_layer + self.segment_size - 1

class ThroughputFunctions:
    """Extended throughput functions with communication modeling"""

    # GPU throughput coefficients (unchanged)
    GPU_THROUGHPUT_COEFFS = {
        'A100': {'seq_len': -0.01, 'batch_size': 5.0, 'num_layers': 2.0, 'constant': 150.0},
        'V100': {'seq_len': -0.008, 'batch_size': 3.0, 'num_layers': 1.5, 'constant': 100.0},
        'H100': {'seq_len': -0.012, 'batch_size': 6.0, 'num_layers': 2.5, 'constant': 200.0},
        'RTX4090': {'seq_len': -0.006, 'batch_size': 2.5, 'num_layers': 1.0, 'constant': 80.0},
        'L20': {'seq_len': -0.009, 'batch_size': 4.0, 'num_layers': 1.8, 'constant': 120.0},
        'A10': {'seq_len': -0.007, 'batch_size': 2.0, 'num_layers': 1.2, 'constant': 70.0},
        'A40': {'seq_len': -0.009, 'batch_size': 4.5, 'num_layers': 1.9, 'constant': 140.0},
        'T4': {'seq_len': -0.005, 'batch_size': 1.5, 'num_layers': 0.8, 'constant': 50.0}
    }

    NETWORK_COEFFS = {
        'bandwidth': 1.2, 'seq_len': -0.001, 'batch_size': -0.1, 'hidden_dim': -0.00001, 'constant': 50.0
    }

    @staticmethod
    def gpu_throughput(gpu_type: str, seq_len: int, batch_size: int, num_layers: int) -> float:
        """Single GPU throughput function (tokens/sec)"""
        coeffs = ThroughputFunctions.GPU_THROUGHPUT_COEFFS[gpu_type]
        throughput = (coeffs['seq_len'] * seq_len +
                     coeffs['batch_size'] * batch_size +
                     coeffs['num_layers'] * num_layers +
                     coeffs['constant'])
        return max(1.0, throughput)

    @staticmethod
    def multi_gpu_throughput(gpu_type: str, seq_len: int, batch_size: int, num_layers: int,
                           num_gpus: int, comm_efficiency: float = 0.85) -> float:
        """Multi-GPU segment throughput with communication overhead"""
        single_gpu_throughput = ThroughputFunctions.gpu_throughput(gpu_type, seq_len, batch_size, num_layers)

        if num_gpus == 1:
            return single_gpu_throughput

        # Each GPU processes batch_size/num_gpus samples
        per_gpu_batch = batch_size / num_gpus
        per_gpu_throughput = ThroughputFunctions.gpu_throughput(gpu_type, seq_len, per_gpu_batch, num_layers)

        # All-reduce communication overhead
        # Communication time increases with num_gpus but efficiency factor compensates
        communication_factor = comm_efficiency * (2 - 1/num_gpus)  # Approaches 2*efficiency as num_gpus → ∞
        effective_throughput = per_gpu_throughput * num_gpus * communication_factor

        return max(1.0, effective_throughput)

    @staticmethod
    def network_throughput(bandwidth_gbps: float, seq_len: int, batch_size: int, hidden_dim: int) -> float:
        """Network throughput function (transfers/sec) - unchanged"""
        coeffs = ThroughputFunctions.NETWORK_COEFFS
        throughput = (coeffs['bandwidth'] * bandwidth_gbps +
                     coeffs['seq_len'] * seq_len +
                     coeffs['batch_size'] * batch_size +
                     coeffs['hidden_dim'] * hidden_dim +
                     coeffs['constant'])
        return max(1.0, throughput)

    @staticmethod
    def memory_usage(seq_len: int, batch_size: int, num_layers: int, layer_weight_gb: float,
                    d_model: int, d_hidden: int, num_gpus: int = 1, bytes_per_element: int = 2) -> float:
        """Memory usage in GB with multi-GPU support"""
        # Model weights (full copy per GPU)
        weight_memory = num_layers * layer_weight_gb

        # Activation memory scales with effective batch size per GPU
        effective_batch = batch_size / num_gpus

        # Attention matrix: batch_size × seq_len × seq_len × d_model
        attention_memory = effective_batch * seq_len * seq_len * d_model * bytes_per_element / (1024**3)

        # K,V cache: 2 × batch_size × seq_len × d_model
        kv_cache_memory = 2 * effective_batch * seq_len * d_model * bytes_per_element / (1024**3)

        # Hidden states: batch_size × seq_len × d_hidden
        hidden_memory = effective_batch * seq_len * d_hidden * bytes_per_element / (1024**3)

        # Intermediate memory per layer
        intermediate_memory_per_layer = (attention_memory + kv_cache_memory + hidden_memory) / 1024
        total_intermediate = intermediate_memory_per_layer * min(num_layers, 2)

        return weight_memory + total_intermediate

class HybridLLMPlacementSolver:
    """Extended solver class for hybrid parallelism optimization"""

    def __init__(self, config_dir: str, enable_symmetry_breaking: bool = True,
                 enable_upper_bound: bool = True, enable_tight_bigm: bool = True,
                 enable_flow_conservation: bool = True, threads: Optional[int] = None,
                 max_threads: int = 32):
        self.options = {
            "WLSACCESSID": "790b9c11-45d0-4785-8d99-a5e6414f9321",
            "WLSSECRET": "adef4738-7bf6-41b8-8dfd-d04e23d53e51",
            "LICENSEID": 2415150,
        }
        self.env = gp.Env(params=self.options)
        self.config_dir = config_dir

        # Optimization flags
        self.enable_symmetry_breaking = enable_symmetry_breaking
        self.enable_upper_bound = enable_upper_bound
        self.enable_tight_bigm = enable_tight_bigm
        self.enable_flow_conservation = enable_flow_conservation
        self.threads = threads
        self.max_threads = max_threads

        # Load configuration files
        gpu_pool_file = os.path.join(config_dir, 'gpu_pool.csv')
        network_file = os.path.join(config_dir, 'network_bandwidth.csv')
        config_file = os.path.join(config_dir, 'config.csv')

        self.gpu_types = self._load_gpu_pool(gpu_pool_file)
        self.total_gpus = sum(gpu_type.count for gpu_type in self.gpu_types.values())
        self.network_bandwidth = self._load_network_bandwidth(network_file)
        self.config = self._load_config(config_file)
        self.model = None
        self.solution = None

        # Validate network matrix matches GPU count
        if self.network_bandwidth.shape[0] != self.total_gpus:
            raise ValueError(f"Network bandwidth matrix size ({self.network_bandwidth.shape[0]}) "
                           f"does not match total GPU count ({self.total_gpus})")

        # Generate valid segments and GPU allocations
        self.valid_segments = self._generate_valid_segments()
        self.max_gpus_per_segment = self._compute_max_gpus_per_segment()
        self.valid_gpu_allocations = self._generate_valid_gpu_allocations()
        self.valid_connections = self._generate_valid_connections()

        # Validate problem size
        self._validate_problem_size()

        logger.info(f"Initialized hybrid solver: {len(self.gpu_types)} GPU types, {self.total_gpus} total GPUs")
        logger.info(f"Model: {self.config.num_decoder_layers} layers, batch_size={self.config.batch_size}")
        logger.info(f"Problem size: {len(self.valid_segments)} segments, {len(self.valid_gpu_allocations)} allocations")

    def _load_gpu_pool(self, filename: str) -> Dict[str, GPUType]:
        """Load GPU pool configuration"""
        df = pd.read_csv(filename)
        gpu_types = {}
        global_id = 0

        for _, row in df.iterrows():
            global_ids = list(range(global_id, global_id + row['count']))
            gpu_types[row['gpu_type']] = GPUType(
                name=row['gpu_type'],
                count=row['count'],
                memory_gb=row['memory_gb'],
                global_ids=global_ids
            )
            global_id += row['count']

        return gpu_types

    def _load_network_bandwidth(self, filename: str) -> np.ndarray:
        """Load network bandwidth matrix"""
        df = pd.read_csv(filename, index_col=0)
        matrix = df.values

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Network bandwidth matrix must be square, got {matrix.shape}")

        return matrix

    def _load_config(self, filename: str) -> Config:
        """Load runtime configuration with hybrid parallelism defaults"""
        df = pd.read_csv(filename)
        config_dict = dict(zip(df['parameter'], df['value']))

        bytes_per_element = int(config_dict.get('bytes_per_element', 2))

        return Config(
            sequence_length=int(config_dict['sequence_length']),
            batch_size=int(config_dict['batch_size']),
            model_name=config_dict['model_name'],
            num_decoder_layers=int(config_dict['num_decoder_layers']),
            d_model=int(config_dict['d_model']),
            d_hidden=int(config_dict['d_hidden']),
            vocab_size=int(config_dict['vocab_size']),
            num_attention_heads=int(config_dict['num_attention_heads']),
            layer_weight_memory_gb=float(config_dict['layer_weight_memory_gb']),
            time_limit_seconds=float(config_dict['time_limit_seconds']),
            optimality_gap=float(config_dict['optimality_gap']),
            bytes_per_element=bytes_per_element,
            # Hybrid parallelism defaults
            enable_data_parallelism=True,
            max_gpus_per_segment=min(8, self.total_gpus // 2),  # Conservative default
            communication_efficiency=0.85
        )

    def _generate_valid_segments(self) -> Set[Segment]:
        """Generate valid layer segments based on memory constraints"""
        valid_segments = set()

        # Calculate maximum layers per GPU type
        max_layers_per_type = {}
        for gpu_type_name, gpu_type in self.gpu_types.items():
            # Binary search for max layers (single GPU)
            max_layers = self._binary_search_max_layers(gpu_type_name, gpu_type.memory_gb)
            max_layers_per_type[gpu_type_name] = max_layers

        # Generate segments with intelligent size constraints
        min_segment_size = 1
        max_segment_size = max(max_layers_per_type.values())

        logger.info(f"Generating segments with size range: {min_segment_size}-{max_segment_size}")

        for start_layer in range(1, self.config.num_decoder_layers + 1):
            for segment_size in range(min_segment_size, min(max_segment_size + 1,
                                                          self.config.num_decoder_layers - start_layer + 2)):
                if start_layer + segment_size - 1 <= self.config.num_decoder_layers:
                    segment = Segment(start_layer, segment_size)
                    valid_segments.add(segment)

        logger.info(f"Generated {len(valid_segments)} valid segments")
        return valid_segments

    def _binary_search_max_layers(self, gpu_type: str, memory_gb: float) -> int:
        """Binary search to find maximum layers for single GPU"""
        left, right = 1, self.config.num_decoder_layers
        max_feasible = 1

        while left <= right:
            mid = (left + right) // 2
            total_memory = ThroughputFunctions.memory_usage(
                self.config.sequence_length, self.config.batch_size, mid,
                self.config.layer_weight_memory_gb, self.config.d_model,
                self.config.d_hidden, num_gpus=1, bytes_per_element=self.config.bytes_per_element
            )

            if total_memory <= memory_gb:
                max_feasible = mid
                left = mid + 1
            else:
                right = mid - 1

        return max_feasible

    def _compute_max_gpus_per_segment(self) -> Dict[Tuple[Segment, str], int]:
        """Compute maximum GPUs that can be allocated to each segment per GPU type"""
        max_gpus = {}

        for segment in self.valid_segments:
            for gpu_type_name, gpu_type in self.gpu_types.items():
                # Check memory constraint for this segment with multiple GPUs
                max_gpus_for_segment = 1

                for num_gpus in range(1, min(gpu_type.count, self.config.max_gpus_per_segment) + 1):
                    memory_needed = ThroughputFunctions.memory_usage(
                        self.config.sequence_length, self.config.batch_size,
                        segment.segment_size, self.config.layer_weight_memory_gb,
                        self.config.d_model, self.config.d_hidden,
                        num_gpus=num_gpus, bytes_per_element=self.config.bytes_per_element
                    )

                    if memory_needed <= gpu_type.memory_gb:
                        max_gpus_for_segment = num_gpus
                    else:
                        break

                max_gpus[(segment, gpu_type_name)] = max_gpus_for_segment

        return max_gpus

    def _generate_valid_gpu_allocations(self) -> List[Tuple[Segment, str, int]]:
        """Generate valid (segment, gpu_type, num_gpus) allocations"""
        valid_allocations = []

        for segment in self.valid_segments:
            for gpu_type_name in self.gpu_types:
                max_gpus = self.max_gpus_per_segment.get((segment, gpu_type_name), 1)
                for num_gpus in range(1, max_gpus + 1):
                    valid_allocations.append((segment, gpu_type_name, num_gpus))

        logger.info(f"Generated {len(valid_allocations)} valid GPU allocations")
        return valid_allocations

    def _generate_valid_connections(self) -> List[Tuple[Segment, Segment]]:
        """Generate valid connections between consecutive segments"""
        valid_connections = []

        # Group segments by ending and starting layers
        segments_by_end = {}
        segments_by_start = {}

        for segment in self.valid_segments:
            end_layer = segment.end_layer
            start_layer = segment.start_layer

            if end_layer not in segments_by_end:
                segments_by_end[end_layer] = []
            segments_by_end[end_layer].append(segment)

            if start_layer not in segments_by_start:
                segments_by_start[start_layer] = []
            segments_by_start[start_layer].append(segment)

        # Generate connections between consecutive layers
        for layer in range(1, self.config.num_decoder_layers):
            ending_segments = segments_by_end.get(layer, [])
            starting_segments = segments_by_start.get(layer + 1, [])

            for seg1 in ending_segments:
                for seg2 in starting_segments:
                    valid_connections.append((seg1, seg2))

        logger.info(f"Generated {len(valid_connections)} valid connections")
        return valid_connections

    def _validate_problem_size(self):
        """Validate that hybrid problem size is manageable"""
        num_segments = len(self.valid_segments)
        num_allocations = len(self.valid_gpu_allocations)
        num_connections = len(self.valid_connections)

        # Estimate total variables
        binary_vars = num_segments + num_allocations + num_connections  # x, n, e variables
        continuous_vars = num_segments + num_connections + self.total_gpus  # throughput variables

        total_vars = binary_vars + continuous_vars

        logger.info(f"Hybrid problem size validation:")
        logger.info(f"  - Segments: {num_segments}")
        logger.info(f"  - GPU allocations: {num_allocations}")
        logger.info(f"  - Connections: {num_connections}")
        logger.info(f"  - Total variables: ~{total_vars} ({binary_vars} binary + {continuous_vars} continuous)")

        if total_vars > 500000:
            logger.warning(f"Very large problem ({total_vars} variables). Consider reducing max_gpus_per_segment.")
        elif total_vars > 100000:
            logger.info(f"Large problem ({total_vars} variables). Solving may take 10-30 minutes.")
        else:
            logger.info(f"Manageable problem size ({total_vars} variables).")

    def build_model(self):
        """Build the hybrid parallelism optimization model"""
        logger.info("Building hybrid parallelism optimization model...")

        # Create model
        self.model = gp.Model("hybrid_llm_placement", env=self.env)

        # Solver parameters (more conservative for larger problems)
        self.model.setParam('Presolve', 2)
        self.model.setParam('Cuts', 1)
        self.model.setParam('Heuristics', 0.1)  # More heuristics time for complex problems
        self.model.setParam('MIPFocus', 1)
        self.model.setParam('NodefileStart', 0.5)
        self.model.setParam('TimeLimit', self.config.time_limit_seconds)
        self.model.setParam('MIPGap', self.config.optimality_gap)
        self.model.setParam('LogToConsole', 1)

        # Create decision variables
        self._create_variables()
        self._create_constraints()
        self._set_objective()

        logger.info("Hybrid model built successfully")

    def _create_variables(self):
        """Create decision variables for hybrid parallelism"""
        logger.info("Creating decision variables...")

        # Segment selection: x[segment] ∈ {0,1}
        self.x = self.model.addVars(
            self.valid_segments,
            vtype=GRB.BINARY,
            name="segment_selection"
        )

        # GPU allocation: n[segment, gpu_type, num_gpus] ∈ {0,1}
        self.n = self.model.addVars(
            self.valid_gpu_allocations,
            vtype=GRB.BINARY,
            name="gpu_allocation"
        )

        # Network connections: e[seg1, seg2] ∈ {0,1}
        self.e = self.model.addVars(
            self.valid_connections,
            vtype=GRB.BINARY,
            name="network_connection"
        )

        # Segment throughputs: τ[segment] ∈ R+
        self.tau_segment = self.model.addVars(
            self.valid_segments,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="segment_throughput"
        )

        # Network throughputs: ρ[seg1, seg2] ∈ R+
        self.rho = self.model.addVars(
            self.valid_connections,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="network_throughput"
        )

        # End-to-end throughput: t ∈ R+
        self.t = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="end_to_end_throughput")

        logger.info(f"Created {len(self.x) + len(self.n) + len(self.e)} binary variables")
        logger.info(f"Created {len(self.tau_segment) + len(self.rho) + 1} continuous variables")

    def _create_constraints(self):
        """Create optimization constraints for hybrid parallelism"""
        logger.info("Creating optimization constraints...")

        # 1. Layer coverage: each layer assigned exactly once
        for layer in range(1, self.config.num_decoder_layers + 1):
            covering_segments = [seg for seg in self.valid_segments
                               if seg.start_layer <= layer <= seg.end_layer]
            self.model.addConstr(
                gp.quicksum(self.x[seg] for seg in covering_segments) == 1,
                name=f"layer_coverage_{layer}"
            )

        # 2. Segment-allocation consistency: if segment selected, exactly one allocation
        for segment in self.valid_segments:
            segment_allocations = [alloc for alloc in self.valid_gpu_allocations
                                 if alloc[0] == segment]
            self.model.addConstr(
                self.x[segment] == gp.quicksum(self.n[alloc] for alloc in segment_allocations),
                name=f"segment_allocation_consistency_{segment.start_layer}_{segment.segment_size}"
            )

        # 3. GPU capacity: each GPU used in at most one segment
        for gpu_type_name, gpu_type in self.gpu_types.items():
            for gpu_id in range(gpu_type.count):
                gpu_allocations = []
                for segment, gt, num_gpus in self.valid_gpu_allocations:
                    if gt == gpu_type_name and num_gpus > gpu_id:
                        gpu_allocations.append((segment, gt, num_gpus))

                if gpu_allocations:
                    self.model.addConstr(
                        gp.quicksum(self.n[alloc] for alloc in gpu_allocations) <= 1,
                        name=f"gpu_capacity_{gpu_type_name}_{gpu_id}"
                    )

        # 4. Network connection logic
        for seg1, seg2 in self.valid_connections:
            # Connection requires both segments to be selected
            self.model.addConstr(
                self.e[seg1, seg2] <= self.x[seg1],
                name=f"connection_req_seg1"
            )
            self.model.addConstr(
                self.e[seg1, seg2] <= self.x[seg2],
                name=f"connection_req_seg2"
            )
            # If both segments selected, connection must exist
            self.model.addConstr(
                self.e[seg1, seg2] >= self.x[seg1] + self.x[seg2] - 1,
                name=f"connection_both_selected"
            )

        # 5. Pipeline connectivity constraints
        self._add_pipeline_connectivity_constraints()

        # 6. Segment throughput definition (with linearization)
        self._add_segment_throughput_constraints()

        # 7. Network throughput constraints
        self._add_network_throughput_constraints()

        # 8. End-to-end throughput constraints
        self._add_end_to_end_throughput_constraints()

        logger.info("Optimization constraints created successfully")

    def _add_pipeline_connectivity_constraints(self):
        """Add constraints ensuring complete pipeline connectivity"""
        # Pipeline must start at layer 1
        first_layer_segments = [seg for seg in self.valid_segments if seg.start_layer == 1]
        if first_layer_segments:
            self.model.addConstr(
                gp.quicksum(self.x[seg] for seg in first_layer_segments) >= 1,
                name="pipeline_starts_at_layer_1"
            )

        # Sequential connectivity for non-terminal layers
        for layer in range(1, self.config.num_decoder_layers):
            segments_ending_here = [seg for seg in self.valid_segments if seg.end_layer == layer]
            segments_starting_next = [seg for seg in self.valid_segments if seg.start_layer == layer + 1]

            if segments_ending_here and segments_starting_next:
                # Outgoing connections for segments ending at this layer
                for seg1 in segments_ending_here:
                    valid_next_connections = [(s1, s2) for s1, s2 in self.valid_connections
                                            if s1 == seg1 and s2 in segments_starting_next]
                    if valid_next_connections:
                        self.model.addConstr(
                            gp.quicksum(self.e[s1, s2] for s1, s2 in valid_next_connections) >= self.x[seg1],
                            name=f"connectivity_out_{layer}"
                        )

                # Incoming connections for segments starting at next layer
                for seg2 in segments_starting_next:
                    valid_prev_connections = [(s1, s2) for s1, s2 in self.valid_connections
                                            if s2 == seg2 and s1 in segments_ending_here]
                    if valid_prev_connections:
                        self.model.addConstr(
                            gp.quicksum(self.e[s1, s2] for s1, s2 in valid_prev_connections) >= self.x[seg2],
                            name=f"connectivity_in_{layer}"
                        )

    def _add_segment_throughput_constraints(self):
        """Add segment throughput constraints with linearization"""
        # For each segment, throughput depends on GPU allocation
        for segment in self.valid_segments:
            segment_allocations = [alloc for alloc in self.valid_gpu_allocations if alloc[0] == segment]

            # Throughput contribution from each allocation
            throughput_expr = gp.LinExpr()
            for seg, gpu_type, num_gpus in segment_allocations:
                throughput_value = ThroughputFunctions.multi_gpu_throughput(
                    gpu_type, self.config.sequence_length, self.config.batch_size,
                    segment.segment_size, num_gpus, self.config.communication_efficiency
                )
                throughput_expr += throughput_value * self.n[seg, gpu_type, num_gpus]

            self.model.addConstr(
                self.tau_segment[segment] == throughput_expr,
                name=f"segment_throughput_{segment.start_layer}_{segment.segment_size}"
            )

    def _add_network_throughput_constraints(self):
        """Add network throughput constraints"""
        # Pre-compute tensor size for transfers
        tensor_size_gb = (self.config.batch_size * self.config.sequence_length *
                         self.config.d_model * self.config.bytes_per_element) / (1024**3)

        for seg1, seg2 in self.valid_connections:
            # Simplified: assume uniform network bandwidth for now
            # In practice, this would depend on specific GPU allocations
            avg_bandwidth = np.mean(self.network_bandwidth)  # Simplification
            max_throughput = avg_bandwidth / tensor_size_gb if tensor_size_gb > 0 else 1000.0

            self.model.addConstr(
                self.rho[seg1, seg2] == max_throughput * self.e[seg1, seg2],
                name=f"network_throughput_{seg1.start_layer}_{seg2.start_layer}"
            )

    def _add_end_to_end_throughput_constraints(self):
        """Add end-to-end throughput constraints"""
        # Compute Big-M values
        max_segment_throughput = 0
        for segment in self.valid_segments:
            for gpu_type in self.gpu_types:
                max_gpus = self.max_gpus_per_segment.get((segment, gpu_type), 1)
                throughput = ThroughputFunctions.multi_gpu_throughput(
                    gpu_type, self.config.sequence_length, self.config.batch_size,
                    segment.segment_size, max_gpus, self.config.communication_efficiency
                )
                max_segment_throughput = max(max_segment_throughput, throughput)

        M_segment = max_segment_throughput * 1.1  # 10% buffer
        M_network = np.max(self.network_bandwidth) * 2  # Conservative network bound

        # Segment throughput constraints
        for segment in self.valid_segments:
            self.model.addConstr(
                self.t <= self.tau_segment[segment] + M_segment * (1 - self.x[segment]),
                name=f"throughput_segment_{segment.start_layer}_{segment.segment_size}"
            )

        # Network throughput constraints
        for seg1, seg2 in self.valid_connections:
            self.model.addConstr(
                self.t <= self.rho[seg1, seg2] + M_network * (1 - self.e[seg1, seg2]),
                name=f"throughput_network_{seg1.start_layer}_{seg2.start_layer}"
            )

    def _set_objective(self):
        """Set optimization objective"""
        self.model.setObjective(self.t, GRB.MAXIMIZE)

    def solve(self) -> bool:
        """Solve the hybrid parallelism optimization problem"""
        # Dynamic thread allocation
        total_vars = len(self.x) + len(self.n) + len(self.e) + len(self.tau_segment) + len(self.rho) + 1
        available_threads = min(self.max_threads, os.cpu_count())

        if self.threads is not None:
            threads = min(self.threads, available_threads)
            logger.info(f"Using manually specified {threads} threads")
        else:
            if total_vars > 100000:
                threads = min(available_threads, 16)
            elif total_vars > 50000:
                threads = min(available_threads, 8)
            else:
                threads = min(available_threads, 4)
            logger.info(f"Auto-scaling to {threads} threads (problem size: {total_vars})")

        self.model.setParam('Threads', threads)

        logger.info("Starting hybrid parallelism optimization...")
        start_time = time.time()

        try:
            self.model.optimize()
            solve_time = time.time() - start_time

            if self.model.status == GRB.OPTIMAL:
                logger.info(f"Optimal solution found in {solve_time:.2f} seconds")
                logger.info(f"Optimal throughput: {self.t.x:.2f} tokens/sec")
                self._extract_solution()
                return True
            elif self.model.status == GRB.TIME_LIMIT:
                if self.model.SolCount > 0:
                    logger.warning(f"Time limit reached. Best solution: {self.t.x:.2f}")
                    self._extract_solution()
                    return True
                else:
                    logger.error("Time limit reached with no feasible solution")
                    return False
            else:
                logger.error(f"No solution found. Status: {self.model.status}")
                return False

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return False

    def _extract_solution(self):
        """Extract solution from solved model"""
        self.solution = {
            'objective_value': self.t.x,
            'segment_assignments': [],
            'gpu_allocations': [],
            'network_connections': [],
            'solve_status': self.model.status
        }

        # Extract segment assignments
        for segment in self.valid_segments:
            if self.x[segment].x > 0.5:
                self.solution['segment_assignments'].append({
                    'start_layer': segment.start_layer,
                    'end_layer': segment.end_layer,
                    'segment_size': segment.segment_size,
                    'throughput': self.tau_segment[segment].x
                })

        # Extract GPU allocations
        for segment, gpu_type, num_gpus in self.valid_gpu_allocations:
            if self.n[segment, gpu_type, num_gpus].x > 0.5:
                self.solution['gpu_allocations'].append({
                    'segment': {'start_layer': segment.start_layer, 'end_layer': segment.end_layer},
                    'gpu_type': gpu_type,
                    'num_gpus': num_gpus,
                    'workload_per_gpu': self.config.batch_size / num_gpus
                })

        # Extract network connections
        for seg1, seg2 in self.valid_connections:
            if self.e[seg1, seg2].x > 0.5:
                self.solution['network_connections'].append({
                    'from_segment': {'start_layer': seg1.start_layer, 'end_layer': seg1.end_layer},
                    'to_segment': {'start_layer': seg2.start_layer, 'end_layer': seg2.end_layer},
                    'throughput': self.rho[seg1, seg2].x
                })

        # Sort by start layer
        self.solution['segment_assignments'].sort(key=lambda x: x['start_layer'])
        self.solution['gpu_allocations'].sort(key=lambda x: x['segment']['start_layer'])

    def print_solution(self):
        """Print the hybrid parallelism solution"""
        if not self.solution:
            logger.error("No solution available")
            return

        print("\n" + "="*90)
        print(f"HYBRID LLM PLACEMENT OPTIMIZATION RESULTS")
        print("="*90)
        print(f"Model: {self.config.model_name} ({self.config.num_decoder_layers} layers)")
        print(f"Batch Size: {self.config.batch_size}, Sequence Length: {self.config.sequence_length}")
        print(f"Optimal End-to-End Throughput: {self.solution['objective_value']:.2f} tokens/sec")
        print()

        print("HYBRID GPU ALLOCATIONS:")
        print("-" * 90)
        print(f"{'Layers':<15} {'GPU Type':<10} {'Num GPUs':<10} {'Workload/GPU':<15} {'Segment Throughput':<20}")
        print("-" * 90)

        for allocation in self.solution['gpu_allocations']:
            layers_str = f"{allocation['segment']['start_layer']}-{allocation['segment']['end_layer']}"
            print(f"{layers_str:<15} {allocation['gpu_type']:<10} {allocation['num_gpus']:<10} "
                  f"{allocation['workload_per_gpu']:<15.1f} "
                  f"{[seg['throughput'] for seg in self.solution['segment_assignments'] if seg['start_layer'] == allocation['segment']['start_layer']][0]:<20.2f}")

        if self.solution['network_connections']:
            print("\nNETWORK CONNECTIONS:")
            print("-" * 60)
            for i, conn in enumerate(self.solution['network_connections']):
                seg1, seg2 = conn['from_segment'], conn['to_segment']
                print(f"Connection {i+1}: Layers {seg1['start_layer']}-{seg1['end_layer']} -> "
                      f"Layers {seg2['start_layer']}-{seg2['end_layer']} "
                      f"[Throughput: {conn['throughput']:.2f}]")

        print("\n" + "="*90)

    def save_solution(self, output_file: str):
        """Save solution to JSON file"""
        if not self.solution:
            logger.error("No solution available to save")
            return

        output_data = {
            'config': {
                'model_name': self.config.model_name,
                'num_decoder_layers': self.config.num_decoder_layers,
                'sequence_length': self.config.sequence_length,
                'batch_size': self.config.batch_size,
                'enable_data_parallelism': self.config.enable_data_parallelism,
                'max_gpus_per_segment': self.config.max_gpus_per_segment
            },
            'solution': self.solution
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Hybrid solution saved to {output_file}")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Hybrid LLM Model Parallelism Placement Optimizer')
    parser.add_argument('--config-dir', required=True, help='Configuration directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--max-gpus-per-segment', type=int, default=4,
                       help='Maximum GPUs per segment (default: 4)')

    # Optimization flags
    parser.add_argument('--enable-symmetry-breaking', action='store_true', default=True)
    parser.add_argument('--disable-symmetry-breaking', dest='enable_symmetry_breaking', action='store_false')
    parser.add_argument('--enable-upper-bound', action='store_true', default=True)
    parser.add_argument('--disable-upper-bound', dest='enable_upper_bound', action='store_false')
    parser.add_argument('--enable-tight-bigm', action='store_true', default=True)
    parser.add_argument('--disable-tight-bigm', dest='enable_tight_bigm', action='store_false')
    parser.add_argument('--enable-flow-conservation', action='store_true', default=True)
    parser.add_argument('--disable-flow-conservation', dest='enable_flow_conservation', action='store_false')
    parser.add_argument('--threads', type=int, help='Number of threads')
    parser.add_argument('--max-threads', type=int, default=32, help='Maximum threads')

    args = parser.parse_args()

    start_time = time.time()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Hybrid parallelism optimization settings:")
    logger.info(f"  - Max GPUs per segment: {args.max_gpus_per_segment}")
    logger.info(f"  - Symmetry breaking: {args.enable_symmetry_breaking}")
    logger.info(f"  - Smart upper bound: {args.enable_upper_bound}")
    logger.info(f"  - Tight Big-M: {args.enable_tight_bigm}")
    logger.info(f"  - Flow conservation: {args.enable_flow_conservation}")

    try:
        # Initialize hybrid solver
        solver = HybridLLMPlacementSolver(
            args.config_dir,
            enable_symmetry_breaking=args.enable_symmetry_breaking,
            enable_upper_bound=args.enable_upper_bound,
            enable_tight_bigm=args.enable_tight_bigm,
            enable_flow_conservation=args.enable_flow_conservation,
            threads=args.threads,
            max_threads=args.max_threads
        )

        # Override max_gpus_per_segment
        solver.config.max_gpus_per_segment = args.max_gpus_per_segment

        # Regenerate allocations with new limit
        solver.max_gpus_per_segment = solver._compute_max_gpus_per_segment()
        solver.valid_gpu_allocations = solver._generate_valid_gpu_allocations()
        solver.valid_connections = solver._generate_valid_connections()
        solver._validate_problem_size()

        # Build and solve model
        solver.build_model()

        if solver.solve():
            solver.print_solution()
            output_file = os.path.join(args.config_dir, 'hybrid_solution.json')
            solver.save_solution(output_file)
        else:
            logger.error("Failed to find optimal solution")
            return 1

    except Exception as e:
        logger.error(f"Hybrid solver failed: {e}")
        return 1

    end_time = time.time()
    logger.info(f"Hybrid solver finished in {end_time - start_time:.0f} seconds")
    return 0


if __name__ == "__main__":
    exit(main())