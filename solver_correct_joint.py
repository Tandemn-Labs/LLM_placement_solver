#!/usr/bin/env python3
"""
LLM Model Parallelism Placement Solver - CORRECT JOINT ILP FORMULATION
Solves the joint optimization of layer placement and specific GPU allocation
using proper Integer Linear Programming without heuristics.

Key Features:
- Specific GPU assignments (not just counts)
- Pre-enumerated valid GPU combinations
- Exact network bandwidth modeling between GPU pairs
- Linear throughput constraints (no division)
- True joint optimization of placement + allocation
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
from typing import Dict, List, Tuple, Optional, Set, FrozenSet
from dataclasses import dataclass
import time
from itertools import combinations

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GPUType:
    """GPU type specification"""
    name: str
    count: int
    memory_gb: float
    global_ids: List[int]

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
    bytes_per_element: int = 2
    max_gpus_per_segment: int = 4
    communication_efficiency: float = 0.85

@dataclass(frozen=True)
class Segment:
    """Immutable segment representation"""
    start_layer: int
    segment_size: int

    @property
    def end_layer(self):
        return self.start_layer + self.segment_size - 1

@dataclass(frozen=True)
class GPUAllocation:
    """Specific GPU allocation for a segment"""
    segment: Segment
    gpu_ids: FrozenSet[int]  # Specific GPU global IDs

    @property
    def num_gpus(self):
        return len(self.gpu_ids)

    @property
    def gpu_types_used(self):
        """Return set of GPU types used in this allocation"""
        types = set()
        for gpu_id in self.gpu_ids:
            # Determine GPU type from global ID
            # This will be filled in by the solver based on gpu_types mapping
            pass
        return types

class ThroughputFunctions:
    """Exact throughput functions for specific GPU combinations"""

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

    @staticmethod
    def single_gpu_throughput(gpu_type: str, seq_len: int, batch_size: int, num_layers: int) -> float:
        """Single GPU throughput (tokens/sec)"""
        coeffs = ThroughputFunctions.GPU_THROUGHPUT_COEFFS[gpu_type]
        throughput = (coeffs['seq_len'] * seq_len +
                     coeffs['batch_size'] * batch_size +
                     coeffs['num_layers'] * num_layers +
                     coeffs['constant'])
        return max(1.0, throughput)

    @staticmethod
    def multi_gpu_throughput(gpu_types: List[str], seq_len: int, batch_size: int,
                           num_layers: int, num_gpus: int, comm_efficiency: float = 0.85) -> float:
        """
        Multi-GPU segment throughput - bottleneck of all GPUs in the group
        Each GPU processes batch_size/num_gpus samples
        """
        if num_gpus == 1:
            return ThroughputFunctions.single_gpu_throughput(gpu_types[0], seq_len, batch_size, num_layers)

        # Each GPU gets equal share of the batch
        per_gpu_batch = batch_size / num_gpus

        # Find bottleneck GPU (minimum throughput)
        min_throughput = float('inf')
        for gpu_type in gpu_types:
            gpu_throughput = ThroughputFunctions.single_gpu_throughput(
                gpu_type, seq_len, per_gpu_batch, num_layers
            )
            min_throughput = min(min_throughput, gpu_throughput)

        # Apply communication efficiency for multi-GPU coordination
        # The segment is limited by the slowest GPU, scaled by communication efficiency
        effective_throughput = min_throughput * num_gpus * comm_efficiency

        return max(1.0, effective_throughput)

    @staticmethod
    def memory_usage_per_gpu(seq_len: int, batch_size: int, num_layers: int,
                           layer_weight_gb: float, d_model: int, d_hidden: int,
                           num_gpus: int = 1, bytes_per_element: int = 2) -> float:
        """Memory usage per GPU in a multi-GPU segment"""
        # Each GPU needs full model weights for its layers
        weight_memory = num_layers * layer_weight_gb

        # Activation memory scales with per-GPU batch size
        per_gpu_batch = batch_size / num_gpus

        # Attention outputs: batch_size × seq_len × d_model
        attention_memory = per_gpu_batch * seq_len * d_model * bytes_per_element / (1024**3)

        # K,V cache: 2 × batch_size × seq_len × d_model
        kv_cache_memory = 2 * per_gpu_batch * seq_len * d_model * bytes_per_element / (1024**3)

        # Hidden states: batch_size × seq_len × d_hidden
        hidden_memory = per_gpu_batch * seq_len * d_hidden * bytes_per_element / (1024**3)

        # Framework overhead
        activation_memory = (attention_memory + kv_cache_memory + hidden_memory) * 1.15

        return weight_memory + activation_memory

class CorrectJointLLMSolver:
    """Correct joint ILP solver for placement + allocation optimization"""

    def __init__(self, config_dir: str, max_gpus_per_segment: int = 4,
                 threads: Optional[int] = None, max_threads: int = 32):
        self.options = {
            "WLSACCESSID": "790b9c11-45d0-4785-8d99-a5e6414f9321",
            "WLSSECRET": "adef4738-7bf6-41b8-8dfd-d04e23d53e51",
            "LICENSEID": 2415150,
        }
        self.env = gp.Env(params=self.options)
        self.config_dir = config_dir
        self.max_gpus_per_segment = max_gpus_per_segment
        self.threads = threads
        self.max_threads = max_threads

        # Load configuration
        self._load_configuration()

        # Generate valid segments and allocations
        self.valid_segments = self._generate_valid_segments()
        self.valid_allocations = self._generate_valid_allocations()
        self.valid_connections = self._generate_valid_connections()

        # Pre-compute throughput values
        self._precompute_allocation_throughputs()
        self._precompute_network_throughputs()

        # Validate problem size
        self._validate_problem_size()

        logger.info(f"Initialized correct joint solver:")
        logger.info(f"  - GPU types: {len(self.gpu_types)}, Total GPUs: {self.total_gpus}")
        logger.info(f"  - Model: {self.config.num_decoder_layers} layers")
        logger.info(f"  - Valid segments: {len(self.valid_segments)}")
        logger.info(f"  - Valid allocations: {len(self.valid_allocations)}")
        logger.info(f"  - Valid connections: {len(self.valid_connections)}")

    def _load_configuration(self):
        """Load all configuration files"""
        # Load GPU pool
        gpu_pool_file = os.path.join(self.config_dir, 'gpu_pool.csv')
        df = pd.read_csv(gpu_pool_file)
        self.gpu_types = {}
        self.global_id_to_type = {}
        global_id = 0

        for _, row in df.iterrows():
            global_ids = list(range(global_id, global_id + row['count']))
            gpu_type = GPUType(
                name=row['gpu_type'],
                count=row['count'],
                memory_gb=row['memory_gb'],
                global_ids=global_ids
            )
            self.gpu_types[row['gpu_type']] = gpu_type

            # Create mapping from global ID to GPU type
            for gid in global_ids:
                self.global_id_to_type[gid] = row['gpu_type']

            global_id += row['count']

        self.total_gpus = global_id

        # Load network bandwidth
        network_file = os.path.join(self.config_dir, 'network_bandwidth.csv')
        df = pd.read_csv(network_file, index_col=0)
        self.network_bandwidth = df.values

        if self.network_bandwidth.shape[0] != self.total_gpus:
            raise ValueError(f"Network matrix size ({self.network_bandwidth.shape[0]}) != GPU count ({self.total_gpus})")

        # Load config
        config_file = os.path.join(self.config_dir, 'config.csv')
        df = pd.read_csv(config_file)
        config_dict = dict(zip(df['parameter'], df['value']))

        self.config = Config(
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
            bytes_per_element=int(config_dict.get('bytes_per_element', 2)),
            max_gpus_per_segment=self.max_gpus_per_segment
        )

    def _generate_valid_segments(self) -> Set[Segment]:
        """Generate all valid layer segments"""
        valid_segments = set()

        # Generate all possible consecutive layer segments
        for start_layer in range(1, self.config.num_decoder_layers + 1):
            for segment_size in range(1, self.config.num_decoder_layers - start_layer + 2):
                if start_layer + segment_size - 1 <= self.config.num_decoder_layers:
                    segment = Segment(start_layer, segment_size)
                    valid_segments.add(segment)

        logger.info(f"Generated {len(valid_segments)} valid segments")
        return valid_segments

    def _is_allocation_feasible(self, segment: Segment, gpu_ids: FrozenSet[int]) -> bool:
        """Check if a GPU allocation is feasible for a segment"""
        # Check memory constraints for each GPU
        for gpu_id in gpu_ids:
            gpu_type = self.global_id_to_type[gpu_id]
            gpu_memory = self.gpu_types[gpu_type].memory_gb

            memory_needed = ThroughputFunctions.memory_usage_per_gpu(
                self.config.sequence_length, self.config.batch_size,
                segment.segment_size, self.config.layer_weight_memory_gb,
                self.config.d_model, self.config.d_hidden,
                len(gpu_ids), self.config.bytes_per_element
            )

            if memory_needed > gpu_memory:
                return False

        return True

    def _generate_valid_allocations(self) -> Set[GPUAllocation]:
        """Generate all valid GPU allocations for all segments"""
        valid_allocations = set()

        for segment in self.valid_segments:
            # Try all possible GPU combinations up to max_gpus_per_segment
            for num_gpus in range(1, min(self.max_gpus_per_segment + 1, self.total_gpus + 1)):
                # Generate all combinations of num_gpus from available GPUs
                for gpu_combination in combinations(range(self.total_gpus), num_gpus):
                    gpu_ids = frozenset(gpu_combination)

                    # Check if this allocation is feasible
                    if self._is_allocation_feasible(segment, gpu_ids):
                        allocation = GPUAllocation(segment, gpu_ids)
                        valid_allocations.add(allocation)

        logger.info(f"Generated {len(valid_allocations)} valid GPU allocations")
        return valid_allocations

    def _generate_valid_connections(self) -> Set[Tuple[GPUAllocation, GPUAllocation]]:
        """Generate valid connections between consecutive segment allocations"""
        valid_connections = set()

        # Group allocations by segment end/start layers
        allocations_by_end = {}
        allocations_by_start = {}

        for allocation in self.valid_allocations:
            segment = allocation.segment
            end_layer = segment.end_layer
            start_layer = segment.start_layer

            if end_layer not in allocations_by_end:
                allocations_by_end[end_layer] = []
            allocations_by_end[end_layer].append(allocation)

            if start_layer not in allocations_by_start:
                allocations_by_start[start_layer] = []
            allocations_by_start[start_layer].append(allocation)

        # Generate connections between consecutive layers
        for layer in range(1, self.config.num_decoder_layers):
            ending_allocations = allocations_by_end.get(layer, [])
            starting_allocations = allocations_by_start.get(layer + 1, [])

            for alloc1 in ending_allocations:
                for alloc2 in starting_allocations:
                    # Don't allow connections between allocations using same GPUs
                    if not alloc1.gpu_ids.intersection(alloc2.gpu_ids):
                        valid_connections.add((alloc1, alloc2))

        logger.info(f"Generated {len(valid_connections)} valid connections")
        return valid_connections

    def _precompute_allocation_throughputs(self):
        """Pre-compute throughput for each valid allocation"""
        self.allocation_throughputs = {}

        for allocation in self.valid_allocations:
            # Get GPU types for this allocation
            gpu_types = [self.global_id_to_type[gpu_id] for gpu_id in allocation.gpu_ids]

            # Calculate throughput
            throughput = ThroughputFunctions.multi_gpu_throughput(
                gpu_types, self.config.sequence_length, self.config.batch_size,
                allocation.segment.segment_size, allocation.num_gpus,
                self.config.communication_efficiency
            )

            self.allocation_throughputs[allocation] = throughput

        logger.info(f"Pre-computed throughputs for {len(self.allocation_throughputs)} allocations")

    def _precompute_network_throughputs(self):
        """Pre-compute network throughput for each valid connection"""
        self.network_throughputs = {}

        # Tensor size for layer-to-layer transfers
        tensor_size_gb = (self.config.batch_size * self.config.sequence_length *
                         self.config.d_model * self.config.bytes_per_element) / (1024**3)

        for alloc1, alloc2 in self.valid_connections:
            # Find minimum bandwidth between any GPU in alloc1 and any GPU in alloc2
            min_bandwidth = float('inf')
            for gpu_i in alloc1.gpu_ids:
                for gpu_j in alloc2.gpu_ids:
                    bandwidth = self.network_bandwidth[gpu_i, gpu_j]
                    min_bandwidth = min(min_bandwidth, bandwidth)

            # Network throughput is bandwidth / tensor_size (transfers per second)
            if tensor_size_gb > 0:
                network_throughput = min_bandwidth / tensor_size_gb
            else:
                network_throughput = 1000.0  # Default high value

            self.network_throughputs[(alloc1, alloc2)] = network_throughput

        logger.info(f"Pre-computed network throughputs for {len(self.network_throughputs)} connections")

    def _validate_problem_size(self):
        """Validate problem size and warn about complexity"""
        num_allocations = len(self.valid_allocations)
        num_connections = len(self.valid_connections)

        # Estimate variables: allocation selection + connections + throughput variables
        binary_vars = num_allocations + num_connections
        continuous_vars = num_allocations + num_connections + 1  # throughputs + objective
        total_vars = binary_vars + continuous_vars

        logger.info(f"Correct joint ILP problem size:")
        logger.info(f"  - Segments: {len(self.valid_segments)}")
        logger.info(f"  - Allocations: {num_allocations}")
        logger.info(f"  - Connections: {num_connections}")
        logger.info(f"  - Total variables: ~{total_vars} ({binary_vars} binary + {continuous_vars} continuous)")

        if total_vars > 100000:
            logger.warning(f"Large problem ({total_vars} variables). Consider reducing max_gpus_per_segment.")
        else:
            logger.info(f"Manageable problem size.")

    def build_model(self):
        """Build the correct joint ILP model"""
        logger.info("Building correct joint ILP model...")

        # Create model
        self.model = gp.Model("correct_joint_llm_placement", env=self.env)

        # Solver parameters
        self.model.setParam('Presolve', 2)
        self.model.setParam('Cuts', 1)
        self.model.setParam('Heuristics', 0.1)
        self.model.setParam('MIPFocus', 1)
        self.model.setParam('NodefileStart', 0.5)
        self.model.setParam('TimeLimit', self.config.time_limit_seconds)
        self.model.setParam('MIPGap', self.config.optimality_gap)
        self.model.setParam('LogToConsole', 1)

        # Create decision variables
        self._create_variables()
        self._create_constraints()
        self._set_objective()

        logger.info("Correct joint ILP model built successfully")

    def _create_variables(self):
        """Create ILP decision variables"""
        logger.info("Creating decision variables...")

        # Allocation selection: z[allocation] ∈ {0,1}
        self.z = self.model.addVars(
            self.valid_allocations,
            vtype=GRB.BINARY,
            name="allocation_selection"
        )

        # Network connections: e[alloc1, alloc2] ∈ {0,1}
        self.e = self.model.addVars(
            self.valid_connections,
            vtype=GRB.BINARY,
            name="network_connection"
        )

        # Allocation throughputs: τ_alloc[allocation] ∈ R+
        self.tau_alloc = self.model.addVars(
            self.valid_allocations,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="allocation_throughput"
        )

        # Network throughputs: ρ[alloc1, alloc2] ∈ R+
        self.rho = self.model.addVars(
            self.valid_connections,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="network_throughput"
        )

        # End-to-end throughput: t ∈ R+
        self.t = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="end_to_end_throughput")

        logger.info(f"Created {len(self.z) + len(self.e)} binary variables")
        logger.info(f"Created {len(self.tau_alloc) + len(self.rho) + 1} continuous variables")

    def _create_constraints(self):
        """Create ILP constraints"""
        logger.info("Creating constraints...")

        # 1. Layer coverage: each layer assigned exactly once
        for layer in range(1, self.config.num_decoder_layers + 1):
            covering_allocations = [
                alloc for alloc in self.valid_allocations
                if alloc.segment.start_layer <= layer <= alloc.segment.end_layer
            ]
            if covering_allocations:
                self.model.addConstr(
                    gp.quicksum(self.z[alloc] for alloc in covering_allocations) == 1,
                    name=f"layer_coverage_{layer}"
                )

        # 2. GPU exclusivity: each GPU used in at most one allocation
        for gpu_id in range(self.total_gpus):
            using_allocations = [
                alloc for alloc in self.valid_allocations
                if gpu_id in alloc.gpu_ids
            ]
            if using_allocations:
                self.model.addConstr(
                    gp.quicksum(self.z[alloc] for alloc in using_allocations) <= 1,
                    name=f"gpu_exclusivity_{gpu_id}"
                )

        # 3. Network connection logic
        for alloc1, alloc2 in self.valid_connections:
            # Connection requires both allocations to be selected
            self.model.addConstr(
                self.e[alloc1, alloc2] <= self.z[alloc1],
                name=f"connection_req_alloc1"
            )
            self.model.addConstr(
                self.e[alloc1, alloc2] <= self.z[alloc2],
                name=f"connection_req_alloc2"
            )

        # 4. Pipeline connectivity: ensure complete path from layer 1 to final layer
        self._add_pipeline_connectivity_constraints()

        # 5. Allocation throughput definition (LINEAR!)
        for allocation in self.valid_allocations:
            throughput_value = self.allocation_throughputs[allocation]
            self.model.addConstr(
                self.tau_alloc[allocation] == throughput_value * self.z[allocation],
                name=f"allocation_throughput_def"
            )

        # 6. Network throughput definition
        for alloc1, alloc2 in self.valid_connections:
            throughput_value = self.network_throughputs[(alloc1, alloc2)]
            self.model.addConstr(
                self.rho[alloc1, alloc2] == throughput_value * self.e[alloc1, alloc2],
                name=f"network_throughput_def"
            )

        # 7. End-to-end throughput constraints (Big-M formulation)
        self._add_end_to_end_throughput_constraints()

        logger.info("Constraints created successfully")

    def _add_pipeline_connectivity_constraints(self):
        """Ensure pipeline connectivity from layer 1 to final layer"""
        # Pipeline must start at layer 1
        first_layer_allocations = [
            alloc for alloc in self.valid_allocations
            if alloc.segment.start_layer == 1
        ]
        if first_layer_allocations:
            self.model.addConstr(
                gp.quicksum(self.z[alloc] for alloc in first_layer_allocations) >= 1,
                name="pipeline_starts_at_layer_1"
            )

        # Pipeline must end at final layer
        final_layer = self.config.num_decoder_layers
        final_layer_allocations = [
            alloc for alloc in self.valid_allocations
            if alloc.segment.end_layer == final_layer
        ]
        if final_layer_allocations:
            self.model.addConstr(
                gp.quicksum(self.z[alloc] for alloc in final_layer_allocations) >= 1,
                name="pipeline_ends_at_final_layer"
            )

        # Sequential connectivity: if allocation ends at layer L, there must be connection to L+1
        for layer in range(1, self.config.num_decoder_layers):
            ending_allocations = [
                alloc for alloc in self.valid_allocations
                if alloc.segment.end_layer == layer
            ]
            starting_allocations = [
                alloc for alloc in self.valid_allocations
                if alloc.segment.start_layer == layer + 1
            ]

            if ending_allocations and starting_allocations:
                # For each allocation ending at this layer
                for alloc1 in ending_allocations:
                    valid_next_connections = [
                        (a1, a2) for a1, a2 in self.valid_connections
                        if a1 == alloc1 and a2 in starting_allocations
                    ]
                    if valid_next_connections:
                        self.model.addConstr(
                            gp.quicksum(self.e[a1, a2] for a1, a2 in valid_next_connections) >= self.z[alloc1],
                            name=f"connectivity_out_{layer}"
                        )

                # For each allocation starting at next layer
                for alloc2 in starting_allocations:
                    valid_prev_connections = [
                        (a1, a2) for a1, a2 in self.valid_connections
                        if a2 == alloc2 and a1 in ending_allocations
                    ]
                    if valid_prev_connections:
                        self.model.addConstr(
                            gp.quicksum(self.e[a1, a2] for a1, a2 in valid_prev_connections) >= self.z[alloc2],
                            name=f"connectivity_in_{layer}"
                        )

    def _add_end_to_end_throughput_constraints(self):
        """Add end-to-end throughput constraints using Big-M"""
        # Compute Big-M values
        max_allocation_throughput = max(self.allocation_throughputs.values()) if self.allocation_throughputs else 1000.0
        max_network_throughput = max(self.network_throughputs.values()) if self.network_throughputs else 1000.0

        M_alloc = max_allocation_throughput * 1.1
        M_network = max_network_throughput * 1.1

        # Allocation throughput constraints
        for allocation in self.valid_allocations:
            self.model.addConstr(
                self.t <= self.tau_alloc[allocation] + M_alloc * (1 - self.z[allocation]),
                name=f"throughput_allocation"
            )

        # Network throughput constraints
        for alloc1, alloc2 in self.valid_connections:
            self.model.addConstr(
                self.t <= self.rho[alloc1, alloc2] + M_network * (1 - self.e[alloc1, alloc2]),
                name=f"throughput_network"
            )

        logger.info(f"Added throughput constraints with M_alloc={M_alloc:.2f}, M_network={M_network:.2f}")

    def _set_objective(self):
        """Set optimization objective"""
        self.model.setObjective(self.t, GRB.MAXIMIZE)

    def solve(self) -> bool:
        """Solve the correct joint ILP problem"""
        # Set thread count
        total_vars = len(self.z) + len(self.e) + len(self.tau_alloc) + len(self.rho) + 1
        available_threads = min(self.max_threads, os.cpu_count())

        if self.threads is not None:
            threads = min(self.threads, available_threads)
        else:
            if total_vars > 50000:
                threads = min(available_threads, 16)
            elif total_vars > 10000:
                threads = min(available_threads, 8)
            else:
                threads = min(available_threads, 4)

        self.model.setParam('Threads', threads)
        logger.info(f"Using {threads} threads for optimization (problem size: {total_vars})")

        logger.info("Starting correct joint ILP optimization...")
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
            'allocations': [],
            'connections': [],
            'solve_status': self.model.status
        }

        # Extract selected allocations
        for allocation in self.valid_allocations:
            if self.z[allocation].x > 0.5:
                # Get GPU types for this allocation
                gpu_types = [self.global_id_to_type[gpu_id] for gpu_id in allocation.gpu_ids]
                gpu_type_counts = {}
                for gt in gpu_types:
                    gpu_type_counts[gt] = gpu_type_counts.get(gt, 0) + 1

                self.solution['allocations'].append({
                    'segment': {
                        'start_layer': allocation.segment.start_layer,
                        'end_layer': allocation.segment.end_layer,
                        'segment_size': allocation.segment.segment_size
                    },
                    'gpu_ids': sorted(list(allocation.gpu_ids)),
                    'gpu_types': gpu_type_counts,
                    'num_gpus': allocation.num_gpus,
                    'throughput': self.tau_alloc[allocation].x,
                    'workload_per_gpu': self.config.batch_size / allocation.num_gpus
                })

        # Extract active connections
        for alloc1, alloc2 in self.valid_connections:
            if self.e[alloc1, alloc2].x > 0.5:
                self.solution['connections'].append({
                    'from_allocation': {
                        'segment': {'start_layer': alloc1.segment.start_layer, 'end_layer': alloc1.segment.end_layer},
                        'gpu_ids': sorted(list(alloc1.gpu_ids))
                    },
                    'to_allocation': {
                        'segment': {'start_layer': alloc2.segment.start_layer, 'end_layer': alloc2.segment.end_layer},
                        'gpu_ids': sorted(list(alloc2.gpu_ids))
                    },
                    'throughput': self.rho[alloc1, alloc2].x
                })

        # Sort by start layer
        self.solution['allocations'].sort(key=lambda x: x['segment']['start_layer'])

    def print_solution(self):
        """Print the correct joint solution"""
        if not self.solution:
            logger.error("No solution available")
            return

        print("\n" + "="*100)
        print(f"CORRECT JOINT ILP PLACEMENT OPTIMIZATION RESULTS")
        print("="*100)
        print(f"Model: {self.config.model_name} ({self.config.num_decoder_layers} layers)")
        print(f"Batch Size: {self.config.batch_size}, Sequence Length: {self.config.sequence_length}")
        print(f"Optimal End-to-End Throughput: {self.solution['objective_value']:.2f} tokens/sec")
        print()

        print("SPECIFIC GPU ALLOCATIONS:")
        print("-" * 100)
        print(f"{'Layers':<15} {'GPU IDs':<25} {'GPU Types':<20} {'Workload/GPU':<15} {'Allocation Throughput':<20}")
        print("-" * 100)

        for allocation in self.solution['allocations']:
            layers_str = f"{allocation['segment']['start_layer']}-{allocation['segment']['end_layer']}"
            gpu_ids_str = f"{allocation['gpu_ids']}"
            gpu_types_str = f"{allocation['gpu_types']}"

            print(f"{layers_str:<15} {gpu_ids_str:<25} {gpu_types_str:<20} "
                  f"{allocation['workload_per_gpu']:<15.1f} {allocation['throughput']:<20.2f}")

        if self.solution['connections']:
            print("\nNETWORK CONNECTIONS:")
            print("-" * 80)
            for i, conn in enumerate(self.solution['connections']):
                from_seg = conn['from_allocation']['segment']
                to_seg = conn['to_allocation']['segment']
                from_gpus = conn['from_allocation']['gpu_ids']
                to_gpus = conn['to_allocation']['gpu_ids']

                print(f"Connection {i+1}: Layers {from_seg['start_layer']}-{from_seg['end_layer']} (GPUs {from_gpus}) -> "
                      f"Layers {to_seg['start_layer']}-{to_seg['end_layer']} (GPUs {to_gpus}) "
                      f"[Throughput: {conn['throughput']:.2f}]")

        print("\n" + "="*100)

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
                'max_gpus_per_segment': self.config.max_gpus_per_segment
            },
            'solution': self.solution
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Correct joint solution saved to {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Correct Joint LLM Placement Optimizer')
    parser.add_argument('--config-dir', required=True, help='Configuration directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--max-gpus-per-segment', type=int, default=3,
                       help='Maximum GPUs per segment (default: 3)')
    parser.add_argument('--threads', type=int, help='Number of threads')
    parser.add_argument('--max-threads', type=int, default=32, help='Maximum threads')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Correct joint ILP optimization settings:")
    logger.info(f"  - Max GPUs per segment: {args.max_gpus_per_segment}")

    start_time = time.time()

    try:
        # Initialize correct joint solver
        solver = CorrectJointLLMSolver(
            args.config_dir,
            max_gpus_per_segment=args.max_gpus_per_segment,
            threads=args.threads,
            max_threads=args.max_threads
        )

        # Build and solve model
        solver.build_model()

        if solver.solve():
            solver.print_solution()
            output_file = os.path.join(args.config_dir, 'correct_joint_solution.json')
            solver.save_solution(output_file)
        else:
            logger.error("Failed to find optimal solution")
            return 1

    except Exception as e:
        logger.error(f"Correct joint solver failed: {e}")
        return 1

    end_time = time.time()
    logger.info(f"Correct joint solver finished in {end_time - start_time:.0f} seconds")
    return 0


if __name__ == "__main__":
    exit(main())