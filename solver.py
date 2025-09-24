#!/usr/bin/env python3
"""
LLM Model Parallelism Placement Solver using Gurobi (FIXED VERSION)
Optimizes layer placement across heterogeneous GPU clusters with network communication.
"""

import os
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import argparse
import json
import logging
from typing import Dict, List, Tuple, Optional
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

class ThroughputFunctions:
    """Throughput functions with configurable coefficients"""

    # FIXED: Positive coefficients to ensure feasibility
    GPU_THROUGHPUT_COEFFS = {
        'A100': {'seq_len': -0.01, 'batch_size': 5.0, 'num_layers': 2.0, 'constant': 150.0},
        'V100': {'seq_len': -0.008, 'batch_size': 3.0, 'num_layers': 1.5, 'constant': 100.0},
        'H100': {'seq_len': -0.012, 'batch_size': 6.0, 'num_layers': 2.5, 'constant': 200.0},
        'RTX4090': {'seq_len': -0.006, 'batch_size': 2.5, 'num_layers': 1.0, 'constant': 80.0}
    }

    NETWORK_COEFFS = {
        'bandwidth': 1.2, 'seq_len': -0.001, 'batch_size': -0.1, 'hidden_dim': -0.00001, 'constant': 50.0
    }
    
    @staticmethod
    def gpu_throughput(gpu_type: str, seq_len: int, batch_size: int, num_layers: int) -> float:
        """Linear GPU throughput function (tokens/sec) - FIXED"""
        coeffs = ThroughputFunctions.GPU_THROUGHPUT_COEFFS[gpu_type]
        throughput = (coeffs['seq_len'] * seq_len + 
                     coeffs['batch_size'] * batch_size + 
                     coeffs['num_layers'] * num_layers + 
                     coeffs['constant'])
        return max(1.0, throughput)  # Ensure positive throughput
    
    @staticmethod
    def network_throughput(bandwidth_gbps: float, seq_len: int, batch_size: int, hidden_dim: int) -> float:
        """Linear network throughput function (transfers/sec) - FIXED"""
        coeffs = ThroughputFunctions.NETWORK_COEFFS
        throughput = (coeffs['bandwidth'] * bandwidth_gbps + 
                     coeffs['seq_len'] * seq_len + 
                     coeffs['batch_size'] * batch_size + 
                     coeffs['hidden_dim'] * hidden_dim + 
                     coeffs['constant'])
        return max(1.0, throughput)  # Ensure positive throughput
    
    @staticmethod
    def memory_usage(seq_len: int, batch_size: int, num_layers: int, layer_weight_gb: float,
                    d_model: int, d_hidden: int) -> float:
        """Memory usage in GB - FIXED"""
        # Model weights
        weight_memory = num_layers * layer_weight_gb

        # FIXED: Correct intermediate tensor memory calculation
        # Attention matrix: batch_size × seq_len × seq_len × d_model (for QK^T)
        attention_memory = batch_size * seq_len * seq_len * d_model * 4 / (1024**3)
        
        # K,V cache: 2 × batch_size × seq_len × d_model
        kv_cache_memory = 2 * batch_size * seq_len * d_model * 4 / (1024**3)
        
        # Hidden states: batch_size × seq_len × d_hidden
        hidden_memory = batch_size * seq_len * d_hidden * 4 / (1024**3)
        
        # Intermediate memory per layer (not total - pipeline processing)
        intermediate_memory_per_layer = (attention_memory + kv_cache_memory + hidden_memory) / 1024  # More reasonable
        
        total_intermediate = intermediate_memory_per_layer * min(num_layers, 2)  # At most 2 layers worth of intermediates
        
        return weight_memory + total_intermediate

class LLMPlacementSolver:
    """Main solver class for LLM placement optimization"""

    def __init__(self, config_dir: str):
        self.options = {
            "WLSACCESSID": "790b9c11-45d0-4785-8d99-a5e6414f9321",
            "WLSSECRET": "adef4738-7bf6-41b8-8dfd-d04e23d53e51",
            "LICENSEID": 2415150,
        }
        self.env = gp.Env(params=self.options)
        self.config_dir = config_dir

        # FIXED: Correct file names
        gpu_pool_file = os.path.join(config_dir, 'gpu_pool.csv')
        network_file = os.path.join(config_dir, 'network_bandwidth.csv')
        config_file = os.path.join(config_dir, 'config.csv')

        self.gpu_types = self._load_gpu_pool(gpu_pool_file)
        self.network_bandwidth = self._load_network_bandwidth(network_file)
        self.config = self._load_config(config_file)
        self.model = None
        self.solution = None
        
        # Derived data
        self.total_gpus = sum(gpu_type.count for gpu_type in self.gpu_types.values())
        self.max_segment_size = self._compute_max_segment_sizes()
        self.valid_segments = self._generate_valid_segments()
        self.valid_connections = self._generate_valid_connections()
        
        logger.info(f"Initialized solver: {len(self.gpu_types)} GPU types, {self.total_gpus} total GPUs")
        logger.info(f"Model: {self.config.num_decoder_layers} layers, batch_size={self.config.batch_size}")
    
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
        return df.values
    
    def _load_config(self, filename: str) -> Config:
        """Load runtime configuration"""
        df = pd.read_csv(filename)
        config_dict = dict(zip(df['parameter'], df['value']))
        
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
            optimality_gap=float(config_dict['optimality_gap'])
        )
    
    def _compute_max_segment_sizes(self) -> Dict[str, int]:
        """Compute maximum segment size for each GPU type based on memory constraints"""
        max_sizes = {}
        
        for gpu_type_name, gpu_type in self.gpu_types.items():
            max_layers = 1
            while max_layers <= self.config.num_decoder_layers:
                memory_needed = ThroughputFunctions.memory_usage(
                    self.config.sequence_length,
                    self.config.batch_size,
                    max_layers,
                    self.config.layer_weight_memory_gb,
                    self.config.d_model,
                    self.config.d_hidden
                )
                
                if memory_needed > gpu_type.memory_gb:
                    break
                max_layers += 1
            
            max_sizes[gpu_type_name] = max(1, max_layers - 1)  # FIXED: Ensure at least 1 layer
            logger.info(f"GPU {gpu_type_name}: max {max_sizes[gpu_type_name]} layers "
                       f"(memory: {gpu_type.memory_gb}GB, estimated usage: {memory_needed:.2f}GB)")
        
        return max_sizes
    
    def _generate_valid_segments(self) -> List[Tuple[str, int, int, int]]:
        """Generate all valid (gpu_type, gpu_id, start_layer, segment_size) combinations"""
        valid_segments = []
        
        for gpu_type_name, gpu_type in self.gpu_types.items():
            max_seg_size = self.max_segment_size[gpu_type_name]
            if max_seg_size == 0:
                logger.warning(f"GPU type {gpu_type_name} cannot hold any layers!")
                continue
                
            for gpu_id in range(gpu_type.count):
                for start_layer in range(1, self.config.num_decoder_layers + 1):
                    for segment_size in range(1, min(max_seg_size + 1,
                                                   self.config.num_decoder_layers - start_layer + 2)):
                        if start_layer + segment_size - 1 <= self.config.num_decoder_layers:
                            valid_segments.append((gpu_type_name, gpu_id, start_layer, segment_size))
        
        logger.info(f"Generated {len(valid_segments)} valid segments")
        return valid_segments
    
    def _generate_valid_connections(self) -> List[Tuple[Tuple[str, int, int, int], Tuple[str, int, int, int]]]:
        """Generate valid network connections between consecutive segments"""
        valid_connections = []
        
        for seg1 in self.valid_segments:
            gpu_type1, gpu_id1, start1, size1 = seg1
            end1 = start1 + size1 - 1
            
            for seg2 in self.valid_segments:
                gpu_type2, gpu_id2, start2, size2 = seg2
                
                # Connection valid if seg2 starts right after seg1 ends and different GPUs
                if start2 == end1 + 1 and (gpu_type1 != gpu_type2 or gpu_id1 != gpu_id2):
                    valid_connections.append((seg1, seg2))
        
        logger.info(f"Generated {len(valid_connections)} valid connections")
        return valid_connections
    
    def _get_global_gpu_id(self, gpu_type: str, gpu_id: int) -> int:
        """Convert (gpu_type, gpu_id) to global GPU ID"""
        return self.gpu_types[gpu_type].global_ids[gpu_id]
    
    def build_model(self):
        """Build the Gurobi optimization model"""
        logger.info("Building optimization model...")
        
        # Create model
        self.model = gp.Model("llm_placement", env=self.env)
        self.model.setParam('Presolve', 2)  # Aggressive presolving
        self.model.setParam('Heuristics', 0.2)  # Spend 20% time on heuristics
        self.model.setParam('Cuts', 2)  # Aggressive cut generation
        self.model.setParam('TimeLimit', self.config.time_limit_seconds)
        self.model.setParam('MIPGap', self.config.optimality_gap)
        self.model.setParam('LogToConsole', 1)
        
        # Decision variables
        self._create_variables()
        self._create_constraints()
        self._set_objective()
        
        logger.info("Model built successfully")
    
    def _create_variables(self):
        """Create decision variables"""
        # Segment assignment variables: x[gpu_type, gpu_id, start_layer, segment_size]
        self.x = self.model.addVars(
            self.valid_segments,
            vtype=GRB.BINARY,
            name="segment_assignment"
        )
        
        # GPU usage indicators: z[gpu_type, gpu_id]
        self.z = self.model.addVars(
            [(gpu_type, gpu_id) for gpu_type, gpu_type_obj in self.gpu_types.items() 
             for gpu_id in range(gpu_type_obj.count)],
            vtype=GRB.BINARY,
            name="gpu_usage"
        )
        
        # Network connection variables: e[seg1, seg2]
        self.e = self.model.addVars(
            self.valid_connections,
            vtype=GRB.BINARY,
            name="network_connection"
        )
        
        # Throughput variables
        self.tau = self.model.addVars(
            [(gpu_type, gpu_id) for gpu_type, gpu_type_obj in self.gpu_types.items() 
             for gpu_id in range(gpu_type_obj.count)],
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="gpu_throughput"
        )
        
        self.rho = self.model.addVars(
            self.valid_connections,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="network_throughput"
        )
        
        # End-to-end throughput
        self.t = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="end_to_end_throughput")
    
    def _create_constraints(self):
        """Create optimization constraints - FIXED"""
        
        # 1. Layer coverage: each layer assigned exactly once
        for layer in range(1, self.config.num_decoder_layers + 1):
            self.model.addConstr(
                gp.quicksum(self.x[seg] for seg in self.valid_segments 
                           if seg[2] <= layer <= seg[2] + seg[3] - 1) == 1,
                name=f"layer_coverage_{layer}"
            )
        
        # 2. GPU capacity: each GPU processes at most one segment
        for gpu_type, gpu_type_obj in self.gpu_types.items():
            for gpu_id in range(gpu_type_obj.count):
                self.model.addConstr(
                    gp.quicksum(self.x[seg] for seg in self.valid_segments 
                               if seg[0] == gpu_type and seg[1] == gpu_id) <= 1,
                    name=f"gpu_capacity_{gpu_type}_{gpu_id}"
                )
        
        # 3. GPU usage indicators
        for gpu_type, gpu_type_obj in self.gpu_types.items():
            for gpu_id in range(gpu_type_obj.count):
                self.model.addConstr(
                    self.z[gpu_type, gpu_id] == 
                    gp.quicksum(self.x[seg] for seg in self.valid_segments 
                               if seg[0] == gpu_type and seg[1] == gpu_id),
                    name=f"gpu_usage_{gpu_type}_{gpu_id}"
                )
        
        # 4. Network connection constraints
        for (seg1, seg2) in self.valid_connections:
            # Connection exists if both segments are selected
            self.model.addConstr(
                self.e[seg1, seg2] <= self.x[seg1],
                name=f"connection_seg1"
            )
            self.model.addConstr(
                self.e[seg1, seg2] <= self.x[seg2],
                name=f"connection_seg2"
            )
            self.model.addConstr(
                self.e[seg1, seg2] >= self.x[seg1] + self.x[seg2] - 1,
                name=f"connection_both"
            )
        
        # 5. GPU throughput definition
        for gpu_type, gpu_type_obj in self.gpu_types.items():
            for gpu_id in range(gpu_type_obj.count):
                gpu_throughput_expr = gp.quicksum(
                    ThroughputFunctions.gpu_throughput(
                        gpu_type, self.config.sequence_length, 
                        self.config.batch_size, seg[3]
                    ) * self.x[seg]
                    for seg in self.valid_segments 
                    if seg[0] == gpu_type and seg[1] == gpu_id
                )
                self.model.addConstr(
                    self.tau[gpu_type, gpu_id] == gpu_throughput_expr,
                    name=f"gpu_throughput_def_{gpu_type}_{gpu_id}"
                )
        
        # 6. Network throughput definition
        for (seg1, seg2) in self.valid_connections:
            gpu_type1, gpu_id1 = seg1[0], seg1[1]
            gpu_type2, gpu_id2 = seg2[0], seg2[1]
            global_id1 = self._get_global_gpu_id(gpu_type1, gpu_id1)
            global_id2 = self._get_global_gpu_id(gpu_type2, gpu_id2)
            
            bandwidth = self.network_bandwidth[global_id1, global_id2]
            net_throughput = ThroughputFunctions.network_throughput(
                bandwidth, self.config.sequence_length,
                self.config.batch_size, self.config.d_model
            )
            
            self.model.addConstr(
                self.rho[seg1, seg2] == net_throughput * self.e[seg1, seg2],
                name=f"network_throughput_def"
            )
        
        # 7. End-to-end throughput constraints
        M = 1e6  # Big M constant

        # GPU throughput constraints - only for used GPUs
        for gpu_type, gpu_type_obj in self.gpu_types.items():
            for gpu_id in range(gpu_type_obj.count):
                self.model.addConstr(
                    self.t <= self.tau[gpu_type, gpu_id] + M * (1 - self.z[gpu_type, gpu_id]),
                    name=f"throughput_gpu_{gpu_type}_{gpu_id}"
                )

        # Network throughput constraints - only for active connections
        for (seg1, seg2) in self.valid_connections:
            self.model.addConstr(
                self.t <= self.rho[seg1, seg2] + M * (1 - self.e[seg1, seg2]),
                name=f"throughput_network"
            )

        # 8. FIXED: Pipeline connectivity constraints
        # Ensure pipeline starts at layer 1
        first_layer_segments = [seg for seg in self.valid_segments if seg[2] == 1]
        if first_layer_segments:
            self.model.addConstr(
                gp.quicksum(self.x[seg] for seg in first_layer_segments) >= 1,
                name="pipeline_starts_at_layer_1"
            )

        # FIXED: Only enforce connectivity for non-terminal layers
        for layer in range(1, self.config.num_decoder_layers):
            # Find segments ending at this layer
            segments_ending_here = [seg for seg in self.valid_segments
                                  if seg[2] + seg[3] - 1 == layer]
            # Find segments starting at next layer
            segments_starting_next = [seg for seg in self.valid_segments
                                    if seg[2] == layer + 1]

            if segments_ending_here and segments_starting_next:
                # If a segment ends at layer i, there must be a connection to layer i+1
                for seg1 in segments_ending_here:
                    valid_next_connections = [(s1, s2) for (s1, s2) in self.valid_connections 
                                            if s1 == seg1 and s2 in segments_starting_next]
                    if valid_next_connections:
                        self.model.addConstr(
                            gp.quicksum(self.e[s1, s2] for (s1, s2) in valid_next_connections) >= self.x[seg1],
                            name=f"connectivity_out_{layer}"
                        )

                # If a segment starts at layer i+1, there must be a connection from layer i
                for seg2 in segments_starting_next:
                    valid_prev_connections = [(s1, s2) for (s1, s2) in self.valid_connections 
                                            if s2 == seg2 and s1 in segments_ending_here]
                    if valid_prev_connections:
                        self.model.addConstr(
                            gp.quicksum(self.e[s1, s2] for (s1, s2) in valid_prev_connections) >= self.x[seg2],
                            name=f"connectivity_in_{layer}"
                        )
    
    def _set_objective(self):
        """Set optimization objective"""
        self.model.setObjective(self.t, GRB.MAXIMIZE)
    
    def solve(self) -> bool:
        """Solve the optimization problem"""
        logger.info("Starting optimization...")
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
                    logger.error("Time limit reached with no feasible solution found")
                    return False
            else:
                logger.error(f"No solution found. Status: {self.model.status}")
                if self.model.status == GRB.INFEASIBLE:
                    logger.error("Model is infeasible - check memory constraints and segment generation")
                return False
                
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return False
    
    def _extract_solution(self):
        """Extract solution from solved model"""
        self.solution = {
            'objective_value': self.t.x,
            'gpu_assignments': [],
            'network_connections': [],
            'solve_status': self.model.status
        }
        
        # Extract GPU assignments
        for seg in self.valid_segments:
            if self.x[seg].x > 0.5:  # Binary variable is 1
                gpu_type, gpu_id, start_layer, segment_size = seg
                global_gpu_id = self._get_global_gpu_id(gpu_type, gpu_id)
                
                assignment = {
                    'gpu_type': gpu_type,
                    'gpu_id': gpu_id,
                    'global_gpu_id': global_gpu_id,
                    'start_layer': start_layer,
                    'end_layer': start_layer + segment_size - 1,
                    'segment_size': segment_size,
                    'throughput': self.tau[gpu_type, gpu_id].x
                }
                self.solution['gpu_assignments'].append(assignment)
        
        # Extract network connections
        for (seg1, seg2) in self.valid_connections:
            if self.e[seg1, seg2].x > 0.5:  # Binary variable is 1
                connection = {
                    'from_segment': seg1,
                    'to_segment': seg2,
                    'throughput': self.rho[seg1, seg2].x
                }
                self.solution['network_connections'].append(connection)
        
        # Sort assignments by start layer
        self.solution['gpu_assignments'].sort(key=lambda x: x['start_layer'])
    
    def print_solution(self):
        """Print the solution in a readable format"""
        if not self.solution:
            logger.error("No solution available")
            return
        
        print("\n" + "="*80)
        print(f"LLM PLACEMENT OPTIMIZATION RESULTS")
        print("="*80)
        print(f"Model: {self.config.model_name} ({self.config.num_decoder_layers} layers)")
        print(f"Batch Size: {self.config.batch_size}, Sequence Length: {self.config.sequence_length}")
        print(f"Optimal End-to-End Throughput: {self.solution['objective_value']:.2f} tokens/sec")
        print()
        
        print("GPU ASSIGNMENTS:")
        print("-" * 80)
        print(f"{'GPU Type':<10} {'GPU ID':<8} {'Global ID':<10} {'Layers':<15} {'Size':<6} {'Throughput':<12}")
        print("-" * 80)
        
        for assignment in self.solution['gpu_assignments']:
            layers_str = f"{assignment['start_layer']}-{assignment['end_layer']}"
            print(f"{assignment['gpu_type']:<10} {assignment['gpu_id']:<8} "
                  f"{assignment['global_gpu_id']:<10} {layers_str:<15} "
                  f"{assignment['segment_size']:<6} {assignment['throughput']:<12.2f}")
        
        if self.solution['network_connections']:
            print("\nNETWORK CONNECTIONS:")
            print("-" * 60)
            for i, conn in enumerate(self.solution['network_connections']):
                seg1, seg2 = conn['from_segment'], conn['to_segment']
                print(f"Connection {i+1}: GPU({seg1[0]},{seg1[1]}) -> GPU({seg2[0]},{seg2[1]}) "
                      f"[Throughput: {conn['throughput']:.2f}]")
        
        print("\n" + "="*80)
    
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
                'd_model': self.config.d_model,
                'd_hidden': self.config.d_hidden
            },
            'solution': self.solution
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Solution saved to {output_file}")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='LLM Model Parallelism Placement Optimizer')
    parser.add_argument('--config-dir', required=True, help='Configuration directory containing all CSV files')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize solver
        solver = LLMPlacementSolver(args.config_dir)
        
        # Build and solve model
        solver.build_model()
        
        if solver.solve():
            solver.print_solution()

            # Save solution to config directory
            output_file = os.path.join(args.config_dir, 'solution.json')
            solver.save_solution(output_file)
        else:
            logger.error("Failed to find optimal solution")
            return 1
            
    except Exception as e:
        logger.error(f"Solver failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())