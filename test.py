#!/usr/bin/env python3
"""
Fixed Test script for LLM Placement Solver
Demonstrates usage with corrected configuration files
"""

import sys
import os
import tempfile
import logging
from solver import LLMPlacementSolver

def create_test_files():
    """Create test configuration files with CORRECTED names and values"""
    
    # GPU Pool CSV
    gpu_pool_content = """gpu_type,count,memory_gb
A100,2,80
V100,2,32"""
    
    # Network Bandwidth Matrix - FIXED: Correct filename and format
    network_content = """,gpu_0,gpu_1,gpu_2,gpu_3
gpu_0,0,400,200,200
gpu_1,400,0,200,200
gpu_2,200,200,0,300
gpu_3,200,200,300,0"""
    
    # Runtime Configuration - FIXED: Smaller model to ensure feasibility
    config_content = """parameter,value
sequence_length,512
batch_size,4
model_name,llama-7b
num_decoder_layers,4
d_model,4096
d_hidden,11008
vocab_size,32000
num_attention_heads,32
layer_weight_memory_gb,1.0
time_limit_seconds,60
optimality_gap,0.01"""
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # FIXED: Use correct filenames
    gpu_pool_file = os.path.join(temp_dir, 'gpu_pool.csv')
    network_file = os.path.join(temp_dir, 'network_bandwidth.csv')  # FIXED name
    config_file = os.path.join(temp_dir, 'config.csv')
    
    with open(gpu_pool_file, 'w') as f:
        f.write(gpu_pool_content)
    
    with open(network_file, 'w') as f:
        f.write(network_content)
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    return temp_dir

def test_solver():
    """Test the solver with example configuration"""
    print("Creating test configuration files...")
    temp_dir = create_test_files()
    
    try:
        print("Initializing solver...")
        
        
        solver = LLMPlacementSolver(temp_dir)
        
        print("Building optimization model...")
        solver.build_model()
        
        print("Solving optimization problem...")
        if solver.solve():
            print("\nSolution found!")
            solver.print_solution()
            
            # Save solution
            output_file = os.path.join(temp_dir, 'solution.json')
            solver.save_solution(output_file)
            print(f"\nSolution saved to: {output_file}")
            
        else:
            print("No solution found!")
            return False
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    return True

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=" * 60)
    print("FIXED LLM PLACEMENT SOLVER TEST")
    print("=" * 60)
    
    if test_solver():
        print("\n✓ Test completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Test failed!")
        sys.exit(1)