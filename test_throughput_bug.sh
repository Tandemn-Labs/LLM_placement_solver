#!/bin/bash
# Quick test to diagnose throughput discrepancy

cd /users/gangmuk/projects/LLM_placement_solver

# Run solver with just the A100 TP=8 budget
python3 solver_constrained_with_tp-2.py \
    --gpu_pool config/medium/gpu_pool.csv \
    --config config/medium/config.csv \
    --network config/medium/network_bandwidth.csv \
    --output config/medium/output-debug \
    --method enumeration \
    --test_budgets 32.77 \
    --log_level INFO



