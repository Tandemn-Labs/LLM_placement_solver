# Method Selection Guide

## Overview

The solver now supports **two optimization methods** for cost/throughput optimization:

| Method | Speed | Optimality | Use Case |
|--------|-------|-----------|----------|
| **Weighted** | ⚡ Fast (1 solve, 2-4 min) | Approximate | Quick results, parameter tuning |
| **Enumeration** | 🐌 Slow (15+ solves, 40-80 min) | Guaranteed optimal | Final production, best $/token |

---

## Command-Line Usage

### Method 1: Weighted (Default)

```bash
# Use weighted objective (fast)
python solver_constrained_with_tp-2.py --config-dir config/medium --method weighted

# Test different weights
python solver_constrained_with_tp-2.py --config-dir config/medium --cost-weight 0.0   # Pure throughput
python solver_constrained_with_tp-2.py --config-dir config/medium --cost-weight 0.5   # Balanced
python solver_constrained_with_tp-2.py --config-dir config/medium --cost-weight 0.95  # Heavy cost focus
```

### Method 2: Enumeration (Guaranteed Optimal)

```bash
# Use enumeration (slow but optimal)
python solver_constrained_with_tp-2.py --config-dir config/medium --method enumeration
```

---

## How Each Method Works

### Weighted Method

**Objective Function**:
```
maximize: w × (throughput/T_norm) - (1-w) × (cost/C_norm)
```

**Parameters**:
- `cost_throughput_weight` (w): Controls trade-off
  - `w = 0.0`: Pure throughput maximization
  - `w = 0.5`: Balanced
  - `w = 1.0`: Pure cost minimization (not useful for $/token)
  - `w = 0.3-0.7`: Good range for $/token approximation

**Pros**:
- Very fast (single solve)
- Good for exploring trade-offs
- Tunable via weight parameter

**Cons**:
- Does NOT directly minimize $/token
- Result depends on normalization scales
- May miss true optimal solution

---

### Enumeration Method

**Strategy**:
1. Generate cost budgets: `[0.30, 0.50, ..., 50.0]`
2. For each budget:
   - Maximize throughput subject to `cost ≤ budget`
3. Select solution with minimum `cost/throughput`

**Pros**:
- Guaranteed to find optimal $/token
- No weight tuning needed
- Mathematically sound

**Cons**:
- Slow (15-20 ILP solves)
- Fixed approach (less flexibility)

**Smart Hybrid Mode** (default):
- Uses iterative search to find ballpark solution
- Enumerates only around ballpark (faster)
- Typically 12-15 solves instead of 20+

---

## Testing Scripts

### Test Different Weights (Weighted Method)

```bash
./test_weight_sweep.sh
```

This will:
- Test weights: 0.0, 0.3, 0.5, 0.7, 0.9, 0.95
- Generate output files: `output_weight_*.txt`
- Print comparison table

### Test Enumeration Method

```bash
python test_optimal_cost_per_token.py
```

---

## Configuration Files

### `config/medium/config.csv`

```csv
cost_throughput_weight,0.95     # Default weight for weighted method
max_hourly_cost,999.0            # Budget constraint (unused by weighted)
max_cost_per_token,0.002         # Target $/token (for comparison only)
throughput_normalization,10000.0 # Scale for weighted objective
cost_normalization,1.0           # Scale for weighted objective
```

**Notes**:
- `cost_throughput_weight`: Only used by weighted method
- `max_hourly_cost`: Only used by enumeration method as budget constraint
- Command-line `--cost-weight` overrides config value

---

## Recommendations

### For Development & Exploration
```bash
# Use weighted method with different weights
python solver_constrained_with_tp-2.py --config-dir config/medium --cost-weight 0.5
```

### For Production & Best $/Token
```bash
# Use enumeration method
python solver_constrained_with_tp-2.py --config-dir config/medium --method enumeration
```

### For Quick Testing
```bash
# Use weighted method with config.csv default
python solver_constrained_with_tp-2.py --config-dir config/medium
```

---

## Current Issue & Why We're Testing

From recent runs, we observed:

| Method | $/token | Throughput | Cost/h | Stages | Time |
|--------|---------|------------|--------|--------|------|
| Weighted (w=0.95) | $0.000000186 | 4768 tok/s | $3.20 | 4×T4 | 2 min |
| Theoretical V100 | $0.000000121 | 5516 tok/s | $2.40 | 1×V100 | N/A |

**The weighted method missed the better solution by 54%!**

**Goals**:
1. Test if different weights (0.3-0.7) can find better solutions
2. Use enumeration to find guaranteed optimal
3. Compare weighted vs enumeration performance

---

## Quick Reference

```bash
# Weighted (fast, approximate)
python solver_constrained_with_tp-2.py --config-dir config/medium --method weighted --cost-weight 0.5

# Enumeration (slow, optimal)
python solver_constrained_with_tp-2.py --config-dir config/medium --method enumeration

# Test all weights
./test_weight_sweep.sh
```


