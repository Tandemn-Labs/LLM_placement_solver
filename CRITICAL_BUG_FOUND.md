# CRITICAL BUG IDENTIFIED IN TIGHT BIG-M COMPUTATION

## Root Cause
The `_compute_tight_bigM()` function (lines 1403-1426) computes Big-M values **per GPU type**, but the global throughput variable `self.t` represents the **system-wide** throughput.

## The Problem
1. For A10 partitions, `max_throughput` is computed based on A10's capabilities (~7.2M tokens/sec)
2. This gives M = max_throughput * 3 ≈ 21.5M for A10 partitions
3. The constraint `self.t <= tau[A10,partition_id] + M*(1-z[A10,partition_id])` becomes:
   - `self.t <= 0 + 21.5M*1 = 21.5M` (for inactive A10 partitions with tau=0, z=0)
4. This **BINDS** `self.t` to 21.5M, preventing it from reaching 23.2M (the actual A100 TP=8 throughput)!

## Evidence
From diagnostic output:
```
A10 partition 12: z=0.000000, tau=0.00, M=21,487,040
Expected constraint RHS: 0.00 + 21,487,040*(1-0) = 21,487,040
Actual self.t: 21,487,039.56 ← BINDING!
```

All 10 A10 partitions (12-21) have M ≈ 21.5M, creating 10 binding constraints that incorrectly limit `self.t`.

## Solution Options
### Option 1: Global Big-M (Simplest)
Compute M as the maximum throughput across **all GPU types**, not per-type:
```python
def _compute_tight_bigM(self) -> Tuple[Dict, float]:
    # Find the maximum possible throughput across ALL GPU types
    max_throughput_global = 0
    for gpu_type, allocations in self.tp_allocations.items():
        tp_degree = self.tp_max_configuration[gpu_type]
        max_size = self.max_segment_size[(gpu_type, tp_degree)]
        if max_size > 0:
            throughput = ThroughputFunctions.gpu_throughput_with_tp(
                gpu_type, self.config.sequence_length,
                self.config.max_batch_size, max_size, 
                self.config.d_model, self.config.bytes_per_element, 
                tp_degree, self.config.d_hidden
            )
            max_throughput_global = max(max_throughput_global, throughput)
    
    # Use the SAME M for ALL partitions
    M_global = max_throughput_global * 3
    M_partition = {key: M_global for key in ...}
    return M_partition, M_network
```

### Option 2: Use separate variables for each partition's throughput (Complex)
Create separate throughput variables per partition instead of a global `self.t`.

### Recommendation
**Use Option 1** - it's simpler, maintains tight Big-M benefits, and fixes the bug.

## Impact
- **Current behavior**: Solver reports suboptimal solution with 21.5M tokens/sec and $0.000424/M tokens
- **Expected behavior**: Solver finds optimal solution with 23.2M tokens/sec and better $/M tokens ratio
- **Cost-per-token error**: ~7.4% too high due to artificially constrained throughput


