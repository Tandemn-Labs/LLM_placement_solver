# Updates to solver_constrained_with_tp.py

## Summary
Updated the TP-aware solver to correctly model memory and network behavior with Tensor Parallelism, accounting for all-reduce communication patterns.

## Changes Made

### 1. Activation Memory Calculation (Lines 297-347)

**Previous Model:**
- Treated activation memory as constant per GPU
- Did not account for TP sharding or all-reduce operations

**Updated Model:**
```python
def _calculate_activation_memory(self, tp_degree: int = 1) -> float:
    # Sharded intermediate tensors during computation
    qkv_memory = 3 * batch * seq_len * (hidden / tp_degree)
    mlp_intermediate = batch * seq_len * (4 * hidden / tp_degree)
    sharded_computation = qkv_memory + mlp_intermediate

    # Full activation after all-reduce (NOT sharded)
    full_activation = batch * seq_len * hidden

    # KV cache (persistently sharded)
    kv_cache = 2 * batch * seq_len * (hidden / tp_degree)

    # Peak memory
    peak = max(sharded_computation, full_activation) + kv_cache
    return peak * 1.15  # Framework overhead
```

**Key Insights:**
- TP has **two all-reduce operations per layer** (after attention, after MLP)
- After all-reduce, each GPU temporarily holds the **full activation tensor**
- KV cache is **persistently sharded** along hidden dimension
- Peak memory = max(sharded computation, full activation) + KV cache

### 2. Max Segment Size Computation (Line 280)

**Change:**
```python
# Before:
activation_memory = self._calculate_activation_memory()

# After:
activation_memory = self._calculate_activation_memory(tp_degree)
```

Now correctly passes `tp_degree` to account for TP-specific memory patterns.

### 3. Network Throughput Model (Lines 440-506)

**Previous Model:**
- Used minimum bandwidth between any two GPUs in different TP groups
- Simplified point-to-point communication

**Updated Model:**
```python
Communication pattern: All-reduce → Master send → All-scatter

Step 1: All-reduce within source TP group
  - Ring all-reduce efficiency: (tp_degree - 1) / tp_degree
  - Bandwidth: efficiency × nvlink_bandwidth

Step 2: Master-to-master inter-stage transfer
  - Master GPU: lowest ID in each partition
  - Bandwidth: network_bandwidth[master1, master2]
  - Full tensor size (NOT sharded)

Step 3: All-scatter within destination TP group
  - Ring all-scatter efficiency: (tp_degree - 1) / tp_degree
  - Bandwidth: efficiency × nvlink_bandwidth

Bottleneck: min(all_reduce_bw, inter_stage_bw, all_scatter_bw)
```

**Key Insights:**
- Matches real LLM frameworks (Megatron-LM, DeepSpeed, vLLM)
- Full tensor transferred between stages (not sharded)
- Accounts for collective communication overhead
- Three-step bottleneck model

### 4. Documentation (Lines 1-28)

Added comprehensive header documenting:
- Memory model with all-reduce semantics
- Network communication pattern
- TP sharding strategy
- Framework alignment

## Validation

### Memory Calculation Test
```
Memory Analysis (batch=8, seq=512, hidden=4096, FP16):
TP=1: 0.323 GB activation memory per GPU
TP=2: 0.162 GB activation memory per GPU
TP=4: 0.081 GB activation memory per GPU
TP=8: 0.045 GB activation memory per GPU
```

Memory correctly decreases with TP degree due to:
- Sharded KV cache (dominant at this batch size)
- Sharded intermediate computation
- Full activation overhead amortized across more GPUs

## Impact

### More Accurate Memory Modeling
- Can now fit more layers per GPU with higher TP degree
- Correctly accounts for KV cache sharding (critical for inference)
- Models all-reduce memory overhead

### More Realistic Network Modeling
- Matches actual framework communication patterns
- Accounts for collective operation overhead
- Identifies true bottlenecks (intra-node vs inter-node)

### Better Optimization Decisions
- More accurate throughput predictions
- Correct memory-communication trade-offs
- TP degree selection based on realistic constraints

## Technical Details

### All-Reduce in TP
Each Transformer layer with TP=4:
```
Input: [batch, seq, hidden] (sharded along hidden)
  ↓
QKV projection (column-parallel): [batch, seq, hidden/4]
  ↓
Attention computation: [batch, seq, hidden/4]
  ↓
Output projection (row-parallel) → ALL-REDUCE #1 → [batch, seq, hidden]
  ↓
MLP layer 1 (column-parallel): [batch, seq, 4*hidden/4]
  ↓
MLP layer 2 (row-parallel) → ALL-REDUCE #2 → [batch, seq, hidden]
  ↓
Output (full tensor for next layer)
```

### Network Communication Between PP Stages
```
TP Group 1: [GPU0, GPU1, GPU2, GPU3]
    ↓ (intra-node all-reduce)
Master GPU0: Full tensor [batch, seq, hidden]
    ↓ (inter-node network transfer)
Master GPU4: Full tensor [batch, seq, hidden]
    ↓ (intra-node all-scatter)
TP Group 2: [GPU4, GPU5, GPU6, GPU7] (each has [batch, seq, hidden/4])
```

## Future Improvements

1. **Configurable NVLink Bandwidth**: Currently hardcoded to 600 Gbps (A100)
2. **Sequence Parallelism**: Could further reduce activation memory
3. **TP Efficiency Calibration**: Profile-based efficiency factors per GPU type
4. **Mixed Precision**: Different precision for weights vs activations
5. **Optimizer State**: ZeRO-style sharding for training scenarios

## References

- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- DeepSpeed: https://www.deepspeed.ai/
- Efficient Large-Scale Language Model Training: https://arxiv.org/abs/2104.04473
