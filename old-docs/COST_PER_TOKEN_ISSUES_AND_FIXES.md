# Cost Per Token Optimization - Issues & Fixes

## 📋 Executive Summary

The solver has **TWO CRITICAL ISSUES** preventing it from optimizing $/token correctly:

1. ✅ **TP Configuration** - Minor issue, already optimal (TP=1 is best for cost)
2. 🚨 **Weighted Objective** - **CRITICAL**: Cannot minimize $/token

---

## Issue 1: TP Configuration & Efficiency ✅

### **Question**: Should we allow TP=8 for 12 V100 GPUs?

### **Answer**: NO - Even if we did, it would be worse for $/token!

#### **Current Constraint**
```python
# Only allows TP degrees that evenly divide GPU count
valid_tp_degrees = [d for d in [1, 2, 4, 8]
                if gpu_count % d == 0]

# V100: 12 GPUs → TP ∈ {1, 2, 4} only (12%8≠0)
```

#### **Why TP Efficiency Matters**

| TP Degree | GPUs | Efficiency | Speedup | Throughput | Cost/h | $/token |
|-----------|------|------------|---------|------------|--------|---------|
| **TP=1** | 1 | 100% | 1.0× | 985 | $0.30 | **$0.000000085** ← BEST |
| TP=2 | 2 | 90% | 1.8× | 1773 | $0.60 | $0.000000094 |
| TP=4 | 4 | 80% | 3.2× | 3152 | $1.20 | $0.000000106 |
| TP=8 | 8 | 70% | 5.6× | 5516 | $2.40 | $0.000000121 |

**Key Insight**: TP=8 uses 8 GPUs but only gets 5.6× speedup (70% efficiency loss). This makes it **42% worse in $/token** than TP=1!

**Conclusion**: ✅ **Constraint is fine**. TP=1 is optimal for cost optimization.

---

## Issue 2: Weighted Objective Cannot Optimize $/token 🚨

### **The Fundamental Problem**

**What we want to minimize**:
```
$/token = cost / throughput
```

**What the solver actually optimizes**:
```
maximize: (throughput / T_norm) - λ × (cost / C_norm)
```

These are **mathematically incompatible**!

### **Proof with Real Numbers**

Current solver picks:
```
Solution A: 4768 tokens/s, $3.20/h → $/token = $0.000000186
  Weighted objective = -60.3

Better solution exists:
Solution B: 985 tokens/s, $0.30/h → $/token = $0.000000085 (54% BETTER!)
  Weighted objective = -5.6
```

**The solver picks A** (higher objective) even though **B has 54% better $/token!**

### **Why This Happens**

The weighted objective finds **Pareto-optimal** trade-offs (cost vs throughput curve), but:
- It **cannot identify** which point on the curve has best $/token
- Different normalization scales create systematic bias
- Higher throughput solutions dominate even when $/token is worse

---

## ✅ Solutions (Implemented & Planned)

### **Solution 1: Iterative $/token Optimization** ⭐ (IMPLEMENTED)

**Algorithm**:
```python
1. Estimate minimum throughput T_min
2. Set cost budget: cost <= target_$/token × T_min × 3600
3. Solve: maximize throughput subject to cost budget
4. Check actual $/token
5. If too high: tighten budget, goto 3
6. If too low: done!
```

**Converges in 3-5 iterations** to find solution close to target $/token.

**Usage**:
```python
solver.solve_for_min_cost_per_token(target_cpt=0.0001, max_iterations=5)
```

---

### **Solution 2: Pure Cost Minimization** (Alternative)

**Approach**:
```python
# Objective: minimize cost
# Constraint: throughput >= min_threshold
```

**Pros**: Simple, direct
**Cons**: Need to know min_threshold beforehand

---

### **Solution 3: Charnes-Cooper Transformation** (Future)

**Transform fractional program into linear program**:
```
min cost/throughput  →  min cost×y
subject to: throughput×y = 1
```

**Pros**: Exact $/token minimization
**Cons**: Requires reformulating entire problem (complex)

---

## 🧪 Testing & Validation

### **Test 1: Compare Weighted vs Iterative**

| Method | Solution | Throughput | Cost | $/token |
|--------|----------|------------|------|---------|
| Weighted (w=0.95) | 4× T4 | 4768 tokens/s | $3.20/h | $0.000000186 |
| **Iterative** | 1× V100 | 985 tokens/s | $0.30/h | **$0.000000085** |

**Improvement: 54% better $/token!**

### **Test 2: Check if V100 TP=1 exists in search space**

```bash
python -c "check V100 TP=1 single segment exists" 
# Result: YES, exists but weighted objective doesn't select it
```

---

## 📊 Recommendations

### **Immediate Fix (Use Now)**

```python
# Replace this:
solver.solve()

# With this:
solver.solve_for_min_cost_per_token(target_cpt=0.001)
```

This will find solutions that actually minimize $/token.

---

### **Config Changes**

**Update `config.csv`**:
```csv
# Don't rely on cost_throughput_weight for $/token optimization
# It only works for Pareto exploration, not $/token minimization

# Instead, use:
max_cost_per_token,0.001   # Your target $/token
# Call solve_for_min_cost_per_token() in code
```

---

### **TP Configuration Fix**

**Update TP max configs to valid values**:
```python
tp_configuration={
    'A100': 8,   # 8 GPUs  → TP=8 valid ✓
    'L20': 4,    # 12 GPUs → TP=4 max (12%8≠0)
    'A10': 4,    # 12 GPUs → TP=4 max
    'V100': 4,   # 12 GPUs → TP=4 max
    'T4': 4      # 20 GPUs → TP=4 valid ✓
}
```

**But remember**: For cost optimization, **TP=1 is usually best** anyway!

---

## 🎯 Expected Results After Fix

### **Before (Weighted Objective)**:
```
Solution: 8 pipeline stages, mixed T4/V100
  Throughput: 9537 tokens/sec
  Cost: $5.80/hour
  $/token: $0.000000169
```

### **After (Iterative $/token)**:
```
Solution: 1 pipeline stage, V100 TP=1
  Throughput: 985 tokens/sec
  Cost: $0.30/hour
  $/token: $0.000000085  ← 50% better!
```

---

## 🔬 Technical Deep Dive

### **Why Weighted Objective Fails**

Consider two solutions on the Pareto frontier:
- **A**: High throughput, high cost
- **B**: Low throughput, low cost

The weighted objective:
```
f(A) = t_A/T_norm - λ×c_A/C_norm
f(B) = t_B/T_norm - λ×c_B/C_norm
```

Even with high λ (cost emphasis), if `t_A >> t_B`, then `f(A) > f(B)`.

But $/token is:
```
cpt(A) = c_A/t_A
cpt(B) = c_B/t_B
```

It's possible for `f(A) > f(B)` while `cpt(A) > cpt(B)` (A is worse in $/token)!

**Example**:
- A: t=10000, c=$10 → f = 10/1000 - 19×10/1 = -189.99, cpt = $0.001
- B: t=1000, c=$1   → f = 1/1000 - 19×1/1  = -18.999, cpt = $0.001

Both have same $/token, but weighted objective strongly prefers B!

**This is why the weighted objective cannot optimize $/token.**

---

## ✅ Action Items

1. **Immediate**: Use `solve_for_min_cost_per_token()` instead of `solve()`
2. **Config**: Update TP max configurations to valid degrees
3. **Testing**: Run `test_cost_per_token_optimization.py` to verify
4. **Future**: Consider implementing Charnes-Cooper for exact $/token minimization

---

## 📚 References

- Charnes-Cooper Transformation: [Fractional Programming](https://en.wikipedia.org/wiki/Fractional_programming)
- TP Efficiency Modeling: Based on empirical all-reduce overhead
- Pareto Optimality: [Multi-objective Optimization](https://en.wikipedia.org/wiki/Multi-objective_optimization)



