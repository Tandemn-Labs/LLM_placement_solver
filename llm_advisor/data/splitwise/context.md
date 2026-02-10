## Splitwise Profiling Data Context

This `data.csv` contains real hardware profiling measurements from DGX-A100 and
DGX-H100 systems, sourced from the SplitwiseSim project
([GitHub](https://github.com/Mutinifni/splitwise-sim)).

### Paper Reference
Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Inigo Goiri,
Saeed Maleki, Ricardo Bianchini.
**"Splitwise: Efficient Generative LLM Inference Using Phase Splitting"**,
International Symposium on Computer Architecture (ISCA 2024), Buenos Aires.

### How the Numbers Were Generated
- Measurements were collected by profiling LLM inference on physical
  **DGX-A100** and **DGX-H100** servers (NVLink-connected GPUs).
- Each row records the timing for a single run of a specific
  `(model, hardware, tensor_parallel, prompt_size, batch_size, token_size)`
  configuration.
- Multiple repetitions exist per configuration for statistical robustness.
- The profiling captures **prompt processing time** (prefill), **per-token
  generation time** (decode), and **end-to-end latency** in milliseconds.
- Power draw (peak and average watts) is also recorded per run.

### Models Profiled
| Model | HuggingFace ID | Architecture | Tensor Parallel |
|-------|---------------|--------------|-----------------|
| llama2-70b | `meta-llama/Llama-2-70b-hf` | LlamaForCausalLM | 2, 4, 8 |
| bloom-176b | `bigscience/bloom` | BloomForCausalLM | 8 |

### Hardware
| Hardware Label | GPU | Memory | Interconnect | Notes |
|---------------|-----|--------|-------------|-------|
| a100-80gb | NVIDIA A100 | 80 GB | NVLink (DGX) | DGX-A100 server |
| h100-80gb | NVIDIA H100 | 80 GB | NVLink (DGX) | DGX-H100 server |
| h100-80gb-pcap | NVIDIA H100 | 80 GB | NVLink (DGX) | Power-capped H100 |

### Configuration Space
- **Prompt sizes (input tokens):** 128, 256, 512, 1024, 2048, 4096, 8192
- **Batch sizes:** 1, 2, 4, 8, 16, 32, 64
- **Token sizes (output tokens):** 128, 256, 512, 1024, 2048, 4096, 8192
- **105 rows per (model, hardware, tensor_parallel) group**, 1260 rows total
- Not all (prompt, batch, token) combinations are present in every group;
  the grid is a subset of the full 7x7x7 cross-product.

### Column Definitions
| Column | Unit | Description |
|--------|------|-------------|
| `model` | — | Short model name (e.g., `llama2-70b`) |
| `hardware` | — | Hardware SKU label |
| `prompt_size` | tokens | Number of input/prompt tokens |
| `batch_size` | requests | Number of concurrent requests in the batch |
| `token_size` | tokens | Number of output tokens to generate |
| `peak_power` | watts (normalized) | Peak GPU power draw during the run |
| `average_power` | watts (normalized) | Average GPU power draw during the run |
| `prompt_time` | ms | Time to process all prompt tokens (prefill phase) |
| `token_time` | ms | Time to generate one output token (decode step) |
| `e2e_time` | ms | End-to-end time for the full request |
| `tensor_parallel` | — | Tensor parallelism degree (number of GPUs) |

### How This Data Is Used in SplitwiseSim
The simulator's `DatabasePerformanceModel` reads this CSV and builds linear
interpolators (via `scipy.interpolate.interp1d`) keyed by
`batch_tokens = prompt_size * batch_size` for each `(model, hardware, tp)`
tuple. These predictors estimate prompt and token times for unseen
configurations by extrapolation. The `generate_perf_db.py` script can further
augment the dataset with interpolated rows for intermediate prompt/batch sizes.

### Reliability
- These are **real hardware measurements** from DGX systems, not simulation
  or analytical estimates.
- Multiple repetitions per configuration allow computing variance/median.
- Power measurements may use normalized values (peak_power near 1.0 for A100
  rows suggests normalization against TDP).
- The H100 `h100-80gb-pcap` variant represents a power-capped configuration,
  useful for studying power-performance tradeoffs.

### What Is Not Captured
- Serving engine or framework (no vLLM/TGI/TensorRT — raw model execution).
- Request scheduling or queuing effects (single-batch isolated measurements).
- KV cache management, continuous batching, or speculative decoding.
- Cloud instance types or pricing (bare-metal DGX systems, not cloud VMs).
- CUDA/driver/NCCL versions used during profiling.
- Pipeline parallelism (all measurements are TP-only, PP=1).
