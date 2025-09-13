# FlowNetwork — Module Presentation

FlowNetwork is an experimental LLM module that combines pattern-based flow routing with attention mechanisms and long-term memory. The goal is to handle long contexts at a lower computational cost.

---

## Key Components

* **FlowLayer** — Generates reduced, multidimensional flow patterns and assembles them into a token-token flow matrix. It includes batched sparsity (top-k) and intensity modulation.
* **EnhancedFlowLayer** — FlowLayer + self-/cross-attention + (optional) memory bank integration.
* **ContextAwareFlowRouter** — Adapts sparsity and routing based on context features and window size (sliding windows for long sequences).
* **FlowMemoryNetwork** — Buffer/memory with secure (buffer) updates, read mechanisms, and EMA updates.
* **FlowNetwork** — the main wrapper combining embeddings, the EnhancedFlowLayer stack, and head to logits.
* **EnhancedFlowTransformer** — an alternative integration of Flow + Transformer, with adaptive head count for dimension compatibility.

---

## Key Features and Design Decisions

* **Pattern-based routing**: Instead of dense, fully attentional matrices, a set of patterns (patterns) is used, mixed with weights per token, which reduces parameterization.
* **Adaptive sparsity**: multi-level top-k selection performed in batches to reduce allocations and speed up operations for long sequences.
* **Memory as buffer**: memory_bank registered as a buffer (`register_buffer`) and updated with `with torch.no_grad()` to maintain autograd safety. * **Parameter safety**: validation of critical parameters (vocab_size, d_model, num_layers) and the `adjust_num_heads` helper for adjusting the number of attention heads.

* **Fallbacks and logging**: defensive JIT calls, safe tensor-to-int conversions (`safe_tensor_to_int`), and logs about automatic adjustments.

---

## Quick Start Guide

1. Requirements:

* Python 3.8+
* PyTorch 1.12+ (CUDA recommended if testing on GPU)
2. Place `flownetwork.py` in the project and import:

```py
from flownetwork import FlowNetwork, EnhancedFlowTransformer
```

3. Sample initialization:

```py
model = FlowNetwork(vocab_size=30000, d_model=512, num_layers=8, max_seq_len=2048)
ids = torch.randint(0, 30000, (2, 128))
logits, metrics = model(ids)
```

---

## API — Most important classes and methods

* `FlowLayer(input_dim, output_dim, num_patterns, base_sparsity=0.1)` 

* `forward(x)` → `(flow_matrix, metrics)`
* `EnhancedFlowLayer(input_dim, output_dim, num_patterns, num_heads, ...)` 

* integrates attention and memory
* `FlowMemoryNetwork(d_model, memory_size, num_memory_heads)` 

* `forward(x, update_memory=True)` → `(out, metrics)`
* `FlowNetwork(...)` and `EnhancedFlowTransformer(...)` - complete models with `.forward(input_ids)` returning `(logits, metrics)`

---

## Testing and validation (recommended)

* Run unit tests for the combination `(d_model, num_heads)` e.g., (64,8), (65,8), (128,12). Verify that `adjust_num_heads` returns a reasonable value and that layers initialize without exception.
* Forward sanity check for short batches and 32/128/512 sequences.
* Memory update test: check for deterministic updates and no injections into the gradient graph.
* Benchmark: throughput tokens/sec and peak memory (`torch.cuda.max_memory_allocated()`), comparison with the baseline Transformer.

--

## Known limitations and recommendations before production

* Standardize all tensor→int conversion helpers (file partially fixed, fragmented locations still possible). Required: Review the entire file and replace the direct `int(tensor)`. * Remove duplicate helpers (e.g., if `_find_compatible_num_heads` appears) and keep a single `adjust_num_heads` implementation.
* Add CIs that run unit tests and benchmarks on target hardware to confirm declared results.
* Limit and document automatic parameter adjustments (logging + clear message to the user).

---

## Changelog (important fixes made)

* `global_memory` changed from `nn.Parameter` to `register_buffer`.
* Critical `int(...)` casts replaced with `safe_tensor_to_int` in key places.
* `EnhancedFlowTransformer` output_head matches `num_heads` with `adjust_num_heads(vocab_size, num_heads)` to avoid an `embed_dim % num_heads` error.
* Batched top-k and scatter in sparsity locations.
* Memory updates performed in `with torch.no_grad()` with `.detach().clone()` and bound checks.

---

## Recommended Next Steps

1. Run full unit and integration tests in CI.
2. Add API documentation and a simple end-to-end demonstration (sample script).
3. Consider 4/8-bit quantization and distributed training if the goal is production.

---
