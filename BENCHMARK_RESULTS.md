# TripoSR Optimization Benchmark Results

All results measured on **Apple M4 Max (MPS backend)** unless noted otherwise.
Quest 3 estimates use 11x bandwidth ratio (M4 Max 546 GB/s vs Quest 3 ~51 GB/s).

---

## Phase 1: Baseline Profiling (unmodified teacher)

**Date:** 2026-02-28
**Device:** M4 Max (MPS)
**Image:** `test_images/examples/flamingo.png`
**Runs:** 10 (first is warmup, stats over runs 2-10)
**Model:** stabilityai/TripoSR, 419.3M params

### Per-Component Timing

| Component | Mean (s) | Std (s) | Min (s) | Max (s) |
|---|---|---|---|---|
| 1. Image preprocessing | 0.0033 | 0.0012 | 0.0023 | 0.0065 |
| 2. Encoder (DINOv2) | 0.0405 | 0.0006 | 0.0398 | 0.0416 |
| 3. Triplane tokenizer | 0.0004 | 0.0000 | 0.0003 | 0.0005 |
| **4. Backbone (total)** | **0.5177** | **0.0101** | **0.5103** | **0.5454** |
| 5. Post-processor | 0.0012 | 0.0001 | 0.0011 | 0.0014 |
| 6. Mesh extraction | 1.2421 | 0.0879 | 1.1967 | 1.4893 |
| **TOTAL** | **1.8052** | | | |

### Per-Layer Backbone Timing

| Layer | Mean (s) |
|---|---|
| Layer 0 | 0.0580 (includes warmup spillover) |
| Layer 1-15 | ~0.0305 each |
| **16 layers total** | **0.5177** |

### Parameter Counts

| Component | Params |
|---|---|
| image_tokenizer (DINOv2) | 86.4M |
| tokenizer (triplane queries) | 3.1M |
| backbone (Transformer1D) | 329.5M |
| post_processor | 0.2M |
| decoder (NeRF MLP) | 0.0M |
| **TOTAL** | **419.3M** |

### Key Observations

- Forward pass (without mesh extraction): **~560ms** on M4 Max
- Mesh extraction (marching cubes 256^3 + vertex color query): **~1.24s** -- dominates total time
- Backbone is **92% of forward pass** time (518ms / 560ms)
- Each backbone layer costs ~31ms uniformly (self-attn dominates)
- Encoder (DINOv2) is only ~40ms -- not worth optimizing
- Quest 3 estimate for forward: ~560ms * 11 = **~6.2s** (mesh extraction runs on CPU, separate concern)

---

## Phase 2: Token Merging (ToMe) on Triplane Tokens

**Date:** 2026-02-28
**Implementation:** `tome_patch.py` -- within-plane bipartite soft matching
**Merge schedule:** layers [4, 8, 12], merge_ratio configurable

### Sanity Check

- ToMe patch applied with `merge_ratio=0.2, merge_layers=[4, 8, 12]`
- Output shape preserved: `(1, 3, 40, 64, 64)` -- matches baseline
- Mean absolute difference in scene_codes vs baseline: 30.49 (expected, since tokens are merged)

### Speed/Quality Results

*(To be filled after `reconstruct_compare.py` runs)*

---

## Phase 3: Image Token Pruning

*(To be filled)*

---

## Phase 4: ONNX Export + Quantization

*(To be filled)*

---

## Cumulative Results Summary

| Variant | Forward (M4 Max) | Quest 3 est. | CD (%) | F@1% | Vol IoU | Status |
|---|---|---|---|---|---|---|
| Baseline (PyTorch) | 560ms | ~6.2s | 0 (ref) | 100 (ref) | 100 (ref) | MEASURED |
| + ToMe (triplane) | TBD | TBD | TBD | TBD | TBD | IN PROGRESS |
| + Image token pruning | TBD | TBD | TBD | TBD | TBD | PENDING |
| + FP16 quantization | TBD | TBD | TBD | TBD | TBD | PENDING |
| + INT8 quantization | TBD | TBD | TBD | TBD | TBD | PENDING |
