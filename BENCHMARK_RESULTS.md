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

### Speed Results (forward pass only, no mesh extraction)

| Variant | Mean (s) | Speedup | Quest 3 est. |
|---|---|---|---|
| Baseline | 0.524 | 1.00x | ~5.8s |
| ToMe r=0.1, layers [4,8,12] | 0.450 | 1.17x | ~5.0s |
| ToMe r=0.2, layers [4,8,12] | 0.375 | 1.40x | ~4.1s |
| ToMe r=0.3, layers [4,8,12] | 0.323 | 1.62x | ~3.6s |

### Quality Results (20-image test set)

**Note:** Baseline meshes regenerated with proper gray-background preprocessing (alpha composited
onto 0.5 gray + resize_foreground, matching official `run.py`). Volume IoU not shown -- marching
cubes meshes aren't watertight, making voxel containment unreliable.

| Variant | Mean CD (%) | Mean F@1% | Mean F@2% | Verts Ratio | Quality Assessment |
|---|---|---|---|---|---|
| **ToMe r=0.1** | 0.708 | 84.2 | 97.3 | 1.01x | Good -- most images acceptable |
| **ToMe r=0.2** | 1.037 | 63.1 | 90.5 | 1.07x | Moderate -- noticeable degradation |
| **ToMe r=0.3** | 1.401 | 48.6 | 80.8 | 1.09x | Poor -- too aggressive |

**Per-image details (ToMe r=0.1, best quality/speed tradeoff):**

| Image | CD (%) | F@1% | F@2% | Status |
|---|---|---|---|---|
| chair.png | 0.639 | 85.7 | 98.9 | PASS |
| flamingo.png | 0.459 | 97.4 | 99.6 | PASS |
| hamburger.png | 0.538 | 95.1 | 100.0 | PASS |
| robot.png | 0.685 | 83.1 | 99.3 | MARGINAL |
| teapot.png | 0.543 | 96.2 | 100.0 | PASS |
| backpack_nobg.png | 0.619 | 89.7 | 100.0 | PASS |
| **book_nobg.png** | **1.343** | **49.1** | 83.6 | **FAIL** |
| bottle_nobg.png | 0.543 | 94.0 | 100.0 | PASS |
| chair_nobg.png | 0.601 | 91.0 | 99.9 | PASS |
| clock_nobg.png | 0.795 | 73.2 | 100.0 | MARGINAL |
| lamp_nobg.png | 0.827 | 76.1 | 93.5 | MARGINAL |
| mug_nobg.png | 0.746 | 77.0 | 100.0 | MARGINAL |
| shoe_nobg.png | 0.469 | 97.6 | 100.0 | PASS |
| teddy_bear_nobg.png | 0.631 | 89.7 | 100.0 | PASS |
| **vase_nobg.png** | **1.325** | **65.8** | 80.0 | **FAIL** |

**Key Findings:**
- Triplane tokens are more sensitive to merging than ViT image tokens -- each encodes a distinct spatial region
- r=0.1 is the practical limit for acceptable quality (2/20 images fail, rest pass/marginal)
- r=0.2+ causes significant geometric distortion, especially on complex/thin structures
- Best tradeoff: **ToMe r=0.1 gives 1.17x speedup with mostly acceptable quality**

---

## Phase 3: Image Token Pruning (DINOv2 output)

**Date:** 2026-04-04
**Implementation:** `prune_image_tokens()` in `tome_patch.py` -- bipartite soft matching on 1024 DINOv2 patch tokens (CLS preserved)

### Speed Results

| Variant | Mean (s) | Speedup |
|---|---|---|
| Baseline | 0.525 | 1.00x |
| Image prune 0.25 only | 0.511 | 1.03x |
| Image prune 0.5 only | 0.486 | 1.08x |
| ToMe r=0.1 + Img prune 0.5 | 0.448 | 1.17x |
| ToMe r=0.1 + Img prune 0.25 | 0.482 | 1.09x |

### Quality Results (20-image test set)

| Variant | Mean CD (%) | Mean F@1% | Mean F@2% | Failures | Assessment |
|---|---|---|---|---|---|
| ToMe r=0.1 alone | 0.71 | 84.3 | 97.3 | 1/20 | Good baseline |
| Image prune 0.5 alone | 0.90 | 76.3 | 93.9 | 3/20 | Worse than ToMe |
| ToMe r=0.1 + Img 0.5 | 0.95 | 70.7 | 92.0 | 3/20 | No speed gain, quality regression |

### Conclusion

**Image token pruning is NOT worth it.** DINOv2 patch tokens carry critical visual detail for
cross-attention. At 50% pruning: only 1.08x speedup, but 3 failures and worse quality than
ToMe r=0.1 alone. Stacking with ToMe gives no additional speedup (matching overhead offsets
cross-attention savings). **Dropped from the optimization stack.**

---

## Phase 4: ONNX Export + Quantization

**Date:** 2026-04-04
**Implementation:** `export_onnx.py` -- legacy TorchScript exporter, opset 15
**Export:** `TripoSRForward` wrapper (image -> scene_codes) + `DecoderWrapper` (NeRF MLP)

### Model Sizes

| Variant | Size | vs FP32 |
|---|---|---|
| FP32 | 1675.5 MB | 100% |
| FP16 (ORT optimizer) | 838.2 MB | 50% |
| INT8 (dynamic, MatMul only) | 435.5 MB | 26% |
| NeRF Decoder FP32 | 0.17 MB | -- |

### Accuracy vs PyTorch Reference

| Variant | Max Relative Error | Mean Relative Error | Status |
|---|---|---|---|
| FP32 | 0.0003% | 0.0000% | PASS |
| FP16 | 0.185% | 0.009% | PASS |
| INT8 | 4.90% | 0.30% | WARN (acceptable) |
| Decoder FP32 | 0.000% | 0.000% | PASS |

### Speed (ORT CPUExecutionProvider, M4 Max)

| Variant | Mean (s) | vs FP32 |
|---|---|---|
| ORT FP32 | 2.745 | 1.00x |
| ORT FP16 | 3.064 | 0.90x (slower -- CPU has no native FP16) |
| ORT INT8 | 2.522 | 1.09x |

Note: ORT CPU is ~5x slower than PyTorch MPS (0.53s) since it doesn't use the GPU.
CoreML EP was tested but crashes on models this large.

### Reconstruction Quality (20-image test set, meshes via ONNX scene_codes + PyTorch decoder)

| Variant | Mean CD (%) | Mean F@1% | Mean F@2% | Failures | Marginals | Overall |
|---|---|---|---|---|---|---|
| **ONNX FP32** | 0.470 | 96.6 | 100.0 | 0/20 | 0/20 | **ALL PASS** |
| **ONNX FP16** | 0.471 | 96.6 | 100.0 | 0/20 | 0/20 | **ALL PASS** |
| ONNX INT8 | 0.558 | 92.5 | 98.5 | 0/20 | 3/20 | MARGINAL |

**FP16 is essentially lossless** — same quality as FP32 across all 20 test images.
INT8 has 3 marginal cases (teapot, mug, vase) but no failures.

### Full ONNX Pipeline Validation (ONNX forward + ONNX NeRF decoder for mesh extraction)

This validates the complete deployment pipeline: ONNX main model outputs scene_codes,
ONNX NeRF decoder (`nerf_decoder.onnx`, 0.17 MB FP32) replaces PyTorch decoder for
triplane querying and mesh extraction. Grid sampling stays in PyTorch/C++ (will be C# on Quest).

| Variant | Mean CD (%) | Mean F@1% | Mean F@2% | Failures | Marginals | Overall |
|---|---|---|---|---|---|---|
| **FP32 + Decoder** | 0.471 | 96.6 | 100.0 | 0/20 | 0/20 | **ALL PASS** |
| **FP16 + Decoder** | 0.471 | 96.5 | 100.0 | 0/20 | 0/20 | **ALL PASS** |

**Full ONNX pipeline produces identical quality to PyTorch baseline.** The NeRF decoder ONNX
model (170 KB) is deployment-ready — no quality loss from ONNX conversion of the decoder MLP.

### Quest 3 Estimates

Quest 3 uses Snapdragon XR2 Gen 2 GPU via NNAPI/QNN execution provider.
M4 Max GPU bandwidth: ~546 GB/s, XR2 GPU bandwidth: ~51 GB/s (ratio ~11x).
Transformer inference is bandwidth-bound, so FP16 halves the bandwidth requirement.

| Variant | Quest 3 GPU est. | Notes |
|---|---|---|
| FP32 (baseline) | ~6.2s | Same as PyTorch estimate (bandwidth-bound) |
| FP16 | ~3.1s | Half the bandwidth of FP32 |
| INT8 | ~1.6-2.5s | 4x less bandwidth, but mobile INT8 GPU support varies |

**FP16 ONNX is the primary deployment target: ~3s forward on Quest 3.** INT8 may
further improve this but needs on-device validation (mobile INT8 GPU support is spotty).

---

## Cumulative Results Summary

| Variant | Forward (M4 Max) | Quest 3 est. | Size | Quality (20 imgs) | Status |
|---|---|---|---|---|---|
| Baseline (PyTorch MPS) | 525ms | ~5.8s | -- | reference | MEASURED |
| + ToMe r=0.1 (PyTorch) | 450ms | ~5.0s | -- | CD=0.71%, 1 fail | BEST PYTORCH |
| ONNX FP32 (CPU) | 2745ms | ~6.2s | 1676 MB | CD=0.47%, 20/20 PASS | MEASURED |
| **ONNX FP16 (CPU)** | 3064ms | **~3.1s** | **838 MB** | **CD=0.47%, 20/20 PASS** | **DEPLOY TARGET** |
| ONNX INT8 (CPU) | 2522ms | ~1.6-2.5s | 436 MB | CD=0.56%, 3 marginal | BACKUP OPTION |
| FP32 + ONNX decoder (full pipeline) | -- | -- | 1676+0.17 MB | CD=0.47%, 20/20 PASS | VALIDATED |
| **FP16 + ONNX decoder (full pipeline)** | -- | -- | **838+0.17 MB** | **CD=0.47%, 20/20 PASS** | **VALIDATED** |
