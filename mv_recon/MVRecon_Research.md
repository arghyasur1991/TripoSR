# MVRecon: Lightweight Multi-View 3D Object Reconstruction

**Date**: 2026-04-10
**Status**: Architecture v2 validated; full-dataset training pending (4,982 objects, 100 epochs)
**Context**: On-device 3D reconstruction for Meta Quest 3 room scanning

---

## 1. Motivation & Background

### 1.1 Problem Statement

During Quest 3 room scanning, YOLO detects objects (chairs, TVs, beds, etc.) and the system collects **3–10 multi-view RGB images** with **known camera poses** from keyframe collection. We need to generate complete 3D meshes of these objects and place them at their world positions — entirely on-device, in under 10 seconds.

### 1.2 Why Not TripoSR?

The prior approach distilled the 419M-parameter TripoSR model (DINO ViT + Transformer decoder + triplane NeRF) into a 50M-parameter student model. Despite extensive optimization (pruning, INT8 quantization, reduced resolution), the deployed pipeline still requires:

| Metric | TripoSR (pruned QDQ 384) |
|--------|--------------------------|
| E2E latency | **54 seconds** |
| Model size | ~170 MB |
| Input | 1 view × 384px |
| Output | Triplane → NeRF decoder → marching cubes |
| Architecture | Transformer cross-attention (O(n²) on 3072 tokens) |

54 seconds is impractical for interactive use. The triplane + NeRF decoder architecture is inherently expensive: it requires dense 3D grid queries through an MLP, and the transformer backbone's quadratic self-attention over 3072 tokens dominates inference time on mobile GPUs.

More fundamentally, TripoSR is a **single-view** model. It must hallucinate unseen geometry from a single image — a much harder task that demands a large model with a strong 3D prior. Our scanning pipeline provides **multiple views with known cameras**, rendering most of that hallucination unnecessary.

### 1.3 Key Insight

With multi-view images and known camera poses, we can replace learned 3D priors with **geometric computation**:

1. **Unproject** 2D image features into 3D using known camera matrices (deterministic, no learning required)
2. Use a **small 3D CNN** to clean up and refine the fused volume (lightweight learning)
3. Output occupancy directly at **64³ resolution** via marching cubes (no NeRF decoder needed)

This eliminates the transformer decoder, the triplane representation, and the NeRF MLP — the three most expensive components.

---

## 2. Architecture

### 2.1 Overview (v2)

```
Input: N views × 160×160 RGB + N × 4×4 camera-to-world matrices
  │
  ├─ FeatureEncoder (shared MobileNetV3-Small, three-scale)
  │    → N × 128ch × 20×20 feature maps
  │
  ├─ GeometricUnprojector (mean + variance fusion)
  │    Project 32³ voxel grid into each view,
  │    sample features via grid_sample,
  │    compute per-voxel mean + variance across views
  │    → 256ch × 32×32×32 feature volume
  │
  ├─ CoarseToFineRefiner (3D CNN, groups=4)
  │    16³ → 32³ → 64³ progressive refinement
  │    → 32ch × 64×64×64
  │
  └─ OccupancyColorHead
       → 1ch × 64³ density logits
       → 3ch × 64³ color logits
```

### 2.2 Component Details

#### FeatureEncoder (0.9M params)

MobileNetV3-Small pretrained on ImageNet. Extracts features at three spatial scales, all upsampled to 20×20 and concatenated:

| Scale | Layers | Spatial | Channels |
|-------|--------|---------|----------|
| Early | 0–3 | 20×20 | 24 |
| Mid | 4–8 | 10×10 → 20×20 | 48 |
| Late | 9–11 | 5×5 → 20×20 | 96 |

Concatenated (168ch) and projected to 128ch via 1×1 conv.

The late layers (added in v2) provide higher-level semantic features — object part understanding, shape priors — while the early/mid layers provide fine spatial detail. The encoder is **shared** across all input views.

**v1→v2 change**: Two-scale (72ch) → three-scale (168ch). The extra pretrained layers add 692K params but provide significantly richer features without any untrained capacity — all weights come from ImageNet pretraining.

#### GeometricUnprojector (0 learned params)

This is the architectural key — it replaces the transformer decoder with deterministic geometry:

1. Create a regular 32³ grid of voxel centers in world space, spanning [-0.55, 0.55]³
2. For each view, compute the world-to-camera transform from the Blender c2w matrix (with OpenGL→OpenCV flip)
3. Project all 32,768 voxel centers into each view's image plane using known intrinsics
4. Sample the 128-channel feature map at each projected location via bilinear `grid_sample`
5. Mask out voxels that fall behind the camera or outside the image bounds
6. Compute **mean** and **variance** of features across views, concatenate → 256ch

The mean+variance fusion (added in v2) replaces simple averaging. The variance channel encodes **view consistency**: low variance at a voxel means multiple views agree on its appearance (likely a real surface), high variance means views disagree (likely occlusion, empty space, or a depth boundary). This gives the refiner a powerful geometric signal with zero additional parameters.

The result is a 256×32×32×32 feature volume. The operation is fully differentiable and batched — no Python loops over views or voxels.

#### CoarseToFineRefiner (1.4M params)

Progressive 3D refinement using grouped convolutions (groups=4):

| Stage | Resolution | Channels | Blocks | Operation |
|-------|-----------|----------|--------|-----------|
| 1 | 32³ → 16³ | 256 | 1 | AvgPool + GroupedResBlock |
| 2 | 16³ → 32³ | 512 → 128 | 2 | Trilinear upsample + skip concat + 1×1 proj + 2× GroupedResBlock |
| 3 | 32³ → 64³ | 128 → 32 | 1 | Trilinear upsample + 1×1 proj + GroupedResBlock |

The coarse stage captures global structure at 16³ with the full 256-channel mean+variance volume. Stage 2 is the workhorse — it gets 2 res blocks and a skip connection from the input volume, operating at the native unprojection resolution. The fine stage adds surface detail at 64³.

**v1→v2 changes**:
- Groups: 8 → 4 (2× more cross-channel mixing per layer)
- Input channels: 128 → 256 (from mean+variance fusion)
- Stage 2: 1 block → 2 blocks (where skip connection enriches features)
- Downsampling: strided conv → AvgPool3d (simpler, no wasted params)
- Total: 296K → 1.41M params (4.8×)

#### OccupancyColorHead (0.002M params)

Two tiny branches from the 32ch refined volume:
- **Density**: 32→32→1 (Conv3d 1×1, GELU, Conv3d 1×1). Bias initialized to -5.0 (sigmoid(-5)≈0.007 — mostly empty at start)
- **Color**: 32→32→3 (Conv3d 1×1, GELU, Conv3d 1×1). Sigmoid applied for RGB

**v1→v2 change**: Intermediate channels doubled (16→32) for slightly more capacity.

### 2.3 Parameter Count

| Component | v1 Params | v2 Params | Change |
|-----------|-----------|-----------|--------|
| FeatureEncoder | 199,864 | 892,192 | +692K (pretrained layers) |
| GeometricUnprojector | 0 | 0 | — |
| CoarseToFineRefiner | 296,224 | 1,413,984 | +1.1M (wider, deeper) |
| OccupancyColorHead | 1,124 | 2,244 | +1.1K |
| **Total** | **497,212 (0.5M)** | **2,308,420 (2.3M)** | **4.6×** |

v2 is still **181× smaller** than TripoSR (419M) and **22× smaller** than the distilled student (50M). Quest inference estimated at 6–7s (within 10s budget, vs 1.5s for v1).

---

## 3. Training

### 3.1 Dataset

**Source**: Objaverse 1.0, curated to 5,000 indoor objects matching 20 YOLO categories.

**Per object**:
- 24 rendered views (8 azimuths × 3 elevations) at 1280×960 with alpha channel
- `cameras.json` with 4×4 camera-to-world poses per view
- GLB mesh file for ground-truth voxelization

**Ground truth voxels**: Each GLB is normalized in Blender (matching the render pipeline's coordinate frame exactly), exported as STL, then voxelized at 64³ using dense surface sampling (1M points). The voxel grid spans [-0.55, 0.55]³ in [z, y, x] layout.

**Train/val split**: 90/10 deterministic split (seed=42) → ~4,484 train, ~498 val.

### 3.2 Loss Function

Four-component hybrid loss:

#### 3.2.1 3D BCE Loss (weight=1.0, was 0.5 in v1)

Direct binary cross-entropy between predicted 64³ occupancy (sigmoid of logits) and GT voxels. Uses class-balanced pos_weight (capped at 20×) since objects are sparse (~2% filled).

This is the geometry workhorse — provides direct voxel-level supervision without any rendering. Weight increased from 0.5 to 1.0 in v2 to make shape accuracy the primary training objective.

#### 3.2.2 Photometric Loss (weight=1.0)

For each supervision view (not seen by the model):
1. Cast `n_rays_per_view` random rays from the camera position
2. March through the predicted density+color volume (64 samples per ray)
3. Alpha-composite using sigmoid occupancy: `alpha = sigmoid(density_logit)`
4. MSE between rendered RGB and GT image pixels at the same ray locations

This teaches the model to produce a volume that *looks correct* from novel viewpoints.

#### 3.2.3 Mask Loss (weight=0.1)

Same ray casting as photometric, but compares accumulated opacity against the GT alpha mask. Teaches the model where the object silhouette should be vs background.

#### 3.2.4 Sparsity Regularizer (weight=0.02)

`mean(sigmoid(all_logits))` — penalizes the model for predicting too many occupied voxels, fighting false-positive noise outside the object.

### 3.3 Training Configuration

| Parameter | v1 | v2 |
|-----------|----|----|
| Optimizer | AdamW (lr=1e-3, wd=1e-4) | same |
| Scheduler | CosineAnnealing (T_max=50) | **5-epoch linear warmup + cosine decay** |
| Epochs | 50 | **100** |
| Effective batch size | 1 | **8 (grad_accum=8)** |
| Input views | 4 (randomly selected from 24) | same |
| Supervision views | 4 (from remaining 20) | same |
| Input resolution | 160×160 (ImageNet-normalized) | same |
| Supervision resolution | 128×128 | same |
| Rays per supervision view | 1024 | same |
| Samples per ray | 64 | same |
| Device | Apple MPS (M4 Max) | same |
| DataLoader workers | 4 | same |
| Gradient clipping | max_norm=1.0 | same |
| Validation | Every 5 epochs | **Every 2 epochs** |
| Early stopping | 50 epochs | **20 epochs** |
| w_bce | 0.5 | **1.0** |
| Color jitter | (0.2, 0.2, 0.15, 0.02) | **(0.15, 0.15, 0.1, 0.02)** |

Key v2 recipe changes:
- **LR warmup** (5 epochs): prevents early training instability with the larger model
- **Gradient accumulation** (8 steps): smooths gradients from the noisy batch-size-1 training
- **Higher w_bce**: focuses the model on getting 3D shape right as the primary objective
- **More frequent validation**: better tracking of generalization trends

### 3.3.1 Data Augmentation

Applied during training to close the domain gap between clean Objaverse renders and noisy Quest camera images:

| Augmentation | Parameters | Purpose |
|---|---|---|
| Random backgrounds | Random solid color behind alpha-masked object | Quest scenes have varied backgrounds |
| Horizontal flip | 50% probability, camera pose mirrored | Double effective dataset size |
| Color jitter | brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02 | Quest camera color variation |
| Gaussian noise | σ ∈ [0.04, 0.08], randomized per sample | Quest sensor noise |

Gaussian noise was retained despite increasing training difficulty because Quest images genuinely contain sensor noise — the larger v2 model has sufficient capacity to learn through it. Color jitter was slightly reduced from v1 values to avoid overwhelming the model during early training.

### 3.4 Training Evolution

The architecture and training went through several iterations during the 11-object overfit phase:

1. **v1 — Triplane volume rendering only**: Learned to match GT silhouettes but plateaued quickly. Volume renders were blurry, mesh shapes approximate but recognizable.

2. **v2 — Added direct 3D BCE loss**: Required GT voxels. Initial voxelization used GDrive GLBs + Blender normalization. Loss dropped faster, IoU improved.

3. **v3 — Coarse-to-fine architecture**: Replaced flat 32³ refiner with progressive 16³→32³→64³ refinement. Output resolution doubled to 64³. Significant quality improvement.

4. **v4 — Added sparsity regularizer**: Reduced false-positive voxels (floating noise fragments outside the object).

5. **v5 — Camera axis fix**: Discovered that render camera orientations didn't match GT voxel axes. Validation script confirmed the fix.

6. **v6 — Full training attempt (v1 architecture)**: 4,484 train / 498 val objects, 50 epochs with augmentation. Val IoU plateaued at ~0.34 from epoch 1 — the 0.5M model lacked capacity to generalize across 4,500 diverse objects.

7. **v7 — Architecture v2 (current)**: Diagnosed capacity bottleneck. Scaled encoder (3-scale features), added mean+variance view fusion, widened refiner (groups 8→4, +1 block at stage 2). 0.5M → 2.3M params. Overfit test confirmed faster learning and higher ceiling.

### 3.5 Generalization Evidence

After overfit training on 11 objects, we tested on 7 **unseen** objects from the same dataset. The model produced incomplete but structurally recognizable meshes — confirming it learned geometric priors (how multi-view images map to 3D shape) rather than memorizing specific objects. This motivated scaling to the full dataset.

---

## 4. Inference & Deployment

### 4.1 ONNX Export

Exported via `torch.onnx.export` (opset 18) with static shapes:
- Input images: `[1, 3, 3, 160, 160]` (batch=1, 3 views)
- Input cameras: `[1, 3, 4, 4]`
- Output density: `[1, 1, 64, 64, 64]`
- Output color: `[1, 3, 64, 64, 64]`

**Critical fix**: `torch.onnx.export` silently stored weights in a `.data` sidecar file. The export script now re-saves with `onnx.save_model(save_as_external_data=False)` to embed all weights inline. Final model: **2.8 MB** self-contained `.onnx`.

### 4.2 Unity Integration

The ONNX model runs via ONNX Runtime (not Unity Sentis) with:
- **macOS/Editor**: CPU Execution Provider (for development)
- **Quest 3**: XNNPACK Execution Provider (optimized ARM CPU kernels)

Unity C# pipeline (`OrtMVReconModel.cs`):
1. Load 3 input images, center-crop to square, resize to 160×160
2. ImageNet-normalize and pack into `DenseTensor<float>`
3. Load camera poses from `cameras.json`, flatten to `float[16]` per view
4. Run ONNX inference → 64³ density volume
5. Apply sigmoid, threshold at 0.5, run marching cubes
6. Generate Unity `Mesh` with vertices, normals, and projected texture

### 4.3 Quest 3 Performance

v1 (0.5M params) was **measured at 1.5 seconds end-to-end** on Quest 3. v2 (2.3M params) is estimated at **6–7 seconds** based on 4.6× parameter increase and heavier 3D convolutions, still within the 10-second budget. Quest measurement pending after full training completion.

| Metric | MVRecon v1 | MVRecon v2 (est.) | TripoSR (pruned QDQ 384) |
|--------|-----------|-------------------|--------------------------|
| E2E latency | 1.5s | **~6–7s** | 54s |
| Model size | 2.8 MB | **~9 MB** | ~170 MB |
| Input | 3 views × 160px | 3 views × 160px | 1 view × 384px |
| Output | 64³ occupancy | 64³ occupancy | Triplane → NeRF decoder |
| Parameters | 0.5M | **2.3M** | ~50M |

v2 compared to TripoSR:
- **~8× faster** than the deployed TripoSR pipeline
- **Within 10-second** interactive target
- **~19× smaller** model file
- **~22× fewer** parameters

---

## 5. Comparison with Prior Approaches

### 5.1 Architectural Comparison

| | TripoSR (Teacher) | TripoSR-Lite (Student) | **MVRecon v2** |
|---|---|---|---|
| **Paradigm** | Single-view, learned prior | Single-view, distilled prior | **Multi-view, geometric fusion** |
| **Encoder** | DINO ViT-B/16 (86M) | MobileNetV3-Large (3.7M) | MobileNetV3-Small 3-scale (0.9M) |
| **View fusion** | N/A | N/A | **Mean+variance unprojection** |
| **3D Representation** | Triplane (3×40×64²) | Triplane (3×40×32²) | **Voxel grid (64³)** |
| **Decoder** | 16L Transformer (330M) + NeRF MLP | 8L Transformer (44M) + NeRF MLP | **3D CNN (1.4M), no MLP** |
| **Camera info** | Not used | Not used | **Required (c2w matrices)** |
| **Multi-view** | N/A (single image) | N/A (single image) | **Native (1–10 views)** |
| **Total params** | 419M | 50M | **2.3M** |

### 5.2 Why Geometric Unprojection Works

The transformer decoder in TripoSR/student learns an **implicit** mapping from 2D tokens to 3D triplane features. This requires hundreds of millions of parameters because it must encode the full distribution of possible 3D structures.

MVRecon replaces this with an **explicit** geometric operation: given a voxel at world position (x,y,z) and a camera at known pose, the 2D pixel location is deterministic trigonometry. The model only needs to learn:
1. What 2D features to extract (MobileNetV3 3-scale: 0.9M params)
2. How to clean up the fused 3D volume (3D CNN: 1.4M params)

The camera-aware unprojection eliminates ~99.5% of the parameters.

### 5.3 Limitations vs TripoSR

- **Requires camera poses**: MVRecon cannot work from a single uncalibrated image. Our pipeline always provides poses from keyframe collection.
- **Resolution**: 64³ voxels = ~1.7cm resolution for a 1m object. TripoSR's triplane at 64² can represent finer detail. Sufficient for our use case (detected objects at room scale).
- **Unseen surfaces**: With 3+ views, most surfaces are observed. Fully occluded surfaces (e.g., bottom of a table) are inferred by the 3D CNN from context, but less accurately than TripoSR's strong learned prior.
- **Color quality**: Vertex colors via projection from input views. TripoSR's NeRF decoder can produce smoother textures.

---

## 6. Data Pipeline

### 6.1 Rendering (Pre-existing)

24 views per object rendered in Blender:
- 8 azimuths (0°, 45°, 90°, ..., 315°)
- 3 elevations (-20°, 20°, 45°)
- 1280×960 resolution with alpha channel
- Camera parameters saved to `cameras.json`

### 6.2 Voxelization

**Pipeline**: GLB → Blender headless (normalize + export STL) → trimesh surface sampling → 64³ binary grid

**Normalization** (critical — must match render pipeline exactly):
1. Import GLB in Blender
2. Apply all transforms (location, rotation, scale)
3. Compute collective bounding box across all mesh objects
4. Center at origin, scale to fit in [-0.5, 0.5]³
5. Export as STL

**Surface sampling**: 1M points sampled uniformly on the mesh surface, mapped to nearest voxel. No flood-fill (removed due to 10GB+ memory usage on complex meshes from `trimesh.voxelized().fill()`). Surface-only gives equivalent quality for training — the model learns to predict solid interiors from the surface pattern.

**Scale**: 4,982/5,000 objects voxelized successfully. 16 failures from degenerate GLBs, 2 from missing renders. Bulk processing with `ProcessPoolExecutor` and per-object Blender subprocess isolation.

### 6.3 Data Transfer

All data (renders 63GB, GLBs 41GB, voxels ~5GB) copied from Google Drive to local SSD via `rclone` with Google Drive API. 16 parallel transfers for large files, `--fast-list` for efficient file enumeration. Total: ~110GB local.

---

## 7. Design Decisions & Rationale

### 7.1 Why MobileNetV3-Small (not Large)?

MobileNetV3-Small (0.2M used) vs Large (3.7M): the encoder only needs to produce 128-channel feature maps at 20×20 spatial resolution. The geometric unprojection handles the 2D→3D mapping, so the encoder doesn't need the representational capacity to encode 3D structure. Small is sufficient and ONNX-exports cleanly.

### 7.2 Why Grouped Convolutions (not Depthwise Separable)?

Groups=4 (v2, previously groups=8 in v1) is a compromise between cross-channel mixing and compute efficiency on Apple MPS. Depthwise convolutions (groups=channels) have a known slow backward pass on MPS. Groups=4 allows each group to mix 64 channels in a 256-channel layer — enough for meaningful cross-feature learning. v1's groups=8 was too restrictive (only 16ch cross-talk per group), contributing to the model's inability to generalize.

### 7.3 Why Sigmoid Alpha (not Softplus Density)?

Standard NeRF uses `softplus(density) × delta` for alpha. At 64³ resolution, rays cross only a few voxels — the continuous density formulation produces near-invisible thin structures. Sigmoid directly outputs opacity per voxel, aligning with the binary occupancy ground truth and the marching cubes iso-surface at 0.5.

### 7.4 Why 32³ Unprojection → 64³ Output?

Unprojecting at 64³ would require 262K voxels × N views grid_sample operations — expensive on mobile. 32³ (32K voxels) is 8× cheaper. The coarse-to-fine refiner then upsamples to 64³, learning to add detail that the coarse grid missed.

### 7.5 Why Both Rendering Loss and BCE Loss?

- **BCE alone**: Fast training, direct geometry signal, but doesn't teach the model about view-dependent appearance or silhouette accuracy
- **Rendering alone**: Teaches appearance but converges slowly and can get stuck in local minima (blurry volumes)
- **Combined**: BCE provides a strong geometric scaffold; rendering loss refines silhouettes and appearance. In v2, BCE weight was raised to 1.0 (from 0.5) to make shape accuracy the dominant training signal.

---

## 8. v1 → v2 Changelog

v1 trained successfully on 11 overfit objects (IoU 0.82) but **failed to generalize** when scaled to 4,484 diverse objects — val IoU plateaued at ~0.34 from epoch 1. Root cause analysis:

| Bottleneck | v1 Issue | v2 Fix |
|---|---|---|
| **Encoder capacity** | 2-scale (72ch), ~200K params. Not enough semantic richness for diverse object categories | 3-scale (168ch), ~892K params. Late MobileNetV3 layers add object-part understanding |
| **View fusion** | Mean pooling discards consistency info. Model can't distinguish "all views agree" from "views disagree" | Mean + variance fusion. Zero-cost signal tells refiner which voxels are reliably seen |
| **Refiner cross-channel mixing** | groups=8 → only 16ch cross-talk per group. Too restrictive for learning complex 3D patterns | groups=4 → 64ch cross-talk per group. 2× more feature interaction |
| **Refiner depth** | 1 block per stage, 296K params. Insufficient capacity for 4500 diverse shapes | 2 blocks at stage 2 (skip-enriched), 1.41M params. Capacity where it matters most |
| **Training stability** | No warmup, effective batch=1 | 5-epoch LR warmup + grad_accum=8 for smoother optimization |
| **Shape supervision** | w_bce=0.5, roughly equal to rendering loss | w_bce=1.0, shape accuracy is the primary objective |

**Total: 0.5M → 2.3M params (4.6×), estimated Quest inference 6–7s (within 10s budget)**

---

## 9. Future Work

### 9.1 Full Training Evaluation (Pending)

100-epoch v2 training on 4,484 train objects (498 val) with early stopping. Key metrics to track:
- Val IoU convergence and comparison to v1's plateau at ~0.34
- Generalization: val-set mesh quality vs train-set
- Failure modes: which object types are hardest?
- Whether the 2.3M model's additional capacity translates to measurably better generalization

### 9.2 Quest Deployment with Trained Model

After training:
1. Export best checkpoint to ONNX
2. Deploy to Unity `StreamingAssets`
3. Visual quality evaluation on Quest with diverse test objects
4. A/B comparison with TripoSR output quality

### 9.3 Potential Improvements

- **Depth input**: Add Quest 3 depth sensor data as a 4th encoder channel for geometric bootstrapping
- **Adaptive view selection**: Prioritize views with maximal angular coverage rather than random selection
- **Higher resolution**: 128³ output with an additional refinement stage (adds ~50K params)
- **FP16/INT8 quantization**: The 2.8MB model likely quantizes well given its simplicity
- **QNN HTP**: If Qualcomm NPU access becomes available, inference could drop below 500ms

---

## 10. References

- **TripoSR**: Tochilkin et al., 2024. "TripoSR: Fast 3D Object Reconstruction from a Single Image." MIT License.
- **LRM**: Hong et al., ICLR 2024. "Large Reconstruction Model for Single Image to 3D."
- **MobileNetV3**: Howard et al., 2019. "Searching for MobileNetV3." ICCV.
- **Objaverse**: Deitke et al., 2023. "Objaverse: A Universe of Annotated 3D Objects." CVPR.
- **Marching Cubes**: Lorensen & Cline, 1987. SIGGRAPH.
- **Geometric multi-view fusion**: Broadly related to MVSNet (Yao et al., 2018) and cost volume approaches, but simplified to direct feature averaging given known cameras and small output resolution.

---

## Appendix A: File Structure

```
TripoSR/mv_recon/
├── model.py          # FeatureEncoder, GeometricUnprojector, CoarseToFineRefiner, MVReconModel
├── dataset.py        # ObjaverseMultiViewDataset (loads renders + cameras.json + voxels)
├── train.py          # Training loop with hybrid loss, train/val split, metrics CSV
├── renderer.py       # Differentiable volume renderer (ray casting + alpha compositing)
├── camera_utils.py   # Blender intrinsics, c2w→w2c conversion, intrinsics adjustment
├── extract_mesh.py   # Inference + marching cubes + texture projection + evaluation
├── voxelize.py       # Blender-based GLB→STL→64³ voxel grid generation
├── export_onnx_v2.py # PyTorch → ONNX export with embedded weights
└── MVRecon_Research.md # This document
```

## Appendix B: Reproducing Results

```bash
# Voxelize (requires Blender, ~2 hours sequential)
python -u -m mv_recon.voxelize \
  --uids filtered_uids.json --manifest object_manifest.json \
  --cache_dir ~/Downloads/mv_recon_data/objaverse_cache \
  --output_dir ~/Downloads/mv_recon_data/voxels --workers 1

# Overfit test (10 objects, 1000 epochs, ~100 min on M4 Max)
python -u -m mv_recon.train \
  --mode overfit --augment \
  --epochs 1000 --lr 1e-3 --warmup_epochs 5 \
  --n_rays_per_view 1024 --n_samples 64 --sup_image_size 128

# Full training v2 (100 epochs, ~48 hours on M4 Max)
python -u -m mv_recon.train \
  --mode full --data_dir ~/Downloads/mv_recon_data \
  --uids_file filtered_uids.json \
  --epochs 100 --val_every 2 --lr 1e-3 --warmup_epochs 5 \
  --grad_accum 8 --early_stop 20 --augment

# Extract meshes
python -u -m mv_recon.extract_mesh \
  --checkpoint output/mv_recon_overfit/<run>/checkpoints/best.pt \
  --volume_size 32

# Export ONNX
python -u -m mv_recon.export_onnx_v2 \
  --checkpoint output/mv_recon_overfit/<run>/checkpoints/best.pt
```
