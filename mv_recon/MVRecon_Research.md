# MVRecon: Lightweight Multi-View 3D Object Reconstruction

**Date**: 2026-04-10
**Status**: Full-dataset training in progress (4,982 objects, 50 epochs)
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

### 2.1 Overview

```
Input: N views × 160×160 RGB + N × 4×4 camera-to-world matrices
  │
  ├─ FeatureEncoder (shared MobileNetV3-Small)
  │    → N × 128ch × 20×20 feature maps
  │
  ├─ GeometricUnprojector
  │    Project 32³ voxel grid into each view,
  │    sample features via grid_sample,
  │    average across views
  │    → 128ch × 32×32×32 feature volume
  │
  ├─ CoarseToFineRefiner (3D CNN)
  │    16³ → 32³ → 64³ progressive refinement
  │    → 32ch × 64×64×64
  │
  └─ OccupancyColorHead
       → 1ch × 64³ density logits
       → 3ch × 64³ color logits
```

### 2.2 Component Details

#### FeatureEncoder (0.2M params)

MobileNetV3-Small pretrained on ImageNet. Extracts multi-scale features:
- Early layers (0–3): 20×20 × 24ch
- Mid layers (4–8): 10×10 × 48ch, upsampled to 20×20
- Concatenated (72ch) and projected to 128ch via 1×1 conv

The encoder is **shared** across all input views — each view is processed independently, then fused geometrically.

#### GeometricUnprojector (0 learned params)

This is the architectural key — it replaces the transformer decoder with deterministic geometry:

1. Create a regular 32³ grid of voxel centers in world space, spanning [-0.55, 0.55]³
2. For each view, compute the world-to-camera transform from the Blender c2w matrix (with OpenGL→OpenCV flip)
3. Project all 32,768 voxel centers into each view's image plane using known intrinsics
4. Sample the 128-channel feature map at each projected location via bilinear `grid_sample`
5. Mask out voxels that fall behind the camera or outside the image bounds
6. Average features across views (weighted by visibility count)

This operation is fully differentiable and batched — no Python loops over views or voxels. The result is a 128×32×32×32 feature volume where each voxel contains the average appearance from all views that see it.

#### CoarseToFineRefiner (0.3M params)

Progressive 3D refinement using grouped convolutions (groups=8):

| Stage | Resolution | Channels | Operation |
|-------|-----------|----------|-----------|
| 1 | 32³ → 16³ | 128 | Strided conv + GroupedResBlock |
| 2 | 16³ → 32³ | 128 → 64 | Trilinear upsample + skip concat + GroupedResBlock |
| 3 | 32³ → 64³ | 64 → 32 | Trilinear upsample + GroupedResBlock |

The coarse stage captures global structure, the fine stage adds detail. Skip connections from the input volume preserve unprojected features.

Groups=8 gives 8× fewer FLOPs than standard conv3d while training efficiently on Apple MPS (unlike depthwise, which has a slow backward pass on MPS).

#### OccupancyColorHead (0.001M params)

Two tiny branches from the 32ch refined volume:
- **Density**: 32→16→1 (Conv3d 1×1, GELU, Conv3d 1×1). Bias initialized to -5.0 (sigmoid(-5)≈0.007 — mostly empty at start)
- **Color**: 32→16→3 (Conv3d 1×1, GELU, Conv3d 1×1). Sigmoid applied for RGB

### 2.3 Parameter Count

| Component | Params | Size (FP32) |
|-----------|--------|-------------|
| FeatureEncoder (MobileNetV3-Small) | 199,864 | 0.8 MB |
| CoarseToFineRefiner | 296,224 | 1.1 MB |
| OccupancyColorHead | 1,124 | 0.004 MB |
| **Total** | **497,212** | **2.0 MB** |

This is **843× smaller** than TripoSR (419M) and **100× smaller** than the distilled student (50M).

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

#### 3.2.1 3D BCE Loss (weight=0.5)

Direct binary cross-entropy between predicted 64³ occupancy (sigmoid of logits) and GT voxels. Uses class-balanced pos_weight (capped at 20×) since objects are sparse (~2% filled).

This is the geometry workhorse — provides direct voxel-level supervision without any rendering.

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

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealing (T_max=50, eta_min=1e-5) |
| Epochs | 50 |
| Batch size | 1 (per-object, sequential) |
| Input views | 4 (randomly selected from 24) |
| Supervision views | 4 (from remaining 20) |
| Input resolution | 160×160 (ImageNet-normalized) |
| Supervision resolution | 128×128 |
| Rays per supervision view | 1024 |
| Samples per ray | 64 |
| Device | Apple MPS (M4 Max) |
| DataLoader workers | 4 |
| Gradient clipping | max_norm=1.0 |
| Validation | Every 5 epochs (mean IoU on val set) |
| Early stopping | 50 epochs without val IoU improvement |
| Checkpointing | Every 5 epochs + best val IoU |

**Estimated training time**: ~38 min/epoch × 50 epochs ≈ 32 hours.

### 3.4 Training Evolution

The architecture and training went through several iterations during the 11-object overfit phase:

1. **v1 — Triplane volume rendering only**: Learned to match GT silhouettes but plateaued quickly. Volume renders were blurry, mesh shapes approximate but recognizable.

2. **v2 — Added direct 3D BCE loss**: Required GT voxels. Initial voxelization used GDrive GLBs + Blender normalization. Loss dropped faster, IoU improved.

3. **v3 — Coarse-to-fine architecture**: Replaced flat 32³ refiner with progressive 16³→32³→64³ refinement. Output resolution doubled to 64³. Significant quality improvement.

4. **v4 — Added sparsity regularizer**: Reduced false-positive voxels (floating noise fragments outside the object).

5. **v5 — Camera axis fix**: Discovered that render camera orientations didn't match GT voxel axes. Validation script confirmed the fix.

6. **v6 — Current full training**: 4,982 objects, all fixes incorporated.

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

**Measured: 1.5 seconds end-to-end** — confirmed on Quest 3 via the debug menu MVRecon tab.

| Metric | MVRecon | TripoSR (pruned QDQ 384) |
|--------|---------|--------------------------|
| E2E latency | **1.5s** | 54s |
| Model size | **2.8 MB** | ~170 MB |
| Input | 3 views × 160px | 1 view × 384px |
| Output | 64³ occupancy → marching cubes | Triplane → NeRF decoder |
| Parameters | **0.5M** | ~50M |

This is:
- **36× faster** than the deployed TripoSR pipeline
- **6.7× faster** than the 10-second target
- **60× smaller** model file
- **100× fewer** parameters

---

## 5. Comparison with Prior Approaches

### 5.1 Architectural Comparison

| | TripoSR (Teacher) | TripoSR-Lite (Student) | **MVRecon** |
|---|---|---|---|
| **Paradigm** | Single-view, learned prior | Single-view, distilled prior | **Multi-view, geometric fusion** |
| **Encoder** | DINO ViT-B/16 (86M) | MobileNetV3-Large (3.7M) | MobileNetV3-Small (0.2M) |
| **3D Representation** | Triplane (3×40×64²) | Triplane (3×40×32²) | **Voxel grid (64³)** |
| **Decoder** | 16L Transformer (330M) + NeRF MLP | 8L Transformer (44M) + NeRF MLP | **3D CNN (0.3M), no MLP** |
| **Camera info** | Not used | Not used | **Required (c2w matrices)** |
| **Multi-view** | N/A (single image) | N/A (single image) | **Native (1–10 views)** |
| **Total params** | 419M | 50M | **0.5M** |

### 5.2 Why Geometric Unprojection Works

The transformer decoder in TripoSR/student learns an **implicit** mapping from 2D tokens to 3D triplane features. This requires hundreds of millions of parameters because it must encode the full distribution of possible 3D structures.

MVRecon replaces this with an **explicit** geometric operation: given a voxel at world position (x,y,z) and a camera at known pose, the 2D pixel location is deterministic trigonometry. The model only needs to learn:
1. What 2D features to extract (MobileNetV3: 0.2M params)
2. How to clean up the fused 3D volume (3D CNN: 0.3M params)

The camera-aware unprojection eliminates ~99.9% of the parameters.

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

Groups=8 is a compromise for training on Apple MPS. Depthwise convolutions (groups=channels) have a known slow backward pass on MPS. Groups=8 gives 8× fewer FLOPs than standard conv while training at full speed. For Quest deployment, these can be converted to depthwise-separable post-training.

### 7.3 Why Sigmoid Alpha (not Softplus Density)?

Standard NeRF uses `softplus(density) × delta` for alpha. At 64³ resolution, rays cross only a few voxels — the continuous density formulation produces near-invisible thin structures. Sigmoid directly outputs opacity per voxel, aligning with the binary occupancy ground truth and the marching cubes iso-surface at 0.5.

### 7.4 Why 32³ Unprojection → 64³ Output?

Unprojecting at 64³ would require 262K voxels × N views grid_sample operations — expensive on mobile. 32³ (32K voxels) is 8× cheaper. The coarse-to-fine refiner then upsamples to 64³, learning to add detail that the coarse grid missed.

### 7.5 Why Both Rendering Loss and BCE Loss?

- **BCE alone**: Fast training, direct geometry signal, but doesn't teach the model about view-dependent appearance or silhouette accuracy
- **Rendering alone**: Teaches appearance but converges slowly and can get stuck in local minima (blurry volumes)
- **Combined**: BCE provides a strong geometric scaffold; rendering loss refines silhouettes and appearance. The BCE loss (weight=0.5) contributes ~40% of total loss.

---

## 8. Future Work

### 8.1 Full Training Evaluation (In Progress)

50-epoch training on 4,982 objects is running. Key metrics to track:
- Val IoU convergence
- Generalization: val-set mesh quality vs train-set
- Failure modes: which object types are hardest?

### 8.2 Quest Deployment with Trained Model

After training:
1. Export best checkpoint to ONNX
2. Deploy to Unity `StreamingAssets`
3. Visual quality evaluation on Quest with diverse test objects
4. A/B comparison with TripoSR output quality

### 8.3 Potential Improvements

- **Depth input**: Add Quest 3 depth sensor data as a 4th encoder channel for geometric bootstrapping
- **Adaptive view selection**: Prioritize views with maximal angular coverage rather than random selection
- **Higher resolution**: 128³ output with an additional refinement stage (adds ~50K params)
- **FP16/INT8 quantization**: The 2.8MB model likely quantizes well given its simplicity
- **QNN HTP**: If Qualcomm NPU access becomes available, inference could drop below 500ms

---

## 9. References

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

# Train (50 epochs, ~32 hours on M4 Max)
python -u -m mv_recon.train \
  --mode full --data_dir ~/Downloads/mv_recon_data \
  --uids_file filtered_uids.json \
  --epochs 50 --val_every 5 --lr 1e-3

# Extract meshes
python -u -m mv_recon.extract_mesh \
  --checkpoint output/mv_recon_overfit/<run>/checkpoints/best.pt \
  --volume_size 32

# Export ONNX
python -u -m mv_recon.export_onnx_v2 \
  --checkpoint output/mv_recon_overfit/<run>/checkpoints/best.pt
```
