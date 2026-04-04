"""Quantitative reconstruction quality comparison across TripoSR variants.

Compares optimized variants against the unmodified PyTorch teacher baseline using:
  - Chamfer Distance (CD): geometry fidelity
  - F-Score @ 1% and 2% of bbox diagonal: completeness/accuracy
  - Volume IoU: overall shape agreement at 64^3 voxel resolution
  - Vertex count ratio: structural preservation
  - Inference latency

Usage:
    python reconstruct_compare.py --baseline              # cache baseline meshes
    python reconstruct_compare.py --tome 0.2              # compare ToMe r=0.2
    python reconstruct_compare.py --tome 0.1 0.2 0.3      # sweep ratios
    python reconstruct_compare.py --onnx models/fp16.onnx # compare ONNX variant
    python reconstruct_compare.py --all                   # run all available variants
    python reconstruct_compare.py --e2e                   # full rembg+onnx pipeline on Unity test images
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from image_utils import prepare_image
from tsr.system import TSR

BASELINE_CACHE = Path(__file__).parent / "output" / "compare_cache" / "baseline"
RESULTS_DIR = Path(__file__).parent / "output" / "compare_results"


def collect_test_images() -> list[Path]:
    """Collect all test images from the standard test set."""
    test_dir = Path(__file__).parent / "test_images"
    images = []
    for subdir in ["examples", "novel", "training"]:
        d = test_dir / subdir
        if not d.exists():
            continue
        if subdir == "novel":
            images.extend(sorted(d.glob("*_nobg.png")))
        else:
            images.extend(sorted(d.glob("*.png")))
    return images


def load_model(device: str) -> TSR:
    model = TSR.from_pretrained(
        "stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt"
    )
    model.to(device)
    model.eval()
    return model


def run_inference(model: TSR, image_path: Path, device: str) -> trimesh.Trimesh:
    """Run full inference and return the mesh."""
    img = prepare_image(image_path)
    with torch.no_grad():
        scene_codes = model(img, device)
        model.set_marching_cubes_resolution(256)
        meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=256)
    return meshes[0]


def time_inference(model: TSR, image_path: Path, device: str, n_runs: int = 5) -> float:
    """Return mean latency in seconds (forward only, excludes mesh extraction)."""
    img = prepare_image(image_path)
    times = []
    with torch.no_grad():
        for i in range(n_runs + 1):
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            t0 = time.perf_counter()
            _ = model(img, device)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            elapsed = time.perf_counter() - t0
            if i > 0:
                times.append(elapsed)
    return float(np.mean(times))


# --- ONNX inference ---

def _preprocess_for_onnx(image_path: Path) -> np.ndarray:
    """Preprocess image to (1, 3, 512, 512) float32 numpy array for ONNX."""
    from tsr.utils import ImagePreprocessor
    img = prepare_image(image_path)
    processor = ImagePreprocessor()
    rgb = processor(img, 512)  # (B, H, W, C)
    return rgb.permute(0, 3, 1, 2).numpy()  # (B, C, H, W)


def _onnx_forward(onnx_session, image_path: Path) -> np.ndarray:
    """Run ONNX forward pass, return scene_codes as numpy."""
    img_np = _preprocess_for_onnx(image_path)
    input_meta = onnx_session.get_inputs()[0]
    if input_meta.type == "tensor(float16)":
        img_np = img_np.astype(np.float16)
    return onnx_session.run(None, {"image": img_np})[0].astype(np.float32)


def rembg_onnx_to_rgba(rembg_session, image_path: Path) -> Image.Image:
    """Run u2netp ONNX on a raw image and return RGBA PIL image with alpha mask.

    Replicates the exact rembg preprocessing:
    1. Resize to 320x320 with LANCZOS
    2. Normalize to [0, max] then subtract ImageNet mean/std
    3. Run inference, min-max normalize output
    4. Resize mask back with LANCZOS
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    resized = img.resize((320, 320), Image.LANCZOS)
    arr = np.array(resized, dtype=np.float32)
    arr = arr / max(np.max(arr), 1e-6)

    # ImageNet normalization per channel
    tmp = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.float32)
    tmp[:, :, 0] = (arr[:, :, 0] - mean[0]) / std[0]
    tmp[:, :, 1] = (arr[:, :, 1] - mean[1]) / std[1]
    tmp[:, :, 2] = (arr[:, :, 2] - mean[2]) / std[2]
    inp = tmp.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 320, 320)

    input_name = rembg_session.get_inputs()[0].name
    pred = rembg_session.run(None, {input_name: inp})[0][:, 0, :, :]

    # Min-max normalization matching rembg
    ma, mi = np.max(pred), np.min(pred)
    pred = (pred - mi) / (ma - mi + 1e-8)
    mask = np.squeeze(pred)

    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.resize((orig_w, orig_h), Image.LANCZOS)

    rgba = img.copy()
    rgba.putalpha(mask_img)
    return rgba


def _preprocess_for_onnx_with_rembg(rembg_session, image_path: Path) -> np.ndarray:
    """Full rembg → preprocess pipeline for raw images. Returns (1, 3, 512, 512)."""
    from tsr.utils import ImagePreprocessor, resize_foreground

    img = Image.open(image_path)
    if img.mode == "RGBA":
        # Already has alpha — use standard pipeline
        rgba = img
    else:
        # Raw image — run rembg ONNX to get alpha
        rgba = rembg_onnx_to_rgba(rembg_session, image_path)

    # Match prepare_image: resize_foreground + composite on gray
    rgba = resize_foreground(rgba, 0.85)
    arr = np.array(rgba).astype(np.float32) / 255.0
    rgb = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
    pil_rgb = Image.fromarray((rgb * 255.0).astype(np.uint8))

    processor = ImagePreprocessor()
    tensor = processor(pil_rgb, 512)  # (B, H, W, C)
    return tensor.permute(0, 3, 1, 2).numpy()  # (B, C, H, W)


def run_e2e_onnx_inference(
    rembg_session,
    onnx_session,
    decoder_session,
    model: TSR,
    image_path: Path,
    device: str,
) -> trimesh.Trimesh:
    """Full end-to-end: rembg ONNX → TripoSR ONNX → decoder ONNX → mesh."""
    img_np = _preprocess_for_onnx_with_rembg(rembg_session, image_path)
    input_meta = onnx_session.get_inputs()[0]
    if input_meta.type == "tensor(float16)":
        img_np = img_np.astype(np.float16)
    scene_codes_np = onnx_session.run(None, {"image": img_np})[0].astype(np.float32)
    scene_codes = torch.from_numpy(scene_codes_np).to(device)
    return _extract_mesh_onnx_decoder(model, scene_codes, decoder_session, device)


def run_onnx_inference(
    onnx_session,
    model: TSR,
    image_path: Path,
    device: str,
    decoder_session=None,
) -> trimesh.Trimesh:
    """Run ONNX forward pass for scene_codes, then mesh extraction.

    If decoder_session is provided, uses ONNX decoder for density/color queries
    (full ONNX pipeline). Otherwise falls back to PyTorch decoder.
    """
    scene_codes_np = _onnx_forward(onnx_session, image_path)
    scene_codes = torch.from_numpy(scene_codes_np).to(device)

    if decoder_session is not None:
        return _extract_mesh_onnx_decoder(model, scene_codes, decoder_session, device)

    with torch.no_grad():
        model.set_marching_cubes_resolution(256)
        meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=256)
    return meshes[0]


def _extract_mesh_onnx_decoder(
    model: TSR,
    scene_codes: torch.Tensor,
    decoder_session,
    device: str,
    resolution: int = 256,
    threshold: float = 25.0,
) -> trimesh.Trimesh:
    """Extract mesh using ONNX decoder instead of PyTorch decoder.

    Replicates TSR.extract_mesh but swaps the decoder MLP call with ONNX inference.
    The triplane grid_sample and marching cubes stay in PyTorch/CPU.
    """
    import torch.nn.functional as F
    from einops import rearrange
    from tsr.models.isosurface import MarchingCubeHelper
    from tsr.utils import scale_tensor

    model.set_marching_cubes_resolution(resolution)
    renderer = model.renderer
    helper = model.isosurface_helper

    scene_code = scene_codes[0]  # single image

    def query_triplane_onnx(positions, triplane):
        """Grid-sample triplane features, then run ONNX decoder."""
        input_shape = positions.shape[:-1]
        positions = positions.view(-1, 3)
        positions = scale_tensor(
            positions, (-renderer.cfg.radius, renderer.cfg.radius), (-1, 1)
        )

        indices2D = torch.stack(
            (positions[..., [0, 1]], positions[..., [0, 2]], positions[..., [1, 2]]),
            dim=-3,
        )
        out = F.grid_sample(
            rearrange(triplane, "Np Cp Hp Wp -> Np Cp Hp Wp", Np=3),
            rearrange(indices2D, "Np N Nd -> Np () N Nd", Np=3),
            align_corners=False,
            mode="bilinear",
        )
        features = rearrange(out, "Np Cp () N -> N (Np Cp)", Np=3)

        # ONNX decoder: (N, 120) -> (N, 4) = [density, r, g, b]
        feat_np = features.cpu().numpy()
        chunk_size = 65536
        results = []
        for i in range(0, feat_np.shape[0], chunk_size):
            chunk = feat_np[i:i+chunk_size]
            result = decoder_session.run(None, {"triplane_features": chunk})[0]
            results.append(result)
        raw = np.concatenate(results, axis=0)
        raw = torch.from_numpy(raw).to(device)

        density = raw[..., 0:1]
        color_features = raw[..., 1:4]

        from tsr.utils import get_activation
        density_act = get_activation(renderer.cfg.density_activation)(
            density + renderer.cfg.density_bias
        )
        color = get_activation(renderer.cfg.color_activation)(color_features)

        return {
            "density_act": density_act.view(*input_shape, -1),
            "color": color.view(*input_shape, -1),
        }

    with torch.no_grad():
        grid_verts = scale_tensor(
            helper.grid_vertices.to(device),
            helper.points_range,
            (-renderer.cfg.radius, renderer.cfg.radius),
        )
        density_result = query_triplane_onnx(grid_verts, scene_code)
        density = density_result["density_act"]

    v_pos, t_pos_idx = helper(-(density - threshold))
    v_pos = scale_tensor(
        v_pos, helper.points_range,
        (-renderer.cfg.radius, renderer.cfg.radius),
    )

    with torch.no_grad():
        color_result = query_triplane_onnx(v_pos, scene_code)
        color = color_result["color"]

    return trimesh.Trimesh(
        vertices=v_pos.cpu().numpy(),
        faces=t_pos_idx.cpu().numpy(),
        vertex_colors=color.cpu().numpy(),
    )


def time_onnx_inference(onnx_session, image_path: Path, n_runs: int = 5) -> float:
    """Return mean ONNX forward-pass latency in seconds."""
    img_np = _preprocess_for_onnx(image_path)

    input_meta = onnx_session.get_inputs()[0]
    if input_meta.type == "tensor(float16)":
        img_np = img_np.astype(np.float16)

    times = []
    for i in range(n_runs + 1):
        t0 = time.perf_counter()
        _ = onnx_session.run(None, {"image": img_np})
        elapsed = time.perf_counter() - t0
        if i > 0:
            times.append(elapsed)
    return float(np.mean(times))


# --- Metrics ---

def chamfer_distance(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, n_samples: int = 10000) -> float:
    """Bidirectional Chamfer distance, normalized by mesh_a bbox diagonal. Returns percentage."""
    pts_a, _ = trimesh.sample.sample_surface(mesh_a, n_samples)
    pts_b, _ = trimesh.sample.sample_surface(mesh_b, n_samples)

    bbox_diag = np.linalg.norm(mesh_a.bounds[1] - mesh_a.bounds[0])
    if bbox_diag < 1e-8:
        return 100.0

    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)

    dist_a2b, _ = tree_b.query(pts_a)
    dist_b2a, _ = tree_a.query(pts_b)

    cd = (dist_a2b.mean() + dist_b2a.mean()) / 2.0
    return float(cd / bbox_diag * 100.0)


def f_score(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, tau_pct: float, n_samples: int = 10000) -> float:
    """F-Score: harmonic mean of precision and recall at distance threshold tau.

    tau_pct is percentage of bbox diagonal (e.g., 1.0 = 1% of diagonal).
    Returns 0-100 score.
    """
    pts_a, _ = trimesh.sample.sample_surface(mesh_a, n_samples)
    pts_b, _ = trimesh.sample.sample_surface(mesh_b, n_samples)

    bbox_diag = np.linalg.norm(mesh_a.bounds[1] - mesh_a.bounds[0])
    tau = bbox_diag * tau_pct / 100.0
    if tau < 1e-10:
        return 0.0

    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)

    dist_a2b, _ = tree_b.query(pts_a)
    dist_b2a, _ = tree_a.query(pts_b)

    precision = (dist_a2b < tau).mean()
    recall = (dist_b2a < tau).mean()

    if precision + recall < 1e-10:
        return 0.0
    return float(2 * precision * recall / (precision + recall) * 100.0)


def volume_iou(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, resolution: int = 64) -> float:
    """Volumetric IoU via shared-grid point containment. Returns 0-100 score.

    Uses a uniform grid covering the union of both bounding boxes to avoid
    alignment issues from independent voxelization.
    """
    try:
        bounds_min = np.minimum(mesh_a.bounds[0], mesh_b.bounds[0])
        bounds_max = np.maximum(mesh_a.bounds[1], mesh_b.bounds[1])
        pitch = max(bounds_max - bounds_min) / resolution

        x = np.arange(bounds_min[0], bounds_max[0], pitch)
        y = np.arange(bounds_min[1], bounds_max[1], pitch)
        z = np.arange(bounds_min[2], bounds_max[2], pitch)
        grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)

        inside_a = mesh_a.contains(grid)
        inside_b = mesh_b.contains(grid)

        intersection = np.logical_and(inside_a, inside_b).sum()
        union = np.logical_or(inside_a, inside_b).sum()

        if union == 0:
            return 0.0
        return float(intersection / union * 100.0)
    except Exception:
        return -1.0


def compute_metrics(baseline_mesh: trimesh.Trimesh, variant_mesh: trimesh.Trimesh) -> dict:
    """Compute all quality metrics for a variant vs baseline."""
    return {
        "cd_pct": chamfer_distance(baseline_mesh, variant_mesh),
        "f_score_1pct": f_score(baseline_mesh, variant_mesh, tau_pct=1.0),
        "f_score_2pct": f_score(baseline_mesh, variant_mesh, tau_pct=2.0),
        "volume_iou": volume_iou(baseline_mesh, variant_mesh),
        "verts_ratio": variant_mesh.vertices.shape[0] / max(baseline_mesh.vertices.shape[0], 1),
    }


def status_for_metrics(m: dict) -> str:
    """Determine PASS/MARGINAL/FAIL based on CD and F-Score thresholds.

    Volume IoU is informational only (marching cubes meshes often aren't watertight).
    """
    cd = m["cd_pct"]
    f1 = m["f_score_1pct"]
    f2 = m["f_score_2pct"]

    if cd < 0.8 and f1 > 85.0 and f2 > 95.0:
        return "PASS"
    elif cd > 1.5 or f1 < 50.0 or f2 < 75.0:
        return "FAIL"
    else:
        return "MARGINAL"


def format_report(variant_name: str, results: list[dict]) -> str:
    """Format a human-readable report table."""
    lines = []
    lines.append(f"=== Reconstruction Quality Report ===")
    lines.append(f"Variant: {variant_name}")
    lines.append(f"Test set: {len(results)} images")
    lines.append("")
    lines.append(f"{'Image':<30} | {'CD (%)':>7} | {'F@1%':>6} | {'F@2%':>6} | {'VolIoU':>6} | {'Verts':>6} | {'Latency':>7} | {'Status':>8}")
    lines.append("-" * 105)

    cds, f1s, f2s, ious, vrs, lats = [], [], [], [], [], []
    statuses = []
    for r in results:
        m = r["metrics"]
        s = status_for_metrics(m)
        statuses.append(s)
        cds.append(m["cd_pct"])
        f1s.append(m["f_score_1pct"])
        f2s.append(m["f_score_2pct"])
        if m["volume_iou"] >= 0:
            ious.append(m["volume_iou"])
        vrs.append(m["verts_ratio"])
        lat = r.get("latency")
        lats.append(lat if lat is not None else 0)
        iou_str = f"{m['volume_iou']:6.1f}" if m["volume_iou"] >= 0 else "  N/A "
        lat_str = f"{lat:6.3f}s" if lat is not None else "   N/A"
        lines.append(
            f"{r['image']:<30} | {m['cd_pct']:7.3f} | {m['f_score_1pct']:6.1f} | {m['f_score_2pct']:6.1f} | "
            f"{iou_str} | {m['verts_ratio']:5.2f}x | {lat_str} | {s:>8}"
        )

    lines.append("-" * 105)
    iou_mean = np.mean(ious) if ious else -1
    iou_std = np.std(ious) if ious else 0
    has_latency = any(r.get("latency") is not None for r in results)
    lat_mean_str = f"{np.mean(lats):6.3f}s" if has_latency else "   N/A"
    lat_std_str = f"{np.std(lats):6.3f}s" if has_latency else "   N/A"
    lines.append(
        f"{'MEAN':<30} | {np.mean(cds):7.3f} | {np.mean(f1s):6.1f} | {np.mean(f2s):6.1f} | "
        f"{iou_mean:6.1f} | {np.mean(vrs):5.2f}x | {lat_mean_str} |"
    )
    lines.append(
        f"{'STD':<30} | {np.std(cds):7.3f} | {np.std(f1s):6.1f} | {np.std(f2s):6.1f} | "
        f"{iou_std:6.1f} | {np.std(vrs):5.2f}x | {lat_std_str} |"
    )

    pass_count = statuses.count("PASS")
    marg_count = statuses.count("MARGINAL")
    fail_count = statuses.count("FAIL")
    overall = "PASS" if fail_count == 0 and marg_count == 0 else ("FAIL" if fail_count > 0 else "MARGINAL")
    lines.append(f"\nOverall: {overall} ({pass_count} pass, {marg_count} marginal, {fail_count} fail)")

    return "\n".join(lines)


def run_variant(
    model: TSR,
    device: str,
    variant_name: str,
    test_images: list[Path],
    measure_latency: bool = True,
    tome_ratio: Optional[float] = None,
    tome_layers: Optional[list[int]] = None,
    image_prune_ratio: Optional[float] = None,
) -> list[dict]:
    """Run a variant on all test images and compute metrics vs cached baseline."""
    from tome_patch import apply_tome, apply_image_tome

    if tome_ratio is not None:
        apply_tome(model.backbone, merge_ratio=tome_ratio, merge_layers=tome_layers or [4, 8, 12])
    if image_prune_ratio is not None:
        apply_image_tome(model, prune_ratio=image_prune_ratio)

    results = []
    for img_path in test_images:
        rel = str(img_path.relative_to(Path(__file__).parent / "test_images"))
        print(f"  [{variant_name}] {rel} ...", end="", flush=True)

        # load cached baseline mesh
        baseline_path = BASELINE_CACHE / f"{img_path.stem}.obj"
        if not baseline_path.exists():
            print(f" SKIP (no baseline cache)")
            continue
        baseline_mesh = trimesh.load(str(baseline_path), process=False)

        # run variant
        variant_mesh = run_inference(model, img_path, device)

        # compute metrics
        metrics = compute_metrics(baseline_mesh, variant_mesh)

        # measure latency
        latency = None
        if measure_latency:
            latency = time_inference(model, img_path, device, n_runs=3)

        # save variant mesh
        variant_dir = RESULTS_DIR / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_mesh.export(str(variant_dir / f"{img_path.stem}.obj"))

        result = {"image": rel, "metrics": metrics, "latency": latency}
        results.append(result)
        status = status_for_metrics(metrics)
        print(f" CD={metrics['cd_pct']:.3f}% F@1%={metrics['f_score_1pct']:.1f} IoU={metrics['volume_iou']:.1f} [{status}]")

    # restore original forwards
    if image_prune_ratio is not None:
        from tome_patch import remove_image_tome
        remove_image_tome(model)
    if tome_ratio is not None:
        from tome_patch import remove_tome
        remove_tome(model.backbone)

    return results



def run_onnx_variant(
    onnx_path: Path,
    model: TSR,
    device: str,
    variant_name: str,
    test_images: list[Path],
    measure_latency: bool = True,
    decoder_path: Path | None = None,
) -> list[dict]:
    """Run an ONNX variant on all test images and compare to cached baseline.

    If decoder_path is provided, the decoder ONNX model is used for mesh extraction
    (full ONNX pipeline validation), otherwise PyTorch decoder is used.
    """
    import onnxruntime as ort

    print(f"  Loading ONNX: {onnx_path.name}")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    decoder_session = None
    if decoder_path is not None:
        print(f"  Loading decoder ONNX: {decoder_path.name}")
        decoder_session = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])

    results = []
    for img_path in test_images:
        rel = str(img_path.relative_to(Path(__file__).parent / "test_images"))
        print(f"  [{variant_name}] {rel} ...", end="", flush=True)

        baseline_path = BASELINE_CACHE / f"{img_path.stem}.obj"
        if not baseline_path.exists():
            print(f" SKIP (no baseline cache)")
            continue
        baseline_mesh = trimesh.load(str(baseline_path), process=False)

        variant_mesh = run_onnx_inference(
            session, model, img_path, device, decoder_session=decoder_session
        )
        metrics = compute_metrics(baseline_mesh, variant_mesh)

        latency = None
        if measure_latency:
            latency = time_onnx_inference(session, img_path, n_runs=3)

        variant_dir = RESULTS_DIR / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_mesh.export(str(variant_dir / f"{img_path.stem}.obj"))

        result = {"image": rel, "metrics": metrics, "latency": latency}
        results.append(result)
        status = status_for_metrics(metrics)
        print(f" CD={metrics['cd_pct']:.3f}% F@1%={metrics['f_score_1pct']:.1f} IoU={metrics['volume_iou']:.1f} [{status}]")

    return results


def generate_baseline(model: TSR, device: str, test_images: list[Path]):
    """Generate and cache baseline meshes."""
    BASELINE_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"Generating baseline meshes ({len(test_images)} images)...")
    for img_path in test_images:
        out_path = BASELINE_CACHE / f"{img_path.stem}.obj"
        if out_path.exists():
            print(f"  [cached] {img_path.stem}")
            continue
        print(f"  [baseline] {img_path.stem} ...", end="", flush=True)
        mesh = run_inference(model, img_path, device)
        mesh.export(str(out_path))
        print(f" {mesh.vertices.shape[0]} verts")


def save_results(variant_name: str, results: list[dict], report: str):
    """Save results as JSON and report as text."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"{variant_name}_{ts}.json"
    txt_path = RESULTS_DIR / f"{variant_name}_{ts}.txt"

    with open(json_path, "w") as f:
        json.dump({"variant": variant_name, "timestamp": ts, "results": results}, f, indent=2)
    with open(txt_path, "w") as f:
        f.write(report)

    print(f"\nResults saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="TripoSR reconstruction quality comparison")
    parser.add_argument("--baseline", action="store_true", help="Generate/cache baseline meshes")
    parser.add_argument("--tome", nargs="+", type=float, help="Test ToMe at given merge ratios")
    parser.add_argument("--tome-layers", nargs="+", type=int, default=[4, 8, 12], help="ToMe merge layers")
    parser.add_argument("--image-prune", nargs="+", type=float, help="Test image token pruning at given ratios")
    parser.add_argument("--onnx", nargs="+", type=Path, help="Test ONNX model variant(s)")
    parser.add_argument("--onnx-decoder", type=Path, help="Decoder ONNX model for full-pipeline validation")
    parser.add_argument("--all", action="store_true", help="Run all available variants")
    parser.add_argument("--e2e", action="store_true",
                        help="Full rembg+triposr+decoder ONNX pipeline on Unity test images (raw + RGBA)")
    parser.add_argument("--e2e-images", nargs="*", type=Path,
                        help="Specific images for --e2e (default: Unity test set)")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, mps, cuda")
    parser.add_argument("--no-latency", action="store_true", help="Skip latency measurement")
    args = parser.parse_args()

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    test_images = collect_test_images()
    print(f"Test set: {len(test_images)} images, device: {device}")

    model = load_model(device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # always ensure baseline exists
    generate_baseline(model, device, test_images)

    if args.baseline and not args.tome and not args.onnx and not args.all:
        print("\nBaseline cached. Use --tome or --all to run comparisons.")
        return

    variants_to_run = []

    if args.tome:
        for ratio in args.tome:
            variants_to_run.append({
                "name": f"tome_r{ratio}_L{'_'.join(map(str, args.tome_layers))}",
                "tome_ratio": ratio, "tome_layers": args.tome_layers,
            })

    if args.image_prune:
        for ratio in args.image_prune:
            variants_to_run.append({
                "name": f"imgprune_{ratio}",
                "image_prune_ratio": ratio,
            })
        if args.tome:
            for tr in args.tome:
                for ir in args.image_prune:
                    variants_to_run.append({
                        "name": f"tome_r{tr}_L{'_'.join(map(str, args.tome_layers))}_imgprune_{ir}",
                        "tome_ratio": tr, "tome_layers": args.tome_layers,
                        "image_prune_ratio": ir,
                    })

    onnx_variants = []

    if args.onnx:
        for onnx_path in args.onnx:
            onnx_variants.append(onnx_path)

    if args.all:
        for ratio in [0.1, 0.2, 0.3]:
            variants_to_run.append({
                "name": f"tome_r{ratio}_L4_8_12",
                "tome_ratio": ratio, "tome_layers": [4, 8, 12],
            })
        models_dir = Path(__file__).parent / "models"
        for onnx_name in ["triposr_fp32.onnx", "triposr_fp16.onnx", "triposr_int8.onnx"]:
            p = models_dir / onnx_name
            if p.exists():
                onnx_variants.append(p)

    for variant in variants_to_run:
        name = variant.pop("name")
        print(f"\n--- Variant: {name} ---")
        results = run_variant(
            model, device, name, test_images,
            measure_latency=not args.no_latency,
            **variant,
        )
        if results:
            report = format_report(name, results)
            print(f"\n{report}")
            save_results(name, results, report)

    for onnx_path in onnx_variants:
        name = f"onnx_{onnx_path.stem}"
        if args.onnx_decoder:
            name += "+decoder"
        print(f"\n--- Variant: {name} ---")
        results = run_onnx_variant(
            onnx_path, model, device, name, test_images,
            measure_latency=not args.no_latency,
            decoder_path=args.onnx_decoder,
        )
        if results:
            report = format_report(name, results)
            print(f"\n{report}")
            save_results(name, results, report)

    # --- E2E: full rembg + triposr + decoder ONNX pipeline ---
    if args.e2e:
        import onnxruntime as ort

        models_dir = Path(__file__).parent / "models"
        rembg_path = models_dir / "u2netp.onnx"
        triposr_path = models_dir / "triposr_fp32.onnx"
        decoder_path = models_dir / "nerf_decoder.onnx"

        for p in [rembg_path, triposr_path, decoder_path]:
            if not p.exists():
                print(f"ERROR: Missing model: {p}")
                sys.exit(1)

        print(f"\n--- E2E: rembg + triposr_fp32 + nerf_decoder (full ONNX) ---")
        rembg_sess = ort.InferenceSession(str(rembg_path), providers=["CPUExecutionProvider"])
        triposr_sess = ort.InferenceSession(str(triposr_path), providers=["CPUExecutionProvider"])
        decoder_sess = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])

        # Collect test images: use --e2e-images if provided, else the Unity test set
        if args.e2e_images:
            e2e_images = [p for p in args.e2e_images if p.exists()]
        else:
            # Unity test images: mix of raw (needs rembg) and RGBA (already processed)
            test_dir = Path(__file__).parent / "test_images"
            unity_names = [
                ("novel", "backpack_raw.jpg"),
                ("examples", "chair.png"),
                ("novel", "clock_raw.jpg"),
                ("examples", "hamburger.png"),
                ("examples", "robot.png"),
                ("novel", "shoe_raw.jpg"),
            ]
            e2e_images = [test_dir / sub / name for sub, name in unity_names if (test_dir / sub / name).exists()]

        if not e2e_images:
            print("No e2e test images found!")
        else:
            print(f"Test images ({len(e2e_images)}):")
            for p in e2e_images:
                print(f"  {p.name} ({'raw→rembg' if p.suffix in ('.jpg', '.jpeg') else 'RGBA'})")

            results = []
            for img_path in e2e_images:
                rel = img_path.name
                print(f"  [e2e] {rel} ...", end="", flush=True)

                # Generate baseline from the standard pipeline (RGBA/nobg input)
                # For raw images, use the corresponding nobg version for baseline
                if "_raw" in img_path.stem:
                    nobg_name = img_path.stem.replace("_raw", "_nobg") + ".png"
                    nobg_path = img_path.parent / nobg_name
                    if nobg_path.exists():
                        baseline_src = nobg_path
                    else:
                        print(f" SKIP (no nobg baseline for {img_path.name})")
                        continue
                else:
                    baseline_src = img_path

                baseline_cache = BASELINE_CACHE / f"{baseline_src.stem}.obj"
                if not baseline_cache.exists():
                    print(f" generating baseline...", end="", flush=True)
                    baseline_mesh = run_inference(model, baseline_src, device)
                    BASELINE_CACHE.mkdir(parents=True, exist_ok=True)
                    baseline_mesh.export(str(baseline_cache))
                else:
                    baseline_mesh = trimesh.load(str(baseline_cache), process=False)

                # Run full e2e ONNX
                t0 = time.perf_counter()
                e2e_mesh = run_e2e_onnx_inference(
                    rembg_sess, triposr_sess, decoder_sess, model, img_path, device)
                elapsed = time.perf_counter() - t0

                metrics = compute_metrics(baseline_mesh, e2e_mesh)

                variant_dir = RESULTS_DIR / "e2e_rembg_onnx"
                variant_dir.mkdir(parents=True, exist_ok=True)
                e2e_mesh.export(str(variant_dir / f"{img_path.stem}.obj"))

                result = {"image": rel, "metrics": metrics, "latency": elapsed}
                results.append(result)
                status = status_for_metrics(metrics)
                print(f" CD={metrics['cd_pct']:.3f}% F@1%={metrics['f_score_1pct']:.1f} "
                      f"IoU={metrics['volume_iou']:.1f} [{status}] {elapsed:.1f}s")

            if results:
                report = format_report("e2e_rembg_onnx", results)
                print(f"\n{report}")
                save_results("e2e_rembg_onnx", results, report)


if __name__ == "__main__":
    main()
