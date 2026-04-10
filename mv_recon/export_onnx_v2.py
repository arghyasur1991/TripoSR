"""Export MVRecon model to ONNX for Unity/Quest deployment.

Creates an ExportableModel wrapper that accepts pre-computed w2c matrices
(OpenCV convention) instead of Blender c2w, eliminating torch.linalg.inv
from the ONNX graph.

Usage:
    python -m mv_recon.export_onnx_v2 --checkpoint output/mv_recon_overfit/.../best.pt
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MVReconModel
from .camera_utils import (
    blender_intrinsics,
    adjust_intrinsics_for_crop_resize,
    BLENDER_RENDER_W,
    BLENDER_RENDER_H,
)


def c2w_blender_to_w2c_cv(c2w_blender: torch.Tensor) -> torch.Tensor:
    """Convert Blender c2w [*, 4, 4] → OpenCV w2c [*, 3, 4].

    This runs on CPU before ONNX inference. The ONNX model expects
    the resulting w2c_cv as input.
    """
    flip = torch.tensor([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1],
    ], dtype=c2w_blender.dtype, device=c2w_blender.device)
    c2w_cv = c2w_blender @ flip
    w2c_cv = torch.linalg.inv(c2w_cv)
    return w2c_cv[..., :3, :]  # drop homogeneous row


class ExportableUnprojector(nn.Module):
    """Geometric unprojector that takes pre-computed w2c (no linalg.inv)."""

    def __init__(self, K: torch.Tensor, voxel_centers: torch.Tensor,
                 input_size: int, volume_size: int):
        super().__init__()
        self.input_size = input_size
        self.volume_size = volume_size
        self.register_buffer('K', K)
        self.register_buffer('voxel_centers', voxel_centers)

    def forward(self, features: torch.Tensor,
                w2c_cv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N, C, h, w]
            w2c_cv:   [B, N, 3, 4] pre-computed world-to-camera (OpenCV)
        Returns:
            volume: [B, 2*C, D, D, D] mean+variance fused volume
        """
        B, N, C, h, w = features.shape
        V = self.volume_size
        P = V * V * V

        ones = torch.ones(P, 1, device=features.device, dtype=features.dtype)
        pts_h = torch.cat([self.voxel_centers, ones], dim=-1)  # [P, 4]

        pts_cam = torch.einsum('bnij,pj->bnpi', w2c_cv, pts_h)

        depth = pts_cam[..., 2]
        safe_depth = depth.clamp(min=0.01)

        u = self.K[0, 0] * pts_cam[..., 0] / safe_depth + self.K[0, 2]
        v = self.K[1, 1] * pts_cam[..., 1] / safe_depth + self.K[1, 2]

        u_norm = 2.0 * u / self.input_size - 1.0
        v_norm = 2.0 * v / self.input_size - 1.0

        valid = ((depth > 0.1) &
                 (u_norm > -1) & (u_norm < 1) &
                 (v_norm > -1) & (v_norm < 1))

        grid = torch.stack([u_norm, v_norm], dim=-1).reshape(B * N, P, 1, 2)
        feat_flat = features.reshape(B * N, C, h, w)

        sampled = F.grid_sample(
            feat_flat, grid, mode='bilinear',
            padding_mode='zeros', align_corners=False
        ).squeeze(-1).reshape(B, N, C, P)

        sampled = sampled * valid.unsqueeze(2).float()

        count = valid.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        count_bc = count.squeeze(1).unsqueeze(1)

        mean = sampled.sum(dim=1) / count_bc
        sq_mean = (sampled ** 2).sum(dim=1) / count_bc
        var = (sq_mean - mean ** 2).clamp(min=0)

        volume = torch.cat([mean, var], dim=1)  # [B, 2C, P]
        return volume.reshape(B, 2 * C, V, V, V)


class ExportableModel(nn.Module):
    """ONNX-exportable wrapper around MVReconModel.

    Accepts pre-computed w2c_cv [B, N, 3, 4] instead of c2w_blender [B, N, 4, 4],
    removing torch.linalg.inv from the graph.
    """

    def __init__(self, trained_model: MVReconModel):
        super().__init__()
        self.encoder = trained_model.encoder
        self.refiner = trained_model.refiner
        self.head = trained_model.head

        up = trained_model.unprojector
        self.unprojector = ExportableUnprojector(
            K=up.K, voxel_centers=up.voxel_centers,
            input_size=up.input_size, volume_size=up.volume_size,
        )

    def forward(self, images: torch.Tensor,
                w2c_cv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: [B, N, 3, H, W] ImageNet-normalized
            w2c_cv: [B, N, 3, 4] pre-computed world-to-camera (OpenCV)
        Returns:
            density: [B, 1, 64, 64, 64] logits
            color:   [B, 3, 64, 64, 64] logits
        """
        B, N, C, H, W = images.shape
        feats = self.encoder(images.reshape(B * N, C, H, W))
        feat_ch, fh, fw = feats.shape[1], feats.shape[2], feats.shape[3]
        feats = feats.reshape(B, N, feat_ch, fh, fw)

        volume = self.unprojector(feats, w2c_cv)
        volume = self.refiner(volume)
        return self.head(volume)


def load_model(checkpoint_path: str, device: str = 'cpu',
               volume_size: int = 32, feat_channels: int = 128,
               input_size: int = 160) -> MVReconModel:
    """Load trained MVReconModel from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    model = MVReconModel(volume_size=volume_size, feat_channels=feat_channels,
                         input_size=input_size)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def export_onnx(model: ExportableModel, output_path: str,
                n_views: int = 3, input_size: int = 160,
                opset: int = 21):
    """Export to ONNX with fixed shapes."""
    model.eval()
    device = next(model.parameters()).device

    dummy_images = torch.randn(1, n_views, 3, input_size, input_size, device=device)
    dummy_w2c = torch.randn(1, n_views, 3, 4, device=device)

    print(f"Exporting ONNX (opset {opset}, N={n_views}, size={input_size})...")

    torch.onnx.export(
        model,
        (dummy_images, dummy_w2c),
        output_path,
        opset_version=opset,
        input_names=['images', 'w2c_cv'],
        output_names=['density', 'color'],
        dynamic_axes=None,  # fixed shapes for Quest deployment
    )

    # Ensure all weights are embedded (torch may create .data sidecar)
    import onnx
    data_path = Path(str(output_path) + ".data")
    if data_path.exists():
        print("Re-saving with embedded weights (removing .data sidecar)...")
        m = onnx.load(str(output_path), load_external_data=True)
        onnx.save_model(m, str(output_path), save_as_external_data=False)
        data_path.unlink()

    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Exported: {output_path} ({file_size:.1f} MB)")
    return output_path


def verify_onnx(onnx_path: str, model: ExportableModel,
                n_views: int = 3, input_size: int = 160):
    """Verify ONNX output matches PyTorch within tolerance."""
    import onnxruntime as ort

    device = next(model.parameters()).device
    model.eval()

    torch.manual_seed(42)
    dummy_images = torch.randn(1, n_views, 3, input_size, input_size, device=device)
    dummy_w2c = torch.randn(1, n_views, 3, 4, device=device)

    with torch.no_grad():
        pt_density, pt_color = model(dummy_images, dummy_w2c)

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_out = sess.run(None, {
        'images': dummy_images.cpu().numpy(),
        'w2c_cv': dummy_w2c.cpu().numpy(),
    })
    ort_density, ort_color = ort_out

    d_diff = np.abs(pt_density.cpu().numpy() - ort_density).max()
    c_diff = np.abs(pt_color.cpu().numpy() - ort_color).max()

    tol = 5e-3
    print(f"Verification — density max diff: {d_diff:.6f}, color max diff: {c_diff:.6f}")
    ok = d_diff < tol and c_diff < tol
    print(f"{'PASS' if ok else 'FAIL'}: tolerance {tol}")
    return ok


def print_model_info(model: ExportableModel, onnx_path: str):
    """Print parameter counts and ONNX op breakdown."""
    import onnx

    total = sum(p.numel() for p in model.parameters())
    for name, mod in [('encoder', model.encoder),
                      ('unprojector', model.unprojector),
                      ('refiner', model.refiner),
                      ('head', model.head)]:
        n = sum(p.numel() for p in mod.parameters())
        print(f"  {name}: {n:,} params")
    print(f"  TOTAL: {total:,} params")

    onnx_model = onnx.load(onnx_path)
    op_counts: dict[str, int] = {}
    for node in onnx_model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    print("\nONNX op breakdown:")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op}: {count}")


def write_camera_config(output_dir: str, input_size: int = 160):
    """Write camera config JSON for C# preprocessing reference."""
    K_orig = blender_intrinsics()
    K = adjust_intrinsics_for_crop_resize(
        K_orig, BLENDER_RENDER_W, BLENDER_RENDER_H, input_size
    )
    config = {
        'input_size': input_size,
        'volume_size': 32,
        'output_resolution': 64,
        'voxel_range': 0.55,
        'blender_fov_deg': 50.0,
        'blender_render_w': BLENDER_RENDER_W,
        'blender_render_h': BLENDER_RENDER_H,
        'intrinsics_3x3': K.tolist(),
        'imagenet_mean': [0.485, 0.456, 0.406],
        'imagenet_std': [0.229, 0.224, 0.225],
        'flip_yz': [[1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, -1, 0],
                     [0, 0, 0, 1]],
        'notes': 'C# must: (1) c2w_cv = c2w_blender @ flip_yz, '
                 '(2) w2c_cv = inverse(c2w_cv), '
                 '(3) pass w2c_cv[:3,:] as [N,3,4] to ONNX model.',
    }
    out_path = Path(output_dir) / 'mv_recon_camera_config.json'
    with open(out_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Camera config: {out_path}")


def optimize_graph(input_path: str, output_path: str = None, level: str = "basic"):
    """Apply ORT graph optimizations (constant folding, CSE, dead node elimination).

    level="basic": safe for all models — no op fusions that break quantization.
    level="all": aggressive fusions (MatMul+Add → Gemm). Use only for final FP32.
    """
    import onnxruntime as ort

    if output_path is None:
        output_path = input_path

    print(f"  Optimizing graph ({level.upper()}): {Path(input_path).name}")
    opt_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if level == "all"
                 else ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)

    so = ort.SessionOptions()
    so.graph_optimization_level = opt_level
    so.optimized_model_filepath = output_path
    ort.InferenceSession(input_path, so, providers=["CPUExecutionProvider"])

    orig_size = Path(input_path).stat().st_size / (1024 * 1024)
    opt_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"    {opt_size:.1f}MB (was {orig_size:.1f}MB)")


def convert_fp16(fp32_path: str, fp16_path: str):
    """Convert FP32 ONNX model to FP16 using ORT transformer optimizer.

    This is more reliable than onnxconverter_common for mixed-precision
    graphs — it uses symbolic shape inference to handle Cast nodes correctly.
    """
    from onnxruntime.transformers.optimizer import optimize_model

    print(f"\nConverting to FP16: {fp16_path}")
    opt = optimize_model(fp32_path, opt_level=0)
    opt.convert_float_to_float16(
        use_symbolic_shape_infer=True,
        keep_io_types=True,
    )
    opt.save_model_to_file(fp16_path)

    fp32_size = Path(fp32_path).stat().st_size / (1024 * 1024)
    fp16_size = Path(fp16_path).stat().st_size / (1024 * 1024)
    print(f"FP16 exported: {fp16_path} ({fp16_size:.1f} MB, was {fp32_size:.1f} MB)")
    return fp16_path


class _MultiInputCalibrationReader:
    """Feeds multi-input samples (dict per sample) to ORT static quantization."""

    def __init__(self, samples: list[dict[str, np.ndarray]]):
        self.samples = samples
        self.index = 0

    def get_next(self):
        if self.index >= len(self.samples):
            return None
        sample = self.samples[self.index]
        self.index += 1
        return sample


def _collect_mvrecon_calibration_data(
    test_dir: str, n_views: int = 3, input_size: int = 160,
) -> list[dict[str, np.ndarray]]:
    """Collect real calibration samples from MVRecon test objects.

    Loads test images + camera poses, applies the same preprocessing as
    training (center-crop, composite on grey bg, resize, ImageNet normalize),
    then converts Blender c2w → OpenCV w2c for the ONNX model input format.
    """
    from PIL import Image
    from torchvision import transforms

    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"  WARNING: calibration dir not found: {test_dir}")
        return []

    imagenet_normalize = transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    samples = []
    for cam_file in sorted(test_path.glob("*/cameras.json")):
        obj_dir = cam_file.parent
        with open(cam_file) as f:
            cams = json.load(f)

        views = cams[:n_views]
        imgs = []
        c2w_list = []
        for v in views:
            img_path = obj_dir / v['filename']
            if not img_path.exists():
                break
            img = Image.open(img_path).convert('RGBA')
            w, h = img.size
            sq = min(w, h)
            left, top = (w - sq) // 2, (h - sq) // 2
            img = img.crop((left, top, left + sq, top + sq))
            r, g, b, a = img.split()
            rgb = Image.merge('RGB', (r, g, b))
            bg = Image.new('RGB', rgb.size, (127, 127, 127))
            rgb = Image.composite(rgb, bg, a)
            t = transforms.functional.to_tensor(
                transforms.functional.resize(rgb, (input_size, input_size)))
            t = imagenet_normalize(t)
            imgs.append(t)
            c2w_list.append(torch.tensor(v['pose'], dtype=torch.float32))

        if len(imgs) < n_views:
            continue

        images = torch.stack(imgs).unsqueeze(0)  # [1, N, 3, H, W]
        c2w = torch.stack(c2w_list).unsqueeze(0)  # [1, N, 4, 4]
        w2c = c2w_blender_to_w2c_cv(c2w)  # [1, N, 3, 4]

        samples.append({
            'images': images.numpy().astype(np.float32),
            'w2c_cv': w2c.numpy().astype(np.float32),
        })

    print(f"  Collected {len(samples)} calibration samples from {test_dir}")
    return samples


def _generate_random_calibration_data(
    n_samples: int = 16, n_views: int = 3, input_size: int = 160,
) -> list[dict[str, np.ndarray]]:
    """Fallback: generate random calibration data when no test images available."""
    samples = []
    for i in range(n_samples):
        torch.manual_seed(i)
        images = torch.randn(1, n_views, 3, input_size, input_size)
        w2c = torch.randn(1, n_views, 3, 4)
        samples.append({
            'images': images.numpy().astype(np.float32),
            'w2c_cv': w2c.numpy().astype(np.float32),
        })
    print(f"  Generated {n_samples} random calibration samples (no test images found)")
    return samples


def quantize_int8_weights_only(fp32_path: str, int8_path: str):
    """Weight-only INT8 quantization using QDQ format for ORT CPU EP.

    Quantizes only Conv weight tensors to INT8 (per-channel, symmetric),
    inserting DequantizeLinear nodes to convert back to FP32 before compute.
    Activations stay in FP32 — no calibration data needed.

    This avoids both:
    - ConvInteger (from quantize_dynamic) which CPU EP doesn't support
    - Activation quantization error accumulation (from quantize_static)
      which destroys accuracy in deep CNNs (72 Conv layers)

    Result: ~4x weight size reduction with near-zero accuracy loss.
    """
    import onnx
    from onnx import numpy_helper, TensorProto, helper

    print(f"\nQuantizing to INT8 (weight-only QDQ): {int8_path}")
    model = onnx.load(fp32_path)

    # Build lookup: initializer name → numpy array
    init_map = {}
    for init in model.graph.initializer:
        init_map[init.name] = init

    # Find Conv nodes and their weight input names
    conv_weight_names = set()
    for node in model.graph.node:
        if node.op_type == "Conv" and len(node.input) >= 2:
            conv_weight_names.add(node.input[1])

    quantized_count = 0
    new_initializers = []
    nodes_to_prepend = []

    for init in model.graph.initializer:
        if init.name not in conv_weight_names:
            new_initializers.append(init)
            continue

        arr = numpy_helper.to_array(init).astype(np.float32)
        if arr.ndim < 3:
            new_initializers.append(init)
            continue

        # Skip tiny output-head Conv layers (density=1ch, color=3ch) —
        # they're most sensitive to quantization and negligible in size.
        if arr.shape[0] <= 4:
            new_initializers.append(init)
            print(f"    Skipping head Conv: {init.name} (shape {arr.shape})")
            continue

        # Per-channel symmetric quantization along axis 0 (output channels)
        out_channels = arr.shape[0]
        scales = np.zeros(out_channels, dtype=np.float32)
        quantized = np.zeros_like(arr, dtype=np.int8)

        for c in range(out_channels):
            channel = arr[c].flatten()
            abs_max = max(float(np.abs(channel).max()), 1e-10)
            scale = abs_max / 127.0
            scales[c] = scale
            quantized[c] = np.clip(np.round(channel / scale), -127, 127).astype(
                np.int8).reshape(arr[c].shape)

        q_name = init.name + "_quantized"
        scale_name = init.name + "_scale"
        zp_name = init.name + "_zero_point"
        dq_output_name = init.name + "_dequantized"

        q_tensor = numpy_helper.from_array(quantized, name=q_name)
        scale_tensor = numpy_helper.from_array(scales, name=scale_name)
        zp_tensor = numpy_helper.from_array(
            np.zeros(out_channels, dtype=np.int8), name=zp_name)

        new_initializers.extend([q_tensor, scale_tensor, zp_tensor])

        dq_node = helper.make_node(
            "DequantizeLinear",
            inputs=[q_name, scale_name, zp_name],
            outputs=[dq_output_name],
            axis=0,
        )
        nodes_to_prepend.append(dq_node)

        # Rewrite Conv node to use dequantized output
        for node in model.graph.node:
            if node.op_type == "Conv" and len(node.input) >= 2:
                if node.input[1] == init.name:
                    node.input[1] = dq_output_name

        quantized_count += 1

    # Replace initializers and prepend DequantizeLinear nodes
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_initializers)

    for n in reversed(nodes_to_prepend):
        model.graph.node.insert(0, n)

    print(f"  Quantized {quantized_count} Conv weight tensors (per-channel symmetric)")

    onnx.save(model, int8_path)

    fp32_size = Path(fp32_path).stat().st_size / (1024 * 1024)
    int8_size = Path(int8_path).stat().st_size / (1024 * 1024)
    ratio = int8_size / fp32_size * 100
    print(f"INT8 weight-only exported: {int8_path} ({int8_size:.1f} MB, {ratio:.0f}% of FP32)")
    return int8_path


def validate_static_shapes(model_path: str) -> bool:
    """Verify all ONNX model I/O shapes are fully static (no symbolic dims).

    Critical for Quest/mobile deployment — dynamic shapes prevent XNNPACK
    graph optimizations and can cause runtime failures.
    """
    import onnx

    model = onnx.load(model_path)
    dynamic_found = []

    for tensor_list, kind in [(model.graph.input, "input"),
                               (model.graph.output, "output")]:
        for tensor in tensor_list:
            shape = tensor.type.tensor_type.shape
            if shape is None:
                dynamic_found.append((kind, tensor.name, "no shape info"))
                continue
            for i, dim in enumerate(shape.dim):
                if dim.dim_param:
                    dynamic_found.append((kind, tensor.name, f"axis {i} = '{dim.dim_param}'"))
                elif dim.dim_value == 0:
                    dynamic_found.append((kind, tensor.name, f"axis {i} = unknown(0)"))

    if dynamic_found:
        print(f"  FAIL static shapes: {Path(model_path).name}")
        for kind, name, detail in dynamic_found:
            print(f"    {kind} '{name}': {detail}")
        return False

    print(f"  PASS static shapes: {Path(model_path).name}")
    return True


def verify_variant(variant_path: str, ref_density: np.ndarray,
                   ref_color: np.ndarray, dummy_images_np: np.ndarray,
                   dummy_w2c_np: np.ndarray, label: str) -> dict:
    """Verify a variant against PyTorch reference using relative error.

    Returns metrics dict with tiered verdict: PASS (<1%), WARN (<5%),
    OK (<15%), FAIL (>15%).
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(variant_path, providers=['CPUExecutionProvider'])

    inp_images = dummy_images_np
    inp_w2c = dummy_w2c_np
    if sess.get_inputs()[0].type == "tensor(float16)":
        inp_images = dummy_images_np.astype(np.float16)
        inp_w2c = dummy_w2c_np.astype(np.float16)

    ort_out = sess.run(None, {
        'images': inp_images, 'w2c_cv': inp_w2c,
    })
    ort_density = ort_out[0].astype(np.float32)
    ort_color = ort_out[1].astype(np.float32)

    d_abs = np.abs(ref_density - ort_density)
    c_abs = np.abs(ref_color - ort_color)
    d_range = max(float(np.abs(ref_density).max()), 1e-8)
    c_range = max(float(np.abs(ref_color).max()), 1e-8)

    d_max_rel = float(d_abs.max()) / d_range
    c_max_rel = float(c_abs.max()) / c_range
    max_rel = max(d_max_rel, c_max_rel)

    d_mean_rel = float(d_abs.mean()) / d_range
    c_mean_rel = float(c_abs.mean()) / c_range
    mean_rel = max(d_mean_rel, c_mean_rel)

    size_mb = Path(variant_path).stat().st_size / (1024 * 1024)

    verdict = ("PASS" if max_rel < 0.01
               else "WARN" if max_rel < 0.05
               else "OK" if max_rel < 0.15
               else "FAIL")

    print(f"  [{label}] {size_mb:.1f}MB | "
          f"density max={d_abs.max():.4f} ({d_max_rel*100:.3f}%) | "
          f"color max={c_abs.max():.4f} ({c_max_rel*100:.3f}%) → {verdict}")

    return {
        'label': label, 'size_mb': size_mb,
        'max_rel': max_rel, 'mean_rel': mean_rel,
        'verdict': verdict,
    }


def main():
    parser = argparse.ArgumentParser(description="Export MVRecon to ONNX")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best.pt checkpoint')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same dir as checkpoint)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ONNX path (overrides output_dir for FP32)')
    parser.add_argument('--n_views', type=int, default=3)
    parser.add_argument('--input_size', type=int, default=160)
    parser.add_argument('--opset', type=int, default=21)
    parser.add_argument('--skip_verify', action='store_true')
    parser.add_argument('--variants', nargs='+',
                        choices=['fp32', 'fp16', 'int8', 'all'],
                        default=['fp32'],
                        help='Which precision variants to export')
    args = parser.parse_args()

    if 'all' in args.variants:
        args.variants = ['fp32', 'fp16', 'int8']

    ckpt_dir = Path(args.checkpoint).parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else ckpt_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = args.output or str(output_dir / 'mv_recon_fp32.onnx')

    print(f"Loading checkpoint: {args.checkpoint}")
    trained = load_model(args.checkpoint)
    exportable = ExportableModel(trained)

    # 1. Export FP32
    print("\nModel info:")
    export_onnx(exportable, fp32_path, args.n_views, args.input_size, args.opset)
    print_model_info(exportable, fp32_path)
    write_camera_config(str(output_dir), args.input_size)

    # 2. Graph optimization (constant folding, CSE, dead node elimination)
    optimize_graph(fp32_path)

    # 3. Static shape validation
    print("\nValidating shapes...")
    validate_static_shapes(fp32_path)

    # 4. Generate PyTorch reference for verification
    torch.manual_seed(42)
    device = next(exportable.parameters()).device
    dummy_images = torch.randn(1, args.n_views, 3, args.input_size, args.input_size, device=device)
    dummy_w2c = torch.randn(1, args.n_views, 3, 4, device=device)
    with torch.no_grad():
        pt_density, pt_color = exportable(dummy_images, dummy_w2c)
    ref_d = pt_density.cpu().numpy()
    ref_c = pt_color.cpu().numpy()
    img_np = dummy_images.cpu().numpy()
    w2c_np = dummy_w2c.cpu().numpy()

    results = []

    if not args.skip_verify:
        print("\nVerifying FP32 ONNX vs PyTorch...")
        results.append(verify_variant(
            fp32_path, ref_d, ref_c, img_np, w2c_np, 'FP32'))

    exported = {'fp32': fp32_path}

    if 'fp16' in args.variants:
        fp16_path = str(output_dir / 'mv_recon_fp16.onnx')
        convert_fp16(fp32_path, fp16_path)
        validate_static_shapes(fp16_path)
        exported['fp16'] = fp16_path
        if not args.skip_verify:
            try:
                results.append(verify_variant(
                    fp16_path, ref_d, ref_c, img_np, w2c_np, 'FP16'))
            except Exception as e:
                print(f"[FP16] CPU verification failed ({e}) — test on device")

    if 'int8' in args.variants:
        int8_path = str(output_dir / 'mv_recon_int8.onnx')
        quantize_int8_weights_only(fp32_path, int8_path)
        validate_static_shapes(int8_path)
        exported['int8'] = int8_path
        if not args.skip_verify:
            results.append(verify_variant(
                int8_path, ref_d, ref_c, img_np, w2c_np, 'INT8 QDQ'))

    # Summary table
    label_map = {'fp32': 'FP32', 'fp16': 'FP16', 'int8': 'INT8 QDQ'}
    print("\n=== Export Summary ===")
    print(f"{'Variant':<10} {'Size':>8} {'Max Rel%':>10} {'Verdict':>8}")
    print("-" * 40)
    for variant, path in exported.items():
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        lbl = label_map.get(variant, variant.upper())
        r = next((x for x in results if x['label'] == lbl), None)
        if r:
            print(f"  {lbl:<8} {size_mb:>7.1f}MB {r['max_rel']*100:>9.3f}% {r['verdict']:>8}")
        else:
            print(f"  {lbl:<8} {size_mb:>7.1f}MB {'':>9} {'N/A':>8}")
    print("Done.")


if __name__ == '__main__':
    main()
