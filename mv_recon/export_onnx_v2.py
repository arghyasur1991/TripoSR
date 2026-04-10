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


def convert_fp16(fp32_path: str, fp16_path: str):
    """Convert FP32 ONNX model to mixed-precision FP16.

    Uses onnx float16 converter with op_block_list to keep ops that are
    numerically sensitive (normalization, reductions) in FP32.
    """
    from onnxconverter_common import float16
    import onnx

    print(f"\nConverting to FP16: {fp16_path}")
    model = onnx.load(fp32_path)
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=True,
        op_block_list=['GroupNormalization', 'ReduceMean', 'GridSample'],
    )
    onnx.save_model(model_fp16, fp16_path, save_as_external_data=False)
    size_mb = Path(fp16_path).stat().st_size / (1024 * 1024)
    print(f"FP16 exported: {fp16_path} ({size_mb:.1f} MB)")
    return fp16_path


def quantize_int8(fp32_path: str, int8_path: str):
    """Dynamic INT8 quantization of ONNX model."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"\nQuantizing to INT8 (dynamic): {int8_path}")
    quantize_dynamic(
        fp32_path, int8_path,
        weight_type=QuantType.QUInt8,
        extra_options={"MatMulConstBOnly": False},
    )
    size_mb = Path(int8_path).stat().st_size / (1024 * 1024)
    print(f"INT8 exported: {int8_path} ({size_mb:.1f} MB)")
    return int8_path


def verify_variant(variant_path: str, ref_model: ExportableModel,
                   label: str, n_views: int = 3, input_size: int = 160,
                   tol: float = 0.05):
    """Verify a quantized variant against PyTorch reference."""
    import onnxruntime as ort

    device = next(ref_model.parameters()).device
    ref_model.eval()

    torch.manual_seed(42)
    dummy_images = torch.randn(1, n_views, 3, input_size, input_size, device=device)
    dummy_w2c = torch.randn(1, n_views, 3, 4, device=device)

    with torch.no_grad():
        pt_density, pt_color = ref_model(dummy_images, dummy_w2c)

    sess = ort.InferenceSession(variant_path, providers=['CPUExecutionProvider'])
    ort_out = sess.run(None, {
        'images': dummy_images.cpu().numpy(),
        'w2c_cv': dummy_w2c.cpu().numpy(),
    })

    d_diff = np.abs(pt_density.cpu().numpy() - ort_out[0]).max()
    c_diff = np.abs(pt_color.cpu().numpy() - ort_out[1]).max()

    print(f"[{label}] density max diff: {d_diff:.6f}, color max diff: {c_diff:.6f}")
    ok = d_diff < tol and c_diff < tol
    print(f"[{label}] {'PASS' if ok else 'FAIL'}: tolerance {tol}")
    return ok


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

    # Always export FP32 first (needed as base for other variants)
    print("\nModel info:")
    export_onnx(exportable, fp32_path, args.n_views, args.input_size, args.opset)
    print_model_info(exportable, fp32_path)
    write_camera_config(str(output_dir), args.input_size)

    if not args.skip_verify:
        print("\nVerifying FP32 ONNX vs PyTorch...")
        verify_onnx(fp32_path, exportable, args.n_views, args.input_size)

    exported = {'fp32': fp32_path}

    if 'fp16' in args.variants:
        fp16_path = str(output_dir / 'mv_recon_fp16.onnx')
        convert_fp16(fp32_path, fp16_path)
        exported['fp16'] = fp16_path
        # FP16 mixed-precision models can't be verified on CPU provider
        # (internal Cast node type mismatches). Verify on Quest with XNNPACK.
        print("[FP16] Skipping CPU verification — test on device (XNNPACK EP)")

    if 'int8' in args.variants:
        int8_path = str(output_dir / 'mv_recon_int8.onnx')
        quantize_int8(fp32_path, int8_path)
        exported['int8'] = int8_path
        if not args.skip_verify:
            verify_variant(int8_path, exportable, 'INT8',
                           args.n_views, args.input_size, tol=0.5)

    print("\n=== Export Summary ===")
    for variant, path in exported.items():
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"  {variant.upper():5s}: {path} ({size_mb:.1f} MB)")
    print("Done.")


if __name__ == '__main__':
    main()
