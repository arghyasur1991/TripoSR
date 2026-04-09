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
            volume: [B, C, D, D, D]
        """
        B, N, C, h, w = features.shape
        V = self.volume_size
        P = V * V * V

        ones = torch.ones(P, 1, device=features.device, dtype=features.dtype)
        pts_h = torch.cat([self.voxel_centers, ones], dim=-1)  # [P, 4]

        # [B, N, 3, 4] x [P, 4]^T → [B, N, 3, P] → transpose → [B, N, P, 3]
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
        volume = sampled.sum(dim=1) / count.squeeze(1).unsqueeze(1)

        return volume.reshape(B, C, V, V, V)


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


def load_model(checkpoint_path: str, device: str = 'cpu') -> MVReconModel:
    """Load trained MVReconModel from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    model = MVReconModel(volume_size=32, feat_channels=128, input_size=160)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def export_onnx(model: ExportableModel, output_path: str,
                n_views: int = 3, input_size: int = 160,
                opset: int = 17):
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


def main():
    parser = argparse.ArgumentParser(description="Export MVRecon to ONNX")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best.pt checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ONNX path (default: same dir as checkpoint)')
    parser.add_argument('--n_views', type=int, default=3)
    parser.add_argument('--input_size', type=int, default=160)
    parser.add_argument('--opset', type=int, default=18)
    parser.add_argument('--skip_verify', action='store_true')
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint).parent.parent
    output_dir = ckpt_dir if args.output is None else Path(args.output).parent
    output_path = args.output or str(ckpt_dir / 'mv_recon.onnx')

    print(f"Loading checkpoint: {args.checkpoint}")
    trained = load_model(args.checkpoint)
    exportable = ExportableModel(trained)

    print("\nModel info:")
    export_onnx(exportable, output_path, args.n_views, args.input_size, args.opset)

    print_model_info(exportable, output_path)
    write_camera_config(str(output_dir), args.input_size)

    if not args.skip_verify:
        print("\nVerifying ONNX vs PyTorch...")
        verify_onnx(output_path, exportable, args.n_views, args.input_size)

    print(f"\nDone. ONNX model: {output_path}")


if __name__ == '__main__':
    main()
