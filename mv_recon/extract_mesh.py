"""Extract meshes from a trained multi-view reconstruction model.

Loads a checkpoint, runs inference on each object, extracts meshes via
marching cubes, and projects input view textures onto vertices.

Usage:
    python -m mv_recon.extract_mesh --checkpoint output/mv_recon_overfit/checkpoints/best.pt --mode overfit
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model import MVReconModel
from .dataset import ObjaverseMultiViewDataset
from .camera_utils import (
    blender_intrinsics,
    adjust_intrinsics_for_crop_resize,
    BLENDER_RENDER_W,
    BLENDER_RENDER_H,
)
from .train import RENDERS_DIR, VOXELS_DIR, ALL_18_UIDS, OVERFIT_11_UIDS

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def marching_cubes_extract(density: np.ndarray,
                           threshold: float = 0.5,
                           voxel_range: float = 0.55) -> tuple:
    """Extract mesh from occupancy volume via marching cubes.

    Returns vertices in (x, y, z) world space, faces, and normals.
    """
    from skimage.measure import marching_cubes

    verts, faces, normals, _ = marching_cubes(density, level=threshold)

    D = density.shape[0]
    verts = verts / (D - 1) * (2 * voxel_range) - voxel_range

    # marching_cubes returns verts in array-index order (dim0, dim1, dim2).
    # Volume layout is [z, y, x], so swap to (x, y, z) world space.
    verts = verts[:, [2, 1, 0]]
    normals = normals[:, [2, 1, 0]]

    return verts, faces, normals


def project_vertex_colors(vertices: np.ndarray, normals: np.ndarray,
                          input_images: np.ndarray, input_c2w: np.ndarray,
                          K: np.ndarray, image_size: int) -> np.ndarray:
    """Project input view pixels onto mesh vertices using known cameras.

    For each vertex, projects into every input view, samples the pixel color
    weighted by how head-on the view sees that vertex. Vertices not visible
    from any view get the average color of visible vertices.

    Args:
        vertices: [N, 3] world-space positions
        normals: [N, 3] vertex normals
        input_images: [V, 3, H, W] RGB in [0, 1] (NOT ImageNet-normalized)
        input_c2w: [V, 4, 4] Blender camera-to-world matrices
        K: [3, 3] intrinsic matrix (adjusted for image_size)
        image_size: spatial size of input images

    Returns:
        vertex_colors: [N, 3] RGB in [0, 1]
    """
    N = len(vertices)
    V = input_images.shape[0]
    H, W = input_images.shape[2], input_images.shape[3]

    flip = np.array([[1, 0, 0, 0], [0, -1, 0, 0],
                     [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    norm_n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

    accumulated_color = np.zeros((N, 3), dtype=np.float64)
    accumulated_weight = np.zeros(N, dtype=np.float64)

    for v_idx in range(V):
        c2w_cv = input_c2w[v_idx] @ flip
        w2c_cv = np.linalg.inv(c2w_cv)
        cam_pos = c2w_cv[:3, 3]

        R = w2c_cv[:3, :3]
        t = w2c_cv[:3, 3]
        pts_cam = (R @ vertices.T + t[:, None]).T  # [N, 3]

        valid_depth = pts_cam[:, 2] > 0.01

        u = fx * pts_cam[:, 0] / (pts_cam[:, 2] + 1e-8) + cx
        v_px = fy * pts_cam[:, 1] / (pts_cam[:, 2] + 1e-8) + cy

        in_frame = (u >= 0) & (u < W - 1) & (v_px >= 0) & (v_px < H - 1) & valid_depth

        view_dir = cam_pos[None, :] - vertices  # [N, 3]
        view_dir = view_dir / (np.linalg.norm(view_dir, axis=1, keepdims=True) + 1e-8)
        cos_angle = np.sum(view_dir * norm_n, axis=1)
        facing = cos_angle > 0.05

        valid = in_frame & facing
        if not valid.any():
            continue

        u_v = u[valid]
        v_v = v_px[valid]

        u0 = np.floor(u_v).astype(int)
        v0 = np.floor(v_v).astype(int)
        u1 = np.minimum(u0 + 1, W - 1)
        v1 = np.minimum(v0 + 1, H - 1)
        du = u_v - u0
        dv = v_v - v0

        img = input_images[v_idx]  # [3, H, W]
        c00 = img[:, v0, u0]  # [3, n_valid]
        c01 = img[:, v0, u1]
        c10 = img[:, v1, u0]
        c11 = img[:, v1, u1]

        sampled = (c00 * (1 - du) * (1 - dv) +
                   c01 * du * (1 - dv) +
                   c10 * (1 - du) * dv +
                   c11 * du * dv).T  # [n_valid, 3]

        weight = cos_angle[valid]
        accumulated_color[valid] += sampled * weight[:, None]
        accumulated_weight[valid] += weight

    seen = accumulated_weight > 0
    vertex_colors = np.full((N, 3), 0.5)
    if seen.any():
        vertex_colors[seen] = accumulated_color[seen] / accumulated_weight[seen, None]
        avg_color = vertex_colors[seen].mean(axis=0)
        vertex_colors[~seen] = avg_color

    return np.clip(vertex_colors, 0, 1).astype(np.float32)


def save_obj_with_colors(path: str, vertices: np.ndarray, faces: np.ndarray,
                         colors: np.ndarray):
    """Save OBJ with per-vertex colors embedded (v x y z r g b).

    Works with macOS Quick Look, MeshLab, and most viewers.
    """
    with open(path, 'w') as f:
        for v, c in zip(vertices, colors):
            r, g, b = np.clip(c, 0, 1)
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r:.4f} {g:.4f} {b:.4f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def save_input_views(dataset: ObjaverseMultiViewDataset, idx: int,
                     out_dir: Path, n_views: int = 4):
    """Save the input views as PNGs for visual reference."""
    item = dataset[idx]
    for v in range(min(n_views, item['input_images'].shape[0])):
        img = item['input_images'][v].numpy()
        img = img.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(out_dir / f'input_view_{v}.png')


def save_gt_mesh(uid: str, out_dir: Path, volume_size: int = 64,
                 voxel_range: float = 0.55):
    """Extract GT mesh from voxelized ground truth for comparison."""
    voxel_path = Path(VOXELS_DIR) / f"{uid}.npy"
    if not voxel_path.exists():
        return
    gt_occ = np.load(voxel_path)
    try:
        verts, faces, _ = marching_cubes_extract(gt_occ, threshold=0.5,
                                                 voxel_range=voxel_range)
        colors = np.full((len(verts), 3), 0.7)
        save_obj_with_colors(str(out_dir / 'gt_mesh.obj'), verts, faces, colors)
    except Exception as e:
        print(f"  GT mesh extraction failed: {e}")


def extract(args):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    uids = OVERFIT_11_UIDS if args.mode == 'overfit' else ALL_18_UIDS
    print(f"Mode: {args.mode}, Objects: {len(uids)}")

    dataset = ObjaverseMultiViewDataset(
        renders_dir=RENDERS_DIR,
        uids=uids,
        n_input_views=args.n_input_views,
        n_sup_views=8,
        image_size=args.image_size,
        voxels_dir=VOXELS_DIR,
    )

    model = MVReconModel(
        volume_size=args.volume_size,
        feat_channels=args.feat_channels,
        input_size=args.image_size,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    K_orig = blender_intrinsics()
    K_input = adjust_intrinsics_for_crop_resize(
        K_orig, BLENDER_RENDER_W, BLENDER_RENDER_H, args.image_size
    ).numpy()

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.parent.name == 'checkpoints':
        out_base = ckpt_path.parent.parent / 'meshes'
    else:
        out_base = Path(args.output_dir) / 'meshes'
    out_base.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_base}")

    for idx in range(len(dataset)):
        uid = dataset.objects[idx]['uid']
        obj_out = out_base / uid
        obj_out.mkdir(parents=True, exist_ok=True)
        print(f"\n[{idx+1}/{len(dataset)}] {uid}")

        item = dataset[idx]
        input_imgs = item['input_images'].unsqueeze(0).to(device)
        input_c2w = item['input_c2w'].unsqueeze(0).to(device)

        with torch.no_grad():
            density = model(input_imgs, input_c2w)

        occ = torch.sigmoid(density[0, 0]).cpu().numpy()

        try:
            verts, faces, normals = marching_cubes_extract(occ, threshold=0.5)
            print(f"  Mesh: {len(verts)} verts, {len(faces)} faces")
        except Exception as e:
            print(f"  Mesh extraction failed: {e}")
            continue

        # Denormalize input images for texture projection
        input_imgs_raw = item['input_images'].numpy()  # [V, 3, H, W]
        input_imgs_raw = input_imgs_raw * IMAGENET_STD[None, :, None, None] \
                       + IMAGENET_MEAN[None, :, None, None]
        input_imgs_raw = np.clip(input_imgs_raw, 0, 1)

        input_c2w_np = item['input_c2w'].numpy()  # [V, 4, 4]

        vcols = project_vertex_colors(
            verts, normals, input_imgs_raw, input_c2w_np,
            K_input, args.image_size,
        )
        n_colored = (vcols.max(axis=1) - vcols.min(axis=1) > 0.02).sum()
        print(f"  Texture: {n_colored}/{len(verts)} vertices colored from views")

        save_obj_with_colors(str(obj_out / 'mesh.obj'), verts, faces, vcols)

        save_input_views(dataset, idx, obj_out, n_views=4)
        save_gt_mesh(uid, obj_out, volume_size=args.volume_size)

        print(f"  Saved to {obj_out}")

    print(f"\nAll meshes saved to {out_base}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mode', choices=['overfit', 'full'], default='overfit')
    parser.add_argument('--volume_size', type=int, default=32)
    parser.add_argument('--feat_channels', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=160)
    parser.add_argument('--n_input_views', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='output/mv_recon_overfit')
    args = parser.parse_args()
    extract(args)


if __name__ == '__main__':
    main()
