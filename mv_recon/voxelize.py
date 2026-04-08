"""Voxelize GT meshes for direct 3D occupancy supervision.

Loads GLB meshes, applies the same normalization as the Blender render pipeline
(center at origin, scale longest bbox dim to 1.0), and voxelizes to a binary
occupancy grid.

Usage:
    python -m mv_recon.voxelize --volume_size 64
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh

from .train import ALL_18_UIDS

GLBS_DIR = Path.home() / "Downloads" / "mv_recon_data" / "glbs"
VOXELS_DIR = Path.home() / "Downloads" / "mv_recon_data" / "voxels"


def load_and_normalize(glb_path: str) -> trimesh.Trimesh:
    """Load GLB and normalize to match Blender render pipeline.

    Normalization (from render_views.py):
      1. Center bounding box at origin
      2. Scale so longest bbox dimension = 1.0
    Result fits in [-0.5, 0.5]^3.
    """
    scene = trimesh.load(glb_path)
    if isinstance(scene, trimesh.Scene):
        mesh = scene.to_geometry()
    else:
        mesh = scene

    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    mesh.vertices -= center

    max_dim = (bounds[1] - bounds[0]).max()
    if max_dim < 1e-6:
        raise ValueError(f"Degenerate mesh: max_dim={max_dim}")
    mesh.vertices /= max_dim

    return mesh


def voxelize_mesh(mesh: trimesh.Trimesh, volume_size: int = 64,
                  voxel_range: float = 0.55) -> np.ndarray:
    """Voxelize a normalized mesh into a binary occupancy grid.

    Uses trimesh voxelization at matching pitch, then snaps to our grid.
    The grid spans [-voxel_range, voxel_range]^3 to match the model's volume.
    Grid layout is [z, y, x] to match the model's volume indexing.
    """
    pitch = (2 * voxel_range) / volume_size

    try:
        vox = mesh.voxelized(pitch).fill()
    except Exception:
        vox = mesh.voxelized(pitch)

    grid = np.zeros((volume_size, volume_size, volume_size), dtype=np.float32)

    for pt in vox.points:
        ix = round((pt[0] + voxel_range) / (2 * voxel_range) * (volume_size - 1))
        iy = round((pt[1] + voxel_range) / (2 * voxel_range) * (volume_size - 1))
        iz = round((pt[2] + voxel_range) / (2 * voxel_range) * (volume_size - 1))
        if 0 <= ix < volume_size and 0 <= iy < volume_size and 0 <= iz < volume_size:
            grid[iz, iy, ix] = 1.0

    return grid


def main():
    parser = argparse.ArgumentParser(description="Voxelize GT meshes")
    parser.add_argument("--volume_size", type=int, default=64)
    parser.add_argument("--glbs_dir", type=str, default=str(GLBS_DIR))
    parser.add_argument("--output_dir", type=str, default=str(VOXELS_DIR))
    args = parser.parse_args()

    glbs_dir = Path(args.glbs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Voxelizing {len(ALL_18_UIDS)} meshes at {args.volume_size}^3")
    print(f"GLBs: {glbs_dir}")
    print(f"Output: {output_dir}")

    success = 0
    for i, uid in enumerate(ALL_18_UIDS):
        glb_path = glbs_dir / f"{uid}.glb"
        if not glb_path.exists():
            print(f"  [{i+1}/{len(ALL_18_UIDS)}] MISSING: {uid}")
            continue

        try:
            mesh = load_and_normalize(str(glb_path))
            grid = voxelize_mesh(mesh, args.volume_size)
            filled = grid.sum()
            total = grid.size
            pct = 100.0 * filled / total

            out_path = output_dir / f"{uid}.npy"
            np.save(out_path, grid)
            print(f"  [{i+1}/{len(ALL_18_UIDS)}] {uid}: "
                  f"{int(filled)}/{total} voxels filled ({pct:.1f}%)")
            success += 1
        except Exception as e:
            print(f"  [{i+1}/{len(ALL_18_UIDS)}] FAILED {uid}: {e}")

    print(f"\nDone: {success}/{len(ALL_18_UIDS)} voxelized")


if __name__ == "__main__":
    main()
