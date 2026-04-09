"""Voxelize GT meshes for direct 3D occupancy supervision.

Uses Blender headlessly to import and normalize GLBs — guaranteeing the exact
same coordinate frame as the render pipeline (render_views.py).  Exports
normalized meshes as STL, then loads with trimesh for voxelization.

Usage:
    python -m mv_recon.voxelize --volume_size 64
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from .train import ALL_18_UIDS

GLBS_DIR = Path.home() / "Downloads" / "mv_recon_data" / "glbs"
VOXELS_DIR = Path.home() / "Downloads" / "mv_recon_data" / "voxels"
BLENDER_PATH = "/Applications/Blender.app/Contents/MacOS/Blender"

# Blender script: import GLB, normalize exactly like render_views.py, export STL.
_BLENDER_NORMALIZE_SCRIPT = '''
import bpy
import os
import sys
import mathutils

mesh_path = "{mesh_path}"
output_stl = "{output_stl}"

bpy.ops.wm.read_homefile(use_empty=True)
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

ext = os.path.splitext(mesh_path)[1].lower()
try:
    if ext in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif ext == '.obj':
        bpy.ops.wm.obj_import(filepath=mesh_path)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=mesh_path)
    elif ext == '.stl':
        bpy.ops.wm.stl_import(filepath=mesh_path)
    elif ext == '.ply':
        bpy.ops.wm.ply_import(filepath=mesh_path)
    else:
        bpy.ops.import_scene.gltf(filepath=mesh_path)
except Exception as e:
    print(f"IMPORT_ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)

mesh_objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not mesh_objects:
    print("NO_MESH_OBJECTS", file=sys.stderr)
    sys.exit(1)

# Apply all transforms (matching render_views.py exactly)
bpy.ops.object.select_all(action='DESELECT')
for obj in mesh_objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = mesh_objects[0]
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Collective world-space bounding box
all_min = [float('inf')] * 3
all_max = [float('-inf')] * 3
for obj in mesh_objects:
    for v in obj.data.vertices:
        world_co = obj.matrix_world @ v.co
        for i in range(3):
            all_min[i] = min(all_min[i], world_co[i])
            all_max[i] = max(all_max[i], world_co[i])

bbox_center = mathutils.Vector(((all_min[i] + all_max[i]) / 2 for i in range(3)))
bbox_size = [all_max[i] - all_min[i] for i in range(3)]
max_dim = max(bbox_size)

if max_dim < 1e-6:
    print("DEGENERATE_OBJECT", file=sys.stderr)
    sys.exit(1)

scale_factor = 1.0 / max_dim
for obj in mesh_objects:
    for v in obj.data.vertices:
        world_co = obj.matrix_world @ v.co
        centered = world_co - bbox_center
        scaled = centered * scale_factor
        v.co = obj.matrix_world.inverted() @ scaled
    obj.data.update()

bpy.context.view_layer.update()

# Join all meshes into one for STL export
bpy.ops.object.select_all(action='DESELECT')
for obj in mesh_objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = mesh_objects[0]
if len(mesh_objects) > 1:
    bpy.ops.object.join()

bpy.ops.wm.stl_export(filepath=output_stl)
print("EXPORT_OK")
'''


def normalize_with_blender(glb_path: str, output_stl: str) -> bool:
    """Run Blender headlessly to normalize a GLB and export as STL."""
    script = _BLENDER_NORMALIZE_SCRIPT.format(
        mesh_path=str(glb_path).replace('\\', '/'),
        output_stl=str(output_stl).replace('\\', '/'),
    )

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [BLENDER_PATH, '--background', '--python', script_path],
            capture_output=True, text=True, timeout=120,
        )
        return 'EXPORT_OK' in result.stdout
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(script_path).unlink(missing_ok=True)


def voxelize_mesh(mesh: trimesh.Trimesh, volume_size: int = 64,
                  voxel_range: float = 0.55,
                  n_surface_samples: int = 500_000) -> np.ndarray:
    """Voxelize a normalized mesh into a binary occupancy grid.

    Two-pass approach:
      1. Dense surface sampling: scatter many points onto the mesh surface
         and mark their enclosing voxels. Captures thin structures reliably.
      2. Solid fill: also fills interior of watertight regions.
    Union of both passes gives the final grid.

    The grid spans [-voxel_range, voxel_range]^3 to match the model's volume.
    Grid layout is [z, y, x] to match the model's volume indexing.
    """
    D = volume_size
    pitch = (2 * voxel_range) / D
    grid = np.zeros((D, D, D), dtype=np.float32)

    # Pass 1: surface sampling — handles thin structures
    pts = mesh.sample(n_surface_samples)
    ix = np.round((pts[:, 0] + voxel_range) / (2 * voxel_range) * (D - 1)).astype(int)
    iy = np.round((pts[:, 1] + voxel_range) / (2 * voxel_range) * (D - 1)).astype(int)
    iz = np.round((pts[:, 2] + voxel_range) / (2 * voxel_range) * (D - 1)).astype(int)
    valid = (
        (ix >= 0) & (ix < D) &
        (iy >= 0) & (iy < D) &
        (iz >= 0) & (iz < D)
    )
    grid[iz[valid], iy[valid], ix[valid]] = 1.0

    # Pass 2: solid fill for watertight interior
    try:
        vox = mesh.voxelized(pitch).fill()
        for pt in vox.points:
            jx = round((pt[0] + voxel_range) / (2 * voxel_range) * (D - 1))
            jy = round((pt[1] + voxel_range) / (2 * voxel_range) * (D - 1))
            jz = round((pt[2] + voxel_range) / (2 * voxel_range) * (D - 1))
            if 0 <= jx < D and 0 <= jy < D and 0 <= jz < D:
                grid[jz, jy, jx] = 1.0
    except Exception:
        pass

    return grid


def main():
    parser = argparse.ArgumentParser(description="Voxelize GT meshes via Blender")
    parser.add_argument("--volume_size", type=int, default=64)
    parser.add_argument("--glbs_dir", type=str, default=str(GLBS_DIR))
    parser.add_argument("--output_dir", type=str, default=str(VOXELS_DIR))
    args = parser.parse_args()

    glbs_dir = Path(args.glbs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="voxelize_stl_"))
    print(f"Voxelizing {len(ALL_18_UIDS)} meshes at {args.volume_size}^3")
    print(f"GLBs: {glbs_dir}")
    print(f"Output: {output_dir}")
    print(f"Temp STLs: {tmp_dir}")

    success = 0
    for i, uid in enumerate(ALL_18_UIDS):
        glb_path = glbs_dir / f"{uid}.glb"
        if not glb_path.exists():
            print(f"  [{i+1}/{len(ALL_18_UIDS)}] MISSING: {uid}")
            continue

        stl_path = tmp_dir / f"{uid}.stl"

        # Step 1: Blender normalizes and exports STL
        ok = normalize_with_blender(str(glb_path), str(stl_path))
        if not ok or not stl_path.exists():
            print(f"  [{i+1}/{len(ALL_18_UIDS)}] BLENDER FAILED: {uid}")
            continue

        try:
            # Step 2: Load STL (coordinates already in Blender Z-up, normalized)
            mesh = trimesh.load(str(stl_path))
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.to_geometry()

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
            print(f"  [{i+1}/{len(ALL_18_UIDS)}] VOXELIZE FAILED {uid}: {e}")

    print(f"\nDone: {success}/{len(ALL_18_UIDS)} voxelized")


if __name__ == "__main__":
    main()
