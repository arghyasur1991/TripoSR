"""Voxelize GT meshes for direct 3D occupancy supervision.

Uses Blender headlessly to import and normalize GLBs — guaranteeing the exact
same coordinate frame as the render pipeline (render_views.py).  Exports
normalized meshes as STL, then loads with trimesh for voxelization.

Usage:
    # Bulk (all UIDs from filtered_uids.json):
    python -m mv_recon.voxelize --uids filtered_uids.json --manifest object_manifest.json --workers 8

    # Small set (legacy):
    python -m mv_recon.voxelize --volume_size 64
"""

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import trimesh

DATA_DIR = Path.home() / "Downloads" / "mv_recon_data"
GLBS_DIR = DATA_DIR / "glbs"
VOXELS_DIR = DATA_DIR / "voxels"
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
                  n_surface_samples: int = 1_000_000) -> np.ndarray:
    """Voxelize a normalized mesh into a binary occupancy grid.

    Uses dense surface sampling only (no flood-fill) to keep memory bounded.
    1M surface samples at 64^3 gives excellent coverage without the 10GB+
    memory cost of trimesh's voxelized().fill() on complex meshes.

    The grid spans [-voxel_range, voxel_range]^3 to match the model's volume.
    Grid layout is [z, y, x] to match the model's volume indexing.
    """
    D = volume_size
    grid = np.zeros((D, D, D), dtype=np.float32)

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

    return grid


def _resolve_glb_path(uid: str, manifest: dict | None,
                      glbs_dir: Path, cache_dir: Path | None) -> Path | None:
    """Find the GLB file for a UID, checking manifest → cache_dir → glbs_dir."""
    if manifest and uid in manifest:
        raw = manifest[uid]
        # Manifest stores absolute GDrive FUSE paths; remap to local cache_dir
        if cache_dir:
            # Extract relative path after "objaverse_cache/"
            marker = "objaverse_cache/"
            idx = raw.find(marker)
            if idx >= 0:
                rel = raw[idx + len(marker):]
                local = cache_dir / rel
                if local.exists():
                    return local
        p = Path(raw)
        if p.exists():
            return p

    flat = glbs_dir / f"{uid}.glb"
    if flat.exists():
        return flat
    return None


def _process_one(args_tuple):
    """Worker function for multiprocessing: voxelize one UID."""
    uid, glb_path_str, output_dir_str, volume_size = args_tuple
    output_dir = Path(output_dir_str)
    out_path = output_dir / f"{uid}.npy"
    if out_path.exists():
        return uid, "skip", 0.0

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"vox_{uid[:8]}_"))
    tmp_stl = tmp_dir / f"{uid}.stl"
    try:
        ok = normalize_with_blender(glb_path_str, str(tmp_stl))
        if not ok or not tmp_stl.exists():
            return uid, "blender_fail", 0.0

        mesh = trimesh.load(str(tmp_stl))
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()

        grid = voxelize_mesh(mesh, volume_size)
        np.save(out_path, grid)
        pct = 100.0 * grid.sum() / grid.size
        return uid, "ok", pct
    except Exception as e:
        return uid, f"error: {e}", 0.0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Voxelize GT meshes via Blender")
    parser.add_argument("--volume_size", type=int, default=64)
    parser.add_argument("--glbs_dir", type=str, default=str(GLBS_DIR))
    parser.add_argument("--output_dir", type=str, default=str(VOXELS_DIR))
    parser.add_argument("--uids", type=str, default=None,
                        help="Path to filtered_uids.json (if omitted, uses legacy 18 UIDs)")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to object_manifest.json mapping UID→GLB path")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Local copy of objaverse_cache/ (rclone download)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel Blender workers")
    args = parser.parse_args()

    # Load UIDs
    if args.uids:
        with open(args.uids) as f:
            uids = json.load(f)
    else:
        from .train import ALL_18_UIDS
        uids = ALL_18_UIDS

    manifest = None
    if args.manifest:
        with open(args.manifest) as f:
            manifest = json.load(f)

    glbs_dir = Path(args.glbs_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build work list, skipping already-voxelized and missing GLBs
    work = []
    skipped = 0
    missing = 0
    for uid in uids:
        out_path = output_dir / f"{uid}.npy"
        if out_path.exists():
            skipped += 1
            continue
        glb_path = _resolve_glb_path(uid, manifest, glbs_dir, cache_dir)
        if glb_path is None:
            missing += 1
            continue
        work.append((uid, str(glb_path), str(output_dir), args.volume_size))

    print(f"Voxelizing {len(work)} meshes at {args.volume_size}^3 "
          f"(skipped {skipped} existing, {missing} GLB missing)")
    print(f"Workers: {args.workers}")
    print(f"Output: {output_dir}")

    t0 = time.time()
    success = 0
    failed_uids = []
    processed = 0

    def _handle_result(uid, status, pct):
        nonlocal success, processed
        processed += 1
        if status == "ok":
            success += 1
        elif status != "skip":
            failed_uids.append(uid)
            if len(failed_uids) <= 50:
                print(f"  FAIL {uid[:12]}.. {status}", flush=True)

        if processed % 50 == 0 or processed == len(work):
            elapsed = time.time() - t0
            rate = processed / max(elapsed, 1)
            eta = (len(work) - processed) / max(rate, 0.01)
            print(f"  [{processed}/{len(work)}] ok={success} fail={len(failed_uids)} "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

    if args.workers <= 1:
        for w in work:
            uid, status, pct = _process_one(w)
            _handle_result(uid, status, pct)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            futures = {exe.submit(_process_one, w): w[0] for w in work}
            for future in as_completed(futures):
                uid_key = futures[future]
                try:
                    uid, status, pct = future.result()
                    _handle_result(uid, status, pct)
                except Exception as e:
                    _handle_result(uid_key, f"executor_error: {e}", 0.0)

    elapsed = time.time() - t0
    print(f"\nDone: {success}/{len(work)} voxelized in {elapsed:.0f}s "
          f"(+{skipped} already existed)", flush=True)

    if failed_uids:
        fail_path = output_dir / "failed.txt"
        fail_path.write_text("\n".join(failed_uids) + "\n")
        print(f"Failed UIDs ({len(failed_uids)}) written to {fail_path}")


if __name__ == "__main__":
    main()
