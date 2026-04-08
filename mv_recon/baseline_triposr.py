"""Generate TripoSR single-view baseline meshes for comparison.

Runs the full TripoSR model on one view per object and saves the resulting
OBJ mesh alongside the mv-recon output for easy visual comparison.
"""

import sys
sys.path.insert(0, '.')

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tsr.system import TSR
from .train import ALL_18_UIDS, OVERFIT_11_UIDS, RENDERS_DIR


def save_obj(path: str, vertices: np.ndarray, faces: np.ndarray,
             colors: np.ndarray = None):
    with open(path, 'w') as f:
        for i, v in enumerate(vertices):
            if colors is not None:
                c = colors[i]
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} "
                        f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
            else:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['overfit', 'full'], default='overfit')
    parser.add_argument('--output_dir', default='output/mv_recon_overfit')
    args = parser.parse_args()

    uids = OVERFIT_11_UIDS if args.mode == 'overfit' else ALL_18_UIDS
    renders_dir = Path(RENDERS_DIR)
    out_base = Path(args.output_dir) / 'meshes' if not args.output_dir.endswith('meshes') else Path(args.output_dir)

    device = 'cpu'
    print("Loading TripoSR model...")
    model = TSR.from_pretrained(
        'stabilityai/TripoSR',
        config_name='config.yaml',
        weight_name='model.ckpt'
    )
    model.to(device)
    model.eval()
    print("Model loaded.")

    for i, uid in enumerate(uids):
        obj_dir = renders_dir / uid
        if not obj_dir.exists():
            print(f"[{i+1}/{len(uids)}] {uid}: MISSING renders, skipping")
            continue

        out_dir = out_base / uid
        out_dir.mkdir(parents=True, exist_ok=True)

        # Use the first available image as input (azimuth=0, elevation=20)
        import json
        with open(obj_dir / 'cameras.json') as f:
            cams = json.load(f)

        fname = cams[1]['filename']  # elevation=20 for a good view
        img_path = obj_dir / fname
        if not img_path.exists():
            img_path = obj_dir / fname.replace('_alpha_', '')
        if not img_path.exists():
            print(f"[{i+1}/{len(uids)}] {uid}: no image found, skipping")
            continue

        print(f"[{i+1}/{len(uids)}] {uid}: processing {img_path.name}...")

        image = Image.open(img_path).convert('RGBA')
        # Composite onto white background (TripoSR expects this)
        bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(bg, image).convert('RGB')

        with torch.no_grad():
            scene_codes = model([image], device=device)
            mesh = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=256)[0]

        verts = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        colors = None
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            vc = np.array(mesh.visual.vertex_colors)
            colors = vc[:, :3].astype(np.float32) / 255.0

        save_obj(str(out_dir / 'triposr_baseline.obj'), verts, faces, colors)
        print(f"  Saved {len(verts)} verts, {len(faces)} faces")

    print("\nAll baselines saved.")


if __name__ == '__main__':
    main()
