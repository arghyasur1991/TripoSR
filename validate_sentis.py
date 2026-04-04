"""Validate Sentis against ONNX at each pipeline stage.

Usage:
    python validate_sentis.py \
        --tensor debug_reconstruction/python_preprocessed.bin \
        --onnx-dir onnx_models/ \
        --debug-dir debug_reconstruction/

Compares scene codes, density fields, and decoder outputs between ONNX and Unity/Sentis.
"""

import argparse
from pathlib import Path

import numpy as np


def load_bin(path: str, shape=None) -> np.ndarray:
    meta_path = path + ".meta.txt"
    meta = {}
    if Path(meta_path).exists():
        with open(meta_path) as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    meta[k] = v

    data = np.fromfile(path, dtype=np.float32)

    if shape is None and "shape" in meta:
        shape = tuple(int(x) for x in meta["shape"].split(","))

    if shape is not None:
        data = data.reshape(shape)

    return data


def print_stats(name, arr):
    print(f"  {name}: shape={arr.shape}, range=[{arr.min():.6f}, {arr.max():.6f}], "
          f"mean={arr.mean():.6f}, std={arr.std():.6f}")


def compare(name, a, b):
    diff = np.abs(a.flatten() - b.flatten())
    print(f"\n=== {name} ===")
    print_stats("A (ONNX)", a)
    print_stats("B (Sentis)", b)
    print(f"  Max diff: {diff.max():.8f}")
    print(f"  Mean diff: {diff.mean():.8f}")
    print(f"  RMSE: {np.sqrt((diff**2).mean()):.8f}")

    for tol in [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]:
        pct = (diff > tol).sum() / diff.size * 100
        print(f"  > {tol}: {pct:.2f}%")

    rel_diff = diff / (np.abs(a.flatten()) + 1e-8)
    print(f"  Max rel diff: {rel_diff.max():.6f}")
    print(f"  Mean rel diff: {rel_diff.mean():.6f}")

    cos_sim = np.dot(a.flatten(), b.flatten()) / (
        np.linalg.norm(a.flatten()) * np.linalg.norm(b.flatten()) + 1e-12)
    print(f"  Cosine similarity: {cos_sim:.8f}")


def run_onnx_forward(tensor_bin, onnx_dir):
    import onnxruntime as ort

    tensor = load_bin(tensor_bin, shape=(1, 3, 512, 512))
    print(f"\nInput tensor: shape={tensor.shape}, range=[{tensor.min():.4f}, {tensor.max():.4f}]")

    triposr_path = str(Path(onnx_dir) / "triposr_fp32.onnx")
    print(f"Loading ONNX TripoSR: {triposr_path}")
    session = ort.InferenceSession(triposr_path)
    input_name = session.get_inputs()[0].name
    scene_codes = session.run(None, {input_name: tensor})[0]
    print_stats("ONNX scene codes", scene_codes)
    return scene_codes, tensor


def run_onnx_density(scene_codes, onnx_dir, resolution=256):
    """Run the full Python pipeline: triplane sampling + ONNX decoder -> density field."""
    import torch
    import torch.nn.functional as F
    import onnxruntime as ort

    scene_codes_t = torch.from_numpy(scene_codes)
    scene_code = scene_codes_t[0]  # (3, 40, 64, 64)

    coords = torch.linspace(-0.5, 0.5, resolution)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)

    # Normalize to [-1, 1] for grid_sample (same as TripoSR renderer.query_triplane)
    normalized = positions * 2.0

    # Project onto three planes
    indices2D = torch.stack(
        (normalized[..., [0, 1]], normalized[..., [0, 2]], normalized[..., [1, 2]]),
        dim=-3,
    )  # (3, N, 2)

    # Sample triplane features
    grid = indices2D.unsqueeze(1)  # (3, 1, N, 2)
    triplane = scene_code.unsqueeze(0).expand(3, -1, -1, -1) if scene_code.dim() == 3 else scene_code
    features = F.grid_sample(
        triplane, grid, align_corners=False, mode="bilinear"
    )  # (3, 40, 1, N)
    features = features.squeeze(2).permute(2, 0, 1).reshape(-1, 120)  # (N, 120)
    features_np = features.numpy()

    print(f"\nTriplane features: shape={features_np.shape}, "
          f"range=[{features_np.min():.4f}, {features_np.max():.4f}]")

    # Run ONNX decoder in chunks
    decoder_path = str(Path(onnx_dir) / "nerf_decoder.onnx")
    decoder = ort.InferenceSession(decoder_path)
    dec_input_name = decoder.get_inputs()[0].name

    chunk_size = 65536
    raw_outputs = []
    for i in range(0, features_np.shape[0], chunk_size):
        chunk = features_np[i:i + chunk_size]
        result = decoder.run(None, {dec_input_name: chunk})[0]
        raw_outputs.append(result)

    raw = np.concatenate(raw_outputs, axis=0)
    print(f"Decoder raw output: shape={raw.shape}, "
          f"range=[{raw.min():.4f}, {raw.max():.4f}]")

    # Apply density activation: trunc_exp(raw_density - 1)
    raw_density = raw[:, 0]
    density = np.exp(np.clip(raw_density - 1.0, -15.0, 15.0))

    print(f"Density field: range=[{density.min():.6f}, {density.max():.6f}], "
          f"mean={density.mean():.4f}")
    above_25 = (density > 25.0).sum()
    print(f"Voxels above threshold 25: {above_25} / {density.size} "
          f"({above_25 / density.size * 100:.1f}%)")

    return density, features_np, raw


def main():
    parser = argparse.ArgumentParser(description="Validate Sentis vs ONNX pipeline")
    parser.add_argument("--tensor", required=True,
                        help="Path to preprocessed tensor .bin (1,3,512,512)")
    parser.add_argument("--onnx-dir", required=True,
                        help="Directory containing triposr_fp32.onnx and nerf_decoder.onnx")
    parser.add_argument("--debug-dir", default=None,
                        help="Directory with Sentis dumps (sentis_scene_codes.bin, etc.)")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--full", action="store_true",
                        help="Run full density pipeline comparison (slower)")
    args = parser.parse_args()

    debug_dir = args.debug_dir or str(Path(args.tensor).parent)

    # Step 1: Run ONNX forward pass
    onnx_scene_codes, tensor = run_onnx_forward(args.tensor, args.onnx_dir)

    # Save ONNX scene codes for reference
    onnx_sc_path = str(Path(debug_dir) / "onnx_scene_codes.bin")
    onnx_scene_codes.astype(np.float32).tofile(onnx_sc_path)
    shape_str = ",".join(str(d) for d in onnx_scene_codes.shape)
    with open(onnx_sc_path + ".meta.txt", "w") as f:
        f.write(f"dtype=float32\nshape={shape_str}\n")
    print(f"Saved ONNX scene codes: {onnx_sc_path}")

    # Step 2: Compare with Sentis scene codes
    sentis_sc_path = str(Path(debug_dir) / "sentis_scene_codes.bin")
    if Path(sentis_sc_path).exists():
        sentis_codes = load_bin(sentis_sc_path)
        if sentis_codes.size == onnx_scene_codes.size:
            sentis_codes = sentis_codes.reshape(onnx_scene_codes.shape)
            compare("Scene Codes (ONNX vs Sentis .sentis)", onnx_scene_codes, sentis_codes)
        else:
            print(f"\n!! Size mismatch: ONNX={onnx_scene_codes.size}, Sentis={sentis_codes.size}")
    else:
        print(f"\n!! Sentis scene codes not found at {sentis_sc_path}")
        print("   Run 'Run from Python Tensor (.bin)' in Unity first to generate it.")

    # Also check ONNX-direct scene codes (bypasses .sentis serialization)
    onnx_direct_path = str(Path(debug_dir) / "sentis_onnx_direct_scene_codes.bin")
    if Path(onnx_direct_path).exists():
        direct_codes = load_bin(onnx_direct_path)
        if direct_codes.size == onnx_scene_codes.size:
            direct_codes = direct_codes.reshape(onnx_scene_codes.shape)
            compare("Scene Codes (ONNX vs Sentis ONNX-direct)", onnx_scene_codes, direct_codes)
        else:
            print(f"\n!! Size mismatch: ONNX={onnx_scene_codes.size}, Direct={direct_codes.size}")

    # Step 3: Compare density fields (if --full)
    if args.full:
        print(f"\n--- Running full ONNX pipeline at resolution {args.resolution} ---")
        onnx_density, onnx_features, onnx_raw = run_onnx_density(
            onnx_scene_codes, args.onnx_dir, args.resolution)

        # Save ONNX density for reference
        onnx_density_path = str(Path(debug_dir) / "onnx_density.bin")
        onnx_density.astype(np.float32).tofile(onnx_density_path)
        with open(onnx_density_path + ".meta.txt", "w") as f:
            N = args.resolution
            f.write(f"dtype=float32\nshape={N*N*N}\n"
                    f"note=flat array, Python ordering: x slowest (i*res*res+j*res+k)\n")
        print(f"Saved ONNX density: {onnx_density_path}")

        # Compare with Unity density dump
        unity_density_path = str(Path(debug_dir) / "sentis_density.bin")
        if Path(unity_density_path).exists():
            unity_density = load_bin(unity_density_path)

            # Unity stores density with x varying fastest: ix + iy*res + iz*res*res
            # Python stores with z varying fastest: i*res*res + j*res + k (where i=x, j=y, k=z)
            # Need to transpose for comparison
            N = args.resolution
            py_3d = onnx_density.reshape(N, N, N)   # [x, y, z] in Python's ij ordering
            unity_3d_flat = unity_density             # flat with x-fastest

            # Reshape Unity density to 3D: index = ix + iy*N + iz*N*N
            # So dim order when reshaped as (N,N,N) treating as [iz, iy, ix]
            unity_3d = unity_3d_flat.reshape(N, N, N)  # This is [iz, iy, ix] since ix varies fastest

            # Transpose to [ix, iy, iz] = [x, y, z] to match Python
            unity_3d_reordered = np.transpose(unity_3d, (2, 1, 0))  # [ix, iy, iz]

            compare("Density Field (ONNX vs Sentis, reordered)",
                    py_3d.flatten(), unity_3d_reordered.flatten())

            # Also check raw comparison without reorder
            print(f"\n  (For reference, raw flat comparison without axis reorder:)")
            diff_raw = np.abs(onnx_density - unity_density)
            print(f"  Raw max diff: {diff_raw.max():.6f}, mean: {diff_raw.mean():.6f}")
        else:
            print(f"\n!! Unity density not found at {unity_density_path}")
            print("   Density dump needs to be enabled in Unity pipeline.")

    # Step 4: Check Sentis density dump (even without --full)
    sentis_density_path = str(Path(debug_dir) / "sentis_density.bin")
    if Path(sentis_density_path).exists():
        d = load_bin(sentis_density_path)
        above = (d > 25.0).sum()
        print(f"\nSentis density stats: range=[{d.min():.4f}, {d.max():.4f}], "
              f"mean={d.mean():.4f}, above_25={above}/{d.size} ({above/d.size*100:.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
