#!/usr/bin/env python3
"""Compare Unity's preprocessed tensor against Python's preprocessing.

Usage:
    python compare_preprocessed.py \
        --unity-bin debug_reconstruction/unity_preprocessed.bin \
        --image Assets/Game/ObjectReconstruction/TestImages/hamburger.png \
        --onnx-dir Assets/Game/ObjectReconstruction/OnnxSource

Loads the Unity tensor dump (.bin + .meta.txt), runs the same image through
Python's preprocessing, and compares numerically. Optionally runs both through
ONNX TripoSR + decoder to isolate whether divergence is in preprocessing or inference.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def load_unity_tensor(bin_path: str) -> np.ndarray:
    meta_path = bin_path + ".meta.txt"
    meta = {}
    with open(meta_path) as f:
        for line in f:
            k, v = line.strip().split("=", 1)
            meta[k] = v
    shape = tuple(int(x) for x in meta["shape"].split(","))
    data = np.fromfile(bin_path, dtype=np.float32).reshape(shape)
    return data


def python_preprocess(image_path: str, foreground_ratio: float = 0.85) -> np.ndarray:
    from PIL import Image
    from tsr.utils import resize_foreground, ImagePreprocessor
    from image_utils import prepare_image
    import torch

    img = Image.open(image_path)

    if img.mode == "RGBA":
        img = resize_foreground(img, foreground_ratio)
        arr = np.array(img).astype(np.float32) / 255.0
        rgb = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        rgb = torch.from_numpy(rgb).unsqueeze(0)
    elif img.mode == "RGB":
        from rembg import remove
        img = remove(img)
        img = resize_foreground(img, foreground_ratio)
        arr = np.array(img).astype(np.float32) / 255.0
        rgb = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        rgb = torch.from_numpy(rgb).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported mode: {img.mode}")

    preprocessor = ImagePreprocessor()
    tensor = preprocessor.convert_and_resize(rgb, 512)
    return tensor.permute(0, 3, 1, 2).numpy()  # NHWC -> NCHW


def compare_tensors(name: str, a: np.ndarray, b: np.ndarray):
    diff = np.abs(a - b)
    print(f"\n=== {name} ===")
    print(f"  Shape:     A={a.shape}, B={b.shape}")
    print(f"  A range:   [{a.min():.6f}, {a.max():.6f}], mean={a.mean():.6f}")
    print(f"  B range:   [{b.min():.6f}, {b.max():.6f}], mean={b.mean():.6f}")
    print(f"  Max diff:  {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  RMSE:      {np.sqrt((diff**2).mean()):.6f}")

    per_channel = []
    if a.ndim == 4 and a.shape[1] == 3:
        for c in range(3):
            cd = np.abs(a[:, c] - b[:, c])
            per_channel.append(f"ch{c}: max={cd.max():.6f} mean={cd.mean():.6f}")
        print(f"  Per-channel: {', '.join(per_channel)}")

    threshold = 0.01
    close = (diff < threshold).mean() * 100
    print(f"  Within {threshold}: {close:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Compare Unity vs Python preprocessing")
    parser.add_argument("--unity-bin", required=True, help="Path to unity_preprocessed.bin")
    parser.add_argument("--image", required=True, help="Path to the test image")
    parser.add_argument("--onnx-dir", default=None, help="ONNX model dir for inference comparison")
    args = parser.parse_args()

    print("Loading Unity tensor...")
    unity = load_unity_tensor(args.unity_bin)
    print(f"  Unity shape: {unity.shape}")

    print("Running Python preprocessing...")
    python = python_preprocess(args.image)
    print(f"  Python shape: {python.shape}")

    compare_tensors("Preprocessed Image (Unity vs Python)", unity, python)

    if args.onnx_dir:
        import onnxruntime as ort

        triposr_path = str(Path(args.onnx_dir) / "triposr_fp32.onnx")
        decoder_path = str(Path(args.onnx_dir) / "nerf_decoder.onnx")

        print(f"\nRunning ONNX TripoSR on both tensors...")
        triposr = ort.InferenceSession(triposr_path)
        input_name = triposr.get_inputs()[0].name

        unity_codes = triposr.run(None, {input_name: unity})[0]
        python_codes = triposr.run(None, {input_name: python})[0]
        compare_tensors("Scene Codes", unity_codes, python_codes)

    print("\nDone.")


if __name__ == "__main__":
    main()
