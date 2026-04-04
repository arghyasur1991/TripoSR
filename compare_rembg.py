"""Compare rembg (u2netp) masks: ONNX Runtime vs Unity Sentis.

Usage:
    python compare_rembg.py --image test_images/novel/chair_raw.jpg
    python compare_rembg.py --image test_images/novel/chair_raw.jpg --unity-mask debug_reconstruction/unity_rembg_mask.png

Outputs saved to debug_reconstruction/:
    python_rembg_input.bin      - preprocessed input tensor (1,3,320,320) float32
    python_rembg_mask.png       - mask visualization (320x320 grayscale)
    python_rembg_mask.bin       - raw mask tensor (1,1,320,320) float32
    python_rembg_composite.png  - final 512x512 composite (for visual comparison)

If --unity-mask is given, prints numerical comparison.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from tsr.utils import resize_foreground


MODELS_DIR = Path(__file__).parent / "models"
DEBUG_DIR = Path(__file__).parent.parent / "debug_reconstruction"


def preprocess_for_rembg(image: Image.Image) -> np.ndarray:
    """Match Unity RembgModel.PrepareInput exactly:
    1. Resize to 320x320
    2. ToTensor (HWC uint8 -> CHW float [0,1])
    3. Divide by per-image max
    4. Subtract ImageNet mean, divide by ImageNet std
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = image.convert("RGB").resize((320, 320), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0  # HWC [0,1]

    # CHW layout
    arr = arr.transpose(2, 0, 1)  # (3, 320, 320)

    max_val = max(arr.max(), 1e-6)
    arr = arr / max_val

    for c in range(3):
        arr[c] = (arr[c] - mean[c]) / std[c]

    return arr[np.newaxis]  # (1, 3, 320, 320)


def minmax_normalize(mask: np.ndarray) -> np.ndarray:
    """Match Unity RembgModel.MinMaxNormalize."""
    mi, ma = mask.min(), mask.max()
    rng = max(ma - mi, 1e-8)
    return (mask - mi) / rng


def run_rembg_onnx(image_path: Path, output_dir: Path):
    """Run u2netp via ONNX Runtime and save all intermediates."""
    output_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    print(f"Input: {image_path} ({img.width}x{img.height})")

    # Preprocess
    input_tensor = preprocess_for_rembg(img)
    print(f"Preprocessed: shape={input_tensor.shape}, "
          f"range=[{input_tensor.min():.4f}, {input_tensor.max():.4f}]")

    # Save input tensor binary (for Unity comparison)
    input_bin = output_dir / "python_rembg_input.bin"
    input_tensor.astype(np.float32).tofile(input_bin)
    np.savetxt(str(input_bin) + ".meta.txt",
               [f"dtype=float32", f"shape={','.join(map(str, input_tensor.shape))}"],
               fmt="%s")
    print(f"Saved input: {input_bin}")

    # Run ONNX
    model_path = MODELS_DIR / "u2netp.onnx"
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: input_tensor})

    raw_mask = outputs[0]  # first output is the primary mask
    print(f"Raw mask: shape={raw_mask.shape}, "
          f"range=[{raw_mask.min():.4f}, {raw_mask.max():.4f}]")

    # MinMax normalize
    mask = minmax_normalize(raw_mask)
    print(f"Normalized mask: range=[{mask.min():.4f}, {mask.max():.4f}], "
          f"mean={mask.mean():.4f}")

    # Save mask binary
    mask_bin = output_dir / "python_rembg_mask.bin"
    mask.astype(np.float32).tofile(mask_bin)
    np.savetxt(str(mask_bin) + ".meta.txt",
               [f"dtype=float32", f"shape={','.join(map(str, mask.shape))}"],
               fmt="%s")
    print(f"Saved mask binary: {mask_bin}")

    # Save mask PNG
    mask_2d = mask[0, 0]  # (320, 320)
    mask_png = output_dir / "python_rembg_mask.png"
    Image.fromarray((mask_2d * 255).astype(np.uint8)).save(mask_png)
    print(f"Saved mask PNG: {mask_png}")

    # Run full composite pipeline for visual comparison
    save_composite(img, mask_2d, output_dir)

    return mask


def save_composite(image: Image.Image, mask_320: np.ndarray, output_dir: Path):
    """Run rembg mask → RGBA → resize_foreground → composite → save.

    Binarizes the mask at 0.5 before creating RGBA, matching rembg's
    post_process_mask behavior. This keeps a tight bbox that maximizes
    object resolution in the final 512x512 frame.
    """
    w, h = image.size

    # Upscale mask to original image size (LANCZOS matches rembg library)
    mask_pil = Image.fromarray((mask_320 * 255).astype(np.uint8))
    mask_full = mask_pil.resize((w, h), Image.LANCZOS)
    mask_arr = np.array(mask_full, dtype=np.float32) / 255.0

    # Binarize alpha at 0.5 — soft edges inflate the bbox and shrink the
    # object in the final 512x512 frame, degrading TripoSR quality.
    rgb = np.array(image.convert("RGB"), dtype=np.float32)
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = (mask_arr > 0.5).astype(np.float32) * 255.0
    rgba_img = Image.fromarray(rgba.astype(np.uint8), "RGBA")

    # resize_foreground (match TripoSR pipeline)
    fg_resized = resize_foreground(rgba_img, 0.85)

    # Composite onto gray 0.5 background
    arr = np.array(fg_resized, dtype=np.float32) / 255.0
    alpha = arr[:, :, 3:4]
    rgb_composite = arr[:, :, :3] * alpha + 0.5 * (1.0 - alpha)
    composite_pil = Image.fromarray((rgb_composite * 255).astype(np.uint8))

    # Resize to 512x512 (match TripoSR input)
    final = composite_pil.resize((512, 512), Image.LANCZOS)
    final_path = output_dir / "python_rembg_composite.png"
    final.save(final_path)
    print(f"Saved composite: {final_path}")


def compare_masks(python_mask: np.ndarray, unity_mask_path: Path):
    """Numerical comparison between Python ONNX and Unity Sentis masks."""
    unity_img = Image.open(unity_mask_path).convert("L")
    unity_arr = np.array(unity_img, dtype=np.float32) / 255.0

    py_2d = python_mask[0, 0]  # (320, 320)

    if unity_arr.shape != py_2d.shape:
        print(f"\nShape mismatch: Python={py_2d.shape}, Unity={unity_arr.shape}")
        unity_pil = Image.fromarray((unity_arr * 255).astype(np.uint8))
        unity_arr = np.array(
            unity_pil.resize((py_2d.shape[1], py_2d.shape[0]), Image.BILINEAR),
            dtype=np.float32) / 255.0
        print(f"Resized Unity mask to {unity_arr.shape} for comparison")

    diff = np.abs(py_2d - unity_arr)
    print(f"\n--- Mask Comparison ---")
    print(f"Python:  mean={py_2d.mean():.4f}, range=[{py_2d.min():.4f}, {py_2d.max():.4f}]")
    print(f"Unity:   mean={unity_arr.mean():.4f}, range=[{unity_arr.min():.4f}, {unity_arr.max():.4f}]")
    print(f"Diff:    max={diff.max():.4f}, mean={diff.mean():.4f}")

    # Cosine similarity
    py_flat = py_2d.flatten()
    u_flat = unity_arr.flatten()
    cos_sim = np.dot(py_flat, u_flat) / (np.linalg.norm(py_flat) * np.linalg.norm(u_flat) + 1e-8)
    print(f"Cosine similarity: {cos_sim:.6f}")

    # IoU at threshold 0.5
    py_fg = py_2d > 0.5
    u_fg = unity_arr > 0.5
    intersection = np.logical_and(py_fg, u_fg).sum()
    union = np.logical_or(py_fg, u_fg).sum()
    iou = intersection / max(union, 1)
    print(f"IoU (threshold=0.5): {iou:.4f}")
    print(f"Python fg pixels: {py_fg.sum()}, Unity fg pixels: {u_fg.sum()}")


def compare_composites(python_composite_path: Path, unity_composite_path: Path):
    """Pixel-level comparison of Python vs Unity 512x512 composites."""
    py_img = Image.open(python_composite_path).convert("RGB")
    u_img = Image.open(unity_composite_path).convert("RGB")

    py_arr = np.array(py_img, dtype=np.float32) / 255.0
    u_arr = np.array(u_img, dtype=np.float32) / 255.0

    if py_arr.shape != u_arr.shape:
        print(f"\nComposite shape mismatch: Python={py_arr.shape}, Unity={u_arr.shape}")
        u_img = u_img.resize((py_arr.shape[1], py_arr.shape[0]), Image.BILINEAR)
        u_arr = np.array(u_img, dtype=np.float32) / 255.0

    diff = np.abs(py_arr - u_arr)
    print(f"\n--- Composite Comparison (512x512 RGB) ---")
    print(f"Python:  mean={py_arr.mean():.4f}, range=[{py_arr.min():.4f}, {py_arr.max():.4f}]")
    print(f"Unity:   mean={u_arr.mean():.4f}, range=[{u_arr.min():.4f}, {u_arr.max():.4f}]")
    print(f"Diff:    max={diff.max():.4f}, mean={diff.mean():.4f}")

    per_pixel = diff.max(axis=2)  # max across RGB channels
    print(f"Pixels > 0.05 diff: {(per_pixel > 0.05).sum()} / {per_pixel.size} "
          f"({100 * (per_pixel > 0.05).mean():.1f}%)")
    print(f"Pixels > 0.10 diff: {(per_pixel > 0.10).sum()} / {per_pixel.size} "
          f"({100 * (per_pixel > 0.10).mean():.1f}%)")

    flat_py = py_arr.flatten()
    flat_u = u_arr.flatten()
    cos_sim = np.dot(flat_py, flat_u) / (np.linalg.norm(flat_py) * np.linalg.norm(flat_u) + 1e-8)
    print(f"Cosine similarity: {cos_sim:.6f}")

    # RMSE
    rmse = np.sqrt(np.mean(diff ** 2))
    print(f"RMSE: {rmse:.6f}")

    # Save diff heatmap
    diff_vis = (per_pixel * 10).clip(0, 1)  # amplify 10x for visibility
    diff_img = Image.fromarray((diff_vis * 255).astype(np.uint8))
    diff_path = python_composite_path.parent / "composite_diff_heatmap.png"
    diff_img.save(diff_path)
    print(f"Saved diff heatmap (10x amplified): {diff_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare rembg masks and composites: ONNX vs Unity")
    parser.add_argument("--image", type=Path, required=True, help="Input image (raw, no bg removal)")
    parser.add_argument("--unity-mask", type=Path, help="Unity rembg mask PNG for comparison")
    parser.add_argument("--unity-composite", type=Path, help="Unity 512x512 composite PNG for comparison")
    parser.add_argument("--output-dir", type=Path, default=DEBUG_DIR, help="Output directory")
    args = parser.parse_args()

    mask = run_rembg_onnx(args.image, args.output_dir)

    if args.unity_mask:
        compare_masks(mask, args.unity_mask)

    python_composite = args.output_dir / "python_rembg_composite.png"
    if args.unity_composite:
        compare_composites(python_composite, args.unity_composite)


if __name__ == "__main__":
    main()
