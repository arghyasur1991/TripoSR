"""Debug script: saves intermediate preprocessing images for comparing with Unity.

Usage:
    python debug_preprocess.py test_images/novel/chair.png
    python debug_preprocess.py test_images/novel/hamburger.png

Outputs to debug_output/:
    - {name}_resized_fg.png    (after resize_foreground, before composite)
    - {name}_preprocessed.png  (after composite on gray, before resize to 512)
    - {name}_final_512.png     (final 512x512 input to TripoSR)
    - {name}_tensor_values.npy (raw float32 tensor values for numerical comparison)
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from tsr.utils import resize_foreground, ImagePreprocessor
from image_utils import prepare_image


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_preprocess.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    name = image_path.stem
    out_dir = Path("debug_output")
    out_dir.mkdir(exist_ok=True)

    print(f"Processing: {image_path}")
    img = Image.open(image_path)
    print(f"  Original: {img.size}, mode={img.mode}")

    # Step 1: resize_foreground (if RGBA)
    if img.mode == "RGBA":
        fg = resize_foreground(img, 0.85)
        fg_pil = Image.fromarray(np.array(fg))
        fg_pil.save(out_dir / f"{name}_resized_fg.png")
        print(f"  After resize_foreground: {fg_pil.size}")

        # Step 2: composite on gray
        arr = np.array(fg).astype(np.float32) / 255.0
        rgb = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        composite = Image.fromarray((rgb * 255.0).astype(np.uint8))
        composite.save(out_dir / f"{name}_preprocessed.png")
        print(f"  Composite: {composite.size}")
    else:
        composite = img.convert("RGB")
        composite.save(out_dir / f"{name}_preprocessed.png")
        print(f"  RGB (no alpha): {composite.size}")

    # Step 3: final 512x512 via ImagePreprocessor (bilinear + antialias)
    processor = ImagePreprocessor()
    tensor = processor(composite, 512)  # (B, H, W, C)
    tensor_nchw = tensor.permute(0, 3, 1, 2)  # (1, 3, 512, 512)

    # Save as image
    final_np = tensor_nchw[0].permute(1, 2, 0).numpy()  # (512, 512, 3)
    final_img = Image.fromarray((final_np * 255).clip(0, 255).astype(np.uint8))
    final_img.save(out_dir / f"{name}_final_512.png")
    print(f"  Final 512x512 saved")

    # Save raw tensor for numerical comparison
    np.save(out_dir / f"{name}_tensor_values.npy", tensor_nchw.numpy())

    # Print some tensor stats for quick comparison
    t = tensor_nchw.numpy()
    print(f"\n  Tensor shape: {t.shape}")
    print(f"  Range: [{t.min():.4f}, {t.max():.4f}]")
    print(f"  Mean:  [{t[:,0].mean():.4f}, {t[:,1].mean():.4f}, {t[:,2].mean():.4f}]")
    print(f"  Center pixel (256,256): R={t[0,0,256,256]:.4f} G={t[0,1,256,256]:.4f} B={t[0,2,256,256]:.4f}")
    print(f"  Corner pixel (0,0): R={t[0,0,0,0]:.4f} G={t[0,1,0,0]:.4f} B={t[0,2,0,0]:.4f}")

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
