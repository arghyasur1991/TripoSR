"""Background removal wrapper using rembg (u2netp model).

Usage:
    python remove_bg.py                          # process all *_raw.* in test_images/novel/
    python remove_bg.py --input path/to/img.jpg  # single image
    python remove_bg.py --input-dir path/to/dir  # all images in directory
    python remove_bg.py --export-model            # export u2netp ONNX to models/
"""

import argparse
import sys
from pathlib import Path

from PIL import Image
from rembg import remove, new_session


def remove_background(
    input_path: Path,
    output_path: Path | None = None,
    session=None,
) -> Path:
    """Remove background from a single image.

    Returns the output path.
    """
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem.replace('_raw', '')}_nobg.png"

    img = Image.open(input_path).convert("RGB")
    result = remove(img, session=session, post_process_mask=True)
    result.save(output_path)
    return output_path


def process_directory(input_dir: Path, session=None):
    """Process all *_raw.* images in a directory."""
    patterns = ["*_raw.jpg", "*_raw.jpeg", "*_raw.png"]
    raw_files = []
    for pattern in patterns:
        raw_files.extend(input_dir.glob(pattern))

    if not raw_files:
        print(f"No *_raw.* files found in {input_dir}")
        return

    for raw_path in sorted(raw_files):
        out_path = raw_path.parent / f"{raw_path.stem.replace('_raw', '')}_nobg.png"
        if out_path.exists():
            print(f"  [skip] {out_path.name} already exists")
            continue
        print(f"  [rembg] {raw_path.name} -> {out_path.name} ...", end="", flush=True)
        remove_background(raw_path, out_path, session=session)
        print(" done")


def export_u2netp_onnx(output_dir: Path):
    """Export the u2netp ONNX model file for on-device deployment."""
    import onnxruntime as ort
    from rembg.sessions import U2netpSession

    output_dir.mkdir(parents=True, exist_ok=True)
    sess = U2netpSession.from_pretrained("u2netp")
    src = Path(sess.inner_session._model_path)
    dst = output_dir / "u2netp.onnx"

    import shutil
    shutil.copy2(src, dst)
    print(f"Exported u2netp ONNX model to {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Background removal using rembg u2netp")
    parser.add_argument("--input", type=Path, help="Single image to process")
    parser.add_argument("--input-dir", type=Path, help="Directory of images to process")
    parser.add_argument("--output", type=Path, help="Output path (for single image mode)")
    parser.add_argument("--model", default="u2netp", help="rembg model name (default: u2netp)")
    parser.add_argument("--export-model", action="store_true", help="Export ONNX model to models/")
    args = parser.parse_args()

    if args.export_model:
        export_u2netp_onnx(Path(__file__).parent / "models")
        return

    print(f"Loading rembg session (model={args.model})...")
    session = new_session(args.model)

    if args.input:
        out = remove_background(args.input, args.output, session=session)
        print(f"Saved: {out}")
    elif args.input_dir:
        process_directory(args.input_dir, session=session)
    else:
        novel_dir = Path(__file__).parent / "test_images" / "novel"
        if novel_dir.exists():
            print(f"Processing {novel_dir}/")
            process_directory(novel_dir, session=session)
        else:
            print(f"No input specified and {novel_dir} does not exist.")
            sys.exit(1)


if __name__ == "__main__":
    main()
