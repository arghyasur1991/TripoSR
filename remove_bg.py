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


def export_u2netp_onnx(output_dir: Path, target_opset: int = 15):
    """Export the u2netp ONNX model file for on-device deployment.

    Converts the rembg-bundled model to the target opset for Sentis compatibility,
    then applies graph optimization (constant folding, dead-node elimination).
    """
    import shutil
    import onnx
    from onnx import version_converter
    from rembg.sessions import U2netpSession

    output_dir.mkdir(parents=True, exist_ok=True)
    src = Path(U2netpSession.download_models())
    raw_dst = output_dir / "u2netp_raw.onnx"
    dst = output_dir / "u2netp.onnx"

    shutil.copy2(src, raw_dst)
    print(f"Copied raw u2netp to {raw_dst} ({raw_dst.stat().st_size / 1e6:.1f} MB)")

    model = onnx.load(str(raw_dst))
    orig_opset = model.opset_import[0].version
    orig_nodes = len(model.graph.node)
    print(f"Original: opset {orig_opset}, {orig_nodes} nodes")

    if orig_opset != target_opset:
        print(f"Converting opset {orig_opset} -> {target_opset}...")
        model = version_converter.convert_version(model, target_opset)
        onnx.checker.check_model(model)
        print(f"After opset conversion: {len(model.graph.node)} nodes")

    try:
        import onnxruntime as ort

        tmp_path = str(output_dir / "u2netp_tmp.onnx")
        onnx.save(model, tmp_path)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.optimized_model_filepath = str(dst)
        _ = ort.InferenceSession(tmp_path, opts, providers=["CPUExecutionProvider"])
        Path(tmp_path).unlink(missing_ok=True)

        opt_model = onnx.load(str(dst))
        opt_nodes = len(opt_model.graph.node)
        print(f"After graph optimization: {opt_nodes} nodes (was {orig_nodes}, -{(1 - opt_nodes / orig_nodes) * 100:.0f}%)")
    except ImportError:
        print("onnxruntime not available, skipping graph optimization")
        onnx.save(model, str(dst))

    raw_dst.unlink(missing_ok=True)
    print(f"Final u2netp saved to {dst} ({dst.stat().st_size / 1e6:.1f} MB)")

    _verify_u2netp(dst)


def _verify_u2netp(model_path: Path):
    """Quick sanity check: run a dummy input through the exported model."""
    import numpy as np
    import onnxruntime as ort

    print("Verifying u2netp output...")
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    dummy = np.random.rand(1, 3, 320, 320).astype(np.float32)
    outputs = sess.run(None, {sess.get_inputs()[0].name: dummy})
    print(f"  Input: {dummy.shape}")
    for i, o in enumerate(outputs):
        print(f"  Output[{i}]: shape={o.shape}, range=[{o.min():.3f}, {o.max():.3f}]")
    print("  Verification PASSED")


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
