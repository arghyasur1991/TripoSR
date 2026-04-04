#!/usr/bin/env python3
"""Split triposr_fp32.onnx into two halves at the transformer block 8 boundary.

Part 1: image_tokenizer + backbone blocks 0-7
  Input:  image [1,3,512,512]
  Outputs: encoder_hidden_states [1,1025,768]
           triplane_features     [1,3,1024,32,32]
           hidden_states         [1,3072,1024]

Part 2: backbone blocks 8-15 + post_processor
  Inputs:  (same 3 tensors above)
  Output:  scene_codes [1,3,40,64,64]

Usage:
  python split_triposr.py [--input PATH] [--verify]
"""

import argparse
from pathlib import Path

import onnx
from onnx.utils import Extractor

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ONNX = SCRIPT_DIR.parent / "SentienceUnity/Assets/Game/ObjectReconstruction/OnnxSource/triposr_fp32.onnx"

CROSS_BOUNDARY_TENSORS = [
    "/Reshape_output_0",                                    # encoder hidden states [1,1025,768]
    "/backbone/transformer_blocks.7/Add_2_output_0",        # hidden states after block 7 [1,3072,1024]
]


def split_model(input_path: Path, output_dir: Path):
    print(f"Loading {input_path} ...")
    model = onnx.load(str(input_path))
    graph = model.graph

    original_input = graph.input[0].name
    original_output = graph.output[0].name

    print(f"  Input: {original_input}")
    print(f"  Output: {original_output}")
    print(f"  Nodes: {len(graph.node)}")

    # --- Part 1: image -> encoder_hidden_states + hidden_states_block7 ---
    print("\nExtracting Part 1 (encoder + blocks 0-7) ...")
    extractor1 = Extractor(model)
    part1 = extractor1.extract_model(
        input_names=[original_input],
        output_names=CROSS_BOUNDARY_TENSORS,
    )
    part1_path = output_dir / "triposr_part1.onnx"
    onnx.save(part1, str(part1_path))
    print(f"  Saved: {part1_path} ({part1_path.stat().st_size / 1e6:.1f} MB, {len(part1.graph.node)} nodes)")

    # --- Part 2: hidden_states + encoder_states -> scene_codes ---
    print("\nExtracting Part 2 (blocks 8-15 + post_processor) ...")
    extractor2 = Extractor(model)
    part2 = extractor2.extract_model(
        input_names=CROSS_BOUNDARY_TENSORS,
        output_names=[original_output],
    )
    part2_path = output_dir / "triposr_part2.onnx"
    onnx.save(part2, str(part2_path))
    print(f"  Saved: {part2_path} ({part2_path.stat().st_size / 1e6:.1f} MB, {len(part2.graph.node)} nodes)")

    return part1_path, part2_path


def verify(original_path: Path, part1_path: Path, part2_path: Path):
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — skipping verification")
        return

    print("\nVerifying numerical equivalence ...")
    rng = np.random.default_rng(42)
    dummy_image = rng.standard_normal((1, 3, 512, 512)).astype(np.float32)

    print("  Running original model ...")
    sess_orig = ort.InferenceSession(str(original_path), providers=["CPUExecutionProvider"])
    [orig_out] = sess_orig.run(None, {"image": dummy_image})

    print("  Running Part 1 ...")
    sess_p1 = ort.InferenceSession(str(part1_path), providers=["CPUExecutionProvider"])
    p1_outs = sess_p1.run(None, {"image": dummy_image})
    p1_output_names = [o.name for o in sess_p1.get_outputs()]

    print("  Running Part 2 ...")
    sess_p2 = ort.InferenceSession(str(part2_path), providers=["CPUExecutionProvider"])
    p2_inputs = {name: val for name, val in zip(p1_output_names, p1_outs)}
    [split_out] = sess_p2.run(None, p2_inputs)

    max_diff = np.max(np.abs(orig_out - split_out))
    mean_diff = np.mean(np.abs(orig_out - split_out))
    cos_sim = np.dot(orig_out.flat, split_out.flat) / (
        np.linalg.norm(orig_out) * np.linalg.norm(split_out) + 1e-12
    )

    print(f"\n  Max diff:  {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")
    print(f"  Cosine:    {cos_sim:.8f}")

    if max_diff < 1e-4:
        print("  PASS — numerically identical")
    elif max_diff < 1e-2:
        print("  PASS — within FP32 tolerance")
    else:
        print("  WARN — significant divergence")


def main():
    parser = argparse.ArgumentParser(description="Split TripoSR ONNX into two halves")
    parser.add_argument("--input", type=Path, default=DEFAULT_ONNX,
                        help="Path to triposr_fp32.onnx")
    parser.add_argument("--verify", action="store_true",
                        help="Verify split produces identical output via ORT")
    args = parser.parse_args()

    output_dir = args.input.parent
    part1_path, part2_path = split_model(args.input, output_dir)

    if args.verify:
        verify(args.input, part1_path, part2_path)

    print("\nDone. Update the Unity wizard to convert both part1 and part2.")


if __name__ == "__main__":
    main()
