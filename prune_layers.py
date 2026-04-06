"""Layer pruning analysis for TripoSR transformer backbone.

Systematically removes transformer blocks to identify which layers contribute
least to reconstruction quality. Works at the PyTorch level (cleaner than ONNX
surgery) — removes blocks from nn.ModuleList, re-runs inference, measures quality.

Usage:
    python prune_layers.py                          # Single-layer removal analysis (all 16)
    python prune_layers.py --blocks 4 5 6           # Test specific blocks
    python prune_layers.py --multi 4 5 6 10         # Remove multiple blocks at once
    python prune_layers.py --multi 4 5 6 10 --export  # Remove + export to ONNX
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).parent))
from image_utils import prepare_image
from tsr.system import TSR

OUTPUT_DIR = Path(__file__).parent / "output" / "layer_pruning"
TEST_IMAGES_DIR = Path(__file__).parent / "test_images"


def collect_test_images(n_max: int = 6) -> list[Path]:
    """Collect a representative set of test images."""
    priority = [
        TEST_IMAGES_DIR / "examples" / "chair.png",
        TEST_IMAGES_DIR / "examples" / "hamburger.png",
        TEST_IMAGES_DIR / "examples" / "robot.png",
        TEST_IMAGES_DIR / "novel" / "backpack_nobg.png",
        TEST_IMAGES_DIR / "novel" / "shoe_nobg.png",
        TEST_IMAGES_DIR / "novel" / "clock_nobg.png",
    ]
    result = [p for p in priority if p.exists()]
    if len(result) < n_max:
        for p in sorted(TEST_IMAGES_DIR.rglob("*.png")):
            if p not in result and len(result) < n_max:
                result.append(p)
    return result[:n_max]


def load_model(device: str) -> TSR:
    model = TSR.from_pretrained(
        "stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt"
    )
    model.to(device)
    model.eval()
    return model


def run_inference(model: TSR, image_path: Path, device: str,
                  resolution: int = 128) -> trimesh.Trimesh:
    """Run forward pass + mesh extraction for a single image."""
    img = prepare_image(image_path)
    with torch.no_grad():
        scene_codes = model(img, device)
        meshes = model.extract_mesh(
            scene_codes, has_vertex_color=False, resolution=resolution)
    return meshes[0]


def compute_metrics(baseline: trimesh.Trimesh, test: trimesh.Trimesh,
                    n_samples: int = 30000) -> dict:
    """Compute Chamfer distance and F-scores between meshes."""
    pts_b = baseline.sample(n_samples)
    pts_t = test.sample(n_samples)

    from scipy.spatial import cKDTree
    tree_b = cKDTree(pts_b)
    tree_t = cKDTree(pts_t)

    d_bt, _ = tree_b.query(pts_t)
    d_tb, _ = tree_t.query(pts_b)

    # Normalize by bounding box diagonal
    bbox = np.ptp(pts_b, axis=0)
    diag = np.linalg.norm(bbox)
    d_bt_pct = d_bt / diag * 100
    d_tb_pct = d_tb / diag * 100

    cd = (d_bt_pct.mean() + d_tb_pct.mean()) / 2

    def f_score(d1, d2, threshold):
        prec = (d1 < threshold).mean()
        rec = (d2 < threshold).mean()
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec) * 100

    return {
        "cd": cd,
        "f1": f_score(d_bt_pct, d_tb_pct, 1.0),
        "f2": f_score(d_bt_pct, d_tb_pct, 2.0),
        "verts": test.vertices.shape[0],
    }


def remove_blocks(model: TSR, block_indices: list[int]) -> TSR:
    """Remove specific transformer blocks from the model (in-place).

    After removal, the remaining blocks are contiguous — block N-1's output
    feeds directly into block N+1 via the residual connection.
    """
    blocks = model.backbone.transformer_blocks
    keep = [i for i in range(len(blocks)) if i not in block_indices]
    new_blocks = torch.nn.ModuleList([blocks[i] for i in keep])
    model.backbone.transformer_blocks = new_blocks
    return model


def run_pruning_experiment(model: TSR, device: str, removed_blocks: list[int],
                           test_images: list[Path], baseline_meshes: dict,
                           save_dir: Path | None = None) -> dict:
    """Remove blocks and evaluate quality."""
    label = "remove_" + "_".join(str(b) for b in removed_blocks)
    n_remaining = 16 - len(removed_blocks)
    removed_params = len(removed_blocks) * 20.46

    pruned = copy.deepcopy(model)
    remove_blocks(pruned, removed_blocks)
    pruned.to(device)
    pruned.eval()

    print(f"\n  [{label}] {n_remaining} blocks, -{removed_params:.1f}M params")

    results = []
    for img_path in test_images:
        name = img_path.stem
        try:
            mesh = run_inference(pruned, img_path, device)
        except Exception as e:
            print(f"    {name}: CRASH ({e})")
            results.append({"image": name, "cd": 100.0, "f1": 0.0, "f2": 0.0})
            continue

        metrics = compute_metrics(baseline_meshes[name], mesh)
        status = "PASS" if metrics["cd"] < 2.0 and metrics["f1"] > 85.0 else "FAIL"
        print(f"    {name}: CD={metrics['cd']:.3f}% F@1%={metrics['f1']:.1f} "
              f"F@2%={metrics['f2']:.1f} [{status}]")
        results.append({"image": name, **metrics})

        if save_dir:
            out_path = save_dir / label / f"{name}.obj"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(str(out_path))

    del pruned
    if device != "cpu":
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    avg_cd = np.mean([r["cd"] for r in results])
    avg_f1 = np.mean([r["f1"] for r in results])
    avg_f2 = np.mean([r["f2"] for r in results])

    print(f"    AVG: CD={avg_cd:.3f}% F@1%={avg_f1:.1f} F@2%={avg_f2:.1f}")

    return {
        "removed": removed_blocks,
        "n_remaining": n_remaining,
        "removed_params_M": removed_params,
        "avg_cd": avg_cd,
        "avg_f1": avg_f1,
        "avg_f2": avg_f2,
        "per_image": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Layer pruning analysis for TripoSR")
    parser.add_argument("--blocks", nargs="+", type=int,
                        help="Specific blocks to test individually (default: all 0-15)")
    parser.add_argument("--multi", nargs="+", type=int,
                        help="Remove multiple blocks simultaneously")
    parser.add_argument("--export", action="store_true",
                        help="Export pruned model to ONNX (only with --multi)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save OBJ meshes")
    args = parser.parse_args()

    if args.device == "auto":
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    test_images = collect_test_images()
    print(f"Test images: {len(test_images)}, device: {device}")
    for p in test_images:
        print(f"  {p.relative_to(TEST_IMAGES_DIR)}")

    model = load_model(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"{len(model.backbone.transformer_blocks)} blocks")

    save_dir = None if args.no_save else OUTPUT_DIR

    # Generate baseline meshes
    print("\nGenerating baseline meshes...")
    baseline_meshes = {}
    for img_path in test_images:
        name = img_path.stem
        print(f"  {name}...", end="", flush=True)
        mesh = run_inference(model, img_path, device)
        baseline_meshes[name] = mesh
        print(f" {mesh.vertices.shape[0]} verts")

        if save_dir:
            bl_dir = save_dir / "baseline"
            bl_dir.mkdir(parents=True, exist_ok=True)
            mesh.export(str(bl_dir / f"{name}.obj"))

    if args.multi:
        print(f"\n{'='*60}")
        print(f"Multi-block removal: removing blocks {args.multi}")
        print(f"{'='*60}")
        result = run_pruning_experiment(
            model, device, args.multi, test_images, baseline_meshes, save_dir)

        if args.export:
            print("\n  Exporting pruned model to ONNX...")
            from export_onnx import (TripoSRForward, export_triposr_fp32,
                                     split_model, transformer_optimize,
                                     optimize_graph, quantize_int8_qdq,
                                     _collect_calibration_images, MODELS_DIR)
            import onnxruntime as ort

            pruned = copy.deepcopy(model)
            remove_blocks(pruned, args.multi)
            pruned.to("cpu")
            pruned.eval()

            n_rem = len(args.multi)
            prefix = f"triposr_pruned{16-n_rem}L"
            out_dir = MODELS_DIR
            fp32_path = out_dir / f"{prefix}_fp32.onnx"

            export_triposr_fp32(TripoSRForward(pruned), fp32_path, 15,
                                f"Pruned {16-n_rem}L FP32", static=True)

            n_blocks = 16 - n_rem
            mid = n_blocks // 2
            from export_onnx import SPLIT_BOUNDARY_TENSORS
            # The split boundary will be different for pruned models
            # Use the midpoint block's Add_2 output
            remaining_blocks = [i for i in range(16) if i not in args.multi]
            mid_block = remaining_blocks[mid - 1]
            pruned_boundary = [
                "/Reshape_output_0",
                f"/backbone/transformer_blocks.{mid_block}/Add_2_output_0",
            ]
            import export_onnx
            old_boundary = export_onnx.SPLIT_BOUNDARY_TENSORS
            export_onnx.SPLIT_BOUNDARY_TENSORS = pruned_boundary

            try:
                p1, p2 = split_model(fp32_path, out_dir, prefix)
                transformer_optimize(p1)
                transformer_optimize(p2)
                optimize_graph(p1)
                optimize_graph(p2)

                # QDQ INT8 quantization
                cal_images = _collect_calibration_images()
                p1_qdq = out_dir / f"{prefix}_part1_int8_qdq.onnx"
                quantize_int8_qdq(p1, p1_qdq, cal_images, input_name="image")

                p1_sess = ort.InferenceSession(str(p1), providers=["CPUExecutionProvider"])
                p2_meta = ort.InferenceSession(str(p2), providers=["CPUExecutionProvider"])
                p2_names = [inp.name for inp in p2_meta.get_inputs()]
                del p2_meta

                p2_cal = []
                for img in cal_images:
                    outs = p1_sess.run(None, {"image": img})
                    out_names = [o.name for o in p1_sess.get_outputs()]
                    p2_cal.append({n: v for n, v in zip(out_names, outs) if n in p2_names})
                del p1_sess

                p2_qdq = out_dir / f"{prefix}_part2_int8_qdq.onnx"
                quantize_int8_qdq(p2, p2_qdq, p2_cal)

                p1_sz = p1_qdq.stat().st_size / 1e6
                p2_sz = p2_qdq.stat().st_size / 1e6
                print(f"\n  Exported: {p1_qdq.name} ({p1_sz:.1f}MB) + "
                      f"{p2_qdq.name} ({p2_sz:.1f}MB) = {p1_sz+p2_sz:.1f}MB total")
            finally:
                export_onnx.SPLIT_BOUNDARY_TENSORS = old_boundary

            del pruned
        return

    # Single-block removal analysis
    blocks_to_test = args.blocks if args.blocks else list(range(16))
    print(f"\n{'='*60}")
    print(f"Single-block removal analysis: {len(blocks_to_test)} blocks")
    print(f"{'='*60}")

    all_results = []
    for block_idx in blocks_to_test:
        result = run_pruning_experiment(
            model, device, [block_idx], test_images, baseline_meshes, save_dir)
        all_results.append(result)

    # Rank by quality impact
    print(f"\n{'='*60}")
    print("LAYER IMPORTANCE RANKING (least important first)")
    print(f"{'='*60}")
    ranked = sorted(all_results, key=lambda r: r["avg_cd"])
    print(f"{'Block':>6} {'Avg CD%':>8} {'Avg F@1%':>9} {'Avg F@2%':>9} {'Status':>8}")
    print("-" * 45)
    for r in ranked:
        block = r["removed"][0]
        status = "SAFE" if r["avg_cd"] < 2.0 and r["avg_f1"] > 85.0 else "RISKY"
        print(f"  {block:>4}  {r['avg_cd']:>7.3f}  {r['avg_f1']:>8.1f}  "
              f"{r['avg_f2']:>8.1f}  {status:>7}")

    # Suggest blocks to remove
    safe_blocks = [r["removed"][0] for r in ranked if r["avg_cd"] < 2.0 and r["avg_f1"] > 85.0]
    print(f"\nSafe to remove (individually): {safe_blocks if safe_blocks else 'None'}")
    if len(safe_blocks) >= 2:
        print(f"Suggested multi-removal test: --multi {' '.join(str(b) for b in safe_blocks[:4])}")


if __name__ == "__main__":
    main()
