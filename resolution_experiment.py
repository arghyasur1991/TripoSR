"""Input resolution experiment for TripoSR.

Tests how reducing the DINOv2 input resolution (from 512) affects reconstruction
quality. DINOv2 supports arbitrary resolutions via positional embedding interpolation
(interpolate_pos_encoding=True). Cross-attention K/V length adjusts dynamically.

No model changes, no training — just smaller input images → fewer encoder tokens.

Usage:
    python resolution_experiment.py --resolutions 384 256
    python resolution_experiment.py --resolutions 384 256 --full-test --vertex-color --benchmark
    python resolution_experiment.py --resolutions 384 256 --full-test --benchmark --resolution 256
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from prune_layers import (
    collect_test_images, load_model, run_inference, compute_metrics,
    remove_blocks,
)
from image_utils import prepare_image

OUTPUT_DIR = Path(__file__).parent / "output" / "resolution_experiment"


def run_resolution_experiment(
    model, device: str, cond_image_size: int,
    test_images: list[Path], baseline_meshes: dict,
    save_dir: Path | None, mesh_resolution: int, vertex_color: bool,
    measure_latency: bool,
) -> dict:
    """Run inference at a given input resolution and compare to baseline."""
    n_patches = (cond_image_size // 16) ** 2
    n_tokens = n_patches + 1  # +1 for CLS
    label = f"res_{cond_image_size}"

    print(f"\nTesting resolution {cond_image_size}x{cond_image_size} "
          f"({n_tokens} tokens)...")

    original_size = model.cfg.cond_image_size
    model.cfg.cond_image_size = cond_image_size

    if measure_latency and test_images:
        run_inference(model, test_images[0], device,
                      resolution=mesh_resolution, vertex_color=False)

    results = []
    fwd_times = []
    for img_path in test_images:
        name = img_path.stem
        try:
            mesh, fwd_t = run_inference(model, img_path, device,
                                        resolution=mesh_resolution,
                                        vertex_color=vertex_color)
            fwd_times.append(fwd_t)
        except Exception as e:
            print(f"    {name}: CRASH ({e})")
            results.append({"image": name, "cd": 100.0, "f1": 0.0, "f2": 0.0})
            continue

        metrics = compute_metrics(baseline_meshes[name], mesh)
        status = "PASS" if metrics["cd"] < 2.0 and metrics["f1"] > 85.0 else "FAIL"
        lat_str = f" {fwd_t:.2f}s" if measure_latency else ""
        print(f"    {name}: CD={metrics['cd']:.3f}% F@1%={metrics['f1']:.1f} "
              f"F@2%={metrics['f2']:.1f} [{status}]{lat_str}")
        results.append({"image": name, "fwd_time": fwd_t, **metrics})

        if save_dir:
            out_path = save_dir / label / f"{name}.obj"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(str(out_path))

    model.cfg.cond_image_size = original_size

    avg_cd = np.mean([r["cd"] for r in results])
    avg_f1 = np.mean([r["f1"] for r in results])
    avg_f2 = np.mean([r["f2"] for r in results])
    avg_fwd = np.mean(fwd_times) if fwd_times else 0.0

    lat_str = f"  Avg fwd: {avg_fwd:.2f}s" if measure_latency else ""
    print(f"    AVG: CD={avg_cd:.3f}% F@1%={avg_f1:.1f} F@2%={avg_f2:.1f}{lat_str}")

    return {
        "cond_image_size": cond_image_size,
        "tokens": n_tokens,
        "avg_cd": avg_cd,
        "avg_f1": avg_f1,
        "avg_f2": avg_f2,
        "avg_fwd_time": avg_fwd,
        "per_image": results,
    }


def run_combined_experiment(
    model, device: str, cond_image_size: int, removed_blocks: list[int],
    test_images: list[Path], baseline_meshes: dict,
    save_dir: Path | None, mesh_resolution: int, vertex_color: bool,
    measure_latency: bool,
) -> dict:
    """Test resolution reduction combined with layer pruning."""
    n_patches = (cond_image_size // 16) ** 2
    n_tokens = n_patches + 1
    n_remaining = 16 - len(removed_blocks)
    label = f"res_{cond_image_size}_{n_remaining}L"

    print(f"\nTesting resolution {cond_image_size}x{cond_image_size} + "
          f"{n_remaining}L (remove blocks {removed_blocks})...")

    pruned = copy.deepcopy(model)
    remove_blocks(pruned, removed_blocks)
    pruned.to(device)
    pruned.eval()
    pruned.cfg.cond_image_size = cond_image_size

    total_params = sum(p.numel() for p in pruned.parameters()) / 1e6
    print(f"  Model: {total_params:.1f}M params, {n_tokens} tokens")

    if measure_latency and test_images:
        run_inference(pruned, test_images[0], device,
                      resolution=mesh_resolution, vertex_color=False)

    results = []
    fwd_times = []
    for img_path in test_images:
        name = img_path.stem
        try:
            mesh, fwd_t = run_inference(pruned, img_path, device,
                                        resolution=mesh_resolution,
                                        vertex_color=vertex_color)
            fwd_times.append(fwd_t)
        except Exception as e:
            print(f"    {name}: CRASH ({e})")
            results.append({"image": name, "cd": 100.0, "f1": 0.0, "f2": 0.0})
            continue

        metrics = compute_metrics(baseline_meshes[name], mesh)
        status = "PASS" if metrics["cd"] < 2.0 and metrics["f1"] > 85.0 else "FAIL"
        lat_str = f" {fwd_t:.2f}s" if measure_latency else ""
        print(f"    {name}: CD={metrics['cd']:.3f}% F@1%={metrics['f1']:.1f} "
              f"F@2%={metrics['f2']:.1f} [{status}]{lat_str}")
        results.append({"image": name, "fwd_time": fwd_t, **metrics})

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
    avg_fwd = np.mean(fwd_times) if fwd_times else 0.0

    lat_str = f"  Avg fwd: {avg_fwd:.2f}s" if measure_latency else ""
    print(f"    AVG: CD={avg_cd:.3f}% F@1%={avg_f1:.1f} F@2%={avg_f2:.1f}{lat_str}")

    return {
        "cond_image_size": cond_image_size,
        "removed_blocks": removed_blocks,
        "n_remaining": n_remaining,
        "tokens": n_tokens,
        "total_params_M": total_params,
        "avg_cd": avg_cd,
        "avg_f1": avg_f1,
        "avg_f2": avg_f2,
        "avg_fwd_time": avg_fwd,
        "per_image": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Input resolution experiment for TripoSR")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[384, 256],
                        help="Input resolutions to test (must be divisible by 16)")
    parser.add_argument("--combine-layers", nargs="+", type=int, default=None,
                        help="Also test combined with layer pruning (block indices to remove)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save OBJ meshes")
    parser.add_argument("--full-test", action="store_true",
                        help="Use full 20-image test set")
    parser.add_argument("--vertex-color", action="store_true",
                        help="Generate meshes with vertex colors")
    parser.add_argument("--resolution", type=int, default=128,
                        help="Mesh extraction resolution (default: 128)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Measure forward pass latency")
    args = parser.parse_args()

    for r in args.resolutions:
        if r % 16 != 0:
            print(f"ERROR: Resolution {r} not divisible by 16 (ViT patch size)")
            sys.exit(1)

    if args.device == "auto":
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    test_images = collect_test_images(full=args.full_test)
    print(f"Test images: {len(test_images)}, device: {device}, "
          f"mesh_resolution: {args.resolution}, vertex_color: {args.vertex_color}")
    for p in test_images:
        print(f"  {p.name}")

    model = load_model(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"baseline cond_image_size: {model.cfg.cond_image_size}")

    save_dir = None if args.no_save else OUTPUT_DIR

    # --- Baseline at 512 ---
    print("\nGenerating baseline meshes (512x512, 1025 tokens)...")
    baseline_meshes = {}
    baseline_fwd_times = []

    if args.benchmark and test_images:
        run_inference(model, test_images[0], device,
                      resolution=args.resolution, vertex_color=False)

    for img_path in test_images:
        name = img_path.stem
        print(f"  {name}...", end="", flush=True)
        mesh, fwd_t = run_inference(model, img_path, device,
                                    resolution=args.resolution,
                                    vertex_color=args.vertex_color)
        baseline_meshes[name] = mesh
        baseline_fwd_times.append(fwd_t)
        lat_str = f" fwd={fwd_t:.2f}s" if args.benchmark else ""
        print(f" {mesh.vertices.shape[0]} verts{lat_str}")

        if save_dir:
            bl_dir = save_dir / "baseline_512"
            bl_dir.mkdir(parents=True, exist_ok=True)
            mesh.export(str(bl_dir / f"{name}.obj"))

    if args.benchmark:
        avg_bl = np.mean(baseline_fwd_times)
        print(f"  Baseline avg forward: {avg_bl:.2f}s ({len(test_images)} images)")

    # --- Test each resolution ---
    all_results = []
    for res in args.resolutions:
        result = run_resolution_experiment(
            model, device, res, test_images, baseline_meshes,
            save_dir, args.resolution, args.vertex_color, args.benchmark)

        if args.benchmark and avg_bl > 0:
            speedup = avg_bl / result["avg_fwd_time"] if result["avg_fwd_time"] > 0 else 0
            print(f"    Speedup vs baseline: {speedup:.2f}x")

        all_results.append(result)

    # --- Combined with layer pruning ---
    if args.combine_layers:
        for res in args.resolutions:
            result = run_combined_experiment(
                model, device, res, args.combine_layers,
                test_images, baseline_meshes,
                save_dir, args.resolution, args.vertex_color, args.benchmark)

            if args.benchmark and avg_bl > 0:
                speedup = avg_bl / result["avg_fwd_time"] if result["avg_fwd_time"] > 0 else 0
                print(f"    Speedup vs baseline: {speedup:.2f}x")

            all_results.append(result)

    # --- Summary table ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':<25} {'Tokens':>6} {'CD%':>7} {'F@1%':>7} {'F@2%':>7}", end="")
    if args.benchmark:
        print(f" {'Fwd(s)':>7} {'Speedup':>7}", end="")
    print()
    print("-" * 70)

    print(f"{'baseline (512)':<25} {'1025':>6} {'0.000':>7} {'100.0':>7} {'100.0':>7}", end="")
    if args.benchmark:
        print(f" {avg_bl:>7.2f} {'1.00x':>7}", end="")
    print()

    for r in all_results:
        if "removed_blocks" in r:
            name = f"res_{r['cond_image_size']}_{r['n_remaining']}L"
        else:
            name = f"res_{r['cond_image_size']}"
        speedup = avg_bl / r["avg_fwd_time"] if args.benchmark and r["avg_fwd_time"] > 0 else 0
        print(f"{name:<25} {r['tokens']:>6} {r['avg_cd']:>7.3f} {r['avg_f1']:>7.1f} "
              f"{r['avg_f2']:>7.1f}", end="")
        if args.benchmark:
            print(f" {r['avg_fwd_time']:>7.2f} {speedup:>6.2f}x", end="")
        print()


if __name__ == "__main__":
    main()
