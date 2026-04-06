"""DINOv2 layer pruning experiment for TripoSR.

Systematically removes layers from the DINOv2 ViT-B/16 image encoder to identify
which layers contribute least to reconstruction quality. The backbone was trained
to cross-attend to DINOv2's output tokens, so pruning changes the feature
representations the backbone sees — quality impact is less predictable than
backbone pruning.

DINOv2 ViT-B/16: 12 transformer layers, 768-dim, 12 heads, ~86.4M params.
Each layer is ~7.1M params. Layers are residual, so removal is straightforward.

Can optionally combine with backbone layer pruning (e.g. deployed 12L config)
to measure the combined effect.

Usage:
    python dino_prune_experiment.py                              # Single-layer removal (all 12)
    python dino_prune_experiment.py --layers 8 9 10 11           # Test specific layers
    python dino_prune_experiment.py --multi 10 11                # Remove multiple layers at once
    python dino_prune_experiment.py --with-backbone-pruning 5 8 12 14  # Combine with backbone pruning
    python dino_prune_experiment.py --cond-image-size 384        # Test at 384x384 (deployed res)
    python dino_prune_experiment.py --full-test --vertex-color   # Full 20-image eval with colors
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

OUTPUT_DIR = Path(__file__).parent / "output" / "dino_pruning"


def remove_dino_layers(model, layer_indices: list[int]):
    """Remove specific layers from DINOv2's ViT encoder (in-place).

    ViT layers are residual — removing a layer means the previous layer's
    output passes directly to the next via the residual stream.
    """
    encoder = model.image_tokenizer.model.encoder
    keep = [i for i in range(len(encoder.layer)) if i not in layer_indices]
    new_layers = torch.nn.ModuleList([encoder.layer[i] for i in keep])
    encoder.layer = new_layers
    return model


def run_dino_pruning_experiment(
    model, device: str, removed_dino_layers: list[int],
    removed_backbone_blocks: list[int] | None,
    test_images: list[Path], baseline_meshes: dict,
    save_dir: Path | None, mesh_resolution: int, vertex_color: bool,
    measure_latency: bool, cond_image_size: int | None,
) -> dict:
    """Remove DINOv2 layers (optionally + backbone blocks) and evaluate quality."""
    n_dino_remaining = 12 - len(removed_dino_layers)
    label = f"dino_remove_{'_'.join(str(l) for l in removed_dino_layers)}"
    if removed_backbone_blocks:
        n_bb = 16 - len(removed_backbone_blocks)
        label += f"_bb{n_bb}L"
    if cond_image_size and cond_image_size != 512:
        label += f"_res{cond_image_size}"

    pruned = copy.deepcopy(model)

    remove_dino_layers(pruned, removed_dino_layers)
    if removed_backbone_blocks:
        remove_blocks(pruned, removed_backbone_blocks)
    if cond_image_size:
        pruned.cfg.cond_image_size = cond_image_size

    pruned.to(device)
    pruned.eval()

    dino_params = sum(p.numel() for p in pruned.image_tokenizer.parameters()) / 1e6
    total_params = sum(p.numel() for p in pruned.parameters()) / 1e6

    desc_parts = [f"DINOv2 {n_dino_remaining}/12 layers"]
    if removed_backbone_blocks:
        desc_parts.append(f"backbone {16 - len(removed_backbone_blocks)}L")
    if cond_image_size and cond_image_size != 512:
        desc_parts.append(f"{cond_image_size}px")
    desc = " + ".join(desc_parts)

    print(f"\n  [{label}] {desc}")
    print(f"    DINOv2: {dino_params:.1f}M params, Total: {total_params:.1f}M params")

    if measure_latency and test_images:
        run_inference(pruned, test_images[0], device, resolution=mesh_resolution,
                      vertex_color=False)

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
        lat_str = f" {fwd_t:.1f}s" if measure_latency else ""
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
        "removed_dino_layers": removed_dino_layers,
        "removed_backbone_blocks": removed_backbone_blocks or [],
        "n_dino_remaining": n_dino_remaining,
        "cond_image_size": cond_image_size or 512,
        "dino_params_M": dino_params,
        "total_params_M": total_params,
        "avg_cd": avg_cd,
        "avg_f1": avg_f1,
        "avg_f2": avg_f2,
        "avg_fwd_time": avg_fwd,
        "per_image": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="DINOv2 layer pruning experiment for TripoSR")
    parser.add_argument("--layers", nargs="+", type=int,
                        help="Specific DINOv2 layers to test individually (default: all 0-11)")
    parser.add_argument("--multi", nargs="+", type=int,
                        help="Remove multiple DINOv2 layers simultaneously")
    parser.add_argument("--with-backbone-pruning", nargs="+", type=int, default=None,
                        metavar="BLOCK",
                        help="Also apply backbone block pruning (e.g. 5 8 12 14 for deployed 12L)")
    parser.add_argument("--cond-image-size", type=int, default=None,
                        help="Input image resolution (default: model default 512; use 384 for deployed)")
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

    if args.device == "auto":
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    test_images = collect_test_images(full=args.full_test)
    print(f"Test images: {len(test_images)}, device: {device}, "
          f"resolution: {args.resolution}, vertex_color: {args.vertex_color}")
    for p in test_images:
        print(f"  {p.name}")

    model = load_model(device)
    n_dino_layers = len(model.image_tokenizer.model.encoder.layer)
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"DINOv2: {n_dino_layers} layers, "
          f"Backbone: {len(model.backbone.transformer_blocks)} blocks")

    if args.with_backbone_pruning:
        bb_str = ", ".join(str(b) for b in args.with_backbone_pruning)
        n_bb_remaining = 16 - len(args.with_backbone_pruning)
        print(f"  + Backbone pruning: remove blocks [{bb_str}] -> {n_bb_remaining}L")

    if args.cond_image_size:
        print(f"  + Input resolution: {args.cond_image_size}x{args.cond_image_size}")

    save_dir = None if args.no_save else OUTPUT_DIR

    # --- Baseline ---
    # Baseline uses the same backbone pruning and resolution config (if specified)
    # so we isolate the DINOv2 pruning effect
    baseline_model = copy.deepcopy(model)
    if args.with_backbone_pruning:
        remove_blocks(baseline_model, args.with_backbone_pruning)
    if args.cond_image_size:
        baseline_model.cfg.cond_image_size = args.cond_image_size
    baseline_model.to(device)
    baseline_model.eval()

    bl_label = "baseline"
    if args.with_backbone_pruning:
        bl_label += f"_{16 - len(args.with_backbone_pruning)}L"
    if args.cond_image_size and args.cond_image_size != 512:
        bl_label += f"_res{args.cond_image_size}"

    bl_params = sum(p.numel() for p in baseline_model.parameters()) / 1e6
    print(f"\nGenerating baseline meshes ({bl_label}, {bl_params:.1f}M params)...")
    baseline_meshes = {}
    baseline_fwd_times = []

    if args.benchmark and test_images:
        run_inference(baseline_model, test_images[0], device,
                      resolution=args.resolution, vertex_color=False)

    for img_path in test_images:
        name = img_path.stem
        print(f"  {name}...", end="", flush=True)
        mesh, fwd_t = run_inference(baseline_model, img_path, device,
                                    resolution=args.resolution,
                                    vertex_color=args.vertex_color)
        baseline_meshes[name] = mesh
        baseline_fwd_times.append(fwd_t)
        lat_str = f" fwd={fwd_t:.2f}s" if args.benchmark else ""
        print(f" {mesh.vertices.shape[0]} verts{lat_str}")

        if save_dir:
            bl_dir = save_dir / bl_label
            bl_dir.mkdir(parents=True, exist_ok=True)
            mesh.export(str(bl_dir / f"{name}.obj"))

    if args.benchmark:
        avg_bl = np.mean(baseline_fwd_times)
        print(f"  Baseline avg forward: {avg_bl:.2f}s ({len(test_images)} images)")

    del baseline_model

    # --- Multi-layer removal ---
    if args.multi:
        for l in args.multi:
            if l < 0 or l >= n_dino_layers:
                print(f"ERROR: DINOv2 layer {l} out of range [0, {n_dino_layers-1}]")
                sys.exit(1)

        print(f"\n{'='*70}")
        print(f"Multi-layer DINOv2 removal: removing layers {args.multi}")
        print(f"{'='*70}")
        result = run_dino_pruning_experiment(
            model, device, args.multi, args.with_backbone_pruning,
            test_images, baseline_meshes, save_dir,
            args.resolution, args.vertex_color, args.benchmark,
            args.cond_image_size)

        if args.benchmark:
            pruned_fwd = result["avg_fwd_time"]
            speedup = avg_bl / pruned_fwd if pruned_fwd > 0 else 0
            print(f"\n  PERFORMANCE: baseline={avg_bl:.2f}s  pruned={pruned_fwd:.2f}s  "
                  f"speedup={speedup:.2f}x")
        return

    # --- Single-layer removal analysis ---
    layers_to_test = args.layers if args.layers else list(range(n_dino_layers))
    print(f"\n{'='*70}")
    print(f"Single DINOv2 layer removal analysis: {len(layers_to_test)} layers")
    print(f"{'='*70}")

    all_results = []
    for layer_idx in layers_to_test:
        if layer_idx < 0 or layer_idx >= n_dino_layers:
            print(f"  Skipping layer {layer_idx} (out of range)")
            continue
        result = run_dino_pruning_experiment(
            model, device, [layer_idx], args.with_backbone_pruning,
            test_images, baseline_meshes, save_dir,
            args.resolution, args.vertex_color, args.benchmark,
            args.cond_image_size)
        all_results.append(result)

    # --- Ranking ---
    print(f"\n{'='*70}")
    print("DINO LAYER IMPORTANCE RANKING (least important first)")
    print(f"{'='*70}")
    ranked = sorted(all_results, key=lambda r: r["avg_cd"])

    header = f"{'Layer':>6} {'Avg CD%':>8} {'Avg F@1%':>9} {'Avg F@2%':>9}"
    if args.benchmark:
        header += f" {'Fwd(s)':>7}"
    header += f" {'Status':>8}"
    print(header)
    print("-" * len(header))

    for r in ranked:
        layer = r["removed_dino_layers"][0]
        status = "SAFE" if r["avg_cd"] < 2.0 and r["avg_f1"] > 85.0 else "RISKY"
        line = f"  {layer:>4}  {r['avg_cd']:>7.3f}  {r['avg_f1']:>8.1f}  {r['avg_f2']:>8.1f}"
        if args.benchmark:
            line += f"  {r['avg_fwd_time']:>6.2f}"
        line += f"  {status:>7}"
        print(line)

    safe_layers = [r["removed_dino_layers"][0] for r in ranked
                   if r["avg_cd"] < 2.0 and r["avg_f1"] > 85.0]
    print(f"\nSafe to remove (individually): {safe_layers if safe_layers else 'None'}")
    if len(safe_layers) >= 2:
        print(f"Suggested multi-removal test: --multi {' '.join(str(l) for l in safe_layers[:4])}")


if __name__ == "__main__":
    main()
