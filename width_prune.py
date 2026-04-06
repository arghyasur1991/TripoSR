"""Width pruning experiment for TripoSR backbone.

Tests reducing the backbone hidden dimension (1024 → 768/512) by removing
attention heads and proportionally shrinking FFN/LayerNorm/embeddings.
No training — just PyTorch-level weight surgery and quality measurement.

Architecture summary (for pruning):
  Triplane tokenizer: embeddings (3, num_channels=1024, 32, 32)
  Transformer1D:
    GroupNorm(32, 1024)
    proj_in: Linear(1024, 1024)
    16 × BasicTransformerBlock:
      norm1: LN(1024)
      attn1 (self): Q/K/V Linear(1024, 1024), Out Linear(1024, 1024)  [16 heads × 64]
      norm2: LN(1024)
      attn2 (cross): Q Linear(1024, 1024), K/V Linear(768, 1024), Out Linear(1024, 1024)
      norm3: LN(1024)
      ff: GEGLU Linear(1024, 8192) + Linear(4096, 1024)
    proj_out: Linear(1024, 1024)
  Post-processor: ConvTranspose2d(1024, 40)

Usage:
    python width_prune.py --widths 768 512
    python width_prune.py --widths 768 --method magnitude --full-test --benchmark
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from prune_layers import (
    collect_test_images, load_model, run_inference, compute_metrics,
)
from tsr.system import TSR

OUTPUT_DIR = Path(__file__).parent / "output" / "width_pruning"

HEAD_DIM = 64  # fixed per the architecture


def compute_head_importance(model: TSR) -> np.ndarray:
    """Rank attention heads by total weight magnitude across all blocks.

    Returns array of shape (16,) with importance scores per head.
    """
    n_heads = model.backbone.num_attention_heads
    importance = np.zeros(n_heads)

    for block in model.backbone.transformer_blocks:
        for attn in [block.attn1, block.attn2]:
            for proj_name in ["to_q", "to_k", "to_v"]:
                proj = getattr(attn, proj_name, None)
                if proj is None:
                    continue
                w = proj.weight.data  # (out_features, in_features)
                out_dim = w.shape[0]
                n_h = out_dim // HEAD_DIM
                for h in range(n_h):
                    start, end = h * HEAD_DIM, (h + 1) * HEAD_DIM
                    importance[h] += w[start:end].norm().item()
            out_proj = attn.to_out[0]
            w = out_proj.weight.data  # (out_dim, inner_dim)
            in_dim = w.shape[1]
            n_h = in_dim // HEAD_DIM
            for h in range(n_h):
                start, end = h * HEAD_DIM, (h + 1) * HEAD_DIM
                importance[h] += w[:, start:end].norm().item()

    return importance


def get_head_indices(n_keep: int, method: str, model: TSR) -> list[int]:
    """Select which head indices to keep."""
    n_heads = model.backbone.num_attention_heads
    if method == "truncate":
        return list(range(n_keep))
    elif method == "magnitude":
        importance = compute_head_importance(model)
        ranked = np.argsort(importance)[::-1]
        kept = sorted(ranked[:n_keep].tolist())
        print(f"  Head importance: {np.array2string(importance, precision=0, separator=', ')}")
        print(f"  Keeping heads: {kept} (dropped: {sorted(set(range(n_heads)) - set(kept))})")
        return kept
    else:
        raise ValueError(f"Unknown method: {method}")


def head_dim_indices(heads: list[int]) -> list[int]:
    """Convert head indices to flattened dim indices."""
    indices = []
    for h in heads:
        indices.extend(range(h * HEAD_DIM, (h + 1) * HEAD_DIM))
    return indices


def prune_linear(linear: nn.Linear, keep_in: list[int] | None, keep_out: list[int] | None) -> nn.Linear:
    """Create a new Linear layer keeping only specified input/output dims."""
    w = linear.weight.data  # (out_features, in_features)
    b = linear.bias.data if linear.bias is not None else None

    if keep_out is not None:
        w = w[keep_out]
        if b is not None:
            b = b[keep_out]
    if keep_in is not None:
        w = w[:, keep_in]

    new = nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
    new.weight.data = w
    if b is not None:
        new.bias.data = b
    return new


def prune_layernorm(ln: nn.LayerNorm, keep: list[int]) -> nn.LayerNorm:
    """Create a pruned LayerNorm."""
    new = nn.LayerNorm(len(keep), elementwise_affine=ln.elementwise_affine)
    if ln.elementwise_affine:
        new.weight.data = ln.weight.data[keep]
        new.bias.data = ln.bias.data[keep]
    return new


def prune_groupnorm(gn: nn.GroupNorm, keep: list[int]) -> nn.GroupNorm:
    """Create a pruned GroupNorm. Requires len(keep) % num_groups == 0."""
    new_channels = len(keep)
    num_groups = gn.num_groups
    while new_channels % num_groups != 0 and num_groups > 1:
        num_groups //= 2
    new = nn.GroupNorm(num_groups, new_channels, eps=gn.eps, affine=gn.affine)
    if gn.affine:
        new.weight.data = gn.weight.data[keep]
        new.bias.data = gn.bias.data[keep]
    return new


def prune_attention(attn, keep_heads_dims: list[int], keep_hidden: list[int],
                    is_cross: bool, cross_dim: int):
    """Prune an Attention module in-place."""
    new_inner = len(keep_heads_dims)
    new_hidden = len(keep_hidden)
    n_heads = new_inner // HEAD_DIM

    attn.to_q = prune_linear(attn.to_q, keep_hidden, keep_heads_dims)

    if is_cross:
        attn.to_k = prune_linear(attn.to_k, None, keep_heads_dims)
        attn.to_v = prune_linear(attn.to_v, None, keep_heads_dims)
    else:
        attn.to_k = prune_linear(attn.to_k, keep_hidden, keep_heads_dims)
        attn.to_v = prune_linear(attn.to_v, keep_hidden, keep_heads_dims)

    attn.to_out[0] = prune_linear(attn.to_out[0], keep_heads_dims, keep_hidden)

    attn.heads = n_heads
    attn.sliceable_head_dim = n_heads
    attn.inner_dim = new_inner
    attn.query_dim = new_hidden
    attn.out_dim = new_hidden
    if not is_cross:
        attn.cross_attention_dim = new_hidden


def prune_ffn(ff, keep_hidden: list[int], new_ffn_inner: int):
    """Prune a FeedForward module in-place."""
    geglu = ff.net[0]  # GEGLU
    out_linear = ff.net[2]  # Linear(ffn_inner, dim)

    old_w = geglu.proj.weight.data  # (inner*2, dim)
    old_b = geglu.proj.bias.data if geglu.proj.bias is not None else None
    old_inner_x2 = old_w.shape[0]
    old_inner = old_inner_x2 // 2

    keep_ffn_gate = list(range(new_ffn_inner))
    keep_ffn_value = [i + old_inner for i in range(new_ffn_inner)]
    keep_ffn_out = keep_ffn_gate + keep_ffn_value

    new_w = old_w[keep_ffn_out][:, keep_hidden]
    new_geglu = type(geglu)(len(keep_hidden), new_ffn_inner)
    new_geglu.proj = nn.Linear(new_w.shape[1], new_w.shape[0],
                               bias=old_b is not None)
    new_geglu.proj.weight.data = new_w
    if old_b is not None:
        new_geglu.proj.bias.data = old_b[keep_ffn_out]
    ff.net[0] = new_geglu

    ff.net[2] = prune_linear(out_linear, keep_ffn_gate, keep_hidden)


def prune_model_width(model: TSR, target_dim: int, method: str) -> TSR:
    """Prune the model to a smaller hidden dimension."""
    assert target_dim % HEAD_DIM == 0, f"target_dim {target_dim} must be divisible by {HEAD_DIM}"
    n_keep_heads = target_dim // HEAD_DIM
    original_dim = model.backbone.num_attention_heads * model.backbone.attention_head_dim

    kept_heads = get_head_indices(n_keep_heads, method, model)
    keep_heads_dims = head_dim_indices(kept_heads)
    keep_hidden = keep_heads_dims  # hidden dim aligned with head dims
    new_ffn_inner = target_dim * 4  # proportional to hidden dim

    # 1. Triplane tokenizer embedding: (3, 1024, 32, 32) → (3, target, 32, 32)
    old_emb = model.tokenizer.embeddings.data
    model.tokenizer.embeddings = nn.Parameter(old_emb[:, keep_hidden])
    model.tokenizer.cfg.num_channels = target_dim

    # 2. Transformer1D wrapper
    backbone = model.backbone
    backbone.norm = prune_groupnorm(backbone.norm, keep_hidden)
    backbone.proj_in = prune_linear(backbone.proj_in, keep_hidden, keep_hidden)
    backbone.proj_out = prune_linear(backbone.proj_out, keep_hidden, keep_hidden)
    backbone.num_attention_heads = n_keep_heads
    backbone.in_channels = target_dim

    # 3. Each transformer block
    for block in backbone.transformer_blocks:
        block.norm1 = prune_layernorm(block.norm1, keep_hidden)
        prune_attention(block.attn1, keep_heads_dims, keep_hidden,
                        is_cross=False, cross_dim=target_dim)

        if block.attn2 is not None:
            block.norm2 = prune_layernorm(block.norm2, keep_hidden)
            prune_attention(block.attn2, keep_heads_dims, keep_hidden,
                            is_cross=True, cross_dim=768)

        block.norm3 = prune_layernorm(block.norm3, keep_hidden)
        prune_ffn(block.ff, keep_hidden, new_ffn_inner)

    # 4. Post-processor: ConvTranspose2d(1024, 40) → ConvTranspose2d(target, 40)
    old_conv = model.post_processor.upsample
    w = old_conv.weight.data  # (in_channels, out_channels, kH, kW)
    b = old_conv.bias.data if old_conv.bias is not None else None
    new_conv = nn.ConvTranspose2d(target_dim, old_conv.out_channels,
                                  kernel_size=old_conv.kernel_size,
                                  stride=old_conv.stride)
    new_conv.weight.data = w[keep_hidden]
    if b is not None:
        new_conv.bias.data = b
    model.post_processor.upsample = new_conv

    return model


def run_width_experiment(
    model: TSR, device: str, target_dim: int, method: str,
    test_images: list[Path], baseline_meshes: dict,
    save_dir: Path | None, mesh_resolution: int, vertex_color: bool,
    measure_latency: bool,
) -> dict:
    """Prune model to target width and evaluate quality."""
    n_heads = target_dim // HEAD_DIM
    label = f"width_{target_dim}_{method}"

    print(f"\n{'='*60}")
    print(f"Testing width {target_dim} ({n_heads} heads, {method})...")
    print(f"{'='*60}")

    pruned = copy.deepcopy(model)
    prune_model_width(pruned, target_dim, method)
    pruned.to(device)
    pruned.eval()

    total_params = sum(p.numel() for p in pruned.parameters()) / 1e6
    orig_params = sum(p.numel() for p in model.parameters()) / 1e6
    removed = orig_params - total_params
    print(f"  Model: {total_params:.1f}M params (was {orig_params:.1f}M, "
          f"removed {removed:.1f}M / {removed/orig_params*100:.1f}%)")

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
        "target_dim": target_dim,
        "method": method,
        "n_heads": n_heads,
        "total_params_M": total_params,
        "avg_cd": avg_cd,
        "avg_f1": avg_f1,
        "avg_f2": avg_f2,
        "avg_fwd_time": avg_fwd,
        "per_image": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Width pruning experiment for TripoSR")
    parser.add_argument("--widths", nargs="+", type=int, default=[768, 512],
                        help="Target hidden dimensions (must be multiples of 64)")
    parser.add_argument("--method", default="both", choices=["truncate", "magnitude", "both"],
                        help="Head selection method")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--full-test", action="store_true",
                        help="Use full 20-image test set")
    parser.add_argument("--vertex-color", action="store_true")
    parser.add_argument("--resolution", type=int, default=128,
                        help="Mesh extraction resolution")
    parser.add_argument("--benchmark", action="store_true",
                        help="Measure forward pass latency")
    args = parser.parse_args()

    for w in args.widths:
        if w % HEAD_DIM != 0:
            print(f"ERROR: Width {w} not divisible by head_dim={HEAD_DIM}")
            sys.exit(1)

    if args.device == "auto":
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    test_images = collect_test_images(full=args.full_test)
    print(f"Test images: {len(test_images)}, device: {device}, "
          f"mesh_resolution: {args.resolution}, vertex_color: {args.vertex_color}")

    model = load_model(device)
    orig_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {orig_params:.1f}M params, {model.backbone.num_attention_heads} heads × "
          f"{model.backbone.attention_head_dim} head_dim = "
          f"{model.backbone.num_attention_heads * model.backbone.attention_head_dim} hidden")

    save_dir = None if args.no_save else OUTPUT_DIR

    # --- Baseline ---
    print("\nGenerating baseline meshes (full 1024-dim)...")
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
            bl_dir = save_dir / "baseline"
            bl_dir.mkdir(parents=True, exist_ok=True)
            mesh.export(str(bl_dir / f"{name}.obj"))

    avg_bl = np.mean(baseline_fwd_times) if args.benchmark else 0

    if args.benchmark:
        print(f"  Baseline avg forward: {avg_bl:.2f}s ({len(test_images)} images)")

    # --- Width experiments ---
    methods = ["truncate", "magnitude"] if args.method == "both" else [args.method]
    all_results = []

    for width in args.widths:
        for method in methods:
            result = run_width_experiment(
                model, device, width, method,
                test_images, baseline_meshes,
                save_dir, args.resolution, args.vertex_color, args.benchmark)

            if args.benchmark and avg_bl > 0:
                speedup = avg_bl / result["avg_fwd_time"] if result["avg_fwd_time"] > 0 else 0
                print(f"    Speedup vs baseline: {speedup:.2f}x")

            all_results.append(result)

    # --- Summary ---
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Variant':<30} {'Params':>8} {'CD%':>7} {'F@1%':>7} {'F@2%':>7}", end="")
    if args.benchmark:
        print(f" {'Fwd(s)':>7} {'Speedup':>7}", end="")
    print()
    print("-" * 80)

    print(f"{'baseline (1024, 16h)':<30} {orig_params:>7.1f}M {'0.000':>7} {'100.0':>7} {'100.0':>7}", end="")
    if args.benchmark:
        print(f" {avg_bl:>7.2f} {'1.00x':>7}", end="")
    print()

    for r in all_results:
        name = f"w{r['target_dim']}_{r['method'][:4]} ({r['n_heads']}h)"
        speedup = avg_bl / r["avg_fwd_time"] if args.benchmark and r["avg_fwd_time"] > 0 else 0
        print(f"{name:<30} {r['total_params_M']:>7.1f}M {r['avg_cd']:>7.3f} {r['avg_f1']:>7.1f} "
              f"{r['avg_f2']:>7.1f}", end="")
        if args.benchmark:
            print(f" {r['avg_fwd_time']:>7.2f} {speedup:>6.2f}x", end="")
        print()


if __name__ == "__main__":
    main()
