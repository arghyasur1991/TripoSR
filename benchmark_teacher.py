"""Baseline profiling of the TripoSR teacher model.

Instruments TSR.forward() and extract_mesh() to measure per-component timing:
  - Image preprocessing
  - Encoder (DINOv2 image_tokenizer)
  - Triplane tokenizer (query generation)
  - Backbone (Transformer1D, per-layer)
  - Post-processor (detokenize + upsample)
  - Mesh extraction (marching cubes + optional vertex color)

Usage:
    python benchmark_teacher.py                         # default: 10 runs, first test image
    python benchmark_teacher.py --image path/to/img.png
    python benchmark_teacher.py --runs 20 --device mps
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from image_utils import prepare_image
from tsr.system import TSR


class Timer:
    """Accumulates timing measurements for named sections."""

    def __init__(self):
        self.records: dict[str, list[float]] = {}
        self._start: float = 0.0

    def start(self):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()

    def stop(self, name: str):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start
        self.records.setdefault(name, []).append(elapsed)

    def report(self, skip_first: int = 1) -> str:
        lines = []
        total_mean = 0.0
        lines.append(f"{'Component':<35} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}  (seconds)")
        lines.append("-" * 80)
        for name, times in self.records.items():
            t = times[skip_first:] if len(times) > skip_first else times
            arr = np.array(t)
            mean, std, mn, mx = arr.mean(), arr.std(), arr.min(), arr.max()
            lines.append(f"{name:<35} {mean:8.4f} {std:8.4f} {mn:8.4f} {mx:8.4f}")
            if not name.startswith("  "):
                total_mean += mean
        lines.append("-" * 80)
        lines.append(f"{'TOTAL (sum of top-level)':<35} {total_mean:8.4f}")
        return "\n".join(lines)


def profile_forward(model: TSR, image: Image.Image, device: str, timer: Timer):
    """Profile a single forward pass with per-component timing."""
    from einops import rearrange

    timer.start()
    rgb_cond = model.image_processor(image, model.cfg.cond_image_size)[:, None].to(device)
    timer.stop("1. Image preprocessing")

    timer.start()
    input_image_tokens = model.image_tokenizer(
        rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1),
    )
    input_image_tokens = rearrange(input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1)
    timer.stop("2. Encoder (DINOv2)")

    batch_size = rgb_cond.shape[0]

    timer.start()
    tokens = model.tokenizer(batch_size)
    timer.stop("3. Triplane tokenizer")

    backbone = model.backbone
    batch, _, seq_len = tokens.shape
    residual = tokens

    timer.start()
    hidden_states = backbone.norm(tokens)
    inner_dim = hidden_states.shape[1]
    hidden_states = hidden_states.permute(0, 2, 1).reshape(batch, seq_len, inner_dim)
    hidden_states = backbone.proj_in(hidden_states)

    for i, block in enumerate(backbone.transformer_blocks):
        layer_start = time.perf_counter()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        hidden_states = block(
            hidden_states,
            encoder_hidden_states=input_image_tokens,
        )
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
        layer_elapsed = time.perf_counter() - layer_start
        timer.records.setdefault(f"  4.{i:02d} backbone layer {i}", []).append(layer_elapsed)

    hidden_states = backbone.proj_out(hidden_states)
    hidden_states = (
        hidden_states.reshape(batch, seq_len, inner_dim)
        .permute(0, 2, 1)
        .contiguous()
    )
    hidden_states = hidden_states + residual
    timer.stop("4. Backbone (total)")

    timer.start()
    scene_codes = model.post_processor(model.tokenizer.detokenize(hidden_states))
    timer.stop("5. Post-processor (detokenize + upsample)")

    timer.start()
    model.set_marching_cubes_resolution(256)
    meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=256)
    timer.stop("6. Mesh extraction")

    return scene_codes, meshes


def main():
    parser = argparse.ArgumentParser(description="Benchmark TripoSR teacher model")
    parser.add_argument("--image", type=Path, help="Input image (default: first test image)")
    parser.add_argument("--runs", type=int, default=10, help="Number of profiling runs")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, mps, cuda")
    parser.add_argument("--skip-mesh", action="store_true", help="Skip mesh extraction timing")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Device: {device}")

    if args.image:
        img_path = args.image
    else:
        test_dir = Path(__file__).parent / "test_images"
        candidates = list(test_dir.glob("examples/*.png")) + list(test_dir.glob("novel/*_nobg.png"))
        if not candidates:
            print("No test images found. Provide --image or populate test_images/")
            sys.exit(1)
        img_path = candidates[0]

    print(f"Image: {img_path}")
    image = prepare_image(img_path)

    print("Loading model from HuggingFace (stabilityai/TripoSR)...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.to(device)
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    timer = Timer()

    print(f"\nRunning {args.runs} profiling passes (first is warmup)...\n")
    with torch.no_grad():
        for i in range(args.runs):
            scene_codes, meshes = profile_forward(model, image, device, timer)
            verts = meshes[0].vertices.shape[0] if meshes else 0
            print(f"  Run {i+1}/{args.runs}: {verts} vertices")

    print(f"\n{timer.report(skip_first=1)}")

    param_counts = {
        "image_tokenizer": sum(p.numel() for p in model.image_tokenizer.parameters()),
        "tokenizer": sum(p.numel() for p in model.tokenizer.parameters()),
        "backbone": sum(p.numel() for p in model.backbone.parameters()),
        "post_processor": sum(p.numel() for p in model.post_processor.parameters()),
        "decoder": sum(p.numel() for p in model.decoder.parameters()),
    }
    print(f"\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count / 1e6:.1f}M")
    print(f"  TOTAL: {sum(param_counts.values()) / 1e6:.1f}M")


if __name__ == "__main__":
    main()
