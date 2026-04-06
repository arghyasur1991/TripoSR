"""ToMe grid search: measure FP32 speedup and quality across configurations.

Usage:
    python tome_experiment.py                 # run all configs
    python tome_experiment.py --configs A B C  # run specific configs
    python tome_experiment.py --speed-only     # skip quality (mesh extraction)
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import trimesh
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from tsr.system import TSR
from tome_patch import apply_tome, remove_tome
from reconstruct_compare import (
    prepare_image, chamfer_distance, f_score, load_model,
)

TEST_IMAGES = [
    Path("test_images/examples/robot.png"),
    Path("test_images/examples/hamburger.png"),
    Path("test_images/examples/chair.png"),
]

CONFIGS = {
    # Phase 1: original grid
    "A": {"r": 0.10, "layers": list(range(0, 16, 2))},        # 8 merge points, gentle
    "B": {"r": 0.15, "layers": list(range(0, 16, 2))},        # 8 merge points, moderate
    "C": {"r": 0.20, "layers": list(range(0, 16, 2))},        # 8 merge points, aggressive
    "D": {"r": 0.10, "layers": list(range(16))},               # 16 merge points, gentle
    "E": {"r": 0.15, "layers": list(range(16))},               # 16 merge points
    "F": {"r": 0.20, "layers": [2, 4, 6, 8, 10, 12]},         # 6 merge points, skip ends
    "G": {"r": 0.30, "layers": [4, 8, 12]},                    # 3 merge points, high r
    "H": {"r": 0.25, "layers": [2, 5, 8, 11, 14]},            # 5 evenly spaced
    # Phase 2: refinement — gentler configs
    "I1": {"r": 0.05, "layers": list(range(0, 16, 2))},       # half of A's ratio
    "I2": {"r": 0.075, "layers": list(range(0, 16, 2))},      # between I1 and A
    "I3": {"r": 0.05, "layers": [4, 8, 12]},                  # very gentle, 3 layers
    "I4": {"r": 0.10, "layers": [8, 10, 12, 14]},             # late-only, 4 layers
    "I5": {"r": 0.10, "layers": [10, 12, 14]},                # late-only, 3 layers
    "I6": {"r": 0.05, "layers": list(range(16))},             # very gentle every layer
    # Phase 2b: fine-tuning around I4/I5 sweet spot
    "J1": {"r": 0.10, "layers": [6, 8, 10, 12, 14]},           # extend I4 one layer earlier
    "J2": {"r": 0.15, "layers": [10, 12, 14]},                  # higher r, late-only 3 layers
    "J3": {"r": 0.15, "layers": [8, 10, 12, 14]},               # higher r, late-only 4 layers
    "J4": {"r": 0.12, "layers": [8, 10, 12, 14]},               # slightly above I4 ratio
}


def estimate_final_tokens(r: float, layers: list, initial: int = 1024) -> int:
    """Estimate tokens per plane after all merges."""
    n = initial
    for _ in layers:
        merge_count = int(n * r)
        if merge_count > 0 and n > 2 * merge_count:
            n -= merge_count
    return n


def forward_pass(model: TSR, image_path: Path, device: str) -> torch.Tensor:
    """Run forward pass only (no mesh extraction), return scene_codes."""
    img = prepare_image(image_path)
    with torch.no_grad():
        scene_codes = model(img, device)
    return scene_codes


def benchmark_speed(model: TSR, image_path: Path, device: str,
                    n_warmup: int = 1, n_runs: int = 3) -> float:
    """Return mean forward-pass latency in seconds."""
    img = prepare_image(image_path)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(img, device)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()

        times = []
        for _ in range(n_runs):
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            t0 = time.perf_counter()
            _ = model(img, device)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            times.append(time.perf_counter() - t0)
    return float(np.mean(times))


def extract_mesh(model: TSR, scene_codes: torch.Tensor) -> trimesh.Trimesh:
    """Extract mesh from scene_codes using the model's built-in method."""
    with torch.no_grad():
        model.set_marching_cubes_resolution(256)
        meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=256)
    return meshes[0]


def measure_quality(mesh: trimesh.Trimesh, baseline_mesh: trimesh.Trimesh) -> dict:
    """Compute CD% and F-scores against baseline."""
    cd = chamfer_distance(mesh, baseline_mesh)
    f1 = f_score(mesh, baseline_mesh, tau_pct=1.0)
    f2 = f_score(mesh, baseline_mesh, tau_pct=2.0)
    return {"cd": cd, "f1": f1, "f2": f2}


def run_experiment(model: TSR, device: str, config_name: str, config: dict,
                   baseline_times: dict, baseline_meshes: dict,
                   speed_only: bool = False) -> dict:
    """Run a single ToMe configuration and return results."""
    r = config["r"]
    layers = config["layers"]
    est_tokens = estimate_final_tokens(r, layers)

    print(f"\n{'='*60}")
    print(f"Config {config_name}: r={r}, layers={layers}")
    print(f"  Est. final tokens/plane: {est_tokens} (from 1024)")
    print(f"{'='*60}")

    apply_tome(model.backbone, merge_ratio=r, merge_layers=layers)

    # Speed benchmark
    times = {}
    for img_path in TEST_IMAGES:
        name = img_path.stem
        t = benchmark_speed(model, img_path, device)
        speedup = baseline_times[name] / t if t > 0 else 0
        times[name] = {"time": t, "speedup": speedup}
        print(f"  {name}: {t:.3f}s (baseline {baseline_times[name]:.3f}s, {speedup:.2f}x)")

    avg_speedup = np.mean([v["speedup"] for v in times.values()])
    avg_time = np.mean([v["time"] for v in times.values()])
    print(f"  AVG: {avg_time:.3f}s ({avg_speedup:.2f}x speedup)")

    # Quality benchmark
    qualities = {}
    if not speed_only:
        for img_path in TEST_IMAGES:
            name = img_path.stem
            scene_codes = forward_pass(model, img_path, device)
            mesh = extract_mesh(model, scene_codes)
            q = measure_quality(mesh, baseline_meshes[name])
            qualities[name] = q
            status = "PASS" if q["cd"] < 2.0 and q["f1"] > 85.0 else "FAIL"
            print(f"  {name}: CD={q['cd']:.3f}% F@1%={q['f1']:.1f} F@2%={q['f2']:.1f} [{status}]")

        avg_cd = np.mean([v["cd"] for v in qualities.values()])
        avg_f1 = np.mean([v["f1"] for v in qualities.values()])
        print(f"  AVG quality: CD={avg_cd:.3f}% F@1%={avg_f1:.1f}")

    remove_tome(model.backbone)

    return {
        "config": config_name,
        "r": r,
        "layers": layers,
        "est_tokens": est_tokens,
        "avg_speedup": float(avg_speedup),
        "avg_time": float(avg_time),
        "times": times,
        "qualities": qualities,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                        help="Which configs to run (default: all)")
    parser.add_argument("--speed-only", action="store_true",
                        help="Skip quality measurement (mesh extraction)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    print("Loading model...")
    model = load_model(device)

    # Baseline measurements
    print("\n" + "="*60)
    print("BASELINE (no ToMe)")
    print("="*60)

    baseline_times = {}
    for img_path in TEST_IMAGES:
        name = img_path.stem
        t = benchmark_speed(model, img_path, device)
        baseline_times[name] = t
        print(f"  {name}: {t:.3f}s")

    baseline_meshes = {}
    if not args.speed_only:
        print("  Generating baseline meshes...")
        for img_path in TEST_IMAGES:
            name = img_path.stem
            scene_codes = forward_pass(model, img_path, device)
            baseline_meshes[name] = extract_mesh(model, scene_codes)
            print(f"    {name}: {baseline_meshes[name].vertices.shape[0]} verts")

    # Run experiments
    results = []
    for config_name in args.configs:
        if config_name not in CONFIGS:
            print(f"Unknown config: {config_name}, skipping")
            continue
        result = run_experiment(
            model, device, config_name, CONFIGS[config_name],
            baseline_times, baseline_meshes, args.speed_only
        )
        results.append(result)

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    header = f"{'Config':>6} {'r':>5} {'#Layers':>7} {'EstTok':>6} {'Speedup':>8} {'AvgTime':>8}"
    if not args.speed_only:
        header += f" {'CD%':>7} {'F@1%':>6} {'Status':>6}"
    print(header)
    print("-" * len(header))

    for r in results:
        line = (f"{r['config']:>6} {r['r']:>5.2f} {len(r['layers']):>7} "
                f"{r['est_tokens']:>6} {r['avg_speedup']:>7.2f}x {r['avg_time']:>7.3f}s")
        if not args.speed_only and r["qualities"]:
            avg_cd = np.mean([v["cd"] for v in r["qualities"].values()])
            avg_f1 = np.mean([v["f1"] for v in r["qualities"].values()])
            status = "PASS" if avg_cd < 2.0 and avg_f1 > 85.0 else "FAIL"
            line += f" {avg_cd:>7.3f} {avg_f1:>5.1f} {status:>6}"
        print(line)

    # Highlight best configs
    if not args.speed_only:
        passing = [r for r in results if r["qualities"]
                   and np.mean([v["cd"] for v in r["qualities"].values()]) < 2.0
                   and np.mean([v["f1"] for v in r["qualities"].values()]) > 85.0]
        if passing:
            best = max(passing, key=lambda r: r["avg_speedup"])
            print(f"\nBest passing config: {best['config']} "
                  f"({best['avg_speedup']:.2f}x speedup)")
        else:
            print("\nNo config passed quality threshold (CD<2%, F@1%>85%)")


if __name__ == "__main__":
    main()
