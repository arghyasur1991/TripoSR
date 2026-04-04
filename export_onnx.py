"""Export TripoSR teacher model to ONNX with FP16 and INT8 quantization.

Exports the forward pass (preprocessed image -> scene_codes) as ONNX.
Mesh extraction (marching cubes + NeRF decoder) stays in Python/C++ on CPU.

Also exports the NeRF decoder separately for on-device mesh extraction.

All target ONNX opset 15 for Unity Sentis 2.5.0 compatibility.

Usage:
    python export_onnx.py                       # FP32 + FP16 + INT8
    python export_onnx.py --fp32-only           # FP32 only
    python export_onnx.py --benchmark           # export + benchmark
    python export_onnx.py --benchmark-only      # benchmark existing models
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

sys.path.insert(0, str(Path(__file__).parent))
from image_utils import prepare_image
from tsr.system import TSR
from tsr.utils import ImagePreprocessor

MODELS_DIR = Path(__file__).parent / "models"


def log(msg: str):
    print(msg, flush=True)


class TripoSRForward(nn.Module):
    """Wrapper: preprocessed image (B,3,512,512) -> scene_codes (B,3,40,64,64)."""

    def __init__(self, model: TSR):
        super().__init__()
        self.image_tokenizer = model.image_tokenizer
        self.tokenizer = model.tokenizer
        self.backbone = model.backbone
        self.post_processor = model.post_processor

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch_size = image.shape[0]

        input_image_tokens = self.image_tokenizer(image.unsqueeze(1))
        input_image_tokens = rearrange(
            input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1
        )

        tokens = self.tokenizer(batch_size)
        tokens = self.backbone(
            tokens,
            encoder_hidden_states=input_image_tokens,
        )

        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))
        return scene_codes


class TripoSRForwardToMe(nn.Module):
    """Wrapper with inline Token Merging: image (B,3,512,512) -> scene_codes (B,3,40,64,64).

    Bakes the ToMe merge/unmerge ops directly into the traced forward graph,
    avoiding Python-level mutable state that can't survive ONNX tracing.
    Merge decisions are data-dependent (computed at runtime from token similarity).
    """

    TOKENS_PER_PLANE = 1024
    NUM_PLANES = 3

    def __init__(self, model: TSR, merge_ratio: float = 0.1,
                 merge_layers: list | None = None):
        super().__init__()
        self.image_tokenizer = model.image_tokenizer
        self.tokenizer = model.tokenizer
        self.post_processor = model.post_processor
        # Store backbone sub-modules directly so tracer sees them
        self.bb_norm = model.backbone.norm
        self.bb_proj_in = model.backbone.proj_in
        self.bb_proj_out = model.backbone.proj_out
        self.bb_blocks = model.backbone.transformer_blocks

        self.merge_ratio = merge_ratio
        _layers = merge_layers or [4, 8, 12]
        self.merge_layer_set = set(_layers)

    @staticmethod
    def _bipartite_soft_matching(metric: torch.Tensor, r: int):
        """Bipartite soft matching — returns (kept_idx, src_idx, dst_merge_target)."""
        B, N, C = metric.shape
        metric = F.normalize(metric, dim=-1)

        a_idx = torch.arange(0, N, 2, device=metric.device)
        b_idx = torch.arange(1, N, 2, device=metric.device)

        scores = torch.bmm(metric[:, a_idx], metric[:, b_idx].transpose(1, 2))
        node_max, node_idx = scores.max(dim=-1)
        _, sorted_indices = node_max.sort(dim=-1, descending=True)

        merge_src_local = sorted_indices[:, :r]
        keep_src_local = sorted_indices[:, r:]

        src_idx = a_idx[merge_src_local]
        dst_local = torch.gather(node_idx, 1, merge_src_local)
        dst_merge_target = b_idx[dst_local]

        keep_a = a_idx[keep_src_local]
        keep_b = b_idx.unsqueeze(0).expand(B, -1)
        kept_idx = torch.cat([keep_a, keep_b], dim=1)
        kept_idx, _ = kept_idx.sort(dim=1)

        return kept_idx, src_idx, dst_merge_target

    @staticmethod
    def _find_positions(kept_idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """ONNX-friendly replacement for torch.searchsorted.

        For each target value, finds its index in the sorted kept_idx tensor
        via broadcast comparison + argmax.
        """
        # kept_idx: (B, K), targets: (B, r)
        match = (kept_idx.unsqueeze(-1) == targets.unsqueeze(1))  # (B, K, r)
        return match.to(torch.int64).argmax(dim=1)  # (B, r)

    @staticmethod
    def _merge_tokens(x: torch.Tensor, kept_idx: torch.Tensor,
                      src_idx: torch.Tensor, dst_merge_target: torch.Tensor):
        B, N, C = x.shape
        merged = torch.gather(x, 1, kept_idx.unsqueeze(-1).expand(-1, -1, C))
        src_tokens = torch.gather(x, 1, src_idx.unsqueeze(-1).expand(-1, -1, C))
        dst_positions = TripoSRForwardToMe._find_positions(kept_idx, dst_merge_target)
        dst_positions = dst_positions.clamp(0, kept_idx.shape[1] - 1)
        merged.scatter_add_(1, dst_positions.unsqueeze(-1).expand(-1, -1, C), src_tokens)
        counts = torch.ones(B, merged.shape[1], 1, device=x.device, dtype=x.dtype)
        ones_r = torch.ones(B, src_idx.shape[1], 1, device=x.device, dtype=x.dtype)
        counts.scatter_add_(1, dst_positions.unsqueeze(-1), ones_r)
        return merged / counts

    @staticmethod
    def _unmerge_tokens(merged: torch.Tensor, kept_idx: torch.Tensor,
                        src_idx: torch.Tensor, dst_merge_target: torch.Tensor,
                        original_n: int):
        B, _, C = merged.shape
        output = torch.zeros(B, original_n, C, device=merged.device, dtype=merged.dtype)
        output.scatter_(1, kept_idx.unsqueeze(-1).expand(-1, -1, C), merged)
        dst_values = torch.gather(output, 1, dst_merge_target.unsqueeze(-1).expand(-1, -1, C))
        output.scatter_(1, src_idx.unsqueeze(-1).expand(-1, -1, C), dst_values)
        return output

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch_size = image.shape[0]

        input_image_tokens = self.image_tokenizer(image.unsqueeze(1))
        input_image_tokens = rearrange(
            input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1
        )

        tokens = self.tokenizer(batch_size)

        # --- Inline backbone forward with ToMe ---
        hidden_states = tokens
        batch, _, seq_len = hidden_states.shape
        residual_full = hidden_states

        hidden_states = self.bb_norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 1).reshape(batch, seq_len, inner_dim)
        hidden_states = self.bb_proj_in(hidden_states)

        # merge_info: list of per-layer lists of (kept_idx, src_idx, dst, orig_n) per plane
        all_merge_info = []
        current_n = [self.TOKENS_PER_PLANE] * self.NUM_PLANES

        for layer_idx in range(len(self.bb_blocks)):
            if layer_idx in self.merge_layer_set:
                # Within-plane merge
                planes = []
                layer_info = []
                offset = 0
                new_n = list(current_n)
                for p in range(self.NUM_PLANES):
                    n_p = current_n[p]
                    plane_tokens = hidden_states[:, offset:offset + n_p, :]
                    r = int(n_p * self.merge_ratio)
                    if r > 0 and n_p > 2 * r:
                        ki, si, dm = self._bipartite_soft_matching(plane_tokens, r)
                        merged_plane = self._merge_tokens(plane_tokens, ki, si, dm)
                        planes.append(merged_plane)
                        layer_info.append((ki, si, dm, n_p))
                        new_n[p] = n_p - r
                    else:
                        planes.append(plane_tokens)
                        layer_info.append(None)
                    offset += n_p
                hidden_states = torch.cat(planes, dim=1)
                all_merge_info.append(layer_info)
                current_n = new_n

            hidden_states = self.bb_blocks[layer_idx](
                hidden_states,
                encoder_hidden_states=input_image_tokens,
            )

        # Unmerge all layers in reverse
        for layer_info in reversed(all_merge_info):
            B_um, N_um, C_um = hidden_states.shape
            planes = []
            offset = 0
            plane_sizes = []
            remaining = N_um
            for p in range(self.NUM_PLANES):
                if layer_info[p] is not None:
                    _, _, _, orig_n = layer_info[p]
                    r = layer_info[p][1].shape[1]
                    plane_sizes.append(orig_n - r)
                    remaining -= (orig_n - r)
                else:
                    sz = remaining // (self.NUM_PLANES - p)
                    plane_sizes.append(sz)
                    remaining -= sz
            for p in range(self.NUM_PLANES):
                n_p = plane_sizes[p]
                plane_tokens = hidden_states[:, offset:offset + n_p, :]
                if layer_info[p] is not None:
                    ki, si, dm, orig_n = layer_info[p]
                    plane_tokens = self._unmerge_tokens(plane_tokens, ki, si, dm, orig_n)
                planes.append(plane_tokens)
                offset += n_p
            hidden_states = torch.cat(planes, dim=1)

        hidden_states = self.bb_proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, seq_len, inner_dim)
            .permute(0, 2, 1)
            .contiguous()
        )
        output = hidden_states + residual_full
        # --- End inline backbone ---

        scene_codes = self.post_processor(self.tokenizer.detokenize(output))
        return scene_codes


class DecoderWrapper(nn.Module):
    """Wrapper for NeRF MLP decoder: triplane features (N, 120) -> density+color (N, 4)."""

    def __init__(self, decoder):
        super().__init__()
        self.layers = decoder.layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def load_teacher(device: str = "cpu") -> TSR:
    model = TSR.from_pretrained(
        "stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt"
    )
    model.to(device)
    model.eval()
    return model


def get_dummy_image(device: str = "cpu") -> torch.Tensor:
    """Preprocessed image tensor (1, 3, 512, 512) in [0, 1]."""
    img = prepare_image(Path(__file__).parent / "test_images" / "examples" / "flamingo.png")
    processor = ImagePreprocessor()
    rgb = processor(img, 512)  # (B, H, W, C)
    return rgb.permute(0, 3, 1, 2).to(device)  # (B, C, H, W)


# ---------------------------------------------------------------------------
# FP32 export
# ---------------------------------------------------------------------------

def _export_wrapper_fp32(wrapper: nn.Module, output_path: Path, opset: int = 15,
                         label: str = "FP32") -> torch.Tensor:
    """Export an nn.Module wrapper (image -> scene_codes) to ONNX FP32."""
    log(f"Exporting {label} to {output_path} (opset {opset})...")
    wrapper = wrapper.cpu()
    wrapper.eval()

    dummy = get_dummy_image("cpu")
    log(f"  Input: {dummy.shape}, dtype={dummy.dtype}")

    with torch.no_grad():
        ref_out = wrapper(dummy)
    log(f"  Output: {ref_out.shape}, range=[{ref_out.min():.2f}, {ref_out.max():.2f}]")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(output_path),
        opset_version=opset,
        input_names=["image"],
        output_names=["scene_codes"],
        dynamic_axes={
            "image": {0: "batch"},
            "scene_codes": {0: "batch"},
        },
        do_constant_folding=True,
        dynamo=False,
    )
    log(f"  Export done in {time.time()-t0:.1f}s")

    fsize = output_path.stat().st_size / 1e6
    log(f"  {label} ONNX: {fsize:.1f}MB")
    return ref_out


def export_fp32(model: TSR, output_path: Path, opset: int = 15):
    """Export the full forward pass (vanilla) to ONNX FP32."""
    wrapper = TripoSRForward(model)
    return _export_wrapper_fp32(wrapper, output_path, opset, "FP32")


def export_tome_fp32(model: TSR, output_path: Path, opset: int = 15,
                     merge_ratio: float = 0.1,
                     merge_layers: list | None = None):
    """Export the forward pass with ToMe to ONNX FP32."""
    wrapper = TripoSRForwardToMe(model, merge_ratio, merge_layers)
    label = f"ToMe r={merge_ratio} FP32"
    return _export_wrapper_fp32(wrapper, output_path, opset, label)


def export_decoder(model: TSR, output_path: Path, opset: int = 15):
    """Export the NeRF MLP decoder to ONNX FP32."""
    log(f"Exporting NeRF decoder to {output_path} (opset {opset})...")
    decoder = model.decoder
    in_ch = decoder.cfg.in_channels
    log(f"  Decoder: {sum(p.numel() for p in decoder.parameters())/1e3:.1f}K params, in_channels={in_ch}")

    wrapper = DecoderWrapper(decoder).cpu()
    wrapper.eval()

    dummy = torch.randn(1024, in_ch)
    with torch.no_grad():
        ref_out = wrapper(dummy)
    log(f"  Output: {ref_out.shape}, range=[{ref_out.min():.2f}, {ref_out.max():.2f}]")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(output_path),
        opset_version=opset,
        input_names=["triplane_features"],
        output_names=["density_color"],
        dynamic_axes={
            "triplane_features": {0: "num_points"},
            "density_color": {0: "num_points"},
        },
        do_constant_folding=True,
        dynamo=False,
    )

    fsize = output_path.stat().st_size / 1e6
    log(f"  Decoder FP32 ONNX: {fsize:.3f}MB")
    return ref_out, dummy


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def convert_fp16(input_path: Path, output_path: Path):
    """Convert FP32 ONNX to FP16 using ORT transformer optimizer."""
    from onnxruntime.transformers.optimizer import optimize_model

    log(f"Converting to FP16: {output_path}")
    t0 = time.time()
    opt = optimize_model(str(input_path), opt_level=0)
    opt.convert_float_to_float16(
        use_symbolic_shape_infer=True,
        keep_io_types=True,
    )
    opt.save_model_to_file(str(output_path))

    fsize = output_path.stat().st_size / 1e6
    fp32_size = input_path.stat().st_size / 1e6
    log(f"  FP16 ONNX: {fsize:.1f}MB (was {fp32_size:.1f}MB, "
        f"{fsize/fp32_size*100:.0f}%) [{time.time()-t0:.1f}s]")


def quantize_int8(input_path: Path, output_path: Path):
    """Apply dynamic INT8 quantization (MatMul weights only)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    log(f"Quantizing to INT8: {output_path}")
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        op_types_to_quantize=["MatMul"],
    )

    fsize = output_path.stat().st_size / 1e6
    fp32_size = input_path.stat().st_size / 1e6
    log(f"  INT8 ONNX: {fsize:.1f}MB (was {fp32_size:.1f}MB, "
        f"{fsize/fp32_size*100:.0f}%)")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_onnx(onnx_path: Path, dummy_input: np.ndarray,
                ref_output: np.ndarray, input_name: str = "image",
                label: str = "") -> dict:
    """Verify ONNX model matches PyTorch output."""
    import onnxruntime as ort

    tag = f" [{label}]" if label else ""
    log(f"Verifying{tag}: {onnx_path.name}")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    input_meta = sess.get_inputs()[0]
    inp = dummy_input
    if input_meta.type == "tensor(float16)":
        inp = dummy_input.astype(np.float16)

    ort_out = sess.run(None, {input_name: inp})[0].astype(np.float32)

    abs_diff = np.abs(ort_out - ref_output)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    output_range = float(np.abs(ref_output).max())
    rel_max = max_diff / (output_range + 1e-8)
    rel_mean = mean_diff / (output_range + 1e-8)

    fsize = onnx_path.stat().st_size / 1e6
    log(f"  Size: {fsize:.1f}MB | "
        f"Max err: {max_diff:.4f} ({rel_max*100:.4f}%) | "
        f"Mean err: {mean_diff:.4f} ({rel_mean*100:.4f}%)")

    if rel_max < 0.01:
        log(f"  PASS (<1% relative error)")
    elif rel_max < 0.05:
        log(f"  WARN (<5% relative, acceptable)")
    elif rel_max < 0.15:
        log(f"  OK (<15% relative, expected for quantized)")
    else:
        log(f"  FAIL ({rel_max*100:.1f}% relative error)")

    return {
        "label": label, "size_mb": fsize,
        "max_abs": max_diff, "mean_abs": mean_diff,
        "max_rel": rel_max, "mean_rel": rel_mean,
    }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_onnx(model_path: Path, n_runs: int = 10) -> float:
    """Benchmark an ONNX model with ORT CPUExecutionProvider."""
    import onnxruntime as ort

    log(f"  Benchmarking {model_path.name} ({n_runs} runs)...")
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    dummy = get_dummy_image("cpu").numpy()
    times = []

    for i in range(n_runs + 1):
        t0 = time.perf_counter()
        outputs = sess.run(None, {"image": dummy})
        elapsed = time.perf_counter() - t0
        if i > 0:
            times.append(elapsed)

    arr = np.array(times)
    log(f"  {model_path.name}: {arr.mean():.3f}s +/- {arr.std():.3f}s "
        f"(min={arr.min():.3f}s)")
    return arr.mean()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export TripoSR teacher to ONNX")
    parser.add_argument("--opset", type=int, default=15)
    parser.add_argument("--fp32-only", action="store_true", help="Skip FP16 and INT8")
    parser.add_argument("--skip-decoder", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark after export")
    parser.add_argument("--benchmark-only", action="store_true", help="Only benchmark existing")
    parser.add_argument("--tome", type=float, default=None,
                        help="Export ToMe variant with given merge ratio (e.g. 0.1)")
    parser.add_argument("--tome-layers", nargs="+", type=int, default=[4, 8, 12],
                        help="ToMe merge layers (default: 4 8 12)")
    parser.add_argument("--tome-only", action="store_true",
                        help="Only export ToMe variant (skip vanilla)")
    parser.add_argument("--runs", type=int, default=10, help="Benchmark runs")
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = out_dir / "triposr_fp32.onnx"
    fp16_path = out_dir / "triposr_fp16.onnx"
    int8_path = out_dir / "triposr_int8.onnx"
    dec_fp32_path = out_dir / "nerf_decoder.onnx"

    # ToMe variant paths
    tome_ratio = args.tome
    tome_tag = f"_tome{tome_ratio}" if tome_ratio else ""
    tome_fp32_path = out_dir / f"triposr_tome_fp32.onnx"
    tome_fp16_path = out_dir / f"triposr_tome_fp16.onnx"
    tome_int8_path = out_dir / f"triposr_tome_int8.onnx"

    if not args.benchmark_only:
        log("Loading teacher model...")
        teacher = load_teacher("cpu")
        img_np = get_dummy_image("cpu").numpy()
        results = []

        # --- Vanilla Main model ---
        if not args.tome_only:
            log("\n" + "=" * 60)
            log("MAIN MODEL: TripoSR Forward Pass (Vanilla)")
            log("=" * 60)

            ref_out = export_fp32(teacher, fp32_path, args.opset)
            ref_np = ref_out.numpy()

            if not args.skip_verify:
                results.append(verify_onnx(fp32_path, img_np, ref_np, "image", "FP32"))

            if not args.fp32_only:
                convert_fp16(fp32_path, fp16_path)
                if not args.skip_verify:
                    results.append(verify_onnx(fp16_path, img_np, ref_np, "image", "FP16"))

                quantize_int8(fp32_path, int8_path)
                if not args.skip_verify:
                    results.append(verify_onnx(int8_path, img_np, ref_np, "image", "INT8"))

        # --- ToMe variant ---
        if tome_ratio is not None:
            log("\n" + "=" * 60)
            log(f"TOME MODEL: ToMe r={tome_ratio}, layers={args.tome_layers}")
            log("=" * 60)

            # Get PyTorch ToMe reference for accuracy comparison
            tome_wrapper = TripoSRForwardToMe(
                teacher, merge_ratio=tome_ratio, merge_layers=args.tome_layers
            ).cpu()
            tome_wrapper.eval()
            with torch.no_grad():
                tome_ref = tome_wrapper(get_dummy_image("cpu"))
            tome_ref_np = tome_ref.numpy()
            log(f"  PyTorch ToMe ref: shape={tome_ref.shape}, "
                f"range=[{tome_ref.min():.2f}, {tome_ref.max():.2f}]")

            tome_fp32_ref = export_tome_fp32(
                teacher, tome_fp32_path, args.opset,
                merge_ratio=tome_ratio, merge_layers=args.tome_layers,
            )

            if not args.skip_verify:
                results.append(verify_onnx(
                    tome_fp32_path, img_np, tome_ref_np,
                    "image", f"ToMe r={tome_ratio} FP32"))

            if not args.fp32_only:
                convert_fp16(tome_fp32_path, tome_fp16_path)
                if not args.skip_verify:
                    results.append(verify_onnx(
                        tome_fp16_path, img_np, tome_ref_np,
                        "image", f"ToMe r={tome_ratio} FP16"))

                quantize_int8(tome_fp32_path, tome_int8_path)
                if not args.skip_verify:
                    results.append(verify_onnx(
                        tome_int8_path, img_np, tome_ref_np,
                        "image", f"ToMe r={tome_ratio} INT8"))

        # --- NeRF Decoder ---
        if not args.skip_decoder and not args.tome_only:
            log("\n" + "=" * 60)
            log("NERF DECODER")
            log("=" * 60)
            ref_dec, dummy_feat = export_decoder(teacher, dec_fp32_path, args.opset)
            if not args.skip_verify:
                results.append(verify_onnx(
                    dec_fp32_path, dummy_feat.numpy(), ref_dec.numpy(),
                    "triplane_features", "Decoder FP32"))

        # --- Summary ---
        if results:
            log("\n" + "=" * 60)
            log("ACCURACY & SIZE SUMMARY")
            log("=" * 60)
            log(f"{'Variant':<30} {'Size':>8} {'Max Err%':>10} {'Mean Err%':>10}")
            log("-" * 60)
            for r in results:
                log(f"{r['label']:<30} {r['size_mb']:>7.1f}MB "
                    f"{r['max_rel']*100:>9.4f}% {r['mean_rel']*100:>9.4f}%")

    if args.benchmark or args.benchmark_only:
        log("\n" + "=" * 60)
        log(f"BENCHMARKING ({args.runs} runs, first is warmup)")
        log("=" * 60)
        bench_results = {}
        all_paths = [fp32_path, fp16_path, int8_path,
                     tome_fp32_path, tome_fp16_path, tome_int8_path]
        for path in all_paths:
            if path.exists():
                bench_results[path.stem] = benchmark_onnx(path, args.runs)

        if bench_results:
            log("\n--- Speed Summary ---")
            base = bench_results.get("triposr_fp32",
                   bench_results.get("triposr_tome_fp32", 1.0))
            for name, t in bench_results.items():
                log(f"  {name}: {t:.3f}s ({base/t:.2f}x vs base)")

    log("\nDone!")


if __name__ == "__main__":
    main()
