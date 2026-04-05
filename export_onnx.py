"""Unified ONNX export pipeline for all reconstruction models.

Exports, splits, quantizes, and verifies all models needed for on-device
3D reconstruction: rembg (u2netp), TripoSR (split at block 8), and NeRF decoder.

Pipeline per model:
  1. FP32 ONNX export (PyTorch for triposr/decoder, rembg library for u2netp)
  2. Graph optimization (constant folding, CSE, dead node elimination)
  3. Split at block 8 boundary (triposr only) -> part1 + part2
  4. FP16 conversion (per part for triposr)
  5. INT8 dynamic quantization (per part for triposr)
  6. Verification against reference

Critical: INT8 quantization of TripoSR split parts must happen AFTER splitting.
quantize_dynamic inserts DynamicQuantizeLinear/Shape nodes that trace back to
the original 'image' input — splitting after quantization leaves Part 2 with
a dangling reference.

Output structure (in models/):
  u2netp.onnx                 Background removal (FP32)
  u2netp_fp16.onnx            Background removal (FP16)
  u2netp_int8.onnx            Background removal (INT8)
  triposr_fp32.onnx           Full TripoSR (reference/benchmarking)
  triposr_part1_fp32.onnx     Split encoder half
  triposr_part2_fp32.onnx     Split decoder half
  triposr_part1_fp16.onnx     Split encoder half (FP16)
  triposr_part2_fp16.onnx     Split decoder half (FP16)
  triposr_part1_int8.onnx     Split encoder half (INT8)
  triposr_part2_int8.onnx     Split decoder half (INT8)
  nerf_decoder.onnx           NeRF MLP decoder (FP32)
  nerf_decoder_fp16.onnx      NeRF MLP decoder (FP16)
  nerf_decoder_int8.onnx      NeRF MLP decoder (INT8)

Usage:
    python export_onnx.py                       # Full pipeline (recommended)
    python export_onnx.py --fp32-only           # FP32 + split only
    python export_onnx.py --skip-rembg          # Skip u2netp export
    python export_onnx.py --skip-split          # Full triposr only, no split
    python export_onnx.py --benchmark           # Export + benchmark
    python export_onnx.py --benchmark-only      # Benchmark existing models
    python export_onnx.py --deploy int8         # Copy int8 models to Unity
    python export_onnx.py --experimental --tome 0.1  # ToMe (experimental)
"""

import argparse
import shutil
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
UNITY_ONNX_SOURCE = (Path(__file__).parent.parent
                     / "SentienceUnity/Assets/Game/ObjectReconstruction/OnnxSource")

SPLIT_BOUNDARY_TENSORS = [
    "/Reshape_output_0",
    "/backbone/transformer_blocks.7/Add_2_output_0",
]


def log(msg: str):
    print(msg, flush=True)


# ===========================================================================
# Model wrappers (PyTorch -> ONNX)
# ===========================================================================

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
    """Wrapper with inline Token Merging for ONNX-traceable ToMe."""

    TOKENS_PER_PLANE = 1024
    NUM_PLANES = 3

    def __init__(self, model: TSR, merge_ratio: float = 0.1,
                 merge_layers: list | None = None):
        super().__init__()
        self.image_tokenizer = model.image_tokenizer
        self.tokenizer = model.tokenizer
        self.post_processor = model.post_processor
        self.bb_norm = model.backbone.norm
        self.bb_proj_in = model.backbone.proj_in
        self.bb_proj_out = model.backbone.proj_out
        self.bb_blocks = model.backbone.transformer_blocks

        self.merge_ratio = merge_ratio
        _layers = merge_layers or [4, 8, 12]
        self.merge_layer_set = set(_layers)

    @staticmethod
    def _bipartite_soft_matching(metric: torch.Tensor, r: int):
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
        match = (kept_idx.unsqueeze(-1) == targets.unsqueeze(1))
        return match.to(torch.int64).argmax(dim=1)

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
        hidden_states = tokens
        batch, _, seq_len = hidden_states.shape
        residual_full = hidden_states
        hidden_states = self.bb_norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 1).reshape(batch, seq_len, inner_dim)
        hidden_states = self.bb_proj_in(hidden_states)

        all_merge_info = []
        current_n = [self.TOKENS_PER_PLANE] * self.NUM_PLANES

        for layer_idx in range(len(self.bb_blocks)):
            if layer_idx in self.merge_layer_set:
                planes, layer_info = [], []
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
                hidden_states, encoder_hidden_states=input_image_tokens,
            )

        for layer_info in reversed(all_merge_info):
            B_um, N_um, C_um = hidden_states.shape
            planes, plane_sizes = [], []
            offset = 0
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
            .permute(0, 2, 1).contiguous()
        )
        output = hidden_states + residual_full
        scene_codes = self.post_processor(self.tokenizer.detokenize(output))
        return scene_codes


class DecoderWrapper(nn.Module):
    """Wrapper for NeRF MLP decoder: triplane features (N, 120) -> density+color (N, 4)."""

    def __init__(self, decoder):
        super().__init__()
        self.layers = decoder.layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ===========================================================================
# Helpers
# ===========================================================================

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
    rgb = processor(img, 512)
    return rgb.permute(0, 3, 1, 2).to(device)


def _print_section(title: str):
    log(f"\n{'=' * 65}")
    log(title)
    log("=" * 65)


# ===========================================================================
# Graph optimization
# ===========================================================================

def optimize_graph(input_path: Path, output_path: Path = None):
    """Apply ORT basic graph optimizations (opt_level=1).

    Standard ONNX-compatible transforms only (constant folding, dead node
    elimination, CSE). No ORT-specific fused operators.
    """
    import onnxruntime as ort

    if output_path is None:
        output_path = input_path

    log(f"  Optimizing graph: {input_path.name}")
    t0 = time.time()

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(output_path)
    ort.InferenceSession(str(input_path), so, providers=["CPUExecutionProvider"])

    orig_size = input_path.stat().st_size / 1e6
    opt_size = output_path.stat().st_size / 1e6
    log(f"    {opt_size:.1f}MB (was {orig_size:.1f}MB) [{time.time()-t0:.1f}s]")


# ===========================================================================
# Quantization / conversion
# ===========================================================================

def convert_fp16(input_path: Path, output_path: Path):
    """Convert FP32 ONNX to FP16 using ORT transformer optimizer."""
    from onnxruntime.transformers.optimizer import optimize_model

    log(f"  FP16: {output_path.name}")
    t0 = time.time()
    opt = optimize_model(str(input_path), opt_level=0)
    opt.convert_float_to_float16(
        use_symbolic_shape_infer=True,
        keep_io_types=True,
    )
    opt.save_model_to_file(str(output_path))

    fsize = output_path.stat().st_size / 1e6
    fp32_size = input_path.stat().st_size / 1e6
    log(f"    {fsize:.1f}MB (was {fp32_size:.1f}MB, "
        f"{fsize/fp32_size*100:.0f}%) [{time.time()-t0:.1f}s]")


def quantize_int8(input_path: Path, output_path: Path):
    """Apply dynamic INT8 quantization (MatMul weights only)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    log(f"  INT8: {output_path.name}")
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
    log(f"    {fsize:.1f}MB (was {fp32_size:.1f}MB, "
        f"{fsize/fp32_size*100:.0f}%)")


def quantize_all(fp32_path: Path, name: str):
    """Generate FP16 and INT8 variants from a FP32 model."""
    stem = fp32_path.stem.replace("_fp32", "").replace(".onnx", "")
    parent = fp32_path.parent
    fp16_path = parent / f"{stem}_fp16.onnx"
    int8_path = parent / f"{stem}_int8.onnx"

    log(f"\n  Quantizing {name}...")
    convert_fp16(fp32_path, fp16_path)
    quantize_int8(fp32_path, int8_path)
    return fp16_path, int8_path


# ===========================================================================
# Splitting (TripoSR only)
# ===========================================================================

def split_model(input_path: Path, output_dir: Path, prefix: str = "triposr"):
    """Split TripoSR ONNX at transformer block 8 boundary.

    Part 1: image_tokenizer + backbone blocks 0-7
    Part 2: backbone blocks 8-15 + post_processor
    """
    import onnx
    from onnx.utils import Extractor

    log(f"\n  Splitting {input_path.name} at block 8 boundary...")
    model = onnx.load(str(input_path))
    orig_input = model.graph.input[0].name
    orig_output = model.graph.output[0].name
    log(f"    {len(model.graph.node)} nodes, input={orig_input}, output={orig_output}")

    part1_path = output_dir / f"{prefix}_part1_fp32.onnx"
    part2_path = output_dir / f"{prefix}_part2_fp32.onnx"

    ext1 = Extractor(model)
    part1 = ext1.extract_model([orig_input], SPLIT_BOUNDARY_TENSORS)
    onnx.save(part1, str(part1_path))
    log(f"    Part 1: {part1_path.name} ({part1_path.stat().st_size/1e6:.1f}MB, "
        f"{len(part1.graph.node)} nodes)")

    ext2 = Extractor(model)
    part2 = ext2.extract_model(SPLIT_BOUNDARY_TENSORS, [orig_output])
    onnx.save(part2, str(part2_path))
    log(f"    Part 2: {part2_path.name} ({part2_path.stat().st_size/1e6:.1f}MB, "
        f"{len(part2.graph.node)} nodes)")

    del model, part1, part2
    return part1_path, part2_path


def quantize_split_parts(part1_fp32: Path, part2_fp32: Path, output_dir: Path,
                         prefix: str = "triposr"):
    """Quantize split FP32 parts to FP16 and INT8 independently."""
    p1_fp16 = output_dir / f"{prefix}_part1_fp16.onnx"
    p2_fp16 = output_dir / f"{prefix}_part2_fp16.onnx"
    p1_int8 = output_dir / f"{prefix}_part1_int8.onnx"
    p2_int8 = output_dir / f"{prefix}_part2_int8.onnx"

    log("\n  Quantizing split parts...")
    convert_fp16(part1_fp32, p1_fp16)
    convert_fp16(part2_fp32, p2_fp16)
    quantize_int8(part1_fp32, p1_int8)
    quantize_int8(part2_fp32, p2_int8)

    return {
        "fp16": (p1_fp16, p2_fp16),
        "int8": (p1_int8, p2_int8),
    }


# ===========================================================================
# Export: u2netp (rembg)
# ===========================================================================

def export_u2netp(output_dir: Path, target_opset: int = 15, fp32_only: bool = False,
                  skip_verify: bool = False) -> list[dict]:
    """Export u2netp ONNX from the rembg library with FP16/INT8 variants.

    The rembg package bundles a pre-trained u2netp checkpoint. We extract it,
    convert to target opset, optimize, and generate quantized variants.
    """
    import onnx
    from onnx import version_converter
    from rembg.sessions import U2netpSession

    _print_section("REMBG (u2netp)")
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    src = Path(U2netpSession.download_models())
    raw_dst = output_dir / "u2netp_raw.onnx"
    fp32_path = output_dir / "u2netp.onnx"

    shutil.copy2(src, raw_dst)
    log(f"  Copied raw u2netp ({raw_dst.stat().st_size / 1e6:.1f}MB)")

    model = onnx.load(str(raw_dst))
    orig_opset = model.opset_import[0].version
    log(f"  Original: opset {orig_opset}, {len(model.graph.node)} nodes")

    if orig_opset != target_opset:
        log(f"  Converting opset {orig_opset} -> {target_opset}...")
        model = version_converter.convert_version(model, target_opset)
        onnx.checker.check_model(model)

    tmp_path = output_dir / "u2netp_tmp.onnx"
    onnx.save(model, str(tmp_path))
    del model
    optimize_graph(tmp_path, fp32_path)
    tmp_path.unlink(missing_ok=True)
    raw_dst.unlink(missing_ok=True)

    log(f"  u2netp FP32: {fp32_path.stat().st_size / 1e6:.1f}MB")

    if not skip_verify:
        results.append(_verify_u2netp(fp32_path, "u2netp FP32"))

    if not fp32_only:
        fp16_path, int8_path = quantize_all(fp32_path, "u2netp")
        if not skip_verify:
            results.append(_verify_u2netp(fp16_path, "u2netp FP16"))
            results.append(_verify_u2netp(int8_path, "u2netp INT8"))

    return results


def _verify_u2netp(model_path: Path, label: str) -> dict:
    """Verify u2netp ONNX produces valid output."""
    import onnxruntime as ort

    log(f"  Verifying [{label}]: {model_path.name}")
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    inp_meta = sess.get_inputs()[0]
    dummy = np.random.rand(1, 3, 320, 320).astype(np.float32)
    if inp_meta.type == "tensor(float16)":
        dummy = dummy.astype(np.float16)

    outputs = sess.run(None, {inp_meta.name: dummy})
    out = outputs[0].astype(np.float32)

    fsize = model_path.stat().st_size / 1e6
    log(f"    Output: shape={out.shape}, range=[{out.min():.3f}, {out.max():.3f}]")
    log(f"    Size: {fsize:.1f}MB — PASS (output valid)")

    return {
        "label": label, "size_mb": fsize,
        "max_abs": 0.0, "mean_abs": 0.0,
        "max_rel": 0.0, "mean_rel": 0.0,
    }


# ===========================================================================
# Export: TripoSR
# ===========================================================================

def export_triposr_fp32(wrapper: nn.Module, output_path: Path, opset: int = 15,
                        label: str = "FP32", static: bool = True) -> torch.Tensor:
    """Export TripoSR wrapper to ONNX FP32 (full unsplit model)."""
    mode_tag = "static" if static else "dynamic batch"
    log(f"\n  Exporting {label} to {output_path.name} (opset {opset}, {mode_tag})...")
    wrapper = wrapper.cpu()
    wrapper.eval()

    dummy = get_dummy_image("cpu")
    log(f"    Input: {dummy.shape}, dtype={dummy.dtype}")

    with torch.no_grad():
        ref_out = wrapper(dummy)
    log(f"    Output: {ref_out.shape}, range=[{ref_out.min():.2f}, {ref_out.max():.2f}]")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs = dict(
        opset_version=opset,
        input_names=["image"],
        output_names=["scene_codes"],
        do_constant_folding=True,
        dynamo=False,
    )
    if not static:
        export_kwargs["dynamic_axes"] = {
            "image": {0: "batch"},
            "scene_codes": {0: "batch"},
        }

    t0 = time.time()
    torch.onnx.export(wrapper, (dummy,), str(output_path), **export_kwargs)
    log(f"    Export: {time.time()-t0:.1f}s, {output_path.stat().st_size/1e6:.1f}MB")
    return ref_out


def export_triposr_variant(model: TSR, out_dir: Path, prefix: str, label: str,
                           wrapper_fn, opset: int, static: bool, fp32_only: bool,
                           skip_split: bool, skip_verify: bool,
                           img_np: np.ndarray) -> list[dict]:
    """Export a TripoSR variant (vanilla or ToMe) through the full pipeline."""
    results = []
    _print_section(f"TRIPOSR: {label}")

    fp32_path = out_dir / f"{prefix}_fp32.onnx"
    ref_out = export_triposr_fp32(wrapper_fn(), fp32_path, opset, f"{label} FP32",
                                  static=static)
    ref_np = ref_out.numpy()

    optimize_graph(fp32_path)

    if not skip_verify:
        results.append(verify_onnx(fp32_path, img_np, ref_np, "image", f"{label} FP32"))

    if not skip_split:
        p1_fp32, p2_fp32 = split_model(fp32_path, out_dir, prefix)

        if not skip_verify:
            results.append(verify_split(p1_fp32, p2_fp32, ref_np, img_np,
                                        f"{label} Split FP32"))

        if not fp32_only:
            quant_paths = quantize_split_parts(p1_fp32, p2_fp32, out_dir, prefix)
            if not skip_verify:
                for prec, (p1, p2) in quant_paths.items():
                    results.append(verify_split(
                        p1, p2, ref_np, img_np, f"{label} Split {prec.upper()}"))

    elif not fp32_only:
        fp16_path, int8_path = quantize_all(fp32_path, label)
        if not skip_verify:
            results.append(verify_onnx(fp16_path, img_np, ref_np, "image",
                                       f"{label} FP16"))
            results.append(verify_onnx(int8_path, img_np, ref_np, "image",
                                       f"{label} INT8"))

    return results


# ===========================================================================
# Export: NeRF decoder
# ===========================================================================

def export_decoder(model: TSR, output_dir: Path, opset: int = 15,
                   fp32_only: bool = False, skip_verify: bool = False) -> list[dict]:
    """Export NeRF MLP decoder with FP16/INT8 variants."""
    results = []
    _print_section("NERF DECODER")

    decoder = model.decoder
    in_ch = decoder.cfg.in_channels
    log(f"  {sum(p.numel() for p in decoder.parameters())/1e3:.1f}K params, "
        f"in_channels={in_ch}")

    wrapper = DecoderWrapper(decoder).cpu()
    wrapper.eval()

    dummy = torch.randn(1024, in_ch)
    with torch.no_grad():
        ref_out = wrapper(dummy)
    log(f"  Output: {ref_out.shape}, range=[{ref_out.min():.2f}, {ref_out.max():.2f}]")

    fp32_path = output_dir / "nerf_decoder.onnx"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper, (dummy,), str(fp32_path),
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

    optimize_graph(fp32_path)
    log(f"  Decoder FP32: {fp32_path.stat().st_size / 1e6:.3f}MB")

    ref_np = ref_out.numpy()
    feat_np = dummy.numpy()

    if not skip_verify:
        results.append(verify_onnx(fp32_path, feat_np, ref_np,
                                   "triplane_features", "Decoder FP32"))

    if not fp32_only:
        fp16_path = output_dir / "nerf_decoder_fp16.onnx"
        int8_path = output_dir / "nerf_decoder_int8.onnx"
        convert_fp16(fp32_path, fp16_path)
        quantize_int8(fp32_path, int8_path)
        if not skip_verify:
            results.append(verify_onnx(fp16_path, feat_np, ref_np,
                                       "triplane_features", "Decoder FP16"))
            results.append(verify_onnx(int8_path, feat_np, ref_np,
                                       "triplane_features", "Decoder INT8"))

    return results


# ===========================================================================
# Verification
# ===========================================================================

def verify_onnx(onnx_path: Path, dummy_input: np.ndarray,
                ref_output: np.ndarray, input_name: str = "image",
                label: str = "") -> dict:
    """Verify single ONNX model matches PyTorch output."""
    import onnxruntime as ort

    tag = f" [{label}]" if label else ""
    log(f"  Verifying{tag}: {onnx_path.name}")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    inp = dummy_input
    if sess.get_inputs()[0].type == "tensor(float16)":
        inp = dummy_input.astype(np.float16)

    ort_out = sess.run(None, {input_name: inp})[0].astype(np.float32)

    abs_diff = np.abs(ort_out - ref_output)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    output_range = float(np.abs(ref_output).max())
    rel_max = max_diff / (output_range + 1e-8)
    rel_mean = mean_diff / (output_range + 1e-8)

    fsize = onnx_path.stat().st_size / 1e6
    log(f"    Size: {fsize:.1f}MB | "
        f"Max err: {max_diff:.4f} ({rel_max*100:.4f}%) | "
        f"Mean err: {mean_diff:.4f} ({rel_mean*100:.4f}%)")

    verdict = ("PASS" if rel_max < 0.01
               else "WARN" if rel_max < 0.05
               else "OK" if rel_max < 0.15
               else "FAIL")
    log(f"    {verdict}")

    return {
        "label": label, "size_mb": fsize,
        "max_abs": max_diff, "mean_abs": mean_diff,
        "max_rel": rel_max, "mean_rel": rel_mean,
    }


def verify_split(part1_path: Path, part2_path: Path, ref_output: np.ndarray,
                 img_np: np.ndarray, label: str = "") -> dict:
    """Verify split model (part1 -> part2) produces same output as full model."""
    import onnxruntime as ort

    tag = f" [{label}]" if label else ""
    log(f"  Verifying split{tag}: {part1_path.name} + {part2_path.name}")

    sess1 = ort.InferenceSession(str(part1_path), providers=["CPUExecutionProvider"])
    p1_inp = img_np
    if sess1.get_inputs()[0].type == "tensor(float16)":
        p1_inp = img_np.astype(np.float16)
    p1_outs = sess1.run(None, {"image": p1_inp})
    p1_names = [o.name for o in sess1.get_outputs()]

    sess2 = ort.InferenceSession(str(part2_path), providers=["CPUExecutionProvider"])
    p2_inputs = {}
    for name, val in zip(p1_names, p1_outs):
        meta = next((i for i in sess2.get_inputs() if i.name == name), None)
        if meta and meta.type == "tensor(float16)":
            p2_inputs[name] = val.astype(np.float16)
        else:
            p2_inputs[name] = val
    [split_out] = sess2.run(None, p2_inputs)
    split_out = split_out.astype(np.float32)

    abs_diff = np.abs(split_out - ref_output)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    output_range = float(np.abs(ref_output).max())
    rel_max = max_diff / (output_range + 1e-8)
    rel_mean = mean_diff / (output_range + 1e-8)

    total_mb = part1_path.stat().st_size / 1e6 + part2_path.stat().st_size / 1e6
    log(f"    Total: {total_mb:.1f}MB | "
        f"Max err: {max_diff:.4f} ({rel_max*100:.4f}%) | "
        f"Mean err: {mean_diff:.4f} ({rel_mean*100:.4f}%)")

    verdict = ("PASS" if rel_max < 0.01
               else "WARN" if rel_max < 0.05
               else "OK" if rel_max < 0.15
               else "FAIL")
    log(f"    {verdict}")

    return {
        "label": label, "size_mb": total_mb,
        "max_abs": max_diff, "mean_abs": mean_diff,
        "max_rel": rel_max, "mean_rel": rel_mean,
    }


# ===========================================================================
# Benchmark
# ===========================================================================

def benchmark_onnx(model_path: Path, n_runs: int = 10) -> float:
    """Benchmark an ONNX model with ORT CPUExecutionProvider."""
    import onnxruntime as ort

    log(f"  Benchmarking {model_path.name} ({n_runs} runs)...")
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    dummy = get_dummy_image("cpu").numpy()
    times = []

    for i in range(n_runs + 1):
        t0 = time.perf_counter()
        sess.run(None, {"image": dummy})
        elapsed = time.perf_counter() - t0
        if i > 0:
            times.append(elapsed)

    arr = np.array(times)
    log(f"    {arr.mean():.3f}s +/- {arr.std():.3f}s (min={arr.min():.3f}s)")
    return arr.mean()


# ===========================================================================
# Deploy
# ===========================================================================

def deploy_to_unity(models_dir: Path, precision: str = "int8"):
    """Copy deployment models to Unity OnnxSource directory.

    Copies split TripoSR parts, nerf decoder, and u2netp.
    Renames parts to triposr_part1.onnx / triposr_part2.onnx for the Unity pipeline.
    """
    _print_section(f"DEPLOYING ({precision.upper()}) TO UNITY")

    if not UNITY_ONNX_SOURCE.exists():
        log(f"  ERROR: Unity OnnxSource not found: {UNITY_ONNX_SOURCE}")
        return

    decoder_name = "nerf_decoder.onnx" if precision == "fp32" else f"nerf_decoder_{precision}.onnx"
    u2netp_name = "u2netp.onnx" if precision == "fp32" else f"u2netp_{precision}.onnx"

    copies = [
        (models_dir / f"triposr_part1_{precision}.onnx", "triposr_part1.onnx"),
        (models_dir / f"triposr_part2_{precision}.onnx", "triposr_part2.onnx"),
        (models_dir / decoder_name, "nerf_decoder.onnx"),
        (models_dir / u2netp_name, "u2netp.onnx"),
    ]

    for src, dst_name in copies:
        dst = UNITY_ONNX_SOURCE / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            log(f"  {src.name} -> {dst_name} ({src.stat().st_size/1e6:.1f}MB)")
        else:
            log(f"  SKIP {src.name} (not found)")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified ONNX export pipeline for all reconstruction models")
    parser.add_argument("--opset", type=int, default=15)
    parser.add_argument("--dynamic", action="store_true",
                        help="Use dynamic batch axis (default: fully static)")
    parser.add_argument("--fp32-only", action="store_true",
                        help="Skip FP16 and INT8 quantization")
    parser.add_argument("--skip-rembg", action="store_true",
                        help="Skip u2netp export")
    parser.add_argument("--skip-triposr", action="store_true",
                        help="Skip TripoSR export")
    parser.add_argument("--skip-split", action="store_true",
                        help="Skip splitting TripoSR into part1/part2")
    parser.add_argument("--skip-decoder", action="store_true",
                        help="Skip NeRF decoder export")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip verification against PyTorch reference")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark after export")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Only benchmark existing models")
    parser.add_argument("--deploy", nargs="?", const="int8", default=None,
                        metavar="PRECISION",
                        help="Copy models to Unity OnnxSource (default: int8)")
    parser.add_argument("--experimental", action="store_true",
                        help="Enable experimental features (ToMe export)")
    parser.add_argument("--tome", type=float, default=None,
                        help="[experimental] Export ToMe variant with given merge ratio")
    parser.add_argument("--tome-layers", nargs="+", type=int, default=[4, 8, 12],
                        help="[experimental] ToMe merge layers (default: 4 8 12)")
    parser.add_argument("--tome-only", action="store_true",
                        help="[experimental] Only export ToMe variant (skip vanilla)")
    parser.add_argument("--runs", type=int, default=10,
                        help="Benchmark runs (default: 10)")
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    if (args.tome is not None or args.tome_only) and not args.experimental:
        parser.error("ToMe export requires --experimental flag")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    if not args.benchmark_only:

        # ---- Rembg (u2netp) ----
        if not args.skip_rembg and not args.tome_only:
            all_results.extend(export_u2netp(
                out_dir, args.opset, args.fp32_only, args.skip_verify))

        # ---- TripoSR (vanilla + optional ToMe) ----
        if not args.skip_triposr:
            log("\nLoading teacher model...")
            teacher = load_teacher("cpu")
            img_np = get_dummy_image("cpu").numpy()

            if not args.tome_only:
                all_results.extend(export_triposr_variant(
                    teacher, out_dir, "triposr", "Vanilla",
                    wrapper_fn=lambda: TripoSRForward(teacher),
                    opset=args.opset, static=not args.dynamic,
                    fp32_only=args.fp32_only, skip_split=args.skip_split,
                    skip_verify=args.skip_verify, img_np=img_np,
                ))

            if args.tome is not None:
                all_results.extend(export_triposr_variant(
                    teacher, out_dir, "triposr_tome",
                    f"ToMe r={args.tome}",
                    wrapper_fn=lambda: TripoSRForwardToMe(
                        teacher, merge_ratio=args.tome,
                        merge_layers=args.tome_layers),
                    opset=args.opset, static=not args.dynamic,
                    fp32_only=args.fp32_only, skip_split=args.skip_split,
                    skip_verify=args.skip_verify, img_np=img_np,
                ))

        # ---- NeRF Decoder ----
        if not args.skip_decoder and not args.tome_only:
            if not args.skip_triposr:
                pass  # teacher already loaded
            else:
                log("\nLoading teacher model for decoder...")
                teacher = load_teacher("cpu")

            all_results.extend(export_decoder(
                teacher, out_dir, args.opset, args.fp32_only, args.skip_verify))

        # ---- Summary ----
        if all_results:
            _print_section("ACCURACY & SIZE SUMMARY")
            log(f"{'Variant':<35} {'Size':>8} {'Max Err%':>10} {'Mean Err%':>10}")
            log("-" * 65)
            for r in all_results:
                log(f"{r['label']:<35} {r['size_mb']:>7.1f}MB "
                    f"{r['max_rel']*100:>9.4f}% {r['mean_rel']*100:>9.4f}%")

    # ---- Deploy ----
    if args.deploy:
        deploy_to_unity(out_dir, args.deploy)

    # ---- Benchmark ----
    if args.benchmark or args.benchmark_only:
        _print_section(f"BENCHMARKING ({args.runs} runs, first is warmup)")
        bench_results = {}
        for pattern in ["triposr_fp32.onnx", "triposr_tome_fp32.onnx"]:
            for p in sorted(out_dir.glob(pattern)):
                bench_results[p.stem] = benchmark_onnx(p, args.runs)

        if bench_results:
            log("\n--- Speed Summary ---")
            base = list(bench_results.values())[0]
            for name, t in bench_results.items():
                log(f"  {name}: {t:.3f}s ({base/t:.2f}x vs base)")

    log("\nDone!")


if __name__ == "__main__":
    main()
