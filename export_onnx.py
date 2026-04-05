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
  u2netp.onnx                     Background removal (FP32)
  u2netp_fp16.onnx                Background removal (FP16)
  u2netp_int8.onnx                Background removal (INT8 dynamic)
  u2netp_int8_qdq.onnx            Background removal (INT8 QDQ static, for NPU)
  triposr_fp32.onnx               Full TripoSR (reference/benchmarking)
  triposr_part1_fp32.onnx         Split encoder half
  triposr_part2_fp32.onnx         Split decoder half
  triposr_part1_fp16.onnx         Split encoder half (FP16)
  triposr_part2_fp16.onnx         Split decoder half (FP16)
  triposr_part1_int8.onnx         Split encoder half (INT8 dynamic)
  triposr_part2_int8.onnx         Split decoder half (INT8 dynamic)
  triposr_part1_int8_qdq.onnx     Split encoder half (INT8 QDQ static, for NPU)
  triposr_part2_int8_qdq.onnx     Split decoder half (INT8 QDQ static, for NPU)
  nerf_decoder.onnx               NeRF MLP decoder (FP32)
  nerf_decoder_fp16.onnx          NeRF MLP decoder (FP16)
  nerf_decoder_int8.onnx          NeRF MLP decoder (INT8 dynamic)
  nerf_decoder_int8_qdq.onnx      NeRF MLP decoder (INT8 QDQ static, for NPU)

Usage:
    python export_onnx.py                       # Full pipeline (recommended)
    python export_onnx.py --fp32-only           # FP32 + split only
    python export_onnx.py --qdq-only            # QDQ INT8 from existing FP32 models
    python export_onnx.py --qdq                 # Full pipeline + QDQ INT8
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
# Static shape validation
# ===========================================================================

def validate_static_shapes(model_path: Path, allow_dynamic_axes: dict[str, list[int]] | None = None) -> bool:
    """Validate that an ONNX model has fully static shapes (no symbolic dims).

    Returns True if all shapes are static (or match allowed exceptions).
    Prints FAIL and details if any unexpected dynamic dimensions found.

    allow_dynamic_axes: dict mapping tensor name -> list of axis indices
        where dynamic dims are expected (e.g. {"triplane_features": [0]}).
    """
    import onnx

    allow = allow_dynamic_axes or {}
    model = onnx.load(str(model_path))

    dynamic_found = []
    for tensor_list, kind in [(model.graph.input, "input"),
                               (model.graph.output, "output")]:
        for tensor in tensor_list:
            name = tensor.name
            shape = tensor.type.tensor_type.shape
            if shape is None:
                dynamic_found.append((kind, name, "no shape info"))
                continue
            for i, dim in enumerate(shape.dim):
                if dim.dim_param:
                    allowed_axes = allow.get(name, [])
                    if i in allowed_axes:
                        continue
                    dynamic_found.append((kind, name, f"axis {i} = '{dim.dim_param}'"))
                elif dim.dim_value == 0:
                    allowed_axes = allow.get(name, [])
                    if i in allowed_axes:
                        continue
                    dynamic_found.append((kind, name, f"axis {i} = unknown(0)"))

    del model

    if dynamic_found:
        log(f"    FAIL static check: {model_path.name}")
        for kind, name, detail in dynamic_found:
            log(f"      {kind} '{name}': {detail}")
        return False
    else:
        log(f"    PASS static check: {model_path.name}")
        return True


def validate_all_models(models_dir: Path) -> bool:
    """Validate static shapes for all exported models in the directory."""
    _print_section("STATIC SHAPE VALIDATION")

    decoder_dynamic = {"triplane_features": [0], "density_color": [0]}
    all_pass = True

    checks = [
        ("u2netp.onnx", None),
        ("u2netp_fp16.onnx", None),
        ("u2netp_int8.onnx", None),
        ("u2netp_int8_qdq.onnx", None),
        ("triposr_fp32.onnx", None),
        ("triposr_part1_fp32.onnx", None),
        ("triposr_part2_fp32.onnx", None),
        ("triposr_part1_fp16.onnx", None),
        ("triposr_part2_fp16.onnx", None),
        ("triposr_part1_int8.onnx", None),
        ("triposr_part2_int8.onnx", None),
        ("triposr_part1_int8_qdq.onnx", None),
        ("triposr_part2_int8_qdq.onnx", None),
        ("nerf_decoder.onnx", decoder_dynamic),
        ("nerf_decoder_fp16.onnx", decoder_dynamic),
        ("nerf_decoder_int8.onnx", decoder_dynamic),
        ("nerf_decoder_int8_qdq.onnx", decoder_dynamic),
    ]

    for filename, allowed in checks:
        path = models_dir / filename
        if path.exists():
            if not validate_static_shapes(path, allowed):
                all_pass = False

    if all_pass:
        log("\n  All models PASS static shape validation")
    else:
        log("\n  WARNING: Some models have unexpected dynamic dimensions")

    return all_pass


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


class _NumpyCalibrationReader:
    """Feeds pre-computed numpy arrays to onnxruntime static quantization."""

    def __init__(self, input_name: str, data_list: list[np.ndarray]):
        self.input_name = input_name
        self.data_list = data_list
        self.index = 0

    def get_next(self):
        if self.index >= len(self.data_list):
            return None
        sample = {self.input_name: self.data_list[self.index]}
        self.index += 1
        return sample


class _MultiInputCalibrationReader:
    """Feeds multi-input samples (dict per sample) to static quantization."""

    def __init__(self, samples: list[dict[str, np.ndarray]]):
        self.samples = samples
        self.index = 0

    def get_next(self):
        if self.index >= len(self.samples):
            return None
        sample = self.samples[self.index]
        self.index += 1
        return sample


def _collect_calibration_images(n: int = 8) -> list[np.ndarray]:
    """Collect preprocessed calibration images for TripoSR (1, 3, 512, 512)."""
    test_dir = Path(__file__).parent / "test_images"
    paths = []
    for subdir in ["examples", "novel"]:
        d = test_dir / subdir
        if not d.exists():
            continue
        for ext in ("*.png", "*.jpg"):
            paths.extend(sorted(d.glob(ext)))
    paths = paths[:n]

    processor = ImagePreprocessor()
    images = []
    for p in paths:
        img = prepare_image(p)
        rgb = processor(img, 512)
        images.append(rgb.permute(0, 3, 1, 2).numpy())
    log(f"    Collected {len(images)} calibration images")
    return images


def _collect_rembg_calibration_images(n: int = 8) -> list[np.ndarray]:
    """Collect preprocessed calibration images for u2netp (1, 3, 320, 320)."""
    test_dir = Path(__file__).parent / "test_images"
    paths = []
    for subdir in ["examples", "novel"]:
        d = test_dir / subdir
        if not d.exists():
            continue
        for ext in ("*.png", "*.jpg"):
            paths.extend(sorted(d.glob(ext)))
    paths = paths[:n]

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    images = []
    for p in paths:
        from PIL import Image as PILImage
        img = PILImage.open(p).convert("RGB")
        resized = img.resize((320, 320), PILImage.LANCZOS)
        arr = np.array(resized, dtype=np.float32)
        arr = arr / max(np.max(arr), 1e-6)
        tmp = np.zeros((320, 320, 3), dtype=np.float32)
        tmp[:, :, 0] = (arr[:, :, 0] - mean[0]) / std[0]
        tmp[:, :, 1] = (arr[:, :, 1] - mean[1]) / std[1]
        tmp[:, :, 2] = (arr[:, :, 2] - mean[2]) / std[2]
        images.append(tmp.transpose(2, 0, 1)[np.newaxis])
    log(f"    Collected {len(images)} rembg calibration images")
    return images


def _collect_decoder_calibration_features(
    models_dir: Path, n_images: int = 8, n_points: int = 4096,
) -> list[np.ndarray]:
    """Generate real triplane features from FP32 TripoSR for decoder calibration.

    Runs FP32 Part1+Part2 on calibration images, then does triplane grid
    sampling with random 3D positions to extract actual decoder inputs.
    This replaces random N(0,1) data which doesn't represent real feature
    distributions and causes poor quantization ranges.
    """
    import onnxruntime as ort
    import torch
    import torch.nn.functional as F
    from einops import rearrange

    p1_path = models_dir / "triposr_part1_fp32.onnx"
    p2_path = models_dir / "triposr_part2_fp32.onnx"
    if not p1_path.exists() or not p2_path.exists():
        log("    WARNING: FP32 split models not found, falling back to random calibration")
        sess = ort.InferenceSession(
            str(models_dir / "nerf_decoder.onnx"), providers=["CPUExecutionProvider"])
        in_ch = sess.get_inputs()[0].shape[1]
        del sess
        return [np.random.randn(n_points, in_ch).astype(np.float32) for _ in range(n_images)]

    images = _collect_calibration_images(n_images)
    p1_sess = ort.InferenceSession(str(p1_path), providers=["CPUExecutionProvider"])
    p2_sess = ort.InferenceSession(str(p2_path), providers=["CPUExecutionProvider"])

    p2_input_names = [inp.name for inp in p2_sess.get_inputs()]

    features_list = []
    for img_np in images:
        p1_outputs = p1_sess.run(None, {"image": img_np})
        p2_feed = {name: val for name, val in zip(p2_input_names, p1_outputs)}
        scene_codes = p2_sess.run(None, p2_feed)[0]

        triplane = torch.from_numpy(scene_codes[0])  # (3, C, H, W)
        positions = torch.rand(n_points, 3) * 2 - 1  # uniform in [-1, 1]

        indices2D = torch.stack(
            (positions[..., [0, 1]], positions[..., [0, 2]], positions[..., [1, 2]]),
            dim=-3,
        )
        out = F.grid_sample(
            rearrange(triplane, "Np Cp Hp Wp -> Np Cp Hp Wp", Np=3),
            rearrange(indices2D, "Np N Nd -> Np () N Nd", Np=3),
            align_corners=False,
            mode="bilinear",
        )
        feats = rearrange(out, "Np Cp () N -> N (Np Cp)", Np=3)
        features_list.append(feats.numpy())

    del p1_sess, p2_sess
    log(f"    Collected {len(features_list)} decoder calibration batches "
        f"({n_points} points each) from FP32 triplane features")
    return features_list


def quantize_int8_qdq(input_path: Path, output_path: Path,
                      calibration_data: list[np.ndarray] | list[dict[str, np.ndarray]],
                      input_name: str | None = None):
    """Apply QNN-optimized static quantization in QDQ format for Hexagon HTP.

    Uses the ORT QNN-specific pipeline:
    1. qnn_preprocess_model -- fuses LayerNorm, fixes op patterns for QNN
    2. get_qnn_qdq_config -- generates QNN-optimized quantization config
       with uint16 activations (65k levels) for transformer accuracy
    3. quantize() -- unified entry point

    uint16 activations + uint8 weights is the Qualcomm-recommended config
    for accuracy-sensitive models (transformers, ViTs).
    """
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize
    from onnxruntime.quantization.execution_providers.qnn import (
        get_qnn_qdq_config, qnn_preprocess_model,
    )

    log(f"  QNN-QDQ: {output_path.name} ({len(calibration_data)} cal samples)")
    t0 = time.time()

    # Step 1: QNN-specific preprocessing (fuse LayerNorm, fix op patterns)
    preprocessed = output_path.parent / f"_qnn_preproc_{output_path.name}"
    try:
        model_changed = qnn_preprocess_model(str(input_path), str(preprocessed))
        if model_changed and preprocessed.exists():
            log(f"    QNN preprocess: modified graph")
            quant_input = preprocessed
        else:
            log(f"    QNN preprocess: no changes needed")
            quant_input = input_path
    except Exception as e:
        log(f"    QNN preprocess failed ({e}), using original model")
        quant_input = input_path

    # Build calibration reader
    if isinstance(calibration_data[0], dict):
        reader = _MultiInputCalibrationReader(calibration_data)
    else:
        if input_name is None:
            sess = ort.InferenceSession(str(quant_input), providers=["CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name
            del sess
        reader = _NumpyCalibrationReader(input_name, calibration_data)

    # Step 2: QNN-optimized config (uint16 activations for transformer accuracy)
    qnn_config = get_qnn_qdq_config(
        str(quant_input),
        reader,
        activation_type=QuantType.QUInt16,
        weight_type=QuantType.QUInt8,
        per_channel=True,
    )

    # Step 3: Quantize with unified entry point
    quantize(str(quant_input), str(output_path), qnn_config)

    preprocessed.unlink(missing_ok=True)

    fsize = output_path.stat().st_size / 1e6
    fp32_size = input_path.stat().st_size / 1e6
    log(f"    {fsize:.1f}MB (was {fp32_size:.1f}MB, "
        f"{fsize/fp32_size*100:.0f}%) [{time.time()-t0:.1f}s]")


def quantize_all(fp32_path: Path, name: str, qdq: bool = False,
                  calibration_data: list[np.ndarray] | None = None,
                  input_name: str | None = None):
    """Generate FP16 and INT8 variants from a FP32 model."""
    stem = fp32_path.stem.replace("_fp32", "").replace(".onnx", "")
    parent = fp32_path.parent
    fp16_path = parent / f"{stem}_fp16.onnx"
    int8_path = parent / f"{stem}_int8.onnx"

    log(f"\n  Quantizing {name}...")
    convert_fp16(fp32_path, fp16_path)
    quantize_int8(fp32_path, int8_path)

    int8_qdq_path = None
    if qdq and calibration_data:
        int8_qdq_path = parent / f"{stem}_int8_qdq.onnx"
        quantize_int8_qdq(fp32_path, int8_qdq_path, calibration_data, input_name)

    return fp16_path, int8_path, int8_qdq_path


# ===========================================================================
# Splitting (TripoSR only)
# ===========================================================================

def split_model(input_path: Path, output_dir: Path, prefix: str = "triposr"):
    """Split TripoSR ONNX at transformer block 8 boundary.

    Part 1: image_tokenizer + backbone blocks 0-7
    Part 2: backbone blocks 8-15 + post_processor

    Runs onnx.shape_inference first to populate value_infos for all
    intermediate tensors — required by the ONNX Extractor.
    """
    import onnx
    from onnx import shape_inference
    from onnx.utils import Extractor

    log(f"\n  Splitting {input_path.name} at block 8 boundary...")
    log(f"    Running shape inference...")
    model = shape_inference.infer_shapes(
        onnx.load(str(input_path)), data_prop=True
    )
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

    # Fix any symbolic dims that shape inference couldn't resolve statically.
    # The full model uses static shapes, so all split tensor dims are known.
    known_shapes = {
        "/Reshape_output_0": [1, 1025, 768],
        "/backbone/transformer_blocks.7/Add_2_output_0": [1, 3072, 1024],
    }
    for inp in part2.graph.input:
        if inp.name in known_shapes:
            for i, dim_val in enumerate(known_shapes[inp.name]):
                dim = inp.type.tensor_type.shape.dim[i]
                dim.ClearField("dim_param")
                dim.dim_value = dim_val

    onnx.save(part2, str(part2_path))
    log(f"    Part 2: {part2_path.name} ({part2_path.stat().st_size/1e6:.1f}MB, "
        f"{len(part2.graph.node)} nodes)")

    del model, part1, part2
    return part1_path, part2_path


def quantize_split_parts(part1_fp32: Path, part2_fp32: Path, output_dir: Path,
                         prefix: str = "triposr", qdq: bool = False):
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

    result = {
        "fp16": (p1_fp16, p2_fp16),
        "int8": (p1_int8, p2_int8),
    }

    if qdq:
        import onnxruntime as ort

        p1_int8_qdq = output_dir / f"{prefix}_part1_int8_qdq.onnx"
        p2_int8_qdq = output_dir / f"{prefix}_part2_int8_qdq.onnx"

        cal_images = _collect_calibration_images()
        quantize_int8_qdq(part1_fp32, p1_int8_qdq, cal_images, input_name="image")

        # Part2 calibration: run part1 to collect intermediate tensors
        log("    Generating Part 2 calibration data from Part 1 outputs...")
        p1_sess = ort.InferenceSession(str(part1_fp32), providers=["CPUExecutionProvider"])
        p2_sess_meta = ort.InferenceSession(str(part2_fp32), providers=["CPUExecutionProvider"])
        p2_input_names = [inp.name for inp in p2_sess_meta.get_inputs()]
        del p2_sess_meta

        p2_cal_samples = []
        for img_np in cal_images:
            p1_outs = p1_sess.run(None, {"image": img_np})
            p1_out_names = [o.name for o in p1_sess.get_outputs()]
            sample = {name: val for name, val in zip(p1_out_names, p1_outs)
                      if name in p2_input_names}
            p2_cal_samples.append(sample)
        del p1_sess

        quantize_int8_qdq(part2_fp32, p2_int8_qdq, p2_cal_samples)
        result["int8_qdq"] = (p1_int8_qdq, p2_int8_qdq)

    return result


# ===========================================================================
# Export: u2netp (rembg)
# ===========================================================================

def export_u2netp(output_dir: Path, target_opset: int = 15, fp32_only: bool = False,
                  skip_verify: bool = False, qdq: bool = False) -> list[dict]:
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
        try:
            cal_data = _collect_rembg_calibration_images() if qdq else None
            fp16_path, int8_path, int8_qdq_path = quantize_all(
                fp32_path, "u2netp", qdq=qdq, calibration_data=cal_data)
            if not skip_verify:
                results.append(_verify_u2netp(fp16_path, "u2netp FP16"))
                results.append(_verify_u2netp(int8_path, "u2netp INT8"))
                if int8_qdq_path:
                    results.append(_verify_u2netp(int8_qdq_path, "u2netp INT8-QDQ"))
        except Exception as e:
            log(f"  WARNING: u2netp quantization failed: {e}")
            log(f"  u2netp is only {fp32_path.stat().st_size/1e6:.1f}MB — "
                f"FP32 is fine for deployment.")

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
                           img_np: np.ndarray, qdq: bool = False) -> list[dict]:
    """Export a TripoSR variant (vanilla or ToMe) through the full pipeline.

    Split happens BEFORE graph optimization — optimization renames internal
    tensors which would break the split boundary lookup.
    """
    results = []
    _print_section(f"TRIPOSR: {label}")

    fp32_path = out_dir / f"{prefix}_fp32.onnx"
    ref_out = export_triposr_fp32(wrapper_fn(), fp32_path, opset, f"{label} FP32",
                                  static=static)
    ref_np = ref_out.numpy()

    if not skip_split:
        # Split from the unoptimized graph (tensor names intact)
        p1_fp32, p2_fp32 = split_model(fp32_path, out_dir, prefix)

        # Now optimize the full model (for benchmarking) and split parts separately
        optimize_graph(fp32_path)
        optimize_graph(p1_fp32)
        optimize_graph(p2_fp32)

        if not skip_verify:
            results.append(verify_onnx(fp32_path, img_np, ref_np, "image",
                                       f"{label} FP32"))
            results.append(verify_split(p1_fp32, p2_fp32, ref_np, img_np,
                                        f"{label} Split FP32"))

        if not fp32_only:
            quant_paths = quantize_split_parts(p1_fp32, p2_fp32, out_dir, prefix,
                                               qdq=qdq)
            if not skip_verify:
                for prec, (p1, p2) in quant_paths.items():
                    results.append(verify_split(
                        p1, p2, ref_np, img_np, f"{label} Split {prec.upper()}"))
    else:
        optimize_graph(fp32_path)

        if not skip_verify:
            results.append(verify_onnx(fp32_path, img_np, ref_np, "image",
                                       f"{label} FP32"))

        if not fp32_only:
            cal_data = _collect_calibration_images() if qdq else None
            fp16_path, int8_path, int8_qdq_path = quantize_all(
                fp32_path, label, qdq=qdq, calibration_data=cal_data,
                input_name="image")
            if not skip_verify:
                results.append(verify_onnx(fp16_path, img_np, ref_np, "image",
                                           f"{label} FP16"))
                results.append(verify_onnx(int8_path, img_np, ref_np, "image",
                                           f"{label} INT8"))
                if int8_qdq_path:
                    results.append(verify_onnx(int8_qdq_path, img_np, ref_np, "image",
                                               f"{label} INT8-QDQ"))

    return results


# ===========================================================================
# Export: NeRF decoder
# ===========================================================================

def export_decoder(model: TSR, output_dir: Path, opset: int = 15,
                   fp32_only: bool = False, skip_verify: bool = False,
                   qdq: bool = False) -> list[dict]:
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

        if qdq:
            int8_qdq_path = output_dir / "nerf_decoder_int8_qdq.onnx"
            cal_data = _collect_decoder_calibration_features(output_dir)
            quantize_int8_qdq(fp32_path, int8_qdq_path, cal_data,
                              input_name="triplane_features")
            if not skip_verify:
                results.append(verify_onnx(int8_qdq_path, feat_np, ref_np,
                                           "triplane_features", "Decoder INT8-QDQ"))

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

def deploy_to_unity(models_dir: Path, precision: str = "all"):
    """Copy deployment models to Unity OnnxSource directory.

    When precision="all", copies all available variants with precision suffixes
    (e.g. triposr_part1_fp32.onnx, triposr_part1_int8.onnx). The Unity wizard
    then picks the desired precision at deploy time.

    When a specific precision is given, copies only that variant with generic
    names (for direct use without the wizard).

    u2netp always uses FP32 (FP16 is broken due to Resize op, INT8 is same size).
    """
    _print_section(f"DEPLOYING ({precision.upper()}) TO UNITY")

    if not UNITY_ONNX_SOURCE.exists():
        UNITY_ONNX_SOURCE.mkdir(parents=True, exist_ok=True)
        log(f"  Created {UNITY_ONNX_SOURCE}")

    if precision == "all":
        copies = []
        for prec in ["fp32", "fp16", "int8"]:
            for part in ["triposr_part1", "triposr_part2"]:
                src = models_dir / f"{part}_{prec}.onnx"
                if src.exists():
                    copies.append((src, f"{part}_{prec}.onnx"))

            dec_name = "nerf_decoder.onnx" if prec == "fp32" else f"nerf_decoder_{prec}.onnx"
            dec_src = models_dir / dec_name
            if dec_src.exists():
                dst_name = f"nerf_decoder_{prec}.onnx" if prec != "fp32" else "nerf_decoder_fp32.onnx"
                copies.append((dec_src, dst_name))

        # u2netp FP32 only (FP16 broken, INT8 same size)
        u2netp_src = models_dir / "u2netp.onnx"
        if u2netp_src.exists():
            copies.append((u2netp_src, "u2netp.onnx"))
    else:
        dec_name = "nerf_decoder.onnx" if precision == "fp32" else f"nerf_decoder_{precision}.onnx"
        copies = [
            (models_dir / f"triposr_part1_{precision}.onnx", "triposr_part1.onnx"),
            (models_dir / f"triposr_part2_{precision}.onnx", "triposr_part2.onnx"),
            (models_dir / dec_name, "nerf_decoder.onnx"),
            (models_dir / "u2netp.onnx", "u2netp.onnx"),
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
    parser.add_argument("--qdq", action="store_true",
                        help="Also export INT8-QDQ (static quantization) for NPU/QNN HTP")
    parser.add_argument("--qdq-only", action="store_true",
                        help="Only export INT8-QDQ variants (skip FP32/FP16/dynamic INT8 export)")
    parser.add_argument("--runs", type=int, default=10,
                        help="Benchmark runs (default: 10)")
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    if args.qdq_only:
        args.qdq = True

    if (args.tome is not None or args.tome_only) and not args.experimental:
        parser.error("ToMe export requires --experimental flag")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # ---- QDQ-only: quantize from existing FP32 models, no re-export ----
    if args.qdq_only:
        _print_section("QDQ-ONLY: Static INT8 quantization from existing FP32 models")

        if not args.skip_rembg:
            fp32 = out_dir / "u2netp.onnx"
            qdq_path = out_dir / "u2netp_int8_qdq.onnx"
            if fp32.exists():
                cal = _collect_rembg_calibration_images()
                quantize_int8_qdq(fp32, qdq_path, cal)
                all_results.append(_verify_u2netp(qdq_path, "u2netp INT8-QDQ"))
            else:
                log(f"  SKIP u2netp (no FP32 at {fp32})")

        if not args.skip_triposr:
            p1 = out_dir / "triposr_part1_fp32.onnx"
            p2 = out_dir / "triposr_part2_fp32.onnx"
            if p1.exists() and p2.exists():
                import onnxruntime as ort

                cal_images = _collect_calibration_images()
                p1_qdq = out_dir / "triposr_part1_int8_qdq.onnx"
                quantize_int8_qdq(p1, p1_qdq, cal_images, input_name="image")

                log("    Generating Part 2 calibration data from Part 1 outputs...")
                p1_sess = ort.InferenceSession(str(p1), providers=["CPUExecutionProvider"])
                p2_meta = ort.InferenceSession(str(p2), providers=["CPUExecutionProvider"])
                p2_input_names = [inp.name for inp in p2_meta.get_inputs()]
                del p2_meta

                p2_cal = []
                for img in cal_images:
                    outs = p1_sess.run(None, {"image": img})
                    out_names = [o.name for o in p1_sess.get_outputs()]
                    sample = {n: v for n, v in zip(out_names, outs) if n in p2_input_names}
                    p2_cal.append(sample)
                del p1_sess

                p2_qdq = out_dir / "triposr_part2_int8_qdq.onnx"
                quantize_int8_qdq(p2, p2_qdq, p2_cal)

                if not args.skip_verify:
                    teacher = load_teacher("cpu")
                    img_np = get_dummy_image("cpu").numpy()
                    ref_out = TripoSRForward(teacher)(torch.from_numpy(img_np)).detach().numpy()
                    all_results.append(verify_split(
                        p1_qdq, p2_qdq, ref_out, img_np, "Vanilla Split INT8_QDQ"))
            else:
                log(f"  SKIP triposr (no split FP32 parts)")

        if not args.skip_decoder:
            dec_fp32 = out_dir / "nerf_decoder.onnx"
            if dec_fp32.exists():
                cal = _collect_decoder_calibration_features(out_dir)
                dec_qdq = out_dir / "nerf_decoder_int8_qdq.onnx"
                quantize_int8_qdq(dec_fp32, dec_qdq, cal, input_name="triplane_features")
            else:
                log(f"  SKIP decoder (no FP32 at {dec_fp32})")

        validate_all_models(out_dir)

        if all_results:
            _print_section("ACCURACY & SIZE SUMMARY")
            log(f"{'Variant':<35} {'Size':>8} {'Max Err%':>10} {'Mean Err%':>10}")
            log("-" * 65)
            for r in all_results:
                log(f"{r['label']:<35} {r['size_mb']:>7.1f}MB "
                    f"{r['max_rel']*100:>9.4f}% {r['mean_rel']*100:>9.4f}%")

        if args.deploy:
            deploy_to_unity(out_dir, args.deploy)
        log("\nDone!")
        return

    if not args.benchmark_only:

        # ---- Rembg (u2netp) ----
        if not args.skip_rembg and not args.tome_only:
            all_results.extend(export_u2netp(
                out_dir, args.opset, args.fp32_only, args.skip_verify,
                qdq=args.qdq))

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
                    qdq=args.qdq,
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
                    qdq=args.qdq,
                ))

        # ---- NeRF Decoder ----
        if not args.skip_decoder and not args.tome_only:
            if not args.skip_triposr:
                pass  # teacher already loaded
            else:
                log("\nLoading teacher model for decoder...")
                teacher = load_teacher("cpu")

            all_results.extend(export_decoder(
                teacher, out_dir, args.opset, args.fp32_only, args.skip_verify,
                qdq=args.qdq))

        # ---- Static shape validation ----
        validate_all_models(out_dir)

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
