"""Split oversized attention MatMul ops in TripoSR ONNX for Quest 3 deployment.

Quest 3's Adreno 740 GPU has a 128 MB per-compute-buffer limit.  The TripoSR
transformer decoder's attention matrices exceed this:
  - Self-attention  (attn1): [1,16,3072,3072] = 576 MB  (blocks 1-15)
  - Cross-attention (attn2): [1,16,3072,1025] = 192 MB  (blocks 0-15)

This script performs lossless ONNX graph surgery: it splits the
  MatMul(Q,K^T) -> Softmax -> MatMul(attn,V)
chain along the head dimension so every intermediate buffer stays under 128 MB.
The output is numerically identical to the original.

Usage:
    python chunk_attention.py [--input PATH] [--output PATH] [--limit-mb 128] [--verify]
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference


MAX_BUFFER_BYTES = 128 * 1024 * 1024  # 128 MB


def get_shape_map(graph):
    """Build tensor-name → static shape mapping from graph value_info + I/O."""
    smap = {}
    for vi in graph.value_info:
        tt = vi.type.tensor_type
        if tt.HasField("shape"):
            dims = [d.dim_value for d in tt.shape.dim]
            if all(d > 0 for d in dims):
                smap[vi.name] = dims
    for collection in (graph.input, graph.output):
        for vi in collection:
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            if all(d > 0 for d in dims):
                smap[vi.name] = dims
    return smap


def buffer_bytes(shape, dtype_bytes=4):
    return int(np.prod(shape)) * dtype_bytes


def find_node_by_output(graph, tensor_name):
    for n in graph.node:
        if tensor_name in n.output:
            return n
    return None


def find_consumers(graph, tensor_name):
    return [n for n in graph.node if tensor_name in n.input]


def find_attention_triplets(graph, shape_map):
    """Find all  MatMul -> Softmax -> MatMul  attention chains where the
    first MatMul output exceeds the buffer limit."""
    triplets = []
    for node in graph.node:
        if node.op_type != "MatMul":
            continue
        out = node.output[0]
        shape = shape_map.get(out)
        if shape is None or buffer_bytes(shape) <= MAX_BUFFER_BYTES:
            continue
        # Must be followed by Softmax
        consumers = find_consumers(graph, out)
        softmax = next((c for c in consumers if c.op_type == "Softmax"), None)
        if softmax is None:
            continue
        # Softmax must be followed by a second MatMul (attn @ V)
        sm_out = softmax.output[0]
        sm_consumers = find_consumers(graph, sm_out)
        matmul_v = next((c for c in sm_consumers if c.op_type == "MatMul"), None)
        if matmul_v is None:
            continue
        triplets.append((node, softmax, matmul_v))
    return triplets


def compute_num_chunks(shape, limit_bytes=MAX_BUFFER_BYTES):
    """Determine how many head-chunks are needed so each chunk's buffer fits."""
    num_heads = shape[1]
    per_head_bytes = buffer_bytes([shape[0], 1, shape[2], shape[3]])
    max_heads = limit_bytes // per_head_bytes
    if max_heads <= 0:
        max_heads = 1
    num_chunks = math.ceil(num_heads / max_heads)
    heads_per_chunk = math.ceil(num_heads / num_chunks)
    return num_chunks, heads_per_chunk


def make_constant_node(name, value, dtype=TensorProto.INT64):
    arr = np.array(value, dtype=np.int64)
    tensor = helper.make_tensor(name + "_val", dtype, arr.shape, arr.flatten().tolist())
    return helper.make_node("Constant", inputs=[], outputs=[name],
                            value=tensor, name=name + "_const")


def chunk_attention_triplet(graph, qk_matmul, softmax, av_matmul, shape_map, idx):
    """Replace one  MatMul->Softmax->MatMul  chain with head-chunked equivalent.
    Returns (new_nodes, new_initializers) to insert, and the names of nodes to remove."""

    prefix = f"chunked_attn_{idx}"
    qk_out_shape = shape_map[qk_matmul.output[0]]
    num_chunks, heads_per = compute_num_chunks(qk_out_shape)
    num_heads = qk_out_shape[1]

    q_name = qk_matmul.input[0]   # [1, H, Q, D]
    k_name = qk_matmul.input[1]   # [1, H, D, S]
    v_name = av_matmul.input[1]    # [1, H, S, D]
    # If softmax output is in input[0] of av_matmul, V is input[1]; otherwise swap
    if av_matmul.input[0] != softmax.output[0]:
        v_name = av_matmul.input[0]

    new_nodes = []
    new_inits = []
    chunk_out_names = []

    for c in range(num_chunks):
        h_start = c * heads_per
        h_end = min(h_start + heads_per, num_heads)
        cp = f"{prefix}_c{c}"

        starts_name = f"{cp}_starts"
        ends_name = f"{cp}_ends"
        axes_name = f"{cp}_axes"

        new_nodes.append(make_constant_node(starts_name, [h_start]))
        new_nodes.append(make_constant_node(ends_name, [h_end]))
        new_nodes.append(make_constant_node(axes_name, [1]))

        # Slice Q along head dim
        q_slice_out = f"{cp}_q_slice"
        new_nodes.append(helper.make_node(
            "Slice", inputs=[q_name, starts_name, ends_name, axes_name],
            outputs=[q_slice_out], name=f"{cp}_slice_q"))

        # Slice K along head dim
        k_slice_out = f"{cp}_k_slice"
        new_nodes.append(helper.make_node(
            "Slice", inputs=[k_name, starts_name, ends_name, axes_name],
            outputs=[k_slice_out], name=f"{cp}_slice_k"))

        # MatMul Q_chunk @ K_chunk (small attention scores)
        scores_out = f"{cp}_scores"
        new_nodes.append(helper.make_node(
            "MatMul", inputs=[q_slice_out, k_slice_out],
            outputs=[scores_out], name=f"{cp}_qk_matmul"))

        # Softmax (per-head, so chunking is mathematically correct)
        attn_out = f"{cp}_attn"
        sm_axis = -1
        for attr in softmax.attribute:
            if attr.name == "axis":
                sm_axis = attr.i
        new_nodes.append(helper.make_node(
            "Softmax", inputs=[scores_out], outputs=[attn_out],
            axis=sm_axis, name=f"{cp}_softmax"))

        # Slice V along head dim
        v_slice_out = f"{cp}_v_slice"
        new_nodes.append(helper.make_node(
            "Slice", inputs=[v_name, starts_name, ends_name, axes_name],
            outputs=[v_slice_out], name=f"{cp}_slice_v"))

        # MatMul attn_chunk @ V_chunk
        out_name = f"{cp}_av_out"
        new_nodes.append(helper.make_node(
            "MatMul", inputs=[attn_out, v_slice_out],
            outputs=[out_name], name=f"{cp}_av_matmul"))

        chunk_out_names.append(out_name)

    # Concat all chunks along head dim → same shape as original av_matmul output
    concat_out = av_matmul.output[0]
    new_nodes.append(helper.make_node(
        "Concat", inputs=chunk_out_names, outputs=[concat_out],
        axis=1, name=f"{prefix}_concat"))

    remove_names = {qk_matmul.name, softmax.name, av_matmul.name}
    return new_nodes, new_inits, remove_names


def chunk_attention(model, limit_mb=128):
    global MAX_BUFFER_BYTES
    MAX_BUFFER_BYTES = int(limit_mb * 1024 * 1024)

    model = shape_inference.infer_shapes(model)
    graph = model.graph
    shape_map = get_shape_map(graph)

    triplets = find_attention_triplets(graph, shape_map)
    if not triplets:
        print("No attention MatMuls exceed the buffer limit. Nothing to do.")
        return model

    print(f"Found {len(triplets)} attention chains exceeding {limit_mb} MB buffer limit.")

    all_new_nodes = []
    all_remove_names = set()

    for idx, (qk, sm, av) in enumerate(triplets):
        qk_shape = shape_map[qk.output[0]]
        n_chunks, hpc = compute_num_chunks(qk_shape)
        chunk_mb = buffer_bytes([qk_shape[0], hpc, qk_shape[2], qk_shape[3]]) / (1024 * 1024)
        print(f"  [{idx:2d}] {qk.name}")
        print(f"       shape={qk_shape} ({buffer_bytes(qk_shape)/(1024*1024):.0f} MB)"
              f" -> {n_chunks} chunks of {hpc} heads ({chunk_mb:.0f} MB each)")

        new_nodes, _, remove_names = chunk_attention_triplet(
            graph, qk, sm, av, shape_map, idx)
        all_new_nodes.extend(new_nodes)
        all_remove_names.update(remove_names)

    # Rebuild node list: replace removed nodes in-place with chunked equivalents
    # to maintain topological order
    final_nodes = []
    inserted = set()
    for node in graph.node:
        if node.name in all_remove_names:
            # Insert chunked replacement nodes at position of the first removed node
            # (the QK MatMul) to maintain topological order
            triplet_idx = None
            for idx, (qk, sm, av) in enumerate(triplets):
                if node.name == qk.name:
                    triplet_idx = idx
                    break
            if triplet_idx is not None and triplet_idx not in inserted:
                prefix = f"chunked_attn_{triplet_idx}_"
                for nn in all_new_nodes:
                    if nn.name.startswith(prefix):
                        final_nodes.append(nn)
                inserted.add(triplet_idx)
        else:
            final_nodes.append(node)

    del graph.node[:]
    graph.node.extend(final_nodes)

    # Remove stale value_info entries for tensors no longer produced by any node
    live_tensors = set()
    for n in graph.node:
        live_tensors.update(n.output)
    for inp in graph.input:
        live_tensors.add(inp.name)
    stale = [vi for vi in graph.value_info if vi.name not in live_tensors]
    for vi in stale:
        graph.value_info.remove(vi)

    # Re-run shape inference to annotate new nodes
    model = shape_inference.infer_shapes(model)

    # Verify no remaining buffers exceed limit
    new_shape_map = get_shape_map(model.graph)
    violations = []
    for name, shape in new_shape_map.items():
        if buffer_bytes(shape) > MAX_BUFFER_BYTES:
            violations.append((name, shape, buffer_bytes(shape) / (1024 * 1024)))
    if violations:
        print(f"\nWARNING: {len(violations)} tensors still exceed {limit_mb} MB:")
        for name, shape, mb in violations[:5]:
            print(f"  {name}: {shape} = {mb:.1f} MB")
    else:
        print(f"\nAll intermediate tensors are now under {limit_mb} MB.")

    return model


def verify_with_ort(original_path, chunked_path):
    """Run both models through ORT and compare outputs."""
    import onnxruntime as ort

    print("\n=== ORT Verification ===")
    rng = np.random.RandomState(42)
    dummy_input = rng.randn(1, 3, 512, 512).astype(np.float32)

    print("Running original model...")
    sess_orig = ort.InferenceSession(str(original_path), providers=["CPUExecutionProvider"])
    out_orig = sess_orig.run(None, {"image": dummy_input})[0]

    print("Running chunked model...")
    sess_chunk = ort.InferenceSession(str(chunked_path), providers=["CPUExecutionProvider"])
    out_chunk = sess_chunk.run(None, {"image": dummy_input})[0]

    max_diff = np.max(np.abs(out_orig - out_chunk))
    mean_diff = np.mean(np.abs(out_orig - out_chunk))
    cos_sim = np.dot(out_orig.flatten(), out_chunk.flatten()) / (
        np.linalg.norm(out_orig) * np.linalg.norm(out_chunk) + 1e-12)

    print(f"  Output shape: {out_orig.shape}")
    print(f"  Max  diff: {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")
    print(f"  Cosine similarity: {cos_sim:.10f}")

    if max_diff < 1e-4:
        print("  PASS: outputs are numerically identical (within FP32 tolerance)")
    elif max_diff < 1e-2:
        print("  WARN: small numerical differences (likely FP32 accumulation order)")
    else:
        print("  FAIL: significant differences detected — graph surgery may have a bug")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    default_input = (Path(__file__).parent.parent /
                     "SentienceUnity/Assets/Game/ObjectReconstruction/OnnxSource/triposr_fp32.onnx")
    default_output = default_input

    parser.add_argument("--input", type=Path, default=default_input,
                        help="Input ONNX model path")
    parser.add_argument("--output", type=Path, default=default_output,
                        help="Output chunked ONNX model path")
    parser.add_argument("--limit-mb", type=int, default=128,
                        help="Max buffer size in MB (default: 128)")
    parser.add_argument("--verify", action="store_true",
                        help="Run ORT verification after chunking")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    model = onnx.load(str(args.input))
    print(f"  opset: {model.opset_import[0].version}")
    print(f"  nodes: {len(model.graph.node)}")

    model = chunk_attention(model, limit_mb=args.limit_mb)

    print(f"\nSaving to {args.output} ...")
    onnx.save(model, str(args.output))
    print(f"  nodes: {len(model.graph.node)}")

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"  file size: {size_mb:.1f} MB")

    if args.verify:
        ok = verify_with_ort(args.input, args.output)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
