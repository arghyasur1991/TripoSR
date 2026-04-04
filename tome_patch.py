"""Token Merging (ToMe) for TripoSR's Transformer1D backbone.

Merges similar triplane tokens between transformer layers to reduce
self-attention cost. Merging is within-plane only (3 planes of 32x32 = 1024
tokens each) to avoid blending XY/XZ/YZ semantics.

Usage:
    from tome_patch import apply_tome, remove_tome

    model = TSR.from_pretrained(...)
    apply_tome(model.backbone, merge_ratio=0.2, merge_layers=[4, 8, 12])
    # ... run inference as normal ...
    # scene_codes = model(image, device)   # internally merges+unmerges
    remove_tome(model.backbone)            # restore original behavior
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


PLANE_SIZE = 32
TOKENS_PER_PLANE = PLANE_SIZE * PLANE_SIZE  # 1024
NUM_PLANES = 3
TOTAL_TOKENS = NUM_PLANES * TOKENS_PER_PLANE  # 3072


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bipartite soft matching for a single plane's tokens.

    Args:
        metric: (B, N, C) token features for similarity computation
        r: number of tokens to merge (remove)

    Returns:
        dst_idx: (B, N-r) indices into original to gather destination tokens
        src_idx: (B, r) indices of source tokens that get merged into destinations
        merge_map: (B, r) for each source, which destination it merges into (index into dst set)
    """
    B, N, C = metric.shape
    if r <= 0 or r >= N // 2:
        return None, None, None

    with torch.no_grad():
        metric = F.normalize(metric, dim=-1)

        # split into two sets: even and odd indices
        a_idx = torch.arange(0, N, 2, device=metric.device)  # src candidates
        b_idx = torch.arange(1, N, 2, device=metric.device)  # dst candidates

        a = metric[:, a_idx]  # (B, N//2, C)
        b = metric[:, b_idx]  # (B, N//2, C)

        # cosine similarity between all pairs
        scores = torch.bmm(a, b.transpose(1, 2))  # (B, N//2, N//2)

        # for each src token, find its most similar dst token
        node_max, node_idx = scores.max(dim=-1)  # (B, N//2)

        # pick the top-r most similar pairs
        _, sorted_indices = node_max.sort(dim=-1, descending=True)  # (B, N//2)
        merge_src_local = sorted_indices[:, :r]   # (B, r) indices into a_idx
        keep_src_local = sorted_indices[:, r:]     # (B, N//2 - r) indices into a_idx

        # map back to original token indices
        src_idx = a_idx[merge_src_local]          # (B, r) original indices of merged tokens
        dst_local = torch.gather(node_idx, 1, merge_src_local)  # (B, r) index into b_idx
        dst_merge_target = b_idx[dst_local]       # (B, r) original indices of merge targets

        # kept tokens: all b tokens + non-merged a tokens
        keep_a = a_idx[keep_src_local]            # (B, N//2 - r)
        keep_b = b_idx                             # (N//2,) broadcast over batch
        keep_b = keep_b.unsqueeze(0).expand(B, -1)

        kept_idx = torch.cat([keep_a, keep_b], dim=1)  # (B, N - r)
        kept_idx, _ = kept_idx.sort(dim=1)

    return kept_idx, src_idx, dst_merge_target


def merge_tokens(
    x: torch.Tensor,
    kept_idx: torch.Tensor,
    src_idx: torch.Tensor,
    dst_merge_target: torch.Tensor,
) -> torch.Tensor:
    """Merge tokens by averaging matched pairs.

    Args:
        x: (B, N, C) input tokens
        kept_idx: (B, N-r) indices of tokens to keep
        src_idx: (B, r) indices of source tokens
        dst_merge_target: (B, r) indices of destination tokens to merge into

    Returns:
        merged: (B, N-r, C) tokens after merging
    """
    B, N, C = x.shape
    r = src_idx.shape[1]

    # gather kept tokens
    merged = torch.gather(x, 1, kept_idx.unsqueeze(-1).expand(-1, -1, C))

    # for each merged pair, average src into its dst
    src_tokens = torch.gather(x, 1, src_idx.unsqueeze(-1).expand(-1, -1, C))

    # find where each dst_merge_target appears in kept_idx
    # kept_idx is sorted, so we can use searchsorted
    dst_positions = torch.searchsorted(kept_idx, dst_merge_target)
    dst_positions = dst_positions.clamp(0, kept_idx.shape[1] - 1)

    # average: merged[dst_pos] = (merged[dst_pos] + src_token) / 2
    merged.scatter_add_(1, dst_positions.unsqueeze(-1).expand(-1, -1, C), src_tokens)

    # build a count tensor to normalize
    counts = torch.ones(B, merged.shape[1], 1, device=x.device, dtype=x.dtype)
    ones = torch.ones(B, r, 1, device=x.device, dtype=x.dtype)
    counts.scatter_add_(1, dst_positions.unsqueeze(-1), ones)
    merged = merged / counts

    return merged


def unmerge_tokens(
    merged: torch.Tensor,
    kept_idx: torch.Tensor,
    src_idx: torch.Tensor,
    dst_merge_target: torch.Tensor,
    original_n: int,
) -> torch.Tensor:
    """Scatter merged tokens back to original positions.

    Each merged-away source token gets the value of its merge target.
    """
    B, _, C = merged.shape
    output = torch.zeros(B, original_n, C, device=merged.device, dtype=merged.dtype)

    # place kept tokens
    output.scatter_(1, kept_idx.unsqueeze(-1).expand(-1, -1, C), merged)

    # for source tokens, copy from their merge target in the output
    dst_values = torch.gather(output, 1, dst_merge_target.unsqueeze(-1).expand(-1, -1, C))
    output.scatter_(1, src_idx.unsqueeze(-1).expand(-1, -1, C), dst_values)

    return output


class ToMeState:
    """Tracks merge state across layers for unmerging."""
    def __init__(self):
        self.merge_info: List[Tuple] = []  # per-plane merge info per layer
        self.active = True
        self.current_n_per_plane: List[int] = [TOKENS_PER_PLANE] * NUM_PLANES


def apply_tome(
    backbone,
    merge_ratio: float = 0.2,
    merge_layers: Optional[List[int]] = None,
):
    """Monkey-patch Transformer1D to apply within-plane Token Merging.

    Args:
        backbone: The Transformer1D module
        merge_ratio: fraction of per-plane tokens to merge at each merge layer (0-0.5)
        merge_layers: which layer indices trigger merging (default: [4, 8, 12])
    """
    if merge_layers is None:
        merge_layers = [4, 8, 12]

    state = ToMeState()
    backbone._tome_state = state
    backbone._tome_merge_ratio = merge_ratio
    backbone._tome_merge_layers = set(merge_layers)

    original_forward = backbone.__class__.forward

    def tome_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        encoder_attention_mask=None,
    ):
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        batch, _, seq_len = hidden_states.shape
        residual_full = hidden_states  # (B, C, 3072) — need to unmerge to this size

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 1).reshape(batch, seq_len, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        state = self._tome_state
        state.merge_info = []
        state.current_n_per_plane = [TOKENS_PER_PLANE] * NUM_PLANES

        for layer_idx, block in enumerate(self.transformer_blocks):
            if layer_idx in self._tome_merge_layers and state.active:
                hidden_states = _merge_step(
                    hidden_states, state, self._tome_merge_ratio
                )

            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )

        # unmerge back to full 3072 tokens
        hidden_states = _unmerge_all(hidden_states, state)

        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, seq_len, inner_dim)
            .permute(0, 2, 1)
            .contiguous()
        )
        output = hidden_states + residual_full
        return output

    backbone.forward = lambda *args, **kwargs: tome_forward(backbone, *args, **kwargs)
    backbone._tome_original_forward = original_forward


def _merge_step(
    hidden_states: torch.Tensor,
    state: ToMeState,
    merge_ratio: float,
) -> torch.Tensor:
    """Apply one round of within-plane merging."""
    B, N, C = hidden_states.shape

    planes = []
    offset = 0
    layer_merge_info = []

    for p in range(NUM_PLANES):
        n_p = state.current_n_per_plane[p]
        plane_tokens = hidden_states[:, offset:offset + n_p, :]
        r = int(n_p * merge_ratio)

        if r > 0 and n_p > 2 * r:
            kept_idx, src_idx, dst_merge_target = bipartite_soft_matching(
                plane_tokens, r
            )
            if kept_idx is not None:
                merged_plane = merge_tokens(plane_tokens, kept_idx, src_idx, dst_merge_target)
                planes.append(merged_plane)
                layer_merge_info.append((kept_idx, src_idx, dst_merge_target, n_p))
                state.current_n_per_plane[p] = merged_plane.shape[1]
            else:
                planes.append(plane_tokens)
                layer_merge_info.append(None)
        else:
            planes.append(plane_tokens)
            layer_merge_info.append(None)

        offset += n_p

    state.merge_info.append(layer_merge_info)
    return torch.cat(planes, dim=1)


def _unmerge_all(
    hidden_states: torch.Tensor,
    state: ToMeState,
) -> torch.Tensor:
    """Unmerge all layers in reverse order to restore original token count."""
    for layer_info in reversed(state.merge_info):
        B, N, C = hidden_states.shape

        planes = []
        offset = 0

        # determine current per-plane sizes from the merged state
        plane_sizes = []
        remaining = N
        for p in range(NUM_PLANES):
            if layer_info[p] is not None:
                _, _, _, original_n = layer_info[p]
                # current size is original_n - r
                r = layer_info[p][1].shape[1]
                current = original_n - r
            else:
                current = remaining // (NUM_PLANES - p)
            plane_sizes.append(current)
            remaining -= current

        for p in range(NUM_PLANES):
            n_p = plane_sizes[p]
            plane_tokens = hidden_states[:, offset:offset + n_p, :]

            if layer_info[p] is not None:
                kept_idx, src_idx, dst_merge_target, original_n = layer_info[p]
                plane_tokens = unmerge_tokens(
                    plane_tokens, kept_idx, src_idx, dst_merge_target, original_n
                )

            planes.append(plane_tokens)
            offset += n_p

        hidden_states = torch.cat(planes, dim=1)

    return hidden_states


def remove_tome(backbone):
    """Remove ToMe patching and restore original forward."""
    if hasattr(backbone, '_tome_original_forward'):
        backbone.forward = lambda *args, **kwargs: backbone._tome_original_forward(backbone, *args, **kwargs)
        del backbone._tome_original_forward
        del backbone._tome_state
        del backbone._tome_merge_ratio
        del backbone._tome_merge_layers
