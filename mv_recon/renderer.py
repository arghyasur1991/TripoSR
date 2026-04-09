"""Differentiable volume renderer for training supervision.

Renders density + color volumes using sigmoid occupancy (alpha = sigmoid(logit)).
This aligns with marching cubes extraction (iso-surface at 0.5) and works well
for discrete low-resolution voxel grids where physically-based density
integration (softplus + delta) produces near-invisible thin structures.

Two modes:
  - render_rays_batch(): random ray sampling for efficient training
  - render_volume(): full-image rendering for evaluation/visualization
"""

import torch
import torch.nn.functional as F


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def _ray_aabb_intersect(rays_o: torch.Tensor, rays_d: torch.Tensor,
                        aabb_min: float = -0.55, aabb_max: float = 0.55
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    inv_d = 1.0 / (rays_d + 1e-10)
    t1 = (aabb_min - rays_o) * inv_d
    t2 = (aabb_max - rays_o) * inv_d
    t_min = torch.minimum(t1, t2)
    t_max = torch.maximum(t1, t2)
    t_near = t_min.max(dim=-1).values.clamp(min=0.01)
    t_far = t_max.min(dim=-1).values
    return t_near, t_far


def _sample_volume(vol: torch.Tensor, points: torch.Tensor,
                   voxel_range: float = 0.55) -> torch.Tensor:
    """Sample from a 3D volume at world-space points via grid_sample.

    Args:
        vol: [C, D, D, D] or [1, C, D, D, D] volume (logits, colors, etc.)
        points: [N, 3] world-space points

    Returns: [N, C] sampled values.
    """
    if vol.dim() == 3:
        vol = vol.unsqueeze(0)  # [1, D, D, D] → treat C=1
    if vol.dim() == 4:
        vol = vol.unsqueeze(0)  # [1, C, D, D, D]
    C = vol.shape[1]
    grid = (points / voxel_range).reshape(1, -1, 1, 1, 3)
    out = F.grid_sample(
        vol, grid,
        mode='bilinear', padding_mode='zeros', align_corners=False,
    )  # [1, C, N, 1, 1]
    return out.reshape(C, -1).T  # [N, C]


def _composite_rays(alpha: torch.Tensor, colors: torch.Tensor,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard alpha compositing (front-to-back).

    Args:
        alpha: [N_rays, N_samples] per-sample opacity in [0, 1]
        colors: [N_rays, N_samples, 3] per-sample RGB

    Returns:
        rgb: [N_rays, 3]
        mask: [N_rays] accumulated opacity
    """
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    T = torch.cat([torch.ones_like(T[:, :1]), T[:, :-1]], dim=-1)
    weights = T * alpha
    rgb = (weights.unsqueeze(-1) * colors).sum(dim=1)
    mask = weights.sum(dim=1)
    return rgb, mask


def _denormalize_images(images: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization: normalized → RGB [0, 1]."""
    mean = IMAGENET_MEAN.to(images.device).reshape(1, 3, 1, 1)
    std = IMAGENET_STD.to(images.device).reshape(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def render_rays_batch(density_vol: torch.Tensor,
                      color_vol: torch.Tensor,
                      sup_c2w: torch.Tensor, K_sup: torch.Tensor,
                      gt_rgbs: torch.Tensor, gt_masks: torch.Tensor,
                      render_h: int = 64, render_w: int = 64,
                      n_samples: int = 64, n_rays_per_view: int = 512,
                      voxel_range: float = 0.55,
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render random rays from learned density + color volumes.

    Args:
        density_vol: [1, D, D, D] raw density logits
        color_vol: [3, D, D, D] raw color values (sigmoid applied inside)
        sup_c2w: [V_sup, 4, 4] supervision cameras (Blender convention)
        K_sup: [3, 3] intrinsics for supervision resolution
        gt_rgbs: [V_sup, 3, H, W] GT supervision images
        gt_masks: [V_sup, H, W] GT alpha masks

    Returns:
        rgb_pred, rgb_gt, mask_pred, mask_gt
    """
    device = density_vol.device
    V_sup = sup_c2w.shape[0]

    flip = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0],
                         [0, 0, -1, 0], [0, 0, 0, 1]],
                        dtype=torch.float32, device=device)

    all_rgb_pred, all_rgb_gt = [], []
    all_mask_pred, all_mask_gt = [], []

    fx, fy = K_sup[0, 0], K_sup[1, 1]
    cx, cy = K_sup[0, 2], K_sup[1, 2]
    bg_color = 0.5

    for v_idx in range(V_sup):
        c2w_cv = sup_c2w[v_idx] @ flip
        cam_pos = c2w_cv[:3, 3]
        R = c2w_cv[:3, :3]

        pixel_u = torch.rand(n_rays_per_view, device=device) * render_w
        pixel_v = torch.rand(n_rays_per_view, device=device) * render_h

        dirs_cam = torch.stack([
            (pixel_u - cx) / fx,
            (pixel_v - cy) / fy,
            torch.ones(n_rays_per_view, device=device),
        ], dim=-1)
        dirs_cam = F.normalize(dirs_cam, dim=-1)
        rays_d = dirs_cam @ R.T
        rays_o = cam_pos.unsqueeze(0).expand(n_rays_per_view, -1)

        t_near, t_far = _ray_aabb_intersect(rays_o, rays_d, -voxel_range, voxel_range)
        valid = t_far > t_near
        if not valid.any():
            continue

        rays_o_v = rays_o[valid]
        rays_d_v = rays_d[valid]
        t_near_v = t_near[valid].unsqueeze(-1)
        t_far_v = t_far[valid].unsqueeze(-1)

        t_vals = torch.linspace(0, 1, n_samples, device=device)
        t_samples = t_near_v + (t_far_v - t_near_v) * t_vals
        pts = rays_o_v.unsqueeze(1) + rays_d_v.unsqueeze(1) * t_samples.unsqueeze(-1)

        N_valid = pts.shape[0]
        pts_flat = pts.reshape(-1, 3)

        # Sample density → sigmoid → alpha
        density_logits = _sample_volume(density_vol, pts_flat, voxel_range)  # [N*S, 1]
        alpha = torch.sigmoid(density_logits[:, 0]).reshape(N_valid, n_samples)

        # Sample color → sigmoid → RGB
        color_raw = _sample_volume(color_vol, pts_flat, voxel_range)  # [N*S, 3]
        colors = torch.sigmoid(color_raw).reshape(N_valid, n_samples, 3)

        rgb_raw, mask_pred = _composite_rays(alpha, colors)
        rgb_pred = rgb_raw + bg_color * (1.0 - mask_pred.unsqueeze(-1))

        # Sample GT at the same pixel locations
        pu_norm = 2.0 * pixel_u[valid] / render_w - 1.0
        pv_norm = 2.0 * pixel_v[valid] / render_h - 1.0
        gt_grid = torch.stack([pu_norm, pv_norm], dim=-1).unsqueeze(0).unsqueeze(0)

        gt_rgb_sampled = F.grid_sample(
            gt_rgbs[v_idx:v_idx+1], gt_grid,
            mode='bilinear', padding_mode='border', align_corners=False,
        ).reshape(3, -1).T

        gt_mask_sampled = F.grid_sample(
            gt_masks[v_idx:v_idx+1].unsqueeze(0), gt_grid,
            mode='bilinear', padding_mode='border', align_corners=False,
        ).reshape(-1)

        all_rgb_pred.append(rgb_pred)
        all_rgb_gt.append(gt_rgb_sampled)
        all_mask_pred.append(mask_pred)
        all_mask_gt.append(gt_mask_sampled)

    if not all_rgb_pred:
        z3 = torch.zeros(1, 3, device=device)
        z1 = torch.zeros(1, device=device)
        return z3, z3, z1, z1

    return (torch.cat(all_rgb_pred), torch.cat(all_rgb_gt),
            torch.cat(all_mask_pred), torch.cat(all_mask_gt))


def render_volume(density: torch.Tensor, c2w_blender: torch.Tensor,
                  K: torch.Tensor, render_h: int = 128, render_w: int = 128,
                  n_samples: int = 96, voxel_range: float = 0.55,
                  color_vol: torch.Tensor | None = None,
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-image rendering for evaluation.

    Returns (rgb [H,W,3], mask [H,W]). If color_vol is None, renders white.
    """
    device = density.device
    flip = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0],
                         [0, 0, -1, 0], [0, 0, 0, 1]],
                        dtype=torch.float32, device=device)
    c2w_cv = c2w_blender @ flip

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    v, u = torch.meshgrid(
        torch.arange(render_h, dtype=torch.float32, device=device) + 0.5,
        torch.arange(render_w, dtype=torch.float32, device=device) + 0.5,
        indexing='ij',
    )
    dirs_cam = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=-1)
    dirs_cam = F.normalize(dirs_cam.reshape(-1, 3), dim=-1)
    R = c2w_cv[:3, :3]
    rays_d = dirs_cam @ R.T
    rays_o = c2w_cv[:3, 3].unsqueeze(0).expand_as(rays_d)

    t_near, t_far = _ray_aabb_intersect(rays_o, rays_d, -voxel_range, voxel_range)
    valid = t_far > t_near
    n_rays = rays_o.shape[0]

    if not valid.any():
        return (torch.full((render_h, render_w, 3), 0.5, device=device),
                torch.zeros(render_h, render_w, device=device))

    t_vals = torch.linspace(0, 1, n_samples, device=device)
    t_near_v = t_near[valid].unsqueeze(-1)
    t_far_v = t_far[valid].unsqueeze(-1)
    t_samples = t_near_v + (t_far_v - t_near_v) * t_vals
    pts = rays_o[valid].unsqueeze(1) + rays_d[valid].unsqueeze(1) * t_samples.unsqueeze(-1)

    N_valid = pts.shape[0]
    pts_flat = pts.reshape(-1, 3)

    density_logits = _sample_volume(density, pts_flat, voxel_range)  # [N*S, 1]
    alpha = torch.sigmoid(density_logits[:, 0]).reshape(N_valid, n_samples)

    if color_vol is not None:
        color_raw = _sample_volume(color_vol, pts_flat, voxel_range)  # [N*S, 3]
        colors = torch.sigmoid(color_raw).reshape(N_valid, n_samples, 3)
    else:
        colors = torch.ones(N_valid, n_samples, 3, device=device)

    rgb_v, mask_v = _composite_rays(alpha, colors)

    mask_full = torch.zeros(n_rays, device=device)
    mask_full[valid] = mask_v
    rgb_full = torch.full((n_rays, 3), 0.5, device=device)
    rgb_full[valid] = rgb_v + 0.5 * (1.0 - mask_v.unsqueeze(-1))

    return rgb_full.reshape(render_h, render_w, 3), mask_full.reshape(render_h, render_w)
