"""Differentiable volume renderer for training supervision.

Two modes:
  - render_volume(): full-image rendering for evaluation/visualization
  - render_rays_batch(): random ray sampling for efficient training
"""

import torch
import torch.nn.functional as F

from .camera_utils import blender_c2w_to_opencv_w2c


def _ray_aabb_intersect(rays_o: torch.Tensor, rays_d: torch.Tensor,
                        aabb_min: float = -0.55, aabb_max: float = 0.55
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Ray-AABB intersection. Returns (t_near, t_far) per ray."""
    inv_d = 1.0 / (rays_d + 1e-10)
    t1 = (aabb_min - rays_o) * inv_d
    t2 = (aabb_max - rays_o) * inv_d
    t_min = torch.minimum(t1, t2)
    t_max = torch.maximum(t1, t2)
    t_near = t_min.max(dim=-1).values.clamp(min=0.01)
    t_far = t_max.min(dim=-1).values
    return t_near, t_far


def _get_rays_opencv(c2w_opencv: torch.Tensor, K: torch.Tensor,
                     H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate rays for all pixels. Returns rays_o, rays_d as [H*W, 3]."""
    device = c2w_opencv.device
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    v, u = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device) + 0.5,
        torch.arange(W, dtype=torch.float32, device=device) + 0.5,
        indexing='ij'
    )
    dirs_cam = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=-1)
    dirs_cam = F.normalize(dirs_cam.reshape(-1, 3), dim=-1)
    R = c2w_opencv[:3, :3]
    rays_d = dirs_cam @ R.T
    rays_o = c2w_opencv[:3, 3].unsqueeze(0).expand_as(rays_d)
    return rays_o, rays_d


def _composite_rays(sigma: torch.Tensor, colors: torch.Tensor,
                    t_samples: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard NeRF alpha compositing.

    Args:
        sigma: [N_rays, N_samples] activated density
        colors: [N_rays, N_samples, 3] RGB
        t_samples: [N_rays, N_samples] sample distances

    Returns:
        rgb: [N_rays, 3]
        mask: [N_rays]
    """
    deltas = t_samples[:, 1:] - t_samples[:, :-1]
    deltas = torch.cat([deltas, torch.full_like(deltas[:, :1], 1e-3)], dim=-1)
    alpha = 1.0 - torch.exp(-sigma * deltas)
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    T = torch.cat([torch.ones_like(T[:, :1]), T[:, :-1]], dim=-1)
    weights = T * alpha
    rgb = (weights.unsqueeze(-1) * colors).sum(dim=1)
    mask = weights.sum(dim=1)
    return rgb, mask


def _sample_volume(density_vol: torch.Tensor, color_vol: torch.Tensor,
                   points: torch.Tensor, voxel_range: float = 0.55
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample density and color from the volume at given 3D points.

    Args:
        density_vol: [1, D, D, D]
        color_vol: [3, D, D, D]
        points: [N, 3] world-space points

    Returns:
        sigma: [N] density (after softplus)
        colors: [N, 3] RGB
    """
    grid = (points / voxel_range).reshape(1, -1, 1, 1, 3)

    d = F.grid_sample(
        density_vol.unsqueeze(0), grid,
        mode='bilinear', padding_mode='zeros', align_corners=False
    ).reshape(-1)

    c = F.grid_sample(
        color_vol.unsqueeze(0), grid,
        mode='bilinear', padding_mode='zeros', align_corners=False
    ).reshape(3, -1).T

    sigma = F.softplus(d)
    return sigma, c


def render_rays_batch(density_vol: torch.Tensor, color_vol: torch.Tensor,
                      c2w_blender_list: torch.Tensor, K: torch.Tensor,
                      gt_rgbs: torch.Tensor, gt_masks: torch.Tensor,
                      render_h: int = 64, render_w: int = 64,
                      n_samples: int = 64, n_rays_per_view: int = 512,
                      voxel_range: float = 0.55,
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Efficient training renderer: sample random rays across all supervision views.

    Instead of rendering full images, sample n_rays_per_view random rays from each
    supervision view. Much faster for training.

    Args:
        density_vol: [1, D, D, D]
        color_vol: [3, D, D, D]
        c2w_blender_list: [V, 4, 4] supervision cameras
        K: [3, 3] intrinsics
        gt_rgbs: [V, 3, H, W] ground truth images
        gt_masks: [V, H, W] ground truth masks
        n_rays_per_view: rays to sample per view

    Returns:
        rgb_pred: [total_rays, 3] composited on gray (0.5) background
        rgb_gt: [total_rays, 3] GT (also composited on gray)
        mask_pred: [total_rays]
        mask_gt: [total_rays]
    """
    device = density_vol.device
    V = c2w_blender_list.shape[0]

    flip = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0],
                         [0, 0, -1, 0], [0, 0, 0, 1]],
                        dtype=torch.float32, device=device)

    all_rgb_pred = []
    all_rgb_gt = []
    all_mask_pred = []
    all_mask_gt = []

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    bg_color = 0.5  # gray background to match GT compositing

    for v_idx in range(V):
        c2w_cv = c2w_blender_list[v_idx] @ flip
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

        # Ray-AABB intersection
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
        sigma, colors = _sample_volume(
            density_vol, color_vol,
            pts.reshape(-1, 3), voxel_range
        )
        sigma = sigma.reshape(N_valid, n_samples)
        colors = colors.reshape(N_valid, n_samples, 3)

        rgb_raw, mask_pred = _composite_rays(sigma, colors, t_samples)
        # Composite onto gray background (same as GT compositing)
        rgb_pred = rgb_raw * mask_pred.unsqueeze(-1) + bg_color * (1.0 - mask_pred.unsqueeze(-1))

        # Normalize pixel coords to [-1, 1] for grid_sample on the GT images
        gt_h, gt_w = gt_rgbs.shape[2], gt_rgbs.shape[3]
        pu_norm = 2.0 * pixel_u[valid] / render_w - 1.0
        pv_norm = 2.0 * pixel_v[valid] / render_h - 1.0
        gt_grid = torch.stack([pu_norm, pv_norm], dim=-1).unsqueeze(0).unsqueeze(0)

        gt_rgb_sampled = F.grid_sample(
            gt_rgbs[v_idx:v_idx+1], gt_grid,
            mode='bilinear', padding_mode='border', align_corners=False
        ).reshape(3, -1).T

        gt_mask_sampled = F.grid_sample(
            gt_masks[v_idx:v_idx+1].unsqueeze(0), gt_grid,
            mode='bilinear', padding_mode='border', align_corners=False
        ).reshape(-1)

        all_rgb_pred.append(rgb_pred)
        all_rgb_gt.append(gt_rgb_sampled)
        all_mask_pred.append(mask_pred)
        all_mask_gt.append(gt_mask_sampled)

    if not all_rgb_pred:
        z3 = torch.zeros(1, 3, device=device)
        z1 = torch.zeros(1, device=device)
        return z3, z3, z1, z1

    return (torch.cat(all_rgb_pred),
            torch.cat(all_rgb_gt),
            torch.cat(all_mask_pred),
            torch.cat(all_mask_gt))


def render_volume(density: torch.Tensor, color: torch.Tensor,
                  c2w_blender: torch.Tensor, K: torch.Tensor,
                  render_h: int = 128, render_w: int = 128,
                  n_samples: int = 96, voxel_range: float = 0.55
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-image rendering for evaluation/visualization."""
    device = density.device
    flip = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0],
                         [0, 0, -1, 0], [0, 0, 0, 1]],
                        dtype=torch.float32, device=device)
    c2w_cv = c2w_blender @ flip
    rays_o, rays_d = _get_rays_opencv(c2w_cv, K, render_h, render_w)

    t_near, t_far = _ray_aabb_intersect(rays_o, rays_d, -voxel_range, voxel_range)
    valid = t_far > t_near
    n_rays = rays_o.shape[0]

    if not valid.any():
        return (torch.zeros(render_h, render_w, 3, device=device),
                torch.zeros(render_h, render_w, device=device))

    t_vals = torch.linspace(0, 1, n_samples, device=device)
    t_near_v = t_near[valid].unsqueeze(-1)
    t_far_v = t_far[valid].unsqueeze(-1)
    t_samples = t_near_v + (t_far_v - t_near_v) * t_vals
    pts = rays_o[valid].unsqueeze(1) + rays_d[valid].unsqueeze(1) * t_samples.unsqueeze(-1)

    N_valid = pts.shape[0]
    sigma, colors = _sample_volume(density, color, pts.reshape(-1, 3), voxel_range)
    sigma = sigma.reshape(N_valid, n_samples)
    colors = colors.reshape(N_valid, n_samples, 3)

    rgb_v, mask_v = _composite_rays(sigma, colors, t_samples)

    rgb_full = torch.zeros(n_rays, 3, device=device)
    mask_full = torch.zeros(n_rays, device=device)
    rgb_full[valid] = rgb_v
    mask_full[valid] = mask_v
    return rgb_full.reshape(render_h, render_w, 3), mask_full.reshape(render_h, render_w)
