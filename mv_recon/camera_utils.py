"""Camera utilities for multi-view reconstruction.

Handles Blender camera conventions, projection, ray generation, and
coordinate system conversions.

Blender cameras: right-handed, look along -Z local axis, Y up.
We convert to OpenCV convention: look along +Z, Y down, for standard projection.
"""

import math
import torch
import torch.nn.functional as F


# Blender render settings from render_views.py
BLENDER_RENDER_W = 1280
BLENDER_RENDER_H = 960
BLENDER_FOV_DEG = 50.0


def blender_intrinsics(render_w: int = BLENDER_RENDER_W,
                       render_h: int = BLENDER_RENDER_H,
                       fov_deg: float = BLENDER_FOV_DEG) -> torch.Tensor:
    """Compute pinhole intrinsics from Blender's horizontal FOV.

    Returns [3, 3] intrinsic matrix in OpenCV convention.
    """
    fov_rad = math.radians(fov_deg)
    fx = (render_w / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx  # square pixels
    cx = render_w / 2.0
    cy = render_h / 2.0
    return torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], dtype=torch.float32)


def adjust_intrinsics_for_crop_resize(K: torch.Tensor,
                                      orig_w: int, orig_h: int,
                                      target_size: int) -> torch.Tensor:
    """Adjust intrinsics after center-crop to square then resize.

    Steps: center-crop orig_w x orig_h to min(w,h) x min(w,h), then resize
    to target_size x target_size.
    """
    sq = min(orig_w, orig_h)
    # Center-crop offsets
    dx = (orig_w - sq) / 2.0
    dy = (orig_h - sq) / 2.0
    scale = target_size / sq

    K_new = K.clone()
    K_new[0, 2] = (K[0, 2] - dx) * scale
    K_new[1, 2] = (K[1, 2] - dy) * scale
    K_new[0, 0] = K[0, 0] * scale
    K_new[1, 1] = K[1, 1] * scale
    return K_new


def blender_c2w_to_opencv_w2c(c2w_blender: torch.Tensor) -> torch.Tensor:
    """Convert Blender camera-to-world to OpenCV world-to-camera.

    Blender: camera looks along -Z, Y up (right-handed).
    OpenCV: camera looks along +Z, Y down.
    Conversion: flip Y and Z axes, then invert.
    """
    # Flip Y and Z to go from Blender to OpenCV camera space
    flip = torch.tensor([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1],
    ], dtype=c2w_blender.dtype, device=c2w_blender.device)

    c2w_cv = c2w_blender @ flip  # camera-to-world in OpenCV convention
    w2c_cv = torch.inverse(c2w_cv)
    return w2c_cv


def project_points(points: torch.Tensor, w2c: torch.Tensor,
                   K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project 3D world points to 2D pixel coordinates.

    Args:
        points: [..., 3] world-space points
        w2c: [4, 4] world-to-camera (OpenCV convention)
        K: [3, 3] intrinsics

    Returns:
        uv: [..., 2] pixel coordinates
        depth: [...] depth values (positive = in front of camera)
    """
    shape = points.shape[:-1]
    pts = points.reshape(-1, 3)

    # World to camera
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    pts_cam = (R @ pts.T).T + t  # [N, 3]

    depth = pts_cam[:, 2]

    # Project
    uv = torch.zeros(pts.shape[0], 2, device=points.device, dtype=points.dtype)
    valid = depth > 1e-6
    if valid.any():
        uv[valid, 0] = K[0, 0] * pts_cam[valid, 0] / pts_cam[valid, 2] + K[0, 2]
        uv[valid, 1] = K[1, 1] * pts_cam[valid, 1] / pts_cam[valid, 2] + K[1, 2]

    return uv.reshape(*shape, 2), depth.reshape(shape)


def get_rays(c2w_opencv: torch.Tensor, K: torch.Tensor,
             H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate camera rays for each pixel.

    Args:
        c2w_opencv: [4, 4] camera-to-world in OpenCV convention
        K: [3, 3] intrinsics
        H, W: image dimensions

    Returns:
        rays_o: [H, W, 3] ray origins (camera position)
        rays_d: [H, W, 3] ray directions (unit vectors, world space)
    """
    device = c2w_opencv.device
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    v, u = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )

    # Ray direction in camera space (OpenCV: +Z forward, Y down)
    dirs_cam = torch.stack([
        (u - cx) / fx,
        (v - cy) / fy,
        torch.ones_like(u),
    ], dim=-1)  # [H, W, 3]

    # Normalize
    dirs_cam = F.normalize(dirs_cam, dim=-1)

    # Transform to world space
    R = c2w_opencv[:3, :3]  # [3, 3]
    dirs_world = (dirs_cam @ R.T)  # [H, W, 3]

    rays_o = c2w_opencv[:3, 3].expand(H, W, 3)  # [H, W, 3]
    return rays_o, dirs_world
