"""Multi-view reconstruction model.

Architecture:
  1. MobileNetV3-Small shared encoder: 160x160 RGB → 5x5x576 feature maps
  2. Geometric unprojection: project 32^3 voxels into each view, sample features
  3. 3D CNN refinement: aggregate + refine multi-view features
  4. Occupancy + Color head: per-voxel density and RGB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from .camera_utils import (
    blender_c2w_to_opencv_w2c,
    adjust_intrinsics_for_crop_resize,
    blender_intrinsics,
    BLENDER_RENDER_W,
    BLENDER_RENDER_H,
)


class FeatureEncoder(nn.Module):
    """MobileNetV3-Small multi-scale feature extractor, shared across views.

    Extracts 10x10 (48ch) and 20x20 (24ch) features, upsamples the 10x10
    to 20x20 and concatenates, giving 20x20 spatial resolution with 72 channels.
    """

    def __init__(self, out_channels: int = 128):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        # layers 0-3: produce 20x20x24
        self.early = backbone.features[:4]
        # layers 4-8: produce 10x10x48
        self.mid = backbone.features[4:9]
        # Project concatenated features (24 + 48 = 72) to out_channels
        self.proj = nn.Conv2d(24 + 48, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_early = self.early(x)   # [B, 24, 20, 20]
        f_mid = self.mid(f_early)  # [B, 48, 10, 10]
        # Upsample mid features to match early spatial size
        f_mid_up = F.interpolate(f_mid, size=f_early.shape[2:],
                                 mode='bilinear', align_corners=False)
        f_cat = torch.cat([f_early, f_mid_up], dim=1)  # [B, 72, 20, 20]
        return self.proj(f_cat)  # [B, out_channels, 20, 20]


class GeometricUnprojector(nn.Module):
    """Projects voxel grid into image planes and samples features.

    Fully batched — no Python loops over views or batch elements.
    """

    def __init__(self, volume_size: int = 32, input_size: int = 160,
                 voxel_range: float = 0.55):
        super().__init__()
        self.volume_size = volume_size
        self.input_size = input_size
        self.voxel_range = voxel_range

        K_orig = blender_intrinsics()
        K_resized = adjust_intrinsics_for_crop_resize(
            K_orig, BLENDER_RENDER_W, BLENDER_RENDER_H, input_size
        )
        self.register_buffer('K', K_resized)

        coords = torch.linspace(-voxel_range, voxel_range, volume_size)
        zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing='ij')
        voxel_centers = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        self.register_buffer('voxel_centers', voxel_centers)

        # OpenGL-to-OpenCV flip matrix
        self.register_buffer('_flip', torch.tensor([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1],
        ], dtype=torch.float32))

    def forward(self, features: torch.Tensor,
                c2w_blender: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N, C, h, w] per-view feature maps
            c2w_blender: [B, N, 4, 4] Blender camera-to-world matrices

        Returns:
            volume: [B, C, D, D, D] aggregated 3D feature volume
        """
        B, N, C, h, w = features.shape
        V = self.volume_size
        P = V * V * V  # total voxels
        device = features.device

        # Compute all w2c matrices at once: [B, N, 4, 4]
        c2w_cv = c2w_blender @ self._flip
        w2c_cv = torch.linalg.inv(c2w_cv)

        # Project all voxels into all views: voxel_centers [P, 3] → homogeneous [P, 4]
        ones = torch.ones(P, 1, device=device)
        pts_h = torch.cat([self.voxel_centers, ones], dim=-1)  # [P, 4]

        # Transform: [B, N, 4, 4] @ [4, P] → [B, N, 4, P] → [B, N, P, 3]
        pts_cam = torch.einsum('bnij,pj->bnpi', w2c_cv[:, :, :3, :], pts_h)  # [B, N, P, 3]

        depth = pts_cam[..., 2]  # [B, N, P]
        safe_depth = depth.clamp(min=0.01)

        u = self.K[0, 0] * pts_cam[..., 0] / safe_depth + self.K[0, 2]
        v = self.K[1, 1] * pts_cam[..., 1] / safe_depth + self.K[1, 2]

        u_norm = 2.0 * u / self.input_size - 1.0
        v_norm = 2.0 * v / self.input_size - 1.0

        # Validity mask: in front of camera and within image bounds
        valid = (depth > 0.1) & \
                (u_norm > -1) & (u_norm < 1) & \
                (v_norm > -1) & (v_norm < 1)  # [B, N, P]

        # grid_sample: flatten B*N for a single batched call
        grid = torch.stack([u_norm, v_norm], dim=-1)  # [B, N, P, 2]
        grid = grid.reshape(B * N, P, 1, 2)
        feat_flat = features.reshape(B * N, C, h, w)

        sampled = F.grid_sample(
            feat_flat, grid, mode='bilinear',
            padding_mode='zeros', align_corners=False
        )  # [B*N, C, P, 1]
        sampled = sampled.squeeze(-1).reshape(B, N, C, P)  # [B, N, C, P]

        # Zero out invalid projections
        sampled = sampled * valid.unsqueeze(2).float()

        # Average across views
        count = valid.float().sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1, P]
        volume = sampled.sum(dim=1) / count.squeeze(1).unsqueeze(1)  # [B, C, P]

        return volume.reshape(B, C, V, V, V)


class GroupedResBlock3D(nn.Module):
    """Residual block with 2 grouped 3D convolutions (groups=8).

    Groups=8 gives 8x fewer FLOPs than standard conv while training
    efficiently on MPS (unlike depthwise which has slow backward on MPS).
    Converts to depthwise-separable for Quest deployment.
    """

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, groups=groups),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1, groups=groups),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


class Refiner3D(nn.Module):
    """3D CNN refiner using grouped convolutions (groups=8).

    3 residual blocks (6 grouped conv layers). ~120 GFLOPS total at 128ch,
    estimated ~2s on Quest CPU at INT8.
    """

    def __init__(self, in_channels: int = 128, mid_channels: int = 128,
                 groups: int = 8):
        super().__init__()
        self.proj_in = nn.Conv3d(in_channels, mid_channels, 1)
        self.blocks = nn.Sequential(
            GroupedResBlock3D(mid_channels, groups),
            GroupedResBlock3D(mid_channels, groups),
            GroupedResBlock3D(mid_channels, groups),
        )
        self.proj_out = nn.Conv3d(mid_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)
        h = self.blocks(h)
        return self.proj_out(h) + x


class OccupancyColorHead(nn.Module):
    """Predicts occupancy logits (1ch) + color logits (3ch) per voxel.

    Separate lightweight branches from the shared feature volume.
    Density bias initialized to -5.0 (mostly empty at start).
    """

    def __init__(self, in_channels: int = 64):
        super().__init__()
        density_out = nn.Conv3d(32, 1, 1)
        nn.init.constant_(density_out.bias, -5.0)
        self.density_branch = nn.Sequential(
            nn.Conv3d(in_channels, 32, 1),
            nn.GELU(),
            density_out,
        )
        self.color_branch = nn.Sequential(
            nn.Conv3d(in_channels, 32, 1),
            nn.GELU(),
            nn.Conv3d(32, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.density_branch(x), self.color_branch(x)


class MVReconModel(nn.Module):
    """Full multi-view reconstruction model.

    Returns density [B, 1, D, D, D] and color [B, 3, D, D, D] logits.
    """

    def __init__(self, volume_size: int = 32, feat_channels: int = 128,
                 input_size: int = 160):
        super().__init__()
        self.volume_size = volume_size
        self.encoder = FeatureEncoder(out_channels=feat_channels)
        self.unprojector = GeometricUnprojector(
            volume_size=volume_size, input_size=input_size
        )
        self.refiner = Refiner3D(in_channels=feat_channels)
        self.head = OccupancyColorHead(in_channels=feat_channels)

    def forward(self, images: torch.Tensor,
                c2w_matrices: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: [B, N, 3, H, W] input views (ImageNet-normalized)
            c2w_matrices: [B, N, 4, 4] Blender camera-to-world matrices

        Returns:
            density: [B, 1, D, D, D] occupancy logits
            color: [B, 3, D, D, D] color logits (apply sigmoid for RGB)
        """
        B, N, C, H, W = images.shape

        flat_imgs = images.reshape(B * N, C, H, W)
        feats = self.encoder(flat_imgs)
        feat_ch = feats.shape[1]
        h, w = feats.shape[2], feats.shape[3]
        feats = feats.reshape(B, N, feat_ch, h, w)

        volume = self.unprojector(feats, c2w_matrices)
        volume = self.refiner(volume)
        return self.head(volume)

    def param_count(self) -> dict:
        counts = {}
        for name, module in [
            ('encoder', self.encoder),
            ('refiner', self.refiner),
            ('head', self.head),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts
