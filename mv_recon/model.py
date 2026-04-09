"""Multi-view reconstruction model.

Architecture:
  1. MobileNetV3-Small shared encoder: 160x160 RGB → 20x20x128 feature maps
  2. Geometric unprojection: project 32^3 voxels into each view, sample features
  3. Coarse-to-fine 3D CNN: 16^3 → 32^3 → 64^3 progressive refinement
  4. Occupancy + Color head: per-voxel density and RGB at 64^3
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


class CoarseToFineRefiner(nn.Module):
    """Progressive 3D refinement: 16^3 → 32^3 → 64^3.

    Input: 32^3 feature volume from the unprojector.
    Stage 1: Downsample to 16^3, 128ch, 1 res block (coarse structure)
    Stage 2: Upsample to 32^3, concat skip, project to 64ch, 1 res block
    Stage 3: Upsample to 64^3, project to 32ch, 1 res block (fine detail)

    Total ~12G FLOPs vs ~24G for the old flat refiner — cheaper AND higher res.
    """

    def __init__(self, in_channels: int = 128, groups: int = 8):
        super().__init__()
        # Stage 1: 16^3, 128ch
        self.down = nn.Conv3d(in_channels, in_channels, 2, stride=2)
        self.stage1 = GroupedResBlock3D(in_channels, groups)

        # Stage 2: upsample to 32^3, concat skip (128+128=256) → 64ch
        self.up1_proj = nn.Conv3d(in_channels + in_channels, 64, 1)
        self.stage2 = GroupedResBlock3D(64, groups=8)

        # Stage 3: upsample to 64^3, 64 → 32ch
        self.up2_proj = nn.Conv3d(64, 32, 1)
        self.stage3 = GroupedResBlock3D(32, groups=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 128, 32, 32, 32] → returns [B, 32, 64, 64, 64]."""
        skip_32 = x

        # Stage 1: 32^3 → 16^3
        h = self.down(x)                                       # [B, 128, 16, 16, 16]
        h = self.stage1(h)                                     # [B, 128, 16, 16, 16]

        # Stage 2: 16^3 → 32^3 + skip
        h = F.interpolate(h, scale_factor=2, mode='trilinear',
                          align_corners=False)                 # [B, 128, 32, 32, 32]
        h = torch.cat([h, skip_32], dim=1)                    # [B, 256, 32, 32, 32]
        h = self.up1_proj(h)                                   # [B, 64, 32, 32, 32]
        h = self.stage2(h)                                     # [B, 64, 32, 32, 32]

        # Stage 3: 32^3 → 64^3
        h = F.interpolate(h, scale_factor=2, mode='trilinear',
                          align_corners=False)                 # [B, 64, 64, 64, 64]
        h = self.up2_proj(h)                                   # [B, 32, 64, 64, 64]
        h = self.stage3(h)                                     # [B, 32, 64, 64, 64]

        return h


class OccupancyColorHead(nn.Module):
    """Predicts occupancy logits (1ch) + color logits (3ch) per voxel.

    Operates on the 32ch output of the coarse-to-fine refiner at 64^3.
    Density bias initialized to -5.0 (mostly empty at start).
    """

    def __init__(self, in_channels: int = 32):
        super().__init__()
        density_out = nn.Conv3d(16, 1, 1)
        nn.init.constant_(density_out.bias, -5.0)
        self.density_branch = nn.Sequential(
            nn.Conv3d(in_channels, 16, 1),
            nn.GELU(),
            density_out,
        )
        self.color_branch = nn.Sequential(
            nn.Conv3d(in_channels, 16, 1),
            nn.GELU(),
            nn.Conv3d(16, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.density_branch(x), self.color_branch(x)


class MVReconModel(nn.Module):
    """Full multi-view reconstruction model.

    Unprojects into 32^3, then coarse-to-fine refines to 64^3.
    Returns density [B, 1, 64, 64, 64] and color [B, 3, 64, 64, 64] logits.
    """

    def __init__(self, volume_size: int = 32, feat_channels: int = 128,
                 input_size: int = 160):
        super().__init__()
        self.volume_size = volume_size
        self.encoder = FeatureEncoder(out_channels=feat_channels)
        self.unprojector = GeometricUnprojector(
            volume_size=volume_size, input_size=input_size
        )
        self.refiner = CoarseToFineRefiner(in_channels=feat_channels)
        self.head = OccupancyColorHead(in_channels=32)

    def forward(self, images: torch.Tensor,
                c2w_matrices: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: [B, N, 3, H, W] input views (ImageNet-normalized)
            c2w_matrices: [B, N, 4, 4] Blender camera-to-world matrices

        Returns:
            density: [B, 1, 64, 64, 64] occupancy logits
            color: [B, 3, 64, 64, 64] color logits (apply sigmoid for RGB)
        """
        B, N, C, H, W = images.shape

        flat_imgs = images.reshape(B * N, C, H, W)
        feats = self.encoder(flat_imgs)
        feat_ch = feats.shape[1]
        h, w = feats.shape[2], feats.shape[3]
        feats = feats.reshape(B, N, feat_ch, h, w)

        volume = self.unprojector(feats, c2w_matrices)  # [B, 128, 32, 32, 32]
        volume = self.refiner(volume)                    # [B, 32, 64, 64, 64]
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
