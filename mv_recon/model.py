"""Multi-view reconstruction model (v2).

Architecture:
  1. MobileNetV3-Small three-scale encoder: 160x160 RGB → 20x20x128 features
  2. Geometric unprojection with mean+variance fusion: 32^3 × 256ch volume
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
    """MobileNetV3-Small three-scale feature extractor, shared across views.

    Extracts features at three scales (20x20, 10x10, 5x5), upsamples all to
    20x20, concatenates (24+48+96=168ch), and projects to out_channels.
    """

    def __init__(self, out_channels: int = 128):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.early = backbone.features[:4]    # 20x20 × 24ch
        self.mid = backbone.features[4:9]     # 10x10 × 48ch
        self.late = backbone.features[9:12]   # 5x5 × 96ch
        self.proj = nn.Conv2d(24 + 48 + 96, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_early = self.early(x)     # [B, 24, 20, 20]
        f_mid = self.mid(f_early)   # [B, 48, 10, 10]
        f_late = self.late(f_mid)   # [B, 96, 5, 5]
        target = f_early.shape[2:]
        f_mid_up = F.interpolate(f_mid, size=target,
                                 mode='bilinear', align_corners=False)
        f_late_up = F.interpolate(f_late, size=target,
                                  mode='bilinear', align_corners=False)
        f_cat = torch.cat([f_early, f_mid_up, f_late_up], dim=1)
        return self.proj(f_cat)     # [B, out_channels, 20, 20]


class GeometricUnprojector(nn.Module):
    """Projects voxel grid into image planes and samples features.

    Returns mean+variance fusion: for each voxel, concatenates the mean and
    variance of features across views, giving 2*C output channels. Variance
    encodes view consistency (low = likely surface, high = occluded/ambiguous).
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
            volume: [B, 2*C, D, D, D] mean+variance fused volume
        """
        B, N, C, h, w = features.shape
        V = self.volume_size
        P = V * V * V
        device = features.device

        c2w_cv = c2w_blender @ self._flip
        w2c_cv = torch.linalg.inv(c2w_cv)

        ones = torch.ones(P, 1, device=device)
        pts_h = torch.cat([self.voxel_centers, ones], dim=-1)

        pts_cam = torch.einsum('bnij,pj->bnpi', w2c_cv[:, :, :3, :], pts_h)

        depth = pts_cam[..., 2]
        safe_depth = depth.clamp(min=0.01)

        u = self.K[0, 0] * pts_cam[..., 0] / safe_depth + self.K[0, 2]
        v = self.K[1, 1] * pts_cam[..., 1] / safe_depth + self.K[1, 2]

        u_norm = 2.0 * u / self.input_size - 1.0
        v_norm = 2.0 * v / self.input_size - 1.0

        valid = (depth > 0.1) & \
                (u_norm > -1) & (u_norm < 1) & \
                (v_norm > -1) & (v_norm < 1)

        grid = torch.stack([u_norm, v_norm], dim=-1).reshape(B * N, P, 1, 2)
        feat_flat = features.reshape(B * N, C, h, w)

        sampled = F.grid_sample(
            feat_flat, grid, mode='bilinear',
            padding_mode='zeros', align_corners=False
        ).squeeze(-1).reshape(B, N, C, P)

        sampled = sampled * valid.unsqueeze(2).float()

        count = valid.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        count_bc = count.squeeze(1).unsqueeze(1)  # [B, 1, P]

        mean = sampled.sum(dim=1) / count_bc
        sq_mean = (sampled ** 2).sum(dim=1) / count_bc
        var = (sq_mean - mean ** 2).clamp(min=0)

        volume = torch.cat([mean, var], dim=1)  # [B, 2C, P]
        return volume.reshape(B, 2 * C, V, V, V)


class GroupedResBlock3D(nn.Module):
    """Residual block with 2 grouped 3D convolutions.

    Groups=4 balances cross-channel mixing with efficiency.
    """

    def __init__(self, channels: int, groups: int = 4):
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
    """Progressive 3D refinement: 16^3 -> 32^3 -> 64^3.

    Input: 32^3 feature volume (256ch from mean+variance fusion).
    Stage 1: Pool to 16^3, 256ch, 2 res blocks (coarse global structure)
    Stage 2: Upsample to 32^3, concat skip, project to 256ch, 3 res blocks
    Stage 3: Upsample to 64^3, project to 64ch, 2 res blocks (fine detail)
    """

    def __init__(self, in_channels: int = 256, mid_channels: int = 256,
                 out_channels: int = 64, groups: int = 4):
        super().__init__()
        self.down = nn.AvgPool3d(2)
        self.stage1a = GroupedResBlock3D(in_channels, groups)
        self.stage1b = GroupedResBlock3D(in_channels, groups)

        self.up1_proj = nn.Conv3d(in_channels * 2, mid_channels, 1)
        self.stage2a = GroupedResBlock3D(mid_channels, groups)
        self.stage2b = GroupedResBlock3D(mid_channels, groups)
        self.stage2c = GroupedResBlock3D(mid_channels, groups)

        self.up2_proj = nn.Conv3d(mid_channels, out_channels, 1)
        self.stage3a = GroupedResBlock3D(out_channels, groups)
        self.stage3b = GroupedResBlock3D(out_channels, groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 256, 32, 32, 32] -> [B, 64, 64, 64, 64]."""
        skip_32 = x

        h = self.down(x)                                       # [B, 256, 16, 16, 16]
        h = self.stage1a(h)                                    # [B, 256, 16, 16, 16]
        h = self.stage1b(h)                                    # [B, 256, 16, 16, 16]

        h = F.interpolate(h, scale_factor=2, mode='trilinear',
                          align_corners=False)                 # [B, 256, 32, 32, 32]
        h = torch.cat([h, skip_32], dim=1)                    # [B, 512, 32, 32, 32]
        h = self.up1_proj(h)                                   # [B, 256, 32, 32, 32]
        h = self.stage2a(h)                                    # [B, 256, 32, 32, 32]
        h = self.stage2b(h)                                    # [B, 256, 32, 32, 32]
        h = self.stage2c(h)                                    # [B, 256, 32, 32, 32]

        h = F.interpolate(h, scale_factor=2, mode='trilinear',
                          align_corners=False)                 # [B, 256, 64, 64, 64]
        h = self.up2_proj(h)                                   # [B, 64, 64, 64, 64]
        h = self.stage3a(h)                                    # [B, 64, 64, 64, 64]
        h = self.stage3b(h)                                    # [B, 64, 64, 64, 64]

        return h


class OccupancyColorHead(nn.Module):
    """Predicts occupancy logits (1ch) + color logits (3ch) per voxel.

    Density bias initialized to -5.0 (mostly empty at start).
    """

    def __init__(self, in_channels: int = 64):
        super().__init__()
        mid = max(in_channels, 32)
        density_out = nn.Conv3d(mid, 1, 1)
        nn.init.constant_(density_out.bias, -5.0)
        self.density_branch = nn.Sequential(
            nn.Conv3d(in_channels, mid, 1),
            nn.GELU(),
            density_out,
        )
        self.color_branch = nn.Sequential(
            nn.Conv3d(in_channels, mid, 1),
            nn.GELU(),
            nn.Conv3d(mid, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.density_branch(x), self.color_branch(x)


class MVReconModel(nn.Module):
    """Full multi-view reconstruction model (v2).

    Three-scale encoder, mean+variance unprojection into 32^3,
    coarse-to-fine refinement to 64^3.
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
        self.refiner = CoarseToFineRefiner(in_channels=2 * feat_channels)
        self.head = OccupancyColorHead(in_channels=self.refiner.up2_proj.out_channels)

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

        volume = self.unprojector(feats, c2w_matrices)  # [B, 256, 32, 32, 32]
        volume = self.refiner(volume)                    # [B, 32, 64, 64, 64]
        return self.head(volume)

    def param_count(self) -> dict:
        counts = {}
        for name, module in [
            ('encoder', self.encoder),
            ('unprojector', self.unprojector),
            ('refiner', self.refiner),
            ('head', self.head),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts
