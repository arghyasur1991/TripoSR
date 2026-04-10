"""Dataset for multi-view reconstruction training.

Loads Objaverse objects rendered from 24 viewpoints with known camera matrices.
Splits views into input (model sees these) and supervision (loss computed on these).
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ImageNet normalization (MobileNetV3 expects this)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ObjaverseMultiViewDataset(Dataset):
    """Loads multi-view renders of Objaverse objects.

    Each item returns:
        - input_images: [N_input, 3, H, W] (ImageNet-normalized)
        - input_c2w: [N_input, 4, 4] camera-to-world matrices
        - sup_images: [N_sup, 3, H, W] (raw [0,1] RGB for loss)
        - sup_masks: [N_sup, H, W] alpha masks
        - sup_c2w: [N_sup, 4, 4] supervision camera matrices
        - gt_occupancy: [D, D, D] binary occupancy grid (if voxels_dir provided)
        - uid: object UID string
    """

    def __init__(self, renders_dir: str, uids: list[str],
                 n_input_views: int = 4, n_sup_views: int = 8,
                 image_size: int = 160, sup_image_size: int = 128,
                 augment: bool = False, voxels_dir: str | None = None):
        self.renders_dir = Path(renders_dir)
        self.uids = uids
        self.n_input_views = n_input_views
        self.n_sup_views = n_sup_views
        self.image_size = image_size
        self.sup_image_size = sup_image_size
        self.augment = augment
        self.voxels_dir = Path(voxels_dir) if voxels_dir else None

        self.imagenet_normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02,
        ) if augment else None

        # Pre-load all camera data
        self.objects = []
        for uid in uids:
            obj_dir = self.renders_dir / uid
            cam_file = obj_dir / 'cameras.json'
            if not cam_file.exists():
                continue
            with open(cam_file) as f:
                cams = json.load(f)
            self.objects.append({
                'uid': uid,
                'dir': obj_dir,
                'cameras': cams,
            })

    def __len__(self) -> int:
        return len(self.objects)

    def _load_image(self, obj_dir: Path, cam_entry: dict) -> Image.Image:
        """Load image, handling the _alpha_ prefix mismatch."""
        fname = cam_entry['filename']
        path = obj_dir / fname
        if not path.exists():
            # Strip _alpha_ prefix
            bare = fname.replace('_alpha_', '')
            path = obj_dir / bare
        return Image.open(path)

    def _center_crop_square(self, img: Image.Image) -> Image.Image:
        """Center-crop to a square."""
        w, h = img.size
        sq = min(w, h)
        left = (w - sq) // 2
        top = (h - sq) // 2
        return img.crop((left, top, left + sq, top + sq))

    def _composite_on_bg(self, rgb: Image.Image, mask: Image.Image,
                         bg_color: tuple[int, int, int]) -> Image.Image:
        bg = Image.new('RGB', rgb.size, bg_color)
        return Image.composite(rgb, bg, mask)

    @staticmethod
    def _flip_c2w(c2w: torch.Tensor) -> torch.Tensor:
        """Flip camera-to-world matrix for horizontal image flip.

        Negate the camera's local X-axis (column 0 of rotation).
        """
        c2w_flip = c2w.clone()
        c2w_flip[:3, 0] = -c2w_flip[:3, 0]
        return c2w_flip

    def __getitem__(self, idx: int) -> dict:
        obj = self.objects[idx]
        cams = obj['cameras']
        n_total = len(cams)

        # Randomly select input and supervision views (reshuffled every call)
        indices = list(range(n_total))
        random.shuffle(indices)
        input_indices = indices[:self.n_input_views]
        remaining = indices[self.n_input_views:]
        sup_indices = remaining[:self.n_sup_views] if len(remaining) >= self.n_sup_views else remaining

        # Per-sample augmentation decisions (consistent across all views)
        do_hflip = self.augment and random.random() < 0.5
        if self.augment:
            bg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            bg_color = (127, 127, 127)

        # --- Input views ---
        input_images = []
        input_c2w = []
        for i in input_indices:
            img = self._load_image(obj['dir'], cams[i])
            img_rgba = img.convert('RGBA')
            img_sq = self._center_crop_square(img_rgba)
            r, g, b, a = img_sq.split()
            rgb = Image.merge('RGB', (r, g, b))
            rgb = self._composite_on_bg(rgb, a, bg_color)

            if do_hflip:
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            if self.color_jitter is not None:
                rgb = self.color_jitter(rgb)

            t = transforms.functional.to_tensor(
                transforms.functional.resize(rgb, (self.image_size, self.image_size))
            )
            if self.augment and random.random() < 0.3:
                sigma = random.uniform(0.04, 0.08)
                t = t + torch.randn_like(t) * sigma
                t = t.clamp(0, 1)
            t = self.imagenet_normalize(t)
            input_images.append(t)

            c2w = torch.tensor(cams[i]['pose'], dtype=torch.float32)
            if do_hflip:
                c2w = self._flip_c2w(c2w)
            input_c2w.append(c2w)

        # --- Supervision views ---
        sup_images = []
        sup_masks = []
        sup_c2w = []
        for i in sup_indices:
            img = self._load_image(obj['dir'], cams[i])
            img_rgba = img.convert('RGBA')
            img_sq = self._center_crop_square(img_rgba)
            r, g, b, a = img_sq.split()
            rgb = Image.merge('RGB', (r, g, b))
            mask = a
            rgb = self._composite_on_bg(rgb, mask, bg_color)

            if do_hflip:
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            sup_tensor = transforms.functional.to_tensor(
                transforms.functional.resize(rgb, (self.sup_image_size, self.sup_image_size))
            )
            mask_tensor = transforms.functional.to_tensor(
                transforms.functional.resize(mask, (self.sup_image_size, self.sup_image_size))
            )

            sup_images.append(sup_tensor)
            sup_masks.append(mask_tensor.squeeze(0))

            c2w = torch.tensor(cams[i]['pose'], dtype=torch.float32)
            if do_hflip:
                c2w = self._flip_c2w(c2w)
            sup_c2w.append(c2w)

        result = {
            'input_images': torch.stack(input_images),         # [N_in, 3, H, W]
            'input_c2w': torch.stack(input_c2w),               # [N_in, 4, 4]
            'sup_images': torch.stack(sup_images),             # [N_sup, 3, H, W]
            'sup_masks': torch.stack(sup_masks),               # [N_sup, H, W]
            'sup_c2w': torch.stack(sup_c2w),                   # [N_sup, 4, 4]
            'uid': obj['uid'],
        }

        if self.voxels_dir is not None:
            voxel_path = self.voxels_dir / f"{obj['uid']}.npy"
            if voxel_path.exists():
                gt_occ = np.load(voxel_path)
                gt_tensor = torch.from_numpy(gt_occ)
                if do_hflip:
                    gt_tensor = gt_tensor.flip(-1)  # flip X axis (last dim in ZYX layout)
                result['gt_occupancy'] = gt_tensor

        return result
