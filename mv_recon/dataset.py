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

        self.input_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        self.sup_transform = transforms.Compose([
            transforms.Resize((sup_image_size, sup_image_size)),
            transforms.ToTensor(),
        ])

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

    def __getitem__(self, idx: int) -> dict:
        obj = self.objects[idx]
        cams = obj['cameras']
        n_total = len(cams)

        # Select diverse input views: pick from different azimuth sectors
        indices = list(range(n_total))
        random.shuffle(indices)
        input_indices = indices[:self.n_input_views]
        remaining = indices[self.n_input_views:]

        if len(remaining) >= self.n_sup_views:
            sup_indices = remaining[:self.n_sup_views]
        else:
            sup_indices = remaining

        input_images = []
        input_c2w = []
        for i in input_indices:
            img = self._load_image(obj['dir'], cams[i])
            img_rgb = img.convert('RGB')
            img_sq = self._center_crop_square(img_rgb)
            input_images.append(self.input_transform(img_sq))
            input_c2w.append(torch.tensor(cams[i]['pose'], dtype=torch.float32))

        sup_images = []
        sup_masks = []
        sup_c2w = []
        for i in sup_indices:
            img = self._load_image(obj['dir'], cams[i])
            img_rgba = img.convert('RGBA')
            img_sq = self._center_crop_square(img_rgba)

            # Split into RGB and alpha
            r, g, b, a = img_sq.split()
            rgb = Image.merge('RGB', (r, g, b))
            mask = a

            # Composite onto gray background (like TripoSR training)
            bg = Image.new('RGB', rgb.size, (127, 127, 127))
            rgb_composited = Image.composite(rgb, bg, mask)

            sup_tensor = self.sup_transform(rgb_composited)
            mask_tensor = transforms.Compose([
                transforms.Resize((self.sup_image_size, self.sup_image_size)),
                transforms.ToTensor(),
            ])(mask)

            sup_images.append(sup_tensor)
            sup_masks.append(mask_tensor.squeeze(0))
            sup_c2w.append(torch.tensor(cams[i]['pose'], dtype=torch.float32))

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
                result['gt_occupancy'] = torch.from_numpy(gt_occ)

        return result
