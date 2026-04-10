"""Training script for multi-view reconstruction model.

Hybrid loss: photometric (projected colors) + mask + 3D BCE.

Usage:
    # Overfit on 11 objects (PoC):
    python -m mv_recon.train --mode overfit --epochs 500

    # Train on all 18 objects:
    python -m mv_recon.train --mode full --epochs 200

    # Train on full dataset (~5000 objects):
    python -m mv_recon.train --mode full --data_dir ~/Downloads/mv_recon_data \
        --uids_file ~/Downloads/mv_recon_data/filtered_uids.json --epochs 50
"""

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import MVReconModel
from .dataset import ObjaverseMultiViewDataset
from .renderer import render_rays_batch
from .camera_utils import (
    blender_intrinsics,
    adjust_intrinsics_for_crop_resize,
    BLENDER_RENDER_W,
    BLENDER_RENDER_H,
)


DEFAULT_DATA_DIR = Path.home() / "Downloads" / "mv_recon_data"

# Legacy aliases for extract_mesh.py and other scripts
RENDERS_DIR = str(DEFAULT_DATA_DIR / "renders")
VOXELS_DIR = str(DEFAULT_DATA_DIR / "voxels")

ALL_18_UIDS = [
    "0bdb81c409e44805b97ad0154c562eeb",
    "0bdc27ca9cfe466ea0ea7b8ac142e38c",
    "0c4a87e96968486dbd601807f1394c6d",
    "0f20e82ba6d64aaca7f2e5364e18c6b2",
    "0f203d3605694e1697eb2fc2611581b8",
    "0fdb921d401149f5b994c3c906c840ff",
    "1a320de881c247398a4b8e9661991215",
    "1ab7279669fa44be8e74e6c75a64819b",
    "1b78be3a0a0e46ebbd7c0c12ee676ec0",
    "1bb0c9be995747e9abe0831d94066fc8",
    "1bf72a656b714fb4bbdceae059556d54",
    "1d62b5272e0a4226911c39f7c4b97a24",
    "ee802eec0b5742f2b036ea986b89ad88",
    "eee65627183040d19077c8362ded93f7",
    "f1f2c4d9262343b89377b16f6265b0f8",
    "f5c7d1000a0344e3afffb51e5615c6ef",
    "f51306c13cd14d38bdcd7a31d8804bfb",
    "fa0021e12f9f4ec1971345c7c9434685",
]

OVERFIT_10_UIDS = ALL_18_UIDS[:10]

OVERFIT_11_UIDS = [
    "0bdb81c409e44805b97ad0154c562eeb",
    "0bdc27ca9cfe466ea0ea7b8ac142e38c",
    "0c4a87e96968486dbd601807f1394c6d",
    "0f20e82ba6d64aaca7f2e5364e18c6b2",
    "0f203d3605694e1697eb2fc2611581b8",
    "0fdb921d401149f5b994c3c906c840ff",
    "1a320de881c247398a4b8e9661991215",
    "1ab7279669fa44be8e74e6c75a64819b",
    "1b78be3a0a0e46ebbd7c0c12ee676ec0",
    "1bb0c9be995747e9abe0831d94066fc8",
    "1bf72a656b714fb4bbdceae059556d54",
]


def discover_uids(data_dir: Path, uids_file: str | None = None) -> list[str]:
    """Load UIDs from file, or discover from renders directory."""
    if uids_file:
        with open(uids_file) as f:
            all_uids = json.load(f)
    else:
        renders_dir = data_dir / "renders"
        all_uids = sorted(
            d.name for d in renders_dir.iterdir()
            if d.is_dir() and (d / "cameras.json").exists()
        )

    # Filter to objects that have both renders and voxels
    renders_dir = data_dir / "renders"
    voxels_dir = data_dir / "voxels"
    valid = [
        uid for uid in all_uids
        if (renders_dir / uid / "cameras.json").exists()
        and (voxels_dir / f"{uid}.npy").exists()
    ]
    return valid


def split_train_val(uids: list[str], val_ratio: float = 0.1,
                    seed: int = 42) -> tuple[list[str], list[str]]:
    """Deterministic train/val split."""
    rng = random.Random(seed)
    shuffled = uids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def compute_loss(model: MVReconModel, batch: dict,
                 device: torch.device,
                 K_sup: torch.Tensor,
                 n_samples: int = 64, n_rays_per_view: int = 512,
                 sup_image_size: int = 128,
                 w_photo: float = 1.0, w_mask: float = 0.1,
                 w_bce: float = 0.5, w_sparse: float = 0.02,
                 ) -> tuple[torch.Tensor, dict]:
    """Hybrid loss: photometric + mask + 3D BCE + sparsity.

    Photometric and mask losses use the learned color volume for gradients.
    3D BCE loss provides direct geometry supervision from GT voxels.
    Sparsity loss penalizes over-prediction of occupied voxels.
    """
    input_imgs = batch['input_images'].to(device)    # [B, V_in, 3, H, W]
    input_c2w = batch['input_c2w'].to(device)        # [B, V_in, 4, 4]
    sup_imgs = batch['sup_images'].to(device)        # [B, V_sup, 3, H, W]
    sup_masks = batch['sup_masks'].to(device)        # [B, V_sup, H, W]
    sup_c2w = batch['sup_c2w'].to(device)            # [B, V_sup, 4, 4]

    density, color = model(input_imgs, input_c2w)    # [B,1,64,64,64], [B,3,64,64,64]

    # Rendering-based losses (per batch element, B=1 expected)
    total_photo = torch.tensor(0.0, device=device)
    total_mask = torch.tensor(0.0, device=device)
    B = density.shape[0]
    for b in range(B):
        rgb_pred, rgb_gt, mask_pred, mask_gt = render_rays_batch(
            density_vol=density[b, 0],   # [64, 64, 64]
            color_vol=color[b],          # [3, 64, 64, 64]
            sup_c2w=sup_c2w[b],
            K_sup=K_sup,
            gt_rgbs=sup_imgs[b],
            gt_masks=sup_masks[b],
            render_h=sup_image_size, render_w=sup_image_size,
            n_samples=n_samples, n_rays_per_view=n_rays_per_view,
        )
        total_photo += F.mse_loss(rgb_pred, rgb_gt)
        total_mask += F.binary_cross_entropy(
            mask_pred.clamp(1e-5, 1 - 1e-5), mask_gt,
        )
    photo_loss = total_photo / B
    mask_loss = total_mask / B

    # Direct 3D BCE loss — model outputs 64^3, GT is 64^3
    bce_loss = torch.tensor(0.0, device=device)
    iou_val = 0.0
    pred_logits = density[:, 0]                      # [B, 64, 64, 64]
    if 'gt_occupancy' in batch:
        gt_occ = batch['gt_occupancy'].to(device)    # [B, 64, 64, 64]

        n_pos = gt_occ.sum().clamp(min=1.0)
        n_neg = (1 - gt_occ).sum().clamp(min=1.0)
        pos_weight = (n_neg / n_pos).clamp(max=20.0)

        bce_loss = F.binary_cross_entropy_with_logits(
            pred_logits, gt_occ, pos_weight=pos_weight,
        )
        with torch.no_grad():
            iou_val = _iou(torch.sigmoid(pred_logits) > 0.5,
                           gt_occ > 0.5).item()

    # Sparsity: penalize mean occupancy to fight false-positive voxels
    sparsity_loss = torch.sigmoid(pred_logits).mean()

    loss = (w_photo * photo_loss + w_mask * mask_loss
            + w_bce * bce_loss + w_sparse * sparsity_loss)

    return loss, {
        'loss': loss.item(),
        'photo': photo_loss.item(),
        'mask': mask_loss.item(),
        'bce': bce_loss.item(),
        'sparse': sparsity_loss.item(),
        'iou': iou_val,
    }


def _iou(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Intersection-over-union for binary occupancy volumes."""
    intersection = (pred & gt).float().sum()
    union = (pred | gt).float().sum()
    return intersection / union.clamp(min=1.0)


@torch.no_grad()
def validate(model: MVReconModel, val_loader: DataLoader,
             device: torch.device, K_sup: torch.Tensor,
             args) -> dict:
    """Run validation loop, return mean metrics."""
    model.eval()
    totals = {}
    n = 0
    for batch in val_loader:
        _, metrics = compute_loss(
            model, batch, device, K_sup=K_sup,
            n_samples=args.n_samples,
            n_rays_per_view=args.n_rays_per_view,
            sup_image_size=args.sup_image_size,
        )
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0) + v
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def train(args):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    renders_dir = str(data_dir / "renders")
    voxels_dir = str(data_dir / "voxels")

    # --- UID selection ---
    if args.mode == 'overfit':
        uids = OVERFIT_11_UIDS
        train_uids, val_uids = uids, []
    else:
        uids = discover_uids(data_dir, args.uids_file)
        if len(uids) > 30:
            train_uids, val_uids = split_train_val(uids, val_ratio=0.1)
        else:
            train_uids, val_uids = uids, []

    print(f"Mode: {args.mode}, Train: {len(train_uids)}, Val: {len(val_uids)}")

    # --- Datasets and loaders ---
    use_workers = args.num_workers if len(train_uids) > 50 else 0
    train_ds = ObjaverseMultiViewDataset(
        renders_dir=renders_dir, uids=train_uids,
        n_input_views=args.n_input_views,
        n_sup_views=args.n_sup_views,
        image_size=args.image_size,
        sup_image_size=args.sup_image_size,
        augment=(args.mode != 'overfit'),
        voxels_dir=voxels_dir,
    )
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=use_workers,
        prefetch_factor=2 if use_workers > 0 else None,
        persistent_workers=use_workers > 0,
    )
    print(f"Train dataset: {len(train_ds)} objects loaded")

    val_loader = None
    if val_uids:
        val_ds = ObjaverseMultiViewDataset(
            renders_dir=renders_dir, uids=val_uids,
            n_input_views=args.n_input_views,
            n_sup_views=args.n_sup_views,
            image_size=args.image_size,
            sup_image_size=args.sup_image_size,
            voxels_dir=voxels_dir,
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=min(use_workers, 2),
            prefetch_factor=2 if use_workers > 0 else None,
        )
        print(f"Val dataset: {len(val_ds)} objects loaded")

    # --- Model ---
    model = MVReconModel(
        volume_size=args.volume_size,
        feat_channels=args.feat_channels,
        input_size=args.image_size,
    ).to(device)

    counts = model.param_count()
    print("Parameters:")
    for k, v in counts.items():
        print(f"  {k}: {v:,} ({v / 1e6:.1f}M)")

    K_orig = blender_intrinsics()
    K_sup = adjust_intrinsics_for_crop_resize(
        K_orig, BLENDER_RENDER_W, BLENDER_RENDER_H, args.sup_image_size
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    start_epoch = 1
    best_val_iou = 0.0
    best_loss = float('inf')
    no_improve_epochs = 0
    log_entries = []

    # --- Resume ---
    if args.resume:
        resume_path = Path(args.resume)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            for _ in range(ckpt['epoch']):
                scheduler.step()
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', ckpt.get('loss', float('inf')))
        best_val_iou = ckpt.get('best_val_iou', 0.0)
        if resume_path.parent.name == 'checkpoints':
            out_dir = resume_path.parent.parent
        else:
            out_dir = resume_path.parent
        ckpt_dir = out_dir / 'checkpoints'
        print(f"Resumed from {resume_path} at epoch {start_epoch}")
        print(f"Output: {out_dir}")
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{args.mode}_v{args.volume_size}_{timestamp}"
        out_dir = Path(args.output_dir) / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = out_dir / 'checkpoints'
        ckpt_dir.mkdir(exist_ok=True)

        config = vars(args).copy()
        config['run_name'] = run_name
        config['timestamp'] = timestamp
        config['n_train'] = len(train_uids)
        config['n_val'] = len(val_uids)
        with open(out_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Save split for reproducibility
        with open(out_dir / 'split.json', 'w') as f:
            json.dump({'train': train_uids, 'val': val_uids}, f)

        print(f"Run: {run_name}")
        print(f"Output: {out_dir}")

    # --- Metrics CSV ---
    csv_path = out_dir / 'metrics.csv'
    csv_fields = ['epoch', 'train_loss', 'train_photo', 'train_mask',
                  'train_bce', 'train_sparse', 'train_iou',
                  'val_loss', 'val_iou', 'lr', 'time_s']
    if not csv_path.exists():
        with open(csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # --- Training loop ---
    grad_accum = args.grad_accum
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {}
        n_batches = 0
        t0 = time.time()

        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            loss, metrics = compute_loss(
                model, batch, device,
                K_sup=K_sup,
                n_samples=args.n_samples,
                n_rays_per_view=args.n_rays_per_view,
                sup_image_size=args.sup_image_size,
            )
            (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += metrics['loss']
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v
            n_batches += 1

            if n_batches % 100 == 0:
                elapsed = time.time() - t0
                avg = epoch_loss / n_batches
                iou = epoch_metrics.get('iou', 0) / n_batches
                eta = elapsed / n_batches * (len(train_loader) - n_batches)
                print(f"  [{n_batches}/{len(train_loader)}] "
                      f"loss={avg:.4f} iou={iou:.4f} "
                      f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

        scheduler.step()
        epoch_time = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_metrics = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}

        # --- Validation ---
        val_metrics = {}
        if val_loader and (epoch % args.val_every == 0 or epoch == 1):
            val_metrics = validate(model, val_loader, device, K_sup, args)
            val_iou = val_metrics.get('iou', 0)
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(model.state_dict(), ckpt_dir / 'best.pt')
                no_improve_epochs = 0
            else:
                no_improve_epochs += args.val_every
            print(f"  VAL loss={val_metrics.get('loss',0):.4f} "
                  f"iou={val_iou:.4f} best_iou={best_val_iou:.4f}")

        # --- CSV logging ---
        row = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'train_photo': avg_metrics.get('photo', 0),
            'train_mask': avg_metrics.get('mask', 0),
            'train_bce': avg_metrics.get('bce', 0),
            'train_sparse': avg_metrics.get('sparse', 0),
            'train_iou': avg_metrics.get('iou', 0),
            'val_loss': val_metrics.get('loss', ''),
            'val_iou': val_metrics.get('iou', ''),
            'lr': scheduler.get_last_lr()[0],
            'time_s': epoch_time,
        }
        with open(csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

        entry = {'epoch': epoch, 'avg_loss': avg_loss, **avg_metrics,
                 'lr': scheduler.get_last_lr()[0], 'time_s': epoch_time}
        if val_metrics:
            entry['val_loss'] = val_metrics.get('loss', 0)
            entry['val_iou'] = val_metrics.get('iou', 0)
        log_entries.append(entry)

        if epoch % args.log_every == 0 or epoch == 1:
            print(f"[Epoch {epoch:4d}/{args.epochs}] "
                  f"loss={avg_loss:.4f} "
                  f"photo={avg_metrics.get('photo', 0):.4f} "
                  f"mask={avg_metrics.get('mask', 0):.4f} "
                  f"bce={avg_metrics.get('bce', 0):.4f} "
                  f"sp={avg_metrics.get('sparse', 0):.4f} "
                  f"iou={avg_metrics.get('iou', 0):.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} "
                  f"({epoch_time:.1f}s)")

        # Save checkpoints (best on train loss if no val set)
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
                'best_val_iou': best_val_iou,
            }, ckpt_dir / f'epoch_{epoch:04d}.pt')
        if not val_loader and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_dir / 'best.pt')

        # Early stopping (only when val set exists)
        if val_loader and args.early_stop > 0 and no_improve_epochs >= args.early_stop:
            print(f"Early stopping: val IoU hasn't improved for {no_improve_epochs} epochs")
            break

    torch.save(model.state_dict(), ckpt_dir / 'final.pt')
    with open(out_dir / 'train_log.json', 'w') as f:
        json.dump(log_entries, f, indent=2)

    print(f"\nTraining complete. Best train loss: {best_loss:.4f}, "
          f"Best val IoU: {best_val_iou:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(description='Multi-view reconstruction training')
    parser.add_argument('--mode', choices=['overfit', 'full'], default='overfit')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--volume_size', type=int, default=32)
    parser.add_argument('--feat_channels', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=160)
    parser.add_argument('--sup_image_size', type=int, default=128)
    parser.add_argument('--n_input_views', type=int, default=4)
    parser.add_argument('--n_sup_views', type=int, default=4)
    parser.add_argument('--n_samples', type=int, default=64)
    parser.add_argument('--n_rays_per_view', type=int, default=1024)
    parser.add_argument('--output_dir', type=str, default='output/mv_recon_overfit')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=50)
    # New args for full-scale training
    parser.add_argument('--data_dir', type=str, default=str(DEFAULT_DATA_DIR),
                        help='Root data directory containing renders/ and voxels/')
    parser.add_argument('--uids_file', type=str, default=None,
                        help='Path to JSON file with UID list (e.g. filtered_uids.json)')
    parser.add_argument('--val_every', type=int, default=5,
                        help='Run validation every N epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker count')
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--early_stop', type=int, default=50,
                        help='Stop if val IoU stalls for N epochs (0=disabled)')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
