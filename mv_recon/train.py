"""Training script for multi-view reconstruction model.

Usage:
    # Overfit on 11 objects (PoC):
    python -m mv_recon.train --mode overfit --epochs 500

    # Train on all 18 objects:
    python -m mv_recon.train --mode full --epochs 200
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import MVReconModel
from .renderer import render_rays_batch
from .dataset import ObjaverseMultiViewDataset
from .camera_utils import (
    blender_intrinsics,
    adjust_intrinsics_for_crop_resize,
    BLENDER_RENDER_W,
    BLENDER_RENDER_H,
)


RENDERS_DIR = "/Users/sur/Downloads/mv_recon_data/renders"

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

# 11 objects already available locally (first batch copied)
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


def compute_loss(model: MVReconModel, batch: dict,
                 K_sup: torch.Tensor, device: torch.device,
                 n_samples: int = 64, n_rays_per_view: int = 512,
                 sup_size: int = 64) -> tuple[torch.Tensor, dict]:
    """Compute rendering-based training loss using random ray sampling."""
    input_imgs = batch['input_images'].to(device)
    input_c2w = batch['input_c2w'].to(device)
    sup_imgs = batch['sup_images'].to(device)
    sup_masks = batch['sup_masks'].to(device)
    sup_c2w = batch['sup_c2w'].to(device)

    density, color = model(input_imgs, input_c2w)

    density_vol = density[0]  # [1, D, D, D]
    color_vol = color[0]      # [3, D, D, D]

    rgb_pred, rgb_gt, mask_pred, mask_gt = render_rays_batch(
        density_vol, color_vol,
        sup_c2w[0], K_sup,
        sup_imgs[0], sup_masks[0],
        render_h=sup_size, render_w=sup_size,
        n_samples=n_samples,
        n_rays_per_view=n_rays_per_view,
    )

    rgb_loss = F.mse_loss(rgb_pred, rgb_gt)
    mask_loss = F.binary_cross_entropy(
        mask_pred.clamp(1e-6, 1 - 1e-6),
        mask_gt.clamp(0, 1)
    )

    occ = torch.sigmoid(density_vol)
    entropy = -(occ * torch.log(occ + 1e-6) + (1 - occ) * torch.log(1 - occ + 1e-6))
    entropy_loss = entropy.mean()

    loss = rgb_loss + 0.1 * mask_loss + 0.01 * entropy_loss

    metrics = {
        'loss': loss.item(),
        'rgb_loss': rgb_loss.item(),
        'mask_loss': mask_loss.item(),
        'entropy_loss': entropy_loss.item(),
    }
    return loss, metrics


def train(args):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.mode == 'overfit':
        uids = OVERFIT_11_UIDS
    else:
        uids = ALL_18_UIDS
    print(f"Mode: {args.mode}, Objects: {len(uids)}")

    dataset = ObjaverseMultiViewDataset(
        renders_dir=RENDERS_DIR, uids=uids,
        n_input_views=args.n_input_views,
        n_sup_views=args.n_sup_views,
        image_size=args.image_size,
        sup_image_size=args.sup_image_size,
    )
    print(f"Dataset: {len(dataset)} objects loaded")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    model = MVReconModel(
        volume_size=args.volume_size,
        feat_channels=args.feat_channels,
        input_size=args.image_size,
    ).to(device)

    counts = model.param_count()
    print("Parameters:")
    for k, v in counts.items():
        print(f"  {k}: {v:,} ({v / 1e6:.1f}M)")

    K_sup = adjust_intrinsics_for_crop_resize(
        blender_intrinsics(), BLENDER_RENDER_W, BLENDER_RENDER_H, args.sup_image_size
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    start_epoch = 1
    best_loss = float('inf')
    log_entries = []

    # Resume from checkpoint if specified
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
        # Recover run directory from checkpoint path
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
        config['n_objects'] = len(uids)
        config['uids'] = uids
        with open(out_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Run: {run_name}")
        print(f"Output: {out_dir}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {}
        n_batches = 0
        t0 = time.time()

        for batch in loader:
            optimizer.zero_grad()
            loss, metrics = compute_loss(
                model, batch, K_sup, device,
                n_samples=args.n_samples,
                n_rays_per_view=args.n_rays_per_view,
                sup_size=args.sup_image_size,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += metrics['loss']
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v
            n_batches += 1

        scheduler.step()
        epoch_time = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_metrics = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}

        entry = {
            'epoch': epoch, 'avg_loss': avg_loss, **avg_metrics,
            'lr': scheduler.get_last_lr()[0], 'time_s': epoch_time,
        }
        log_entries.append(entry)

        if epoch % args.log_every == 0 or epoch == 1:
            print(f"[Epoch {epoch:4d}/{args.epochs}] "
                  f"loss={avg_loss:.4f} "
                  f"rgb={avg_metrics.get('rgb_loss', 0):.4f} "
                  f"mask={avg_metrics.get('mask_loss', 0):.4f} "
                  f"ent={avg_metrics.get('entropy_loss', 0):.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} "
                  f"({epoch_time:.1f}s)")

        if epoch % args.save_every == 0 or avg_loss < best_loss:
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), ckpt_dir / 'best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
            }, ckpt_dir / f'epoch_{epoch:04d}.pt')

    torch.save(model.state_dict(), ckpt_dir / 'final.pt')
    with open(out_dir / 'train_log.json', 'w') as f:
        json.dump(log_entries, f, indent=2)

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(description='Multi-view reconstruction training')
    parser.add_argument('--mode', choices=['overfit', 'full'], default='overfit')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--volume_size', type=int, default=32)
    parser.add_argument('--feat_channels', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=160)
    parser.add_argument('--sup_image_size', type=int, default=64)
    parser.add_argument('--n_input_views', type=int, default=4)
    parser.add_argument('--n_sup_views', type=int, default=4)
    parser.add_argument('--n_samples', type=int, default=64)
    parser.add_argument('--n_rays_per_view', type=int, default=512)
    parser.add_argument('--output_dir', type=str, default='output/mv_recon_overfit')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=50)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
