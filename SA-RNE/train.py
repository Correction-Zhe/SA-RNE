import os
import random

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import config
from dataset import InfraredDataset
from lcp_model import LCP
from loss import LCPGuidedLoss
from main_model import SARNEResidualEstimator


def _prepare_output_dirs(outputs_dir):
    patches_dir = os.path.join(outputs_dir, 'patches')
    visuals_dir = os.path.join(outputs_dir, 'visuals')
    ckpt_dir = os.path.join(outputs_dir, 'checkpoints')
    for path in (patches_dir, visuals_dir, ckpt_dir):
        os.makedirs(path, exist_ok=True)
    return patches_dir, visuals_dir, ckpt_dir


def _load_full_image(path):
    """Load a complete grayscale image as a (1, 1, H, W) tensor."""
    img = Image.open(path).convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def _save_patch_grid(path, batch, denoised, noise_pred, lcp_map):
    n = min(4, batch.size(0))
    grid = torch.cat(
        [
            batch[:n],
            denoised[:n].clamp(0, 1),
            noise_pred[:n].clamp(0, 1),
            lcp_map[:n],
        ],
        dim=0,
    )
    save_image(grid, path, nrow=n)


def _save_full_visual(path, model, full_img_tensor, device):
    """Save side-by-side: noisy input, denoised output, and normalized residual."""
    model.eval()
    with torch.no_grad():
        img = full_img_tensor.to(device)
        denoised, noise_pred = model(img)
        noise_min = noise_pred.amin(dim=(2, 3), keepdim=True)
        noise_max = noise_pred.amax(dim=(2, 3), keepdim=True)
        noise_vis = (noise_pred - noise_min) / (noise_max - noise_min + 1e-8)
        grid = torch.cat([img.clamp(0, 1), denoised.clamp(0, 1), noise_vis], dim=0)
        save_image(grid, path, nrow=3)
    model.train()


def _save_random_full_visuals(epoch, dataset, model, visuals_dir, device):
    if len(dataset.image_paths) == 0:
        return

    n_samples = min(getattr(config, 'vis_num_images', 4), len(dataset.image_paths))
    sampled_paths = random.sample(dataset.image_paths, k=n_samples)

    for index, image_path in enumerate(sampled_paths, start=1):
        sample = _load_full_image(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(visuals_dir, f'full_epoch_{epoch:04d}_{index:02d}_{base_name}.png')
        _save_full_visual(out_path, model, sample, device)


def main():
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    patches_dir, visuals_dir, ckpt_dir = _prepare_output_dirs(config.outputs_dir)

    train_dataset = InfraredDataset(root_dir=config.dataset_dir, patch_size=config.patch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    print(f'Dataset size: {len(train_dataset)} images')

    residual_estimator = SARNEResidualEstimator(
        depth=config.depth,
        n_channels=config.n_channels,
        image_channels=config.image_channels,
        kernel_size=config.kernel_size,
    ).to(device)

    lcp_model = LCP(
        kernel_sizes=config.lcp_kernel_sizes,
        use_attention=config.lcp_use_attention,
        window_size=config.lcp_window_size,
    ).to(device)

    criterion = LCPGuidedLoss(
        lcp_module=lcp_model,
        lambda_tv=config.lambda_tv,
        lambda_grad=config.lambda_grad,
        epsilon=config.epsilon,
    ).to(device)

    optimizer = optim.Adam(
        list(residual_estimator.parameters()) + list(lcp_model.parameters()),
        lr=config.learning_rate,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    save_every = getattr(config, 'vis_save_every', 5)
    ckpt_every = getattr(config, 'ckpt_save_every', 10)

    for epoch in range(1, config.epochs + 1):
        residual_estimator.train()
        lcp_model.train()

        epoch_loss = 0.0
        epoch_loss_lcp = 0.0
        epoch_loss_grad = 0.0
        epoch_loss_tv = 0.0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            denoised, noise_pred = residual_estimator(batch)
            total_loss, loss_lcp, loss_tv, loss_grad = criterion(denoised, batch)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(residual_estimator.parameters()) + list(lcp_model.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_loss_lcp += loss_lcp.item()
            epoch_loss_grad += loss_grad.item()
            epoch_loss_tv += loss_tv.item()

        scheduler.step()

        n_batches = max(1, len(train_loader))
        print(
            f'Epoch [{epoch:04d}/{config.epochs}] '
            f'Loss: {epoch_loss / n_batches:.5f} | '
            f'LCP: {epoch_loss_lcp / n_batches:.5f} | '
            f'TV: {epoch_loss_tv / n_batches:.5f} | '
            f'Grad: {epoch_loss_grad / n_batches:.5f} | '
            f'LR: {scheduler.get_last_lr()[0]:.6f}'
        )

        if epoch % save_every == 0:
            with torch.no_grad():
                lcp_map = lcp_model(batch)
                lcp_sharp = torch.sigmoid((lcp_map - 0.5) * 8.0)
                print(
                    f'LCP raw mean:{lcp_map.mean():.3f} min:{lcp_map.min():.3f} '
                    f'max:{lcp_map.max():.3f} std:{lcp_map.std():.3f}'
                )
                print(
                    f'LCP sharp mean:{lcp_sharp.mean():.3f} min:{lcp_sharp.min():.3f} '
                    f'max:{lcp_sharp.max():.3f} std:{lcp_sharp.std():.3f}'
                )

            patch_path = os.path.join(patches_dir, f'epoch_{epoch:04d}.png')
            _save_patch_grid(
                patch_path,
                batch.cpu(),
                denoised.detach().cpu(),
                noise_pred.detach().cpu(),
                lcp_map.detach().cpu(),
            )
            _save_random_full_visuals(epoch, train_dataset, residual_estimator, visuals_dir, device)

        if epoch % ckpt_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'ckpt_epoch_{epoch:04d}.pth')
            torch.save(
                {
                    'epoch': epoch,
                    'residual_estimator_state_dict': residual_estimator.state_dict(),
                    'lcp_state_dict': lcp_model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                },
                ckpt_path,
            )
            print(f'Checkpoint saved: {ckpt_path}')

    print('Training complete.')


if __name__ == '__main__':
    main()
