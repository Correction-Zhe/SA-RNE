import torch
import torch.nn as nn
import torch.nn.functional as F


class LCPGuidedLoss(nn.Module):
    """Self-supervised denoising loss guided by a Local Correlation Prior."""

    def __init__(self, lcp_module, lambda_tv, lambda_grad, epsilon=1e-3):
        super(LCPGuidedLoss, self).__init__()
        self.lcp = lcp_module
        self.lambda_tv = lambda_tv
        self.lambda_grad = lambda_grad
        self.epsilon = epsilon

        self.register_buffer(
            'sobel_x',
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.register_buffer(
            'sobel_y',
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3),
        )

    def _tv_loss(self, x, weight_map):
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return (weight_map[:, :, 1:, :] * diff_h).mean() + (weight_map[:, :, :, 1:] * diff_w).mean()

    def _grad_loss(self, denoised, original, weight_map):
        gd_x = F.conv2d(denoised, self.sobel_x, padding=1)
        gd_y = F.conv2d(denoised, self.sobel_y, padding=1)
        go_x = F.conv2d(original, self.sobel_x, padding=1)
        go_y = F.conv2d(original, self.sobel_y, padding=1)
        grad_d = torch.sqrt(gd_x**2 + gd_y**2 + self.epsilon)
        grad_o = torch.sqrt(go_x**2 + go_y**2 + self.epsilon)
        return (weight_map * torch.abs(grad_d - grad_o)).mean()

    def _brc_target(self, x, half_r=4):
        """
        Bilateral Region Contrast (BRC) target.

        For each pixel, four directions split the local neighborhood into two
        half regions. A Fisher-style contrast score highlights stable structure
        while suppressing noisy block boundaries.
        """
        r = half_r
        size = 2 * r + 1

        def _half_kernel(direction):
            k_pos = torch.zeros(size, size, device=x.device, dtype=x.dtype)
            k_neg = torch.zeros(size, size, device=x.device, dtype=x.dtype)
            center = r

            if direction == 'h':
                k_pos[center, center + 1:] = 1.0
                k_neg[center, :center] = 1.0
            elif direction == 'v':
                k_pos[:center, center] = 1.0
                k_neg[center + 1:, center] = 1.0
            elif direction == 'd1':
                for i in range(1, r + 1):
                    k_pos[center - i, center - i] = 1.0
                    k_neg[center + i, center + i] = 1.0
            elif direction == 'd2':
                for i in range(1, r + 1):
                    k_pos[center - i, center + i] = 1.0
                    k_neg[center + i, center - i] = 1.0
            else:
                raise ValueError(f'Unsupported direction: {direction}')

            n_pos = k_pos.sum().clamp(min=1)
            n_neg = k_neg.sum().clamp(min=1)
            return (k_pos / n_pos).view(1, 1, size, size), (k_neg / n_neg).view(1, 1, size, size)

        brc_maps = []
        for direction in ('h', 'v', 'd1', 'd2'):
            k_pos, k_neg = _half_kernel(direction)
            mu_pos = F.conv2d(x, k_pos, padding=r)
            mu_neg = F.conv2d(x, k_neg, padding=r)

            x2 = x * x
            var_pos = F.conv2d(x2, k_pos, padding=r) - mu_pos**2
            var_neg = F.conv2d(x2, k_neg, padding=r) - mu_neg**2
            var_pos = var_pos.clamp(min=0)
            var_neg = var_neg.clamp(min=0)

            brc = (mu_pos - mu_neg) ** 2 / (var_pos + var_neg + self.epsilon)
            brc_maps.append(brc)

        brc_max = torch.stack(brc_maps, dim=0).max(dim=0).values
        brc_max = brc_max / brc_max.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        brc_max = torch.sqrt(brc_max + 1e-6)
        return brc_max / brc_max.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)

    def forward(self, denoised, original):
        lcp_map = self.lcp(original)
        lcp_map = torch.sigmoid((lcp_map - 0.5) * 8.0)
        flat_map = 1.0 - lcp_map

        brc_target = self._brc_target(original).detach()
        loss_lcp = F.mse_loss(lcp_map, brc_target)

        loss_grad = self.lambda_grad * self._grad_loss(denoised, original, lcp_map.detach())
        loss_tv = self.lambda_tv * self._tv_loss(denoised, flat_map.detach())

        total_loss = loss_lcp + loss_tv + loss_grad
        return total_loss, loss_lcp, loss_tv, loss_grad
