import torch
import torch.nn as nn
import torch.nn.functional as F


class _LocalWindowAttention(nn.Module):
    def __init__(self, in_channels, window_size=8):
        super(_LocalWindowAttention, self).__init__()
        self.window_size = window_size
        self.mid = max(1, in_channels // 4)
        self.query = nn.Conv2d(in_channels, self.mid, 1, bias=False)
        self.key = nn.Conv2d(in_channels, self.mid, 1, bias=False)
        self.value = nn.Conv2d(in_channels, self.mid, 1, bias=False)
        self.proj = nn.Conv2d(self.mid, in_channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)

    def _partition(self, x):
        ws = self.window_size
        batch_size, channels, height, width = x.shape
        pad_h = (ws - height % ws) % ws
        pad_w = (ws - width % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        _, _, padded_h, padded_w = x.shape
        n_h, n_w = padded_h // ws, padded_w // ws
        x = x.view(batch_size, channels, n_h, ws, n_w, ws)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(batch_size * n_h * n_w, channels, ws * ws)
        return x, batch_size, channels, height, width, padded_h, padded_w, n_h, n_w

    def _unpartition(self, x, batch_size, channels, height, width, padded_h, padded_w, n_h, n_w):
        ws = self.window_size
        x = x.view(batch_size, n_h, n_w, channels, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(batch_size, channels, padded_h, padded_w)
        return x[:, :, :height, :width]

    def forward(self, x):
        q_map = self.query(x)
        k_map = self.key(x)
        v_map = self.value(x)
        q, batch_size, _, height, width, padded_h, padded_w, n_h, n_w = self._partition(q_map)
        k, *_ = self._partition(k_map)
        v, *_ = self._partition(v_map)

        scale = self.mid ** 0.5
        attention = F.softmax(torch.bmm(q.permute(0, 2, 1), k) / scale, dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = self._unpartition(out, batch_size, self.mid, height, width, padded_h, padded_w, n_h, n_w)
        return x + self.gamma * self.proj(out)


class LCP(nn.Module):
    """
    Local Correlation Prior.

    Input: noisy infrared image with shape (B, 1, H, W).
    Output: soft structure map in [0, 1] with shape (B, 1, H, W).
    """

    def __init__(self, kernel_sizes=(3, 5, 7), use_attention=True, window_size=8):
        super(LCP, self).__init__()
        self.use_attention = use_attention

        self.branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size, padding=padding, bias=False),
                    nn.BatchNorm2d(16),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(16, 16, kernel_size, padding=padding, bias=False),
                    nn.BatchNorm2d(16),
                    nn.LeakyReLU(0.1, inplace=True),
                )
            )

        fused_channels = 16 * len(kernel_sizes)

        self.register_buffer(
            'sobel_x',
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.register_buffer(
            'sobel_y',
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3),
        )

        if use_attention:
            self.attention = _LocalWindowAttention(fused_channels, window_size=window_size)

        self.fusion = nn.Sequential(
            nn.Conv2d(fused_channels + 1, 32, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, 1, bias=True),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        final_conv = self.fusion[-2]
        nn.init.constant_(final_conv.bias, -0.85)

    def forward(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
        grad_mag = grad_mag / (grad_mag.amax(dim=(2, 3), keepdim=True) + 1e-6)

        branch_outputs = [branch(x) for branch in self.branches]
        fused = torch.cat(branch_outputs, dim=1)

        if self.use_attention:
            fused = self.attention(fused)

        fused = torch.cat([fused, grad_mag], dim=1)
        return self.fusion(fused)


if __name__ == '__main__':
    model = LCP(kernel_sizes=(3, 5, 7), use_attention=True, window_size=8)
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    print('LCP output shape:', out.shape)
    print('Value range: [{:.3f}, {:.3f}]'.format(out.min().item(), out.max().item()))
    print('Mean activation: {:.3f}'.format(out.mean().item()))
