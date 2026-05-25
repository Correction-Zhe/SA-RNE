import torch
import torch.nn as nn


class SARNEResidualEstimator(nn.Module):
    """Compact Conv+ReLU residual noise estimator used by SA-RNE."""

    def __init__(self, depth=7, n_channels=32, image_channels=1, kernel_size=3):
        super(SARNEResidualEstimator, self).__init__()
        padding = kernel_size // 2
        layers = []

        # Entry layer: Conv + ReLU
        layers.append(nn.Conv2d(image_channels, n_channels, kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers: Conv + BN + ReLU
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))

        # Exit layer: Conv only
        layers.append(nn.Conv2d(n_channels, image_channels, kernel_size, padding=padding, bias=True))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        noise_pred = self.network(x)
        denoised = x - noise_pred
        return denoised, noise_pred


if __name__ == '__main__':
    model = SARNEResidualEstimator(depth=7, n_channels=32, image_channels=1, kernel_size=3)
    x = torch.randn(4, 1, 128, 128)
    denoised, noise_pred = model(x)
    print('denoised:', denoised.shape)    # (4, 1, 128, 128)
    print('noise_pred:', noise_pred.shape)  # (4, 1, 128, 128)
