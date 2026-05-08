"""
Supervised baselines and the sparse autoencoder used for filter recovery.

HWNet:        biological frontend + small classifier head
BaselineCNN:  parameter-matched standard CNN baseline
SparseAutoencoder: Olshausen-Field-style probe for "do natural-image patches
                   produce Gabor receptive fields?" diagnostic.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bio_layers import HWFrontEnd, HWFrontEndPlus


class HWNet(nn.Module):
    """HW frontend + 3-conv classifier head. ~61K head params."""
    def __init__(self, num_classes: int = 10, n_orientations: int = 8,
                 n_scales: int = 2, freeze_frontend: bool = False,
                 use_end_stopped: bool = False, input_size: int = 32):
        super().__init__()
        if use_end_stopped:
            self.frontend = HWFrontEndPlus(n_orientations=n_orientations,
                                           n_scales=n_scales)
        else:
            self.frontend = HWFrontEnd(n_orientations=n_orientations,
                                       n_scales=n_scales)
        c = self.frontend.out_channels
        self.head = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )
        if freeze_frontend:
            for p in self.frontend.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.frontend(x))


class BaselineCNN(nn.Module):
    """Parameter-matched standard CNN. Replaces the bio frontend with a
    learnable conv layer of similar receptive field. Same downstream head."""
    def __init__(self, num_classes: int = 10, frontend_channels: int = 16,
                 input_channels: int = 3):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=11, padding=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, frontend_channels, kernel_size=1),
            nn.BatchNorm2d(frontend_channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(frontend_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.frontend(x))


class SparseAutoencoder(nn.Module):
    """Tied-weights sparse autoencoder for the Olshausen-Field probe.

    Train this on whitened natural-image patches; the decoder weights should
    converge to Gabor-like filters. Diagnostic check that the V1 frontend is
    well-aligned with natural-image statistics.
    """
    def __init__(self, patch_size: int = 12, n_features: int = 64,
                 in_channels: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.n_features = n_features
        self.in_channels = in_channels
        input_dim = patch_size * patch_size * in_channels
        self.decoder = nn.Linear(n_features, input_dim, bias=False)
        nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(F.linear(x, self.decoder.weight.t()))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z

    def get_filters(self) -> torch.Tensor:
        """(n_features, in_channels, patch_size, patch_size)."""
        W = self.decoder.weight.t()
        return W.view(self.n_features, self.in_channels,
                      self.patch_size, self.patch_size)


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
