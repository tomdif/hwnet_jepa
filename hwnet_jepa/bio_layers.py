"""
Hubel-Wiesel-inspired biological vision layers.

Stages (all faithful to specific H&W findings):
  Stage 0: Retinal/LGN ON+OFF channels via Difference of Gaussians (Kuffler 1953)
  Stage 1: V1 simple-cell oriented quadrature filters (Hubel & Wiesel 1959-1962),
           parameterized as Gabors so the biological prior is hard-coded
  Stage 2: V1 complex cells via energy pooling (Adelson & Bergen 1985)
  Stage 3: Divisive normalization (Heeger 1992, Carandini & Heeger 2012)
  Stage 4: End-stopped / hypercomplex cells (Hubel & Wiesel 1965)

Two assemblies provided:
  HWFrontEnd: stages 0-3 (the canonical minimal pipeline)
  HWFrontEndPlus: stages 0-4 with end-stopped channels concatenated
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- Stage 0: Retinal/LGN ----------------------------- #


def make_dog_kernel(size: int, sigma_center: float, sigma_surround: float,
                    polarity: str = "on") -> torch.Tensor:
    """Difference-of-Gaussians kernel approximating Kuffler-style center-surround."""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    r2 = xx ** 2 + yy ** 2
    g_center = torch.exp(-r2 / (2 * sigma_center ** 2))
    g_center /= g_center.sum()
    g_surround = torch.exp(-r2 / (2 * sigma_surround ** 2))
    g_surround /= g_surround.sum()
    dog = g_center - g_surround
    if polarity == "off":
        dog = -dog
    dog = dog - dog.mean()
    return dog


class RetinalLGN(nn.Module):
    """ON and OFF channel preprocessing via DoG filters.

    For RGB input we first compute luminance (BT.601), then apply ON and OFF
    DoG filters. With rectify=True (default) we apply half-wave rectification
    to mirror separated ON/OFF cell populations. With rectify=False we keep
    signed responses; useful when feeding sparse-coding pretraining where you
    want full signal magnitude.
    """
    def __init__(self, in_channels: int = 3, kernel_size: int = 7,
                 sigma_center: float = 0.8, sigma_surround: float = 1.6,
                 rectify: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.rectify = rectify
        on_kernel = make_dog_kernel(kernel_size, sigma_center, sigma_surround, "on")
        off_kernel = make_dog_kernel(kernel_size, sigma_center, sigma_surround, "off")
        self.register_buffer("on_kernel", on_kernel.view(1, 1, kernel_size, kernel_size))
        self.register_buffer("off_kernel", off_kernel.view(1, 1, kernel_size, kernel_size))
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 3:
            lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            lum = x.mean(dim=1, keepdim=True)
        on = F.conv2d(lum, self.on_kernel, padding=self.padding)
        off = F.conv2d(lum, self.off_kernel, padding=self.padding)
        if self.rectify:
            on = F.relu(on)
            off = F.relu(off)
        return torch.cat([on, off], dim=1)


# ----------------------------- Stage 1: V1 simple cells ----------------------------- #


class V1SimpleQuadrature(nn.Module):
    """Oriented quadrature-pair filters parameterized as Gabors.

    Each (orientation, scale) pair produces TWO output channels: a cos-phase
    (even) and a sin-phase (odd) Gabor. The pairs feed the energy model in
    Stage 2. The Gabor *parameters* (theta, sigma, lambda) are learnable
    rather than the raw pixel weights, which keeps the biological prior intact
    throughout training.
    """
    def __init__(self, in_channels: int = 2, n_orientations: int = 8,
                 n_scales: int = 2, kernel_size: int = 11):
        super().__init__()
        self.n_orientations = n_orientations
        self.n_scales = n_scales
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.padding = kernel_size // 2

        init_thetas = torch.linspace(0, math.pi, n_orientations + 1)[:-1]
        init_sigmas = torch.linspace(1.5, 3.0, n_scales)
        init_lambdas = torch.linspace(3.0, 6.0, n_scales)
        thetas = init_thetas.unsqueeze(1).expand(n_orientations, n_scales).clone()
        sigmas = init_sigmas.unsqueeze(0).expand(n_orientations, n_scales).clone()
        lambdas = init_lambdas.unsqueeze(0).expand(n_orientations, n_scales).clone()

        self.thetas = nn.Parameter(thetas)
        self.log_sigmas = nn.Parameter(torch.log(sigmas))
        self.log_lambdas = nn.Parameter(torch.log(lambdas))

        n_filters = n_orientations * n_scales * 2
        mix = torch.zeros(n_filters, in_channels)
        if in_channels == 2:
            mix[:, 0] = 1.0
            mix[:, 1] = -1.0
        else:
            mix[:] = 1.0 / in_channels
        self.input_mix = nn.Parameter(mix)

    @property
    def n_filters(self) -> int:
        return self.n_orientations * self.n_scales * 2

    def build_kernels(self) -> torch.Tensor:
        """Construct (n_filters, in_channels, k, k) tensor from Gabor parameters."""
        device = self.thetas.device
        K = self.kernel_size
        coords = torch.arange(K, device=device, dtype=torch.float32) - (K - 1) / 2.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        sigmas = torch.exp(self.log_sigmas)
        lambdas = torch.exp(self.log_lambdas)
        thetas = self.thetas

        n_pairs = self.n_orientations * self.n_scales
        thetas_f = thetas.flatten()
        sigmas_f = sigmas.flatten()
        lambdas_f = lambdas.flatten()

        cos_t = torch.cos(thetas_f).view(n_pairs, 1, 1)
        sin_t = torch.sin(thetas_f).view(n_pairs, 1, 1)
        sigmas_b = sigmas_f.view(n_pairs, 1, 1)
        lambdas_b = lambdas_f.view(n_pairs, 1, 1)
        xx_b = xx.unsqueeze(0)
        yy_b = yy.unsqueeze(0)
        x_t = xx_b * cos_t + yy_b * sin_t
        y_t = -xx_b * sin_t + yy_b * cos_t

        env = torch.exp(-(x_t ** 2 + y_t ** 2) / (2 * sigmas_b ** 2))
        g_cos = env * torch.cos(2 * math.pi * x_t / lambdas_b)
        g_sin = env * torch.sin(2 * math.pi * x_t / lambdas_b)

        g_cos = g_cos - g_cos.mean(dim=(-2, -1), keepdim=True)
        g_cos = g_cos / (g_cos.flatten(1).norm(dim=1).view(n_pairs, 1, 1).clamp_min(1e-8))
        g_sin = g_sin - g_sin.mean(dim=(-2, -1), keepdim=True)
        g_sin = g_sin / (g_sin.flatten(1).norm(dim=1).view(n_pairs, 1, 1).clamp_min(1e-8))

        kernels = torch.stack([g_cos, g_sin], dim=1).view(n_pairs * 2, K, K)
        kernels = kernels.unsqueeze(1) * self.input_mix.unsqueeze(-1).unsqueeze(-1)
        return kernels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K = self.build_kernels()
        return F.conv2d(x, K, padding=self.padding)


# ----------------------------- Stage 2: V1 complex cells ----------------------------- #


class V1ComplexEnergy(nn.Module):
    """Adelson-Bergen energy model. Phase-invariant per (orientation, scale)."""
    def __init__(self, n_orientations: int, n_scales: int):
        super().__init__()
        self.n_orientations = n_orientations
        self.n_scales = n_scales

    def forward(self, simple_out: torch.Tensor) -> torch.Tensor:
        B, _, H, W = simple_out.shape
        x = simple_out.view(B, self.n_orientations, self.n_scales, 2, H, W)
        cos_part = x[:, :, :, 0]
        sin_part = x[:, :, :, 1]
        energy = torch.sqrt(cos_part ** 2 + sin_part ** 2 + 1e-8)
        return energy.reshape(B, self.n_orientations * self.n_scales, H, W)


# ----------------------------- Stage 3: Divisive normalization ----------------------------- #


class DivisiveNormalization(nn.Module):
    """R_i = C_i^n / (sigma^n + sum_j C_j^n) over local neighborhood."""
    def __init__(self, n_channels: int, spatial_size: int = 3, n: float = 2.0):
        super().__init__()
        self.n = n
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self.spatial_size = spatial_size
        self.padding = spatial_size // 2
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        x_n = x ** self.n
        pooled_channels = x_n.sum(dim=1, keepdim=True)
        pooled = F.avg_pool2d(pooled_channels, kernel_size=self.spatial_size,
                              stride=1, padding=self.padding)
        pooled = pooled.expand_as(x_n)
        sigma_n = torch.exp(self.log_sigma) ** self.n
        denom = sigma_n + pooled
        return x_n / (denom + 1e-8)


# ----------------------------- Stage 4: End-stopped / hypercomplex ----------------------------- #


class EndStoppedCells(nn.Module):
    """Hubel-Wiesel hypercomplex cells: respond to oriented bars of LIMITED length.

    Constructed as (R - shift(R))_+ where shift is along the preferred orientation
    by a learnable distance. Positive at line endings, corners, curvature.
    """
    def __init__(self, n_orientations: int = 8, n_scales: int = 2,
                 init_shift: float = 3.0):
        super().__init__()
        self.n_orientations = n_orientations
        self.n_scales = n_scales
        thetas = torch.linspace(0, math.pi, n_orientations + 1)[:-1]
        self.register_buffer("thetas", thetas)
        self.log_shift = nn.Parameter(torch.full((n_orientations, n_scales),
                                                 math.log(init_shift)))

    def forward(self, complex_out: torch.Tensor) -> torch.Tensor:
        B, C, H, W = complex_out.shape
        x = complex_out.view(B, self.n_orientations, self.n_scales, H, W)
        shifts = torch.exp(self.log_shift)

        ys = torch.linspace(-1, 1, H, device=x.device)
        xs = torch.linspace(-1, 1, W, device=x.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack([xx, yy], dim=-1)

        out = torch.zeros_like(x)
        for o in range(self.n_orientations):
            theta = self.thetas[o]
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            for s in range(self.n_scales):
                shift_pix = shifts[o, s]
                dx_norm = 2.0 * shift_pix * cos_t / max(W - 1, 1)
                dy_norm = 2.0 * shift_pix * sin_t / max(H - 1, 1)
                grid = base_grid.clone()
                grid[..., 0] = grid[..., 0] + dx_norm
                grid[..., 1] = grid[..., 1] + dy_norm
                grid_b = grid.unsqueeze(0).expand(B, -1, -1, -1)
                feat = x[:, o, s].unsqueeze(1)
                shifted = F.grid_sample(feat, grid_b, mode="bilinear",
                                        padding_mode="zeros", align_corners=True)
                end_stopped = F.relu(feat - shifted)
                out[:, o, s] = end_stopped.squeeze(1)
        return out.view(B, C, H, W)


# ----------------------------- Assemblies ----------------------------- #


class HWFrontEnd(nn.Module):
    """Stages 0-3: retinal -> V1 simple -> V1 complex -> divisive norm.

    out_channels = n_orientations * n_scales (default 16).
    """
    def __init__(self, n_orientations: int = 8, n_scales: int = 2,
                 v1_kernel_size: int = 11, dog_kernel_size: int = 7,
                 input_channels: int = 3):
        super().__init__()
        self.retinal = RetinalLGN(in_channels=input_channels, kernel_size=dog_kernel_size)
        self.v1_simple = V1SimpleQuadrature(in_channels=2,
                                            n_orientations=n_orientations,
                                            n_scales=n_scales,
                                            kernel_size=v1_kernel_size)
        self.v1_complex = V1ComplexEnergy(n_orientations, n_scales)
        n_complex = n_orientations * n_scales
        self.norm = DivisiveNormalization(n_complex)
        self.out_channels = n_complex

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.retinal(x)
        x = self.v1_simple(x)
        x = self.v1_complex(x)
        x = self.norm(x)
        return x


class HWFrontEndPlus(nn.Module):
    """Stages 0-4: adds end-stopped channels concatenated with complex channels.

    out_channels = 2 * (n_orientations * n_scales) (default 32).

    Note: in our experiments on synthetic data this *hurt* downstream performance
    (-9 to -11 points) because the synthetic class structure didn't have shape
    boundaries for end-stopped cells to detect. On real natural images with
    object boundaries and corners this should help. Provided here for that case.
    """
    def __init__(self, n_orientations: int = 8, n_scales: int = 2,
                 v1_kernel_size: int = 11, dog_kernel_size: int = 7,
                 input_channels: int = 3):
        super().__init__()
        self.retinal = RetinalLGN(in_channels=input_channels, kernel_size=dog_kernel_size)
        self.v1_simple = V1SimpleQuadrature(in_channels=2,
                                            n_orientations=n_orientations,
                                            n_scales=n_scales,
                                            kernel_size=v1_kernel_size)
        self.v1_complex = V1ComplexEnergy(n_orientations, n_scales)
        n_complex = n_orientations * n_scales
        self.norm = DivisiveNormalization(n_complex)
        self.end_stopped = EndStoppedCells(n_orientations, n_scales)
        self.norm_es = DivisiveNormalization(n_complex)
        self.out_channels = 2 * n_complex

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.retinal(x)
        x = self.v1_simple(x)
        c = self.v1_complex(x)
        c_norm = self.norm(c)
        es = self.end_stopped(c_norm)
        es_norm = self.norm_es(es)
        return torch.cat([c_norm, es_norm], dim=1)
