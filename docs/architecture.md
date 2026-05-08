# Architecture notes

## The Hubel & Wiesel front end

We implement five stages, faithful to specific findings from the H&W program:

### Stage 0: Retinal/LGN ON+OFF channels (Kuffler 1953)
Difference of Gaussians filtering on the luminance channel produces ON-center and OFF-center responses. Half-wave rectification mirrors the separated ON and OFF cell populations in the retina/LGN. We use fixed (non-learned) DoG kernels because retinal/LGN receptive field structure is largely innate.

### Stage 1: V1 simple cells (Hubel & Wiesel 1959-1962)
Oriented filters with separate ON and OFF subregions. Each (orientation, scale) pair produces TWO output channels: a cos-phase (even) and a sin-phase (odd) Gabor. The cos/sin pairs feed the energy model in stage 2.

We parameterize the filters as Gabors with learnable (theta, sigma, lambda) parameters rather than free pixel weights. This keeps the biological prior intact throughout training and uses only ~100 parameters for the entire V1 layer.

### Stage 2: V1 complex cells (Adelson & Bergen 1985)
Phase-invariant oriented responses via the energy model: `C = sqrt(S_cos^2 + S_sin^2)`. This replaces standard max-pooling with the actual mathematical model of complex-cell behavior.

### Stage 3: Divisive normalization (Heeger 1992, Carandini & Heeger 2012)
`R_i = C_i^n / (sigma^n + sum_j C_j^n)` over a local neighborhood. Produces contrast-invariant responses and is one of the most well-attested canonical computations in neuroscience.

### Stage 4: End-stopped / hypercomplex cells (Hubel & Wiesel 1965)
`E = (R - shift(R, delta))_+` where shift is along the preferred orientation by a learnable distance. Positive response at line endings, corners, and curvature.

In our synthetic-data experiments stage 4 hurt performance because the synthetic classes had no shape boundaries. On real natural images with object boundaries it should help. Keep `--use_end_stopped` available but off by default.

## I-JEPA on top of the bio frontend

Standard I-JEPA (Assran et al. 2023) but with the bio frontend producing the input feature map instead of raw pixels. Components:

- **Patch projector**: a strided conv that aggregates the 16-channel V1 feature map into patch tokens (default 4x4 patches, 64 patches for 32x32 input).
- **Context encoder**: small transformer (3 layers, 4 heads, 64-dim) over visible patch tokens.
- **Target encoder**: EMA of the context encoder, no gradients flow through it.
- **Predictor**: 2-layer transformer that takes context tokens plus mask tokens at target positions, predicts target latents.
- **Loss**: smooth-L1 between predicted and target latents at masked positions, plus a VICReg-style variance regularizer (`max(0, var_target - std(predictions))`) to prevent representation collapse.

EMA decay is scheduled from 0.95 to 0.999 over training - looser at start (target encoder follows context closely so the loss has signal), tighter at end (target encoder stabilizes).

## Why the variance regularizer matters

Without variance regularization, JEPA collapses: the EMA target gradually moves toward a near-constant solution, the predictor trivially matches it, and predictions/targets both have variance ~0.001. This was visible in our early experiments and is a documented failure mode in the JEPA literature.

The fix (penalize standard deviation of predictions falling below a target value) is from VICReg (Bardes et al. 2022). Variance is computed across the batch dimension at each embedding dimension; penalizing low std forces the encoder to spread its outputs across the embedding space.

## Why end-to-end training of HW-Net struggles at low data

The HW-Net frontend has only ~113 parameters but they're entangled in a constrained Gabor parameterization. Moving theta affects the orientation of all cos/sin filter pairs at that scale simultaneously. The loss landscape is pinched compared to free-form conv weights, and gradient descent with few examples (n=50, n=100) cannot navigate it reliably.

JEPA pretraining solves this by giving the encoder a rich self-supervised signal that drives Gabor parameters to good values without needing labels. By the time the linear probe runs, the Gabor parameters are already near-optimal.

## Why supervised features transfer poorly

The supervised HWNet frontend, after training on a 10-class task, has Gabor parameters tuned to the *specific* orientations, scales, and spatial frequencies that distinguish those 10 classes. On a different downstream task, those tuned parameters are mismatched - especially at low data, where the new linear head can't compensate.

JEPA features, by contrast, are tuned to the predictable structure of the visual distribution rather than to any particular discrimination task. They transfer because they encode something more general.

## Things we haven't built but should

- **Top-down feedback**: real V1 receives more connections from V2 than it sends. Implementable as cross-attention from later transformer blocks back to earlier ones.
- **Binocular pathway**: the disparity energy model (Ohzawa et al. 1990) extends the energy model to combine left/right eye filters at variable offsets. Useful for stereo data.
- **Direction selectivity**: temporal-difference filtering plus orientation tuning for video. Connects naturally to V-JEPA / world-model extensions.
- **Block-masking with attention masks**: current block-mask implementation truncates to the min visible/target count across a batch, losing some samples' coverage. Proper implementation uses attention masks instead of truncation.
