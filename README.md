# HW-Net + I-JEPA

A biologically-structured visual encoder combining Hubel & Wiesel's findings on cat V1 with I-JEPA-style self-supervised pretraining.

## What this is

This repo implements a small-scale research prototype of the idea that **biological priors plus self-supervised predictive pretraining produce sample-efficient visual encoders**. The architecture has three layers of design:

- **A biologically-structured front end** based on Hubel & Wiesel's cat V1 work: ON/OFF retinal channels via Difference of Gaussians, V1 simple cells parameterized as oriented Gabor quadrature pairs, V1 complex cells via the Adelson-Bergen energy model, divisive normalization, and (optionally) end-stopped/hypercomplex cells.
- **An I-JEPA self-supervised pretraining stage** that operates on top of this frontend. Masked patches of the bio-feature map are predicted from visible context patches in latent space, with VICReg-style variance regularization to prevent representation collapse.
- **A small classifier head** for downstream evaluation, with three readout options (mean-pool linear, attention-pool linear, kNN).

## Why

Standard end-to-end CNNs eventually rediscover Gabor-like filters in their first layer when given enough data. Building those filters in from the start - faithfully to what H&W actually measured - gives you a useful prior, but training the constrained Gabor parameters supervised gets stuck at very low data counts (the "n=100 collapse" we observed). Self-supervised pretraining via I-JEPA fixes this: the encoder adapts its Gabor parameters using *unlabeled* data, then a small linear probe on top dominates supervised baselines at low labeled-data counts.

## Synthetic experiment results (reproducible from this repo)

10-class synthetic classification task, 6 seeds, class-balanced subsets:

| n_train | HW supervised | BaselineCNN | JEPA frozen | JEPA fine-tuned |
|---------|---------------|-------------|-------------|-----------------|
| 50 | 19.4% ± 5.6% | 13.6% ± 3.3% | **73.9% ± 4.0%** | 75.8% ± 2.5% |
| 100 | 27.8% ± 10.0% | 64.8% ± 12.0% | **81.6% ± 1.1%** | 82.1% ± 0.2% |
| 200 | 70.4% ± 4.4% | 67.0% ± 13.2% | **82.0% ± 0.7%** | 82.4% ± 2.7% |
| 500 | 95.9% ± 1.3% | 86.1% ± 6.5% | 83.1% ± 0.7% | **89.7% ± 0.8%** |

Transfer to a *different* 8-class compositional task (the JEPA encoder never saw these class boundaries during pretraining):

| n_train | JEPA attn-pool | Random encoder | HW E2E | Sup HW frozen |
|---------|---------------|----------------|--------|---------------|
| 40 | **65.3% ± 3.3%** | 44.5% ± 5.2% | 14.8% ± 2.5% | 17.1% ± 5.1% |
| 80 | **70.4% ± 3.3%** | 42.7% ± 7.1% | 32.0% ± 6.1% | 28.2% ± 9.8% |
| 200 | **71.0% ± 3.2%** | 49.7% ± 5.0% | 50.6% ± 7.4% | 63.1% ± 4.7% |

JEPA features transfer; supervised features do not, especially at low data.

## Caveats up front

- All numbers above are on synthetic data with hand-crafted class-defining patterns. Real-data validation is the natural next step and is what the `--source cifar10`, `--source stl10`, `--source imagefolder` options exist for.
- We tested architectural extensions (image augmentation, end-stopped cells) and both *hurt* on synthetic data. They're expected to help on real natural images. Match augmentations to your data.
- Most experiments ran on CPU at small scale. GPU + larger pretraining datasets should improve absolute numbers across the board.

## Installation

```bash
pip install -e .
```

Or just `pip install -r requirements.txt` and run scripts directly.

## Reproducing the synthetic results

```bash
# Supervised baselines (HW-Net and BaselineCNN at multiple sample sizes)
python scripts/run_supervised.py --source synthetic_10class

# JEPA pretraining + linear/attn-pool probe
python scripts/run_jepa.py --source synthetic_10class --pretrain_epochs 12

# Transfer experiment
python scripts/run_transfer.py \
    --pretrain_source synthetic_10class \
    --transfer_source synthetic_transfer

# Plot the curves
python scripts/plot_results.py \
    --supervised_results results/supervised_synthetic_10class.json \
    --jepa_results results/jepa_results_synthetic_10class.json \
    --out results/synthetic_curves.png
```

## Running on real data

The data abstraction supports several common sources. CIFAR-10 is the cleanest first step:

```bash
# CIFAR-10 supervised baselines
python scripts/run_supervised.py --source cifar10 --image_size 32

# CIFAR-10 JEPA pretraining + linear probe (uses augmentation by default for natural images)
python scripts/run_jepa.py --source cifar10 --image_size 32 \
    --pretrain_epochs 50 --use_augmentation \
    --n_pretrain_max 50000

# STL-10 - has 100K unlabeled images explicitly designed for SSL
python scripts/run_jepa.py --source stl10 --image_size 64 \
    --pretrain_epochs 100 --use_augmentation --use_block_masks \
    --patch_size 8 --n_pretrain_max 100000

# Transfer: CIFAR-10 pretraining -> CIFAR-100 evaluation
python scripts/run_transfer.py \
    --pretrain_source cifar10 --transfer_source cifar100 \
    --image_size 32

# Custom dataset organized as ImageFolder
python scripts/run_jepa.py --source imagefolder \
    --data_root /path/to/your/dataset --image_size 64 \
    --pretrain_epochs 50 --use_augmentation
```

Note: torchvision will download CIFAR-10/100/STL-10 to `--data_root` on first run if your machine has internet access.

## Olshausen-Field probe

Verify the architectural prior is well-aligned with natural-image statistics by training a sparse autoencoder on whitened patches and checking whether the learned filters are Gabor-like:

```bash
python scripts/sparse_coding_probe.py --source cifar10 --image_size 32 --n_epochs 50
```

This produces a filter visualization (`results/sae_filters_cifar10.png`) and Gabor-fit scores. On real natural-image data the fraction of filters with Gabor-fit-score > 0.5 should be substantial.

## File structure

```
hwnet_jepa/
  hwnet_jepa/                    # main package
    __init__.py
    bio_layers.py                # H&W layers (DoG, Gabor V1, energy pool, divisive norm, end-stopped)
    networks.py                  # HWNet, BaselineCNN, SparseAutoencoder
    jepa.py                      # I-JEPA architecture
    augment.py                   # augmentation + readout heads
    train.py                     # supervised + jepa_pretrain + linear_probe
    data.py                      # dataset loaders (synthetic, cifar10/100, stl10, imagefolder)
    data_synthetic.py            # synthetic dataset generators
  scripts/                       # CLI entry points
    run_supervised.py
    run_jepa.py
    run_transfer.py
    sparse_coding_probe.py
    plot_results.py
  configs/                       # hyperparameter presets
    synthetic_baseline.yaml
    cifar10_baseline.yaml
    stl10_baseline.yaml
  results/                       # outputs go here
  docs/                          # design notes
    architecture.md
    findings.md
  requirements.txt
  setup.py
```

## Architectural notes

See `docs/architecture.md` for the design rationale and `docs/findings.md` for what we learned (positive and negative results) on the synthetic data.

## Known TODOs / gaps

- Block-masking is implemented but uses a min-length truncation across the batch instead of full padding-aware attention. Cleaner to do this with attention masks in `JEPAEncoder.forward_subset`.
- We don't have a kNN-with-fine-tuning option (kNN is frozen by construction).
- Top-down feedback (V2 -> V1) is in the design notes but not implemented.
- No data parallelism / multi-GPU. Single-GPU only.
- The synthetic data on disk is regenerated each run; for very large pretraining pools this could be cached.

## License

MIT.
