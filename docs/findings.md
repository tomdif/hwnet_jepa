# Findings from synthetic-data development

This document records what we learned during the development sessions, including negative results that are scientifically informative even though they don't appear in the README's headline numbers.

## What worked

### 1. Biological priors + JEPA self-supervision is genuinely synergistic

The combination dominates either alone at low labeled-data counts. At n=50 (5 examples per class), JEPA frozen probe scored 73.9% vs HW-Net supervised 19.4% and BaselineCNN supervised 13.6%. The 54-point gap is not subtle.

The mechanism: HW-Net supervised gets stuck because the Gabor parameters need labeled data to adapt and there isn't enough at n=50. JEPA pretraining adapts those parameters using the unlabeled pretraining pool, and the linear probe afterward only has to learn 778 weights instead of also fitting the encoder.

### 2. Variance reduction

JEPA standard deviations across seeds were 5-15x tighter than supervised baselines at every sample size below n=500. Beyond mean accuracy, the result is more *reliable* run to run.

### 3. Attention-pool readout

Replacing mean-pool linear readout with a single-query attention-pool added 3-5 points at every sample size, for free. Mean-pool is a weak readout and leaves real signal on the table.

### 4. Fine-tuning the encoder closes the high-data gap

At n=500 the frozen-probe ceiling (83%) was below HW-Net supervised (96%). Fine-tuning the encoder with a small LR (5e-4) brought JEPA up to 89.7% - most of the gap closed without losing the low-data advantage.

### 5. Transfer to a different task structure

The JEPA encoder, pretrained on 10-class single-pattern data, achieved 65-71% on an *8-class compositional* task it never saw during pretraining. The supervised HWNet frontend hit only 17-28% on the same transfer task at low data. This is the strongest evidence we have that JEPA learned general visual features rather than memorizing the original class signatures.

## What didn't work, and why

### 1. Augmentation hurts on synthetic data

We added standard SSL augmentation (random crop+resize, horizontal flip, brightness/contrast jitter) to JEPA pretraining. Result: -9 to -12 points on the synthetic 10-class task at low data.

The reason: our synthetic classes are defined by *exact* features. A horizontal flip turns a "diag-up" class image into a "diag-down" class image. Random crops can remove the top/bottom-emphasis patterns. The augmentations destroy class-relevant information.

This is a real lesson, not just a synthetic-data artifact: SSL augmentations encode invariances, and the invariances must match what's task-irrelevant in your data. On natural images a flipped cat is still a cat. On our synthetic data a flipped class is a different class.

### 2. End-stopped cells hurt on synthetic data

Adding stage 4 (end-stopped/hypercomplex cells) to the frontend, doubling the channel count from 16 to 32, *reduced* downstream accuracy by 9-11 points.

The reason: synthetic classes had no shape boundaries, line endings, or corners. End-stopped cells fire at exactly those features, so the extra 16 channels added noise without adding signal. On real natural images with object boundaries, end-stopped cells should help.

### 3. The "n=200 dip" disappeared with more seeds

Our initial 2-seed run showed HW-Net losing badly at n=200 (24.3% vs 58.8% baseline). With 6 seeds the picture flipped: HW-Net 70.4% vs baseline 67.0%. The "dip" was random variance from too few seeds. **Lesson: 2 seeds is not enough for a result you want to interpret.**

### 4. Sparse-coding pretraining of V1 was hard to make work

We tried Olshausen-Field-style sparse coding to derive Gabor filter parameters automatically from natural image patches. On synthetic pink-noise data with random oriented edges, the sparse autoencoder struggled to find clean Gabor solutions. We worked around this by initializing V1 with hand-set Gabor parameters spanning orientation/scale uniformly. On real natural-image data the sparse-coding probe should converge to Gabors more easily and could be re-enabled.

## Things we'd test on real data

- Whether augmentation flips sign (helps instead of hurts) on natural images.
- Whether end-stopped cells help on real data with object boundaries.
- Whether the sample-efficiency advantage holds on CIFAR-10 / STL-10. Synthetic classes were partly designed to be V1-detectable; real classes are messier.
- Whether the transfer story (CIFAR-10 -> CIFAR-100) gives the same pattern as the synthetic transfer.
- Whether top-down feedback (V2 -> V1 cross-attention) adds anything.

## Overall summary

The architecture is more than the sum of its parts in two distinct ways: biological priors solve the rich-feature-init problem, and JEPA self-supervision solves the low-data-optimization problem the priors otherwise create. Together they produce features that are better at low data AND more transferable than supervised features.

What we have not validated: whether this works on real images.
