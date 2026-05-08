"""
HW-Net + I-JEPA: biologically-structured visual encoder with self-supervised
predictive pretraining.

Components:
- bio_layers: Hubel-Wiesel-inspired front end (DoG retinal, Gabor V1, energy
  pooling, divisive normalization, end-stopped cells).
- networks: HWNet (supervised baseline using the bio frontend), BaselineCNN
  (parameter-matched standard CNN), SparseAutoencoder (Olshausen-Field probe).
- jepa: I-JEPA (Assran et al. 2023) on top of the bio frontend.
- readout: linear/attention-pool/kNN readouts and AttnPoolClassifier.
- data: synthetic and real-image data loaders with a common interface.
- train: training loops for supervised and JEPA pretraining.
- augment: image augmentation utilities for JEPA pretraining.
"""

__version__ = "0.1.0"
