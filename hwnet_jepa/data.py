"""
Real-image data loaders.

Supported sources:
- CIFAR-10 / CIFAR-100 (via torchvision; needs internet on first run)
- STL-10 (via torchvision; 96x96 natural images, designed for SSL)
- ImageFolder (any directory of class-organized images)
- Tiny ImageNet (via direct download of the Stanford archive)
- HuggingFace datasets (via datasets library; useful for ImageNet-1k subsets)

All loaders return a common interface:
- pretrain_images: (N, 3, H, W) float in [0, 1] - unlabeled images for SSL
- (train_x, train_y, val_x, val_y): labeled splits for downstream eval

H is configurable via --image_size; loaders resize as needed.

When running on a machine without internet for these hosts:
- Pre-download datasets to a local path and pass --data_root <path>
- Use --source imagefolder to point at any local class-organized directory
"""
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

try:
    import torchvision
    import torchvision.transforms as T
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


# ----------------------------- Common interface ----------------------------- #


def _to_tensor_dataset(dataset, n_max: Optional[int] = None,
                       image_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a torch Dataset of (image, label) pairs into stacked tensors."""
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),  # -> [0, 1] float
    ])
    images, labels = [], []
    for i in range(len(dataset)):
        if n_max is not None and i >= n_max:
            break
        img, lab = dataset[i]
        if hasattr(img, "convert"):  # PIL image
            img = transform(img)
        elif isinstance(img, torch.Tensor):
            img = T.functional.resize(img, (image_size, image_size))
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
        images.append(img)
        labels.append(lab)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels


# ----------------------------- CIFAR-10/100 ----------------------------- #


def load_cifar10(data_root: str = "./data", image_size: int = 32,
                 n_per_class_train: Optional[int] = None,
                 n_per_class_val: int = 200,
                 n_pretrain_max: Optional[int] = 10000):
    """CIFAR-10 with per-class subset support and an unlabeled pretrain pool.

    n_per_class_train: cap per-class training examples (None = all 5000)
    n_per_class_val: number of test-set examples per class to use as val
    n_pretrain_max: cap unlabeled pretraining pool size (uses train set images)

    Returns:
      pretrain_images: (n_pretrain, 3, H, W)
      train_x, train_y: labeled training data
      val_x, val_y: validation data
      n_classes: 10
    """
    assert HAS_TORCHVISION, "torchvision required for CIFAR-10"
    transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform)

    # Pretraining pool: random subset of training images, unlabeled
    n_pretrain = min(n_pretrain_max or len(train_set), len(train_set))
    pretrain_indices = torch.randperm(len(train_set))[:n_pretrain]
    pretrain_images = torch.stack([train_set[i][0] for i in pretrain_indices], dim=0)

    # Labeled subsets - balanced per class
    train_x, train_y = _balanced_subset_from_dataset(
        train_set, n_classes=10, n_per_class=n_per_class_train)
    val_x, val_y = _balanced_subset_from_dataset(
        test_set, n_classes=10, n_per_class=n_per_class_val)

    return pretrain_images, train_x, train_y, val_x, val_y, 10


def load_cifar100(data_root: str = "./data", image_size: int = 32,
                  n_per_class_train: Optional[int] = None,
                  n_per_class_val: int = 50,
                  n_pretrain_max: Optional[int] = 10000):
    assert HAS_TORCHVISION
    transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    train_set = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=transform)
    n_pretrain = min(n_pretrain_max or len(train_set), len(train_set))
    pretrain_indices = torch.randperm(len(train_set))[:n_pretrain]
    pretrain_images = torch.stack([train_set[i][0] for i in pretrain_indices], dim=0)
    train_x, train_y = _balanced_subset_from_dataset(
        train_set, n_classes=100, n_per_class=n_per_class_train)
    val_x, val_y = _balanced_subset_from_dataset(
        test_set, n_classes=100, n_per_class=n_per_class_val)
    return pretrain_images, train_x, train_y, val_x, val_y, 100


# ----------------------------- STL-10 (designed for SSL) ----------------------------- #


def load_stl10(data_root: str = "./data", image_size: int = 32,
               n_per_class_train: Optional[int] = None,
               n_per_class_val: int = 200,
               n_pretrain_max: Optional[int] = 10000):
    """STL-10: 10 classes at 96x96. Includes a 100K unlabeled split designed for SSL.

    We use the unlabeled split for SSL pretraining (this is the right tool here),
    and the labeled train/test splits for downstream eval.
    """
    assert HAS_TORCHVISION
    transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    unlabeled = torchvision.datasets.STL10(
        root=data_root, split="unlabeled", download=True, transform=transform)
    train_set = torchvision.datasets.STL10(
        root=data_root, split="train", download=True, transform=transform)
    test_set = torchvision.datasets.STL10(
        root=data_root, split="test", download=True, transform=transform)

    n_pretrain = min(n_pretrain_max or len(unlabeled), len(unlabeled))
    pretrain_indices = torch.randperm(len(unlabeled))[:n_pretrain]
    pretrain_images = torch.stack([unlabeled[i][0] for i in pretrain_indices], dim=0)

    train_x, train_y = _balanced_subset_from_dataset(
        train_set, n_classes=10, n_per_class=n_per_class_train)
    val_x, val_y = _balanced_subset_from_dataset(
        test_set, n_classes=10, n_per_class=n_per_class_val)
    return pretrain_images, train_x, train_y, val_x, val_y, 10


# ----------------------------- ImageFolder (custom local data) ----------------------------- #


def load_imagefolder(data_root: str, image_size: int = 32,
                     n_per_class_train: Optional[int] = None,
                     n_per_class_val: int = 100,
                     n_pretrain_max: Optional[int] = 10000,
                     val_split: float = 0.2):
    """Load any directory of class-organized images.

    Expected structure:
      data_root/
        class_name_1/
          image_1.jpg
          ...
        class_name_2/
          ...

    Splits: random val_split fraction held out as validation.
    Pretraining pool: unlabeled subset of training images.
    """
    assert HAS_TORCHVISION
    transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    full_set = torchvision.datasets.ImageFolder(root=data_root, transform=transform)
    n_classes = len(full_set.classes)

    indices_per_class = [[] for _ in range(n_classes)]
    for idx, (_, lab) in enumerate(full_set.samples):
        indices_per_class[lab].append(idx)

    train_indices, val_indices = [], []
    for lab, idxs in enumerate(indices_per_class):
        torch.manual_seed(42 + lab)
        perm = torch.randperm(len(idxs)).tolist()
        n_val = int(len(idxs) * val_split)
        val_indices.extend([idxs[i] for i in perm[:n_val]])
        train_indices.extend([idxs[i] for i in perm[n_val:]])

    # Pretraining pool: subset of train indices, unlabeled
    n_pretrain = min(n_pretrain_max or len(train_indices), len(train_indices))
    pretrain_perm = torch.randperm(len(train_indices))[:n_pretrain]
    pretrain_imgs = []
    for i in pretrain_perm:
        img, _ = full_set[train_indices[i]]
        pretrain_imgs.append(img)
    pretrain_images = torch.stack(pretrain_imgs, dim=0)

    # Build balanced labeled subsets via a thin Subset wrapper
    train_subset = torch.utils.data.Subset(full_set, train_indices)
    val_subset = torch.utils.data.Subset(full_set, val_indices)
    train_x, train_y = _balanced_subset_from_dataset(
        train_subset, n_classes=n_classes, n_per_class=n_per_class_train)
    val_x, val_y = _balanced_subset_from_dataset(
        val_subset, n_classes=n_classes, n_per_class=n_per_class_val)

    return pretrain_images, train_x, train_y, val_x, val_y, n_classes


# ----------------------------- Helpers ----------------------------- #


def _balanced_subset_from_dataset(dataset, n_classes: int,
                                   n_per_class: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a class-balanced (image, label) tensor pair from a torch Dataset."""
    indices_per_class = {c: [] for c in range(n_classes)}
    # Iterate dataset to find per-class indices
    if hasattr(dataset, "targets"):
        # CIFAR-style: targets is a list/array
        targets = dataset.targets
        if isinstance(targets, list):
            for i, t in enumerate(targets):
                indices_per_class[int(t)].append(i)
        else:
            for i, t in enumerate(targets):
                indices_per_class[int(t.item() if torch.is_tensor(t) else t)].append(i)
    elif hasattr(dataset, "labels"):
        # STL10 uses .labels
        for i, t in enumerate(dataset.labels):
            indices_per_class[int(t)].append(i)
    else:
        # Fallback: walk the dataset
        for i in range(len(dataset)):
            _, t = dataset[i]
            indices_per_class[int(t)].append(i)

    selected_indices = []
    for c in range(n_classes):
        idxs = indices_per_class[c]
        if n_per_class is not None:
            idxs = idxs[:n_per_class]
        selected_indices.extend(idxs)

    images, labels = [], []
    for i in selected_indices:
        img, lab = dataset[i]
        images.append(img)
        labels.append(lab)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels


def make_balanced_subset(images: torch.Tensor, labels: torch.Tensor,
                         n_per_class: int, n_classes: int,
                         seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample n_per_class examples of each class from a tensor pair."""
    import numpy as np
    rng = np.random.RandomState(seed)
    selected = []
    for c in range(n_classes):
        cls_idx = (labels == c).nonzero(as_tuple=True)[0].tolist()
        rng.shuffle(cls_idx)
        selected.extend(cls_idx[:n_per_class])
    selected = torch.tensor(selected)
    perm = torch.randperm(len(selected))
    return images[selected[perm]], labels[selected[perm]]


# ----------------------------- Unified loader ----------------------------- #


def load_dataset(source: str, **kwargs):
    """Dispatch to the appropriate loader.

    source: 'synthetic_10class', 'synthetic_transfer', 'cifar10', 'cifar100',
            'stl10', 'imagefolder'
    """
    if source == "synthetic_10class":
        from .data_synthetic import (generate_pretrain_dataset,
                                      generate_classification_dataset)
        n_pretrain = kwargs.get("n_pretrain_max", 2000) or 2000
        n_per_class = kwargs.get("n_per_class_train", 200) or 200
        n_val = kwargs.get("n_per_class_val", 50) or 50
        pretrain = generate_pretrain_dataset(n_images=n_pretrain, seed=7)
        images, labels = generate_classification_dataset(
            n_per_class=n_per_class + n_val, n_classes=10, seed=42)
        # Split per-class
        train_idx, val_idx = [], []
        for c in range(10):
            ci = (labels == c).nonzero(as_tuple=True)[0].tolist()
            train_idx.extend(ci[:n_per_class])
            val_idx.extend(ci[n_per_class:n_per_class + n_val])
        train_x = images[torch.tensor(train_idx)]
        train_y = labels[torch.tensor(train_idx)]
        val_x = images[torch.tensor(val_idx)]
        val_y = labels[torch.tensor(val_idx)]
        return pretrain, train_x, train_y, val_x, val_y, 10

    elif source == "synthetic_transfer":
        from .data_synthetic import (generate_pretrain_dataset,
                                      generate_transfer_dataset)
        n_pretrain = kwargs.get("n_pretrain_max", 2000) or 2000
        n_per_class = kwargs.get("n_per_class_train", 150) or 150
        n_val = kwargs.get("n_per_class_val", 50) or 50
        pretrain = generate_pretrain_dataset(n_images=n_pretrain, seed=7)
        images, labels = generate_transfer_dataset(
            n_per_class=n_per_class + n_val, n_classes=8, seed=123)
        train_idx, val_idx = [], []
        for c in range(8):
            ci = (labels == c).nonzero(as_tuple=True)[0].tolist()
            train_idx.extend(ci[:n_per_class])
            val_idx.extend(ci[n_per_class:n_per_class + n_val])
        train_x = images[torch.tensor(train_idx)]
        train_y = labels[torch.tensor(train_idx)]
        val_x = images[torch.tensor(val_idx)]
        val_y = labels[torch.tensor(val_idx)]
        return pretrain, train_x, train_y, val_x, val_y, 8

    elif source == "cifar10":
        return load_cifar10(**kwargs)
    elif source == "cifar100":
        return load_cifar100(**kwargs)
    elif source == "stl10":
        return load_stl10(**kwargs)
    elif source == "imagefolder":
        return load_imagefolder(**kwargs)
    else:
        raise ValueError(f"Unknown source: {source}")
