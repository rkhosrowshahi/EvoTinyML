"""Dataset loading and class-balanced batch sampling."""

from __future__ import annotations

import ssl

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

# macOS / Homebrew Python often lacks system CA certs; torchvision downloads fail
# with CERTIFICATE_VERIFY_FAILED unless the default HTTPS context uses certifi.
try:
    import certifi

    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where()
    )
except ImportError:
    pass


DATASETS = ("mnist", "mnist_2cls", "cifar10")


def _filter_labels(dataset: Dataset, keep_labels: tuple[int, ...]) -> Subset:
    """Keep only samples whose label is in ``keep_labels`` (labels unchanged)."""
    labels = _labels_of(dataset)
    keep = set(int(c) for c in keep_labels)
    indices = np.flatnonzero(np.isin(labels, list(keep))).tolist()
    if not indices:
        raise ValueError(f"No samples found for labels {sorted(keep)}.")
    return Subset(dataset, indices)


def load_dataset(name: str, root: str = "./data", train: bool = True) -> tuple[Dataset, int]:
    """Load a supported image dataset with a simple ToTensor transform.

    Returns ``(dataset, num_classes)``. ``mnist_2cls`` keeps digits {0, 1}
    (labels already in ``{0, 1}``), so CWRM / class-balanced sampling use
    ``n_obj = num_classes = 2``.
    """
    name = name.lower()
    transform = transforms.ToTensor()
    if name == "mnist":
        dataset: Dataset = datasets.MNIST(
            root=root, train=train, download=True, transform=transform
        )
        return dataset, 10
    if name == "mnist_2cls":
        full = datasets.MNIST(root=root, train=train, download=True, transform=transform)
        return _filter_labels(full, (0, 1)), 2
    if name == "cifar10":
        dataset = datasets.CIFAR10(
            root=root, train=train, download=True, transform=transform
        )
        return dataset, 10
    raise ValueError(f"Unsupported dataset: {name!r}. Use one of {DATASETS}.")


def _labels_of(dataset: Dataset) -> np.ndarray:
    """Extract labels as a numpy array for common torchvision datasets / subsets."""
    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = np.asarray(dataset.indices)
        if hasattr(base, "targets"):
            return np.asarray(base.targets)[indices]
        if hasattr(base, "labels"):
            return np.asarray(base.labels)[indices]

    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)

    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))])


EVAL_MODES = ("single", "multi")  # pool size per generation; sampler draws each gen


def _materialize_batch(
    dataset: Dataset, indices: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    images = []
    targets = []
    for i in indices:
        x, y = dataset[int(i)]
        images.append(x)
        targets.append(y)
    return torch.stack(images, dim=0), torch.tensor(targets, dtype=torch.long)


class RandomBatchSampler:
    """Sample mini-batches uniformly at random (PyTorch ``RandomSampler`` style).

    Draws ``batch_size`` indices without replacement. Intended for metrics
    computed on pooled predictions across one or more batches.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_classes: int,
        seed: int | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if batch_size > len(dataset):
            raise ValueError(
                f"batch_size ({batch_size}) cannot exceed dataset size ({len(dataset)})."
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.rng = np.random.default_rng(seed)
        self.all_indices = np.arange(len(dataset))

    def sample_indices(self) -> np.ndarray:
        return self.rng.choice(self.all_indices, size=self.batch_size, replace=False)

    def sample_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _materialize_batch(self.dataset, self.sample_indices())

    def sample_batches(self, n_batches: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [self.sample_batch() for _ in range(max(1, int(n_batches)))]


class ClassBalancedSampler:
    """Sample batches that try to include every class at least once.

    With ``num_classes=10`` and ``batch_size=32``, each batch contains one
    mandatory sample per class, then the remaining 22 slots are filled
    uniformly at random (without replacement within the batch).
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_classes: int,
        seed: int | None = None,
    ) -> None:
        if batch_size < num_classes:
            raise ValueError(
                f"batch_size ({batch_size}) must be >= num_classes ({num_classes}) "
                "to guarantee at least one sample per class."
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.rng = np.random.default_rng(seed)

        labels = _labels_of(dataset)
        self.class_indices: dict[int, np.ndarray] = {
            c: np.flatnonzero(labels == c) for c in range(num_classes)
        }
        for c, idxs in self.class_indices.items():
            if len(idxs) == 0:
                raise ValueError(f"Dataset has no samples for class {c}.")

        self.all_indices = np.arange(len(dataset))

    def sample_indices(self) -> np.ndarray:
        """Return index array of length ``batch_size`` with all classes present."""
        chosen: list[int] = []
        chosen_set: set[int] = set()

        # Guarantee one sample from each class.
        for c in range(self.num_classes):
            idx = int(self.rng.choice(self.class_indices[c]))
            chosen.append(idx)
            chosen_set.add(idx)

        # Fill remaining slots without replacement.
        remaining = self.batch_size - self.num_classes
        if remaining > 0:
            pool = np.setdiff1d(
                self.all_indices,
                np.fromiter(chosen_set, dtype=np.int64),
                assume_unique=False,
            )
            if len(pool) < remaining:
                extra = self.rng.choice(self.all_indices, size=remaining, replace=True)
            else:
                extra = self.rng.choice(pool, size=remaining, replace=False)
            chosen.extend(int(i) for i in extra)

        self.rng.shuffle(chosen)
        return np.asarray(chosen, dtype=np.int64)

    def sample_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize one balanced batch of ``(inputs, targets)``."""
        return _materialize_batch(self.dataset, self.sample_indices())

    def sample_batches(self, n_batches: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [self.sample_batch() for _ in range(max(1, int(n_batches)))]
