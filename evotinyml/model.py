"""Minimal CNN architectures for MNIST and CIFAR-10."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIVATIONS = ("relu", "tanh")


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name!r}. Use one of {ACTIVATIONS}.")


def apply_activation(name: str, x: torch.Tensor) -> torch.Tensor:
    name = name.lower()
    if name == "relu":
        return F.relu(x)
    if name == "tanh":
        return torch.tanh(x)
    raise ValueError(f"Unsupported activation: {name!r}. Use one of {ACTIVATIONS}.")


class TinyCNN_MNIST_4K(nn.Module):
    """Small CNN for MNIST (28x28); matches esde ``cnn_forward`` (c1=4, c2=8).

    Architecture (4,266 params total):
        Conv(1→4, 3x3) -> Act -> MaxPool 2x2   # 40 params  -> (B, 4, 14, 14)
        Conv(4→8, 3x3) -> Act -> MaxPool 2x2   # 296 params -> (B, 8, 7, 7)
        Flatten -> Linear(392 → 10)             # 3,930 params
    """

    def __init__(
        self,
        num_classes: int = 10,
        c1: int = 4,
        c2: int = 8,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.activation_name = activation.lower()
        act1 = get_activation(self.activation_name)
        act2 = get_activation(self.activation_name)

        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1),
            act1,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            act2,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(7 * 7 * c2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TinyCNN_CIFAR_4K(nn.Module):
    """Tiny CNN for CIFAR-10 (32x32); c1=4, c2=12, c3=12.

    Same topology as ``TinyCNN_CIFAR_34K``, with narrower channels to stay under 4K.

    Architecture (3,794 params total):
        Conv(3→4, 3x3)  -> Act -> MaxPool 2x2  # 112 params  -> (B, 4, 16, 16)
        Conv(4→12, 3x3) -> Act -> MaxPool 2x2  # 444 params  -> (B, 12, 8, 8)
        Conv(12→12, 3x3)-> Act -> MaxPool 2x2  # 1,308 params -> (B, 12, 4, 4)
        Flatten -> Linear(192 → 10)             # 1,930 params
    """

    def __init__(
        self,
        num_classes: int = 10,
        c1: int = 4,
        c2: int = 12,
        c3: int = 12,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.activation_name = activation.lower()
        act1 = get_activation(self.activation_name)
        act2 = get_activation(self.activation_name)
        act3 = get_activation(self.activation_name)

        self.features = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1),
            act1,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            act2,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            act3,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(4 * 4 * c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TinyCNN_CIFAR_34K(nn.Module):
    """Small CNN for CIFAR-10 (32x32); c1=16, c2=32, c3=64.

    Architecture (33,834 params total):
        Conv(3→16, 3x3)  -> Act -> MaxPool 2x2  # 448 params   -> (B, 16, 16, 16)
        Conv(16→32, 3x3) -> Act -> MaxPool 2x2  # 4,640 params -> (B, 32, 8, 8)
        Conv(32→64, 3x3) -> Act -> MaxPool 2x2  # 18,496 params -> (B, 64, 4, 4)
        Flatten -> Linear(1024 → 10)             # 10,250 params
    """

    def __init__(
        self,
        num_classes: int = 10,
        c1: int = 16,
        c2: int = 32,
        c3: int = 64,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.activation_name = activation.lower()
        act1 = get_activation(self.activation_name)
        act2 = get_activation(self.activation_name)
        act3 = get_activation(self.activation_name)

        self.features = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1),
            act1,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            act2,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            act3,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(4 * 4 * c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------------------
# ButterflyNet (~4K): butterfly 1x1 mixing, weight-tied stages, no BatchNorm (ES-friendly)
# --------------------------------------------------------------------------------------


class Butterfly1x1(nn.Module):
    """Pointwise mix with O(C log C) params via log2(C) stages of 2x2 blocks.

    Collapses to one fused CxC 1x1 conv per forward (no eval-time weight cache, so
    ES ``set_weights`` updates are always visible).
    """

    def __init__(self, c: int) -> None:
        super().__init__()
        if c & (c - 1) != 0:
            raise ValueError(f"channel count must be a power of two, got {c}")
        self.c = c
        self.stages = c.bit_length() - 1
        theta = torch.rand(self.stages, c // 2) * (2 * torch.pi)
        cos, sin = torch.cos(theta), torch.sin(theta)
        w = torch.stack([cos, -sin, sin, cos], dim=-1).view(self.stages, c // 2, 2, 2)
        self.weight = nn.Parameter(w)

    def dense(self) -> torch.Tensor:
        c = self.c
        m = torch.eye(c, device=self.weight.device, dtype=self.weight.dtype)
        for s in range(self.stages):
            step = 1 << s
            g = c // (2 * step)
            ws = self.weight[s].view(g, step, 2, 2)
            m = torch.einsum("gikn,gkoi->gokn", m.view(g, 2, step, c), ws).reshape(c, c)
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.dense()
        return F.conv2d(x, m.view(self.c, self.c, 1, 1))


class FiLM(nn.Module):
    """Per-channel scale and shift (2C params)."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(c))
        self.b = nn.Parameter(torch.zeros(c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.g.view(1, -1, 1, 1) + self.b.view(1, -1, 1, 1)


class TiedStage(nn.Module):
    """Residual depthwise + butterfly block, unrolled ``iters`` times with shared weights.

    Shared: 9C (depthwise) + 2C·log2(C) (butterfly).
    Per-iter: 4C (two FiLMs). No BatchNorm.
    """

    def __init__(self, c: int, iters: int, activation: str = "relu") -> None:
        super().__init__()
        self.iters = iters
        self.activation_name = activation.lower()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.bf = Butterfly1x1(c)
        self.m1 = nn.ModuleList(FiLM(c) for _ in range(iters))
        self.m2 = nn.ModuleList(FiLM(c) for _ in range(iters))
        for m in self.m2:
            nn.init.zeros_(m.g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.iters):
            y = apply_activation(self.activation_name, self.m1[i](self.dw(x)))
            y = self.m2[i](self.bf(y))
            x = apply_activation(self.activation_name, x + y)
        return x


class Reduce(nn.Module):
    """Stride-2 depthwise downsample; optional free width doubling via avg-pool concat."""

    def __init__(self, c_in: int, expand: bool, activation: str = "relu") -> None:
        super().__init__()
        c_out = c_in * 2 if expand else c_in
        self.expand = expand
        self.activation_name = activation.lower()
        self.dw = nn.Conv2d(c_in, c_in, 3, stride=2, padding=1, groups=c_in, bias=False)
        self.bf = Butterfly1x1(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        if self.expand:
            y = torch.cat([y, F.avg_pool2d(x, 2)], dim=1)
        return apply_activation(self.activation_name, self.bf(y))


class FrozenCosineHead(nn.Module):
    """Fixed orthonormal class vectors; only logit temperature is learned (1 param)."""

    def __init__(self, dim: int, num_classes: int, seed: int = 0) -> None:
        super().__init__()
        if dim < num_classes:
            raise ValueError(f"dim ({dim}) must be >= num_classes ({num_classes})")
        g = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(dim, num_classes, generator=g))
        self.register_buffer("W", F.normalize(q, dim=0))
        self.scale = nn.Parameter(torch.tensor(16.0))

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return self.scale * (F.normalize(d, dim=1) @ self.W)


class ButterflyNet_CIFAR_4K(nn.Module):
    """Butterfly CIFAR-10 CNN without BatchNorm; 3,713 params (w=16, iters=3×3×3).

    Architecture:
        Stem: 1x1 (3→w) + DW 3x3
        TiedStage(w)×3 @ 32x32
        Reduce expand → 2w @ 16x16
        TiedStage(2w)×3 @ 16x16
        Reduce → 2w @ 8x8
        TiedStage(2w)×3 @ 8x8
        Multi-pool descriptor (224-d) → frozen cosine head (1 param)
    """

    def __init__(
        self,
        num_classes: int = 10,
        w: int = 16,
        iters: tuple[int, int, int] = (3, 3, 3),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.activation_name = activation.lower()
        act = get_activation(self.activation_name)

        self.stem = nn.Sequential(
            nn.Conv2d(3, w, 1, bias=False),
            act,
            nn.Conv2d(w, w, 3, padding=1, groups=w, bias=False),
            get_activation(self.activation_name),
        )
        self.stage1 = TiedStage(w, iters[0], activation=self.activation_name)
        self.red1 = Reduce(w, expand=True, activation=self.activation_name)
        self.stage2 = TiedStage(2 * w, iters[1], activation=self.activation_name)
        self.red2 = Reduce(2 * w, expand=False, activation=self.activation_name)
        self.stage3 = TiedStage(2 * w, iters[2], activation=self.activation_name)

        dim = 2 * w + 2 * w * 6  # 32 + 192 = 224 for w=16
        self.head = FrozenCosineHead(dim, num_classes)

    def descriptor(self, f2: torch.Tensor, f3: torch.Tensor) -> torch.Tensor:
        parts = [
            F.adaptive_avg_pool2d(f2, 1).flatten(1),
            F.adaptive_avg_pool2d(f3, 1).flatten(1),
            F.adaptive_max_pool2d(f3, 1).flatten(1),
            F.adaptive_avg_pool2d(f3, 2).flatten(1),
        ]
        return torch.cat(parts, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.red1(x)
        f2 = self.stage2(x)
        f3 = self.stage3(self.red2(f2))
        return self.head(self.descriptor(f2, f3))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


MODELS: dict[str, type[nn.Module]] = {
    "tinycnn_mnist_4k": TinyCNN_MNIST_4K,
    "tinycnn_cifar_4k": TinyCNN_CIFAR_4K,
    "tinycnn_cifar_34k": TinyCNN_CIFAR_34K,
    "butterflynet_cifar_4k": ButterflyNet_CIFAR_4K,
}

MODEL_DATASETS: dict[str, frozenset[str]] = {
    "tinycnn_mnist_4k": frozenset({"mnist", "mnist_2cls"}),
    "tinycnn_cifar_4k": frozenset({"cifar10"}),
    "tinycnn_cifar_34k": frozenset({"cifar10"}),
    "butterflynet_cifar_4k": frozenset({"cifar10"}),
}

DEFAULT_MODELS: dict[str, str] = {
    "mnist": "tinycnn_mnist_4k",
    "mnist_2cls": "tinycnn_mnist_4k",
    "cifar10": "tinycnn_cifar_34k",
}


def resolve_model_name(dataset: str, model: str | None = None) -> str:
    """Return a registered model name (lowercase), defaulting from ``dataset`` when unset."""
    dataset = dataset.lower()
    name = (model.lower() if model else None) or DEFAULT_MODELS.get(dataset)
    if name is None:
        raise ValueError(
            f"Unsupported dataset: {dataset!r}. Use 'mnist', 'mnist_2cls', or 'cifar10'."
        )
    if name not in MODELS:
        raise ValueError(
            f"Unsupported model: {model!r}. Use one of {sorted(MODELS)}."
        )
    allowed = MODEL_DATASETS[name]
    if dataset not in allowed:
        raise ValueError(
            f"Model {name!r} is incompatible with dataset {dataset!r}; "
            f"expected one of {sorted(allowed)}."
        )
    return name


def build_model(
    dataset: str,
    num_classes: int,
    activation: str = "relu",
    model: str | None = None,
) -> nn.Module:
    """Build a TinyCNN by registered ``model`` name (see ``MODELS``; case-insensitive)."""
    name = resolve_model_name(dataset, model)
    return MODELS[name](num_classes=num_classes, activation=activation)
