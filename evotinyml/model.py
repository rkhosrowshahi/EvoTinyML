"""Minimal CNN architectures for MNIST and CIFAR-10."""

from __future__ import annotations

import torch
import torch.nn as nn


ACTIVATIONS = ("relu", "tanh")


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name!r}. Use one of {ACTIVATIONS}.")


class TinyCNN_MNIST(nn.Module):
    """Small CNN for MNIST (28x28); matches esde ``cnn_forward`` (c1=4, c2=8).

    Architecture (~4,266 params):
        Conv(SAME 3x3) -> Activation -> MaxPool 2x2  # -> (B, c1, 14, 14)
        Conv(SAME 3x3) -> Activation -> MaxPool 2x2  # -> (B, c2, 7, 7)
        Flatten -> Linear(7*7*c2 -> num_classes)
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


class TinyCNN_CIFAR(nn.Module):
    """Small CNN for CIFAR-10 (32x32).

    Architecture:
        Conv(SAME 3x3) -> Activation -> MaxPool 2x2  # -> (B, c1, 16, 16)
        Conv(SAME 3x3) -> Activation -> MaxPool 2x2  # -> (B, c2, 8, 8)
        Conv(SAME 3x3) -> Activation -> MaxPool 2x2  # -> (B, c3, 4, 4)
        Flatten -> Linear(4*4*c3 -> num_classes)
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


def build_model(
    dataset: str,
    num_classes: int,
    activation: str = "relu",
) -> nn.Module:
    dataset = dataset.lower()
    if dataset in {"mnist", "mnist_2cls"}:
        return TinyCNN_MNIST(num_classes=num_classes, activation=activation)
    if dataset == "cifar10":
        return TinyCNN_CIFAR(num_classes=num_classes, activation=activation)
    raise ValueError(
        f"Unsupported dataset: {dataset!r}. Use 'mnist', 'mnist_2cls', or 'cifar10'."
    )
