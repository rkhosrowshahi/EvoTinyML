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


class TinyCNN(nn.Module):
    """Smallest practical CNN for MNIST / CIFAR-10.

    Architecture:
        Conv(in -> 8, 3x3, stride 2) -> Activation
        Conv(8 -> 16, 3x3, stride 2) -> Activation
        AdaptiveAvgPool(1) -> Linear(16 -> num_classes)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 10,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.activation_name = activation.lower()
        act1 = get_activation(self.activation_name)
        act2 = get_activation(self.activation_name)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1),
            act1,
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            act2,
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(16, num_classes)

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
) -> TinyCNN:
    dataset = dataset.lower()
    if dataset in {"mnist", "mnist_2cls"}:
        return TinyCNN(in_channels=1, num_classes=num_classes, activation=activation)
    if dataset == "cifar10":
        return TinyCNN(in_channels=3, num_classes=num_classes, activation=activation)
    raise ValueError(
        f"Unsupported dataset: {dataset!r}. Use 'mnist', 'mnist_2cls', or 'cifar10'."
    )