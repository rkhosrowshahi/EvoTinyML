"""Train TinyCNN with SGD (lr=0.1)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evotinyml.data import load_dataset
from evotinyml.model import ACTIVATIONS, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyCNN with SGD (lr=0.1).")
    parser.add_argument(
        "--dataset",
        choices=("mnist", "cifar10"),
        required=True,
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--activation",
        choices=ACTIVATIONS,
        default="relu",
        help="Hidden activation: relu or tanh.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=0.1, help="SGD learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="SGD weight decay.")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed.")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache root.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to save the trained model checkpoint (.pt).",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        total_loss += float(loss.item())
        pred = logits.argmax(dim=1)
        correct += int((pred == targets).sum().item())
        n += targets.size(0)
    return total_loss / n, correct / n


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total_loss += float(loss.item()) * targets.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == targets).sum().item())
            n += targets.size(0)

    return total_loss / n, correct / n


def run(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_ds, num_classes = load_dataset(args.dataset, root=args.data_root, train=True)
    test_ds, _ = load_dataset(args.dataset, root=args.data_root, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(args.dataset, num_classes, activation=args.activation).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    print(
        f"dataset={args.dataset}  activation={args.activation}  "
        f"params={model.num_parameters()}  "
        f"epochs={args.epochs}  batch_size={args.batch_size}  "
        f"lr={args.lr}  momentum={args.momentum}  device={device}"
    )

    best_acc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, test_acc)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )
        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}"
        )

    print(f"Best test accuracy: {best_acc:.4f}")

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "dataset": args.dataset,
                "activation": args.activation,
                "num_classes": num_classes,
                "epochs": args.epochs,
                "lr": args.lr,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
                "best_test_acc": best_acc,
                "history": history,
            },
            out_path,
        )
        print(f"Saved checkpoint to {out_path}")

    return model, history


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
