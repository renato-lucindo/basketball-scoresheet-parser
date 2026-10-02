from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any


@dataclass(frozen=True, slots=True)
class DigitTrainingConfig:
    data_dir: Path
    output: Path
    epochs: int = 5
    batch_size: int = 128
    learning_rate: float = 1e-3
    seed: int = 20260930
    max_train_samples: int | None = None
    max_test_samples: int | None = None


class DigitCNN:
    """Wrapper que constroi a CNN sem importar torch no pacote base."""

    @staticmethod
    def build() -> Any:
        import torch.nn as nn

        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 10),
        )


def _torch_modules() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, transforms
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "O treinamento requer as dependencias opcionais de ML. "
            "Instale com: python -m pip install -e \".[ml]\""
        ) from exc
    return torch, DataLoader, Subset, (datasets, transforms)


def _emnist_loaders(config: DigitTrainingConfig) -> tuple[Any, Any]:
    torch, DataLoader, Subset, vision = _torch_modules()
    datasets, transforms = vision
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train = datasets.EMNIST(
        root=config.data_dir,
        split="digits",
        train=True,
        download=True,
        transform=transform,
    )
    test = datasets.EMNIST(
        root=config.data_dir,
        split="digits",
        train=False,
        download=True,
        transform=transform,
    )
    if config.max_train_samples is not None:
        train = Subset(train, range(min(config.max_train_samples, len(train))))
    if config.max_test_samples is not None:
        test = Subset(test, range(min(config.max_test_samples, len(test))))

    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    test_loader = DataLoader(
        test,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader


def evaluate_digit_model(model: Any, loader: Any, device: Any) -> dict[str, Any]:
    torch, _, _, _ = _torch_modules()
    model.eval()
    total = 0
    correct = 0
    per_class_total = [0] * 10
    per_class_correct = [0] * 10
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            total += labels.numel()
            correct += int((predictions == labels).sum().item())
            for label in range(10):
                mask = labels == label
                count = int(mask.sum().item())
                per_class_total[label] += count
                per_class_correct[label] += int(
                    ((predictions == labels) & mask).sum().item()
                )
    return {
        "samples": total,
        "accuracy": correct / total if total else 0.0,
        "accuracy_by_digit": {
            str(label): (
                per_class_correct[label] / per_class_total[label]
                if per_class_total[label]
                else 0.0
            )
            for label in range(10)
        },
    }


def train_emnist_digits(config: DigitTrainingConfig) -> dict[str, Any]:
    if config.epochs < 1:
        raise ValueError("epochs deve ser maior ou igual a 1")
    if config.batch_size < 1:
        raise ValueError("batch_size deve ser maior ou igual a 1")

    torch, _, _, _ = _torch_modules()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    train_loader, test_loader = _emnist_loaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN.build().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.numel()
            total += labels.numel()
            correct += int((logits.argmax(dim=1) == labels).sum().item())
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / total if total else 0.0,
                "train_accuracy": correct / total if total else 0.0,
            }
        )

    metrics = evaluate_digit_model(model, test_loader, device)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "emnist_digit_cnn",
            "labels": [str(value) for value in range(10)],
            "normalization": {"mean": [0.5], "std": [0.5]},
            "state_dict": model.state_dict(),
            "training": {
                **asdict(config),
                "data_dir": str(config.data_dir),
                "output": str(config.output),
                "device": str(device),
                "history": history,
            },
            "metrics": metrics,
        },
        config.output,
    )
    return {
        "model": str(config.output),
        "device": str(device),
        "history": history,
        "test": metrics,
    }
