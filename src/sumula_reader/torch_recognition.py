from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .evaluation import PredictionRecord, choose_acceptance_threshold
from .models import DecisionStatus
from .recognition import FOUL_LABELS, RecognitionCandidate, RecognitionResult
from .scoring import colored_ink_masks


BLANK_DIGIT = 10
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 64


@dataclass(frozen=True, slots=True)
class SequenceTrainingConfig:
    data_dir: Path
    output: Path
    manifest: Path | None = None
    epochs: int = 5
    batch_size: int = 128
    learning_rate: float = 1e-3
    synthetic_train_samples: int = 20_000
    synthetic_test_samples: int = 4_000
    seed: int = 20261001


def _torch() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        from torch.utils.data import ConcatDataset, DataLoader, Dataset
        from torchvision import datasets, transforms
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "Reconhecimento manuscrito requer torch e torchvision compativeis"
        ) from exc
    return torch, Dataset, ConcatDataset, DataLoader, (datasets, transforms)


class JerseySequenceCNN:
    @staticmethod
    def build() -> Any:
        torch, _, _, _, _ = _torch()
        nn = torch.nn

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.AdaptiveAvgPool2d((4, 8)),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64 * 4 * 8, 128),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                    nn.Linear(128, 2 * 11),
                )

            def forward(self, image: Any) -> Any:
                return self.classifier(self.features(image)).reshape(-1, 2, 11)

        return Model()


class FoulSymbolCNN:
    @staticmethod
    def build(labels: int = len(FOUL_LABELS)) -> Any:
        torch, _, _, _, _ = _torch()
        nn = torch.nn
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 8)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, labels),
        )


def prepare_handwriting_image(image: np.ndarray) -> Any:
    torch, _, _, _, _ = _torch()
    red, blue = colored_ink_masks(np.asarray(image, dtype=np.uint8))
    mask = red | blue
    if not mask.any():
        rgb = np.asarray(image, dtype=np.uint8)
        gray = rgb.mean(axis=2) if rgb.ndim == 3 else rgb
        mask = gray < 180
    yy, xx = np.nonzero(mask)
    canvas = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
    if len(xx):
        cropped = (mask[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1] * 255).astype(
            np.uint8
        )
        pil = Image.fromarray(cropped, mode="L")
        scale = min(54 / max(pil.width, 1), 26 / max(pil.height, 1))
        size = (max(1, round(pil.width * scale)), max(1, round(pil.height * scale)))
        pil = pil.resize(size, Image.Resampling.BILINEAR)
        x0 = (IMAGE_WIDTH - pil.width) // 2
        y0 = (IMAGE_HEIGHT - pil.height) // 2
        canvas[y0 : y0 + pil.height, x0 : x0 + pil.width] = (
            np.asarray(pil, dtype=np.float32) / 255.0
        )
    return torch.from_numpy(canvas).unsqueeze(0).sub(0.5).div(0.5)


def _encode_jersey(label: str) -> tuple[int, int]:
    normalized = label.strip()
    if not normalized.isdigit() or len(normalized) not in {1, 2}:
        raise ValueError(f"Camisa deve conter um ou dois digitos: {label!r}")
    if len(normalized) == 1:
        return int(normalized), BLANK_DIGIT
    return int(normalized[0]), int(normalized[1])


def _jersey_probability(probabilities: Any, label: str) -> float:
    first, second = _encode_jersey(label)
    return float(probabilities[0, first] * probabilities[1, second])


def _manifest_items(manifest: Path | None, field_type: str, split: str) -> list[dict]:
    if manifest is None or not manifest.exists():
        return []
    items: list[dict] = []
    for item in (
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ):
        review_state = item.get("review_state")
        reviewed_ok = (
            review_state in {"accepted", "adjusted"}
            if review_state is not None
            else item.get("status") == "labeled"
        )
        if (
            item.get("field_type") == field_type
            and reviewed_ok
            and item.get("split") == split
            and item.get("label") is not None
        ):
            items.append(item)
    return items


def _manifest_dataset(manifest: Path, items: list[dict], labels: Sequence[str] | None):
    torch, Dataset, _, _, _ = _torch()
    label_to_index = {label: index for index, label in enumerate(labels or ())}

    class ManifestDataset(Dataset):
        def __len__(self) -> int:
            return len(items)

        def __getitem__(self, index: int):
            item = items[index]
            path = manifest.parent / item["crop_path"]
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                expected_hash = item.get("crop_image_hash")
                if expected_hash is not None:
                    actual_hash = hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
                    if actual_hash != str(expected_hash):
                        raise ValueError(
                            f"Crop revisado obsoleto: {item.get('crop_id') or path.name}"
                        )
                bbox = item.get("bbox_revised")
                if isinstance(bbox, list) and len(bbox) == 4:
                    rgb = rgb.crop(tuple(int(value) for value in bbox))
                array = np.asarray(rgb)
            tensor = prepare_handwriting_image(array)
            label = str(item["label"])
            if labels is None:
                target = torch.tensor(_encode_jersey(label), dtype=torch.long)
            else:
                target = torch.tensor(label_to_index[label], dtype=torch.long)
            return tensor, target

    return ManifestDataset()


def _synthetic_jersey_dataset(
    data_dir: Path,
    *,
    train: bool,
    samples: int,
    seed: int,
):
    torch, Dataset, _, _, vision = _torch()
    datasets, transforms = vision
    base = datasets.EMNIST(
        root=data_dir,
        split="digits",
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )
    targets = np.asarray(base.targets)
    indexes = {digit: np.flatnonzero(targets == digit).tolist() for digit in range(10)}

    class SyntheticJerseys(Dataset):
        def __len__(self) -> int:
            return samples

        def __getitem__(self, index: int):
            rng = random.Random(seed + index + (0 if train else 10_000_000))
            if rng.random() < 0.40:
                label = str(rng.randint(0, 9))
            else:
                label = str(rng.randint(10, 99))
            canvas = torch.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.float32)
            positions = [20] if len(label) == 1 else [7, 33]
            for character, x0 in zip(label, positions):
                digit = int(character)
                sample_index = rng.choice(indexes[digit])
                image, _ = base[sample_index]
                resized = torch.nn.functional.interpolate(
                    image.unsqueeze(0),
                    size=(24, 24),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                x = max(0, min(IMAGE_WIDTH - 24, x0 + rng.randint(-3, 3)))
                y = max(0, min(IMAGE_HEIGHT - 24, 4 + rng.randint(-3, 3)))
                canvas[:, y : y + 24, x : x + 24] = torch.maximum(
                    canvas[:, y : y + 24, x : x + 24], resized
                )
            canvas = canvas.sub(0.5).div(0.5)
            return canvas, torch.tensor(_encode_jersey(label), dtype=torch.long)

    return SyntheticJerseys()


def _font_paths() -> list[Path]:
    candidates = [
        Path("C:/Windows/Fonts/segoepr.ttf"),
        Path("C:/Windows/Fonts/comic.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    return [path for path in candidates if path.exists()]


def _synthetic_foul_dataset(*, train: bool, samples: int, seed: int):
    torch, Dataset, _, _, _ = _torch()
    fonts = _font_paths()

    class SyntheticFouls(Dataset):
        def __len__(self) -> int:
            return samples

        def __getitem__(self, index: int):
            rng = random.Random(seed + index + (0 if train else 20_000_000))
            label_index = rng.randrange(len(FOUL_LABELS))
            label = FOUL_LABELS[label_index]
            image = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), 0)
            size = rng.randint(19, 27)
            if fonts:
                font = ImageFont.truetype(str(rng.choice(fonts)), size=size)
            else:
                font = ImageFont.load_default(size=size)
            draw = ImageDraw.Draw(image)
            bounds = draw.textbbox((0, 0), label, font=font)
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            x = max(0, (IMAGE_WIDTH - width) // 2 + rng.randint(-4, 4))
            y = max(0, (IMAGE_HEIGHT - height) // 2 - bounds[1] + rng.randint(-3, 3))
            draw.text((x, y), label, fill=255, font=font, stroke_width=rng.randint(0, 1))
            image = image.rotate(rng.uniform(-8, 8), resample=Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array.copy()).unsqueeze(0).sub(0.5).div(0.5)
            return tensor, torch.tensor(label_index, dtype=torch.long)

    return SyntheticFouls()


def _train_model(
    model: Any,
    train_dataset: Any,
    test_dataset: Any,
    config: SequenceTrainingConfig,
    *,
    task: str,
    labels: Sequence[str],
) -> dict[str, Any]:
    torch, _, _, DataLoader, _ = _torch()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(config.seed),
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            if task == "jersey":
                loss = loss_function(logits[:, 0], targets[:, 0]) + loss_function(
                    logits[:, 1], targets[:, 1]
                )
                matches = (logits.argmax(dim=2) == targets).all(dim=1)
            else:
                loss = loss_function(logits, targets)
                matches = logits.argmax(dim=1) == targets
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * targets.shape[0]
            total += targets.shape[0]
            correct += int(matches.sum().item())
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / max(total, 1),
                "train_accuracy": correct / max(total, 1),
            }
        )

    model.eval()
    predictions: list[PredictionRecord] = []
    with torch.no_grad():
        for images, targets in test_loader:
            logits = model(images.to(device)).cpu()
            if task == "jersey":
                probabilities = logits.softmax(dim=2)
                predicted_slots = probabilities.argmax(dim=2)
                for probability, predicted, expected in zip(
                    probabilities, predicted_slots, targets
                ):
                    predicted_label = (
                        str(int(predicted[0]))
                        if int(predicted[1]) == BLANK_DIGIT
                        else f"{int(predicted[0])}{int(predicted[1])}"
                    )
                    expected_label = (
                        str(int(expected[0]))
                        if int(expected[1]) == BLANK_DIGIT
                        else f"{int(expected[0])}{int(expected[1])}"
                    )
                    confidence = math.sqrt(
                        float(probability[0, predicted[0]] * probability[1, predicted[1]])
                    )
                    predictions.append(
                        PredictionRecord(
                            expected_label,
                            predicted_label,
                            confidence,
                            "synthetic-test",
                        )
                    )
            else:
                probabilities = logits.softmax(dim=1)
                confidence, predicted = probabilities.max(dim=1)
                for score, predicted_index, expected_index in zip(
                    confidence, predicted, targets
                ):
                    predictions.append(
                        PredictionRecord(
                            labels[int(expected_index)],
                            labels[int(predicted_index)],
                            float(score),
                            "synthetic-test",
                        )
                    )
    correct = sum(record.correct for record in predictions)
    threshold_metrics = choose_acceptance_threshold(
        predictions,
        max_accepted_error_rate=0.01,
        min_automation_rate=0.10,
    )
    threshold = max(0.70, threshold_metrics.threshold) if threshold_metrics else 1.0
    metrics = {
        "samples": len(predictions),
        "accuracy": correct / len(predictions) if predictions else 0.0,
        "acceptance_threshold": threshold,
        "automation_rate": (
            threshold_metrics.automation_rate if threshold_metrics else 0.0
        ),
        "accepted_error_rate": (
            threshold_metrics.accepted_error_rate if threshold_metrics else 0.0
        ),
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "task": task,
            "labels": list(labels),
            "state_dict": model.state_dict(),
            "threshold": threshold,
            "config": {
                **asdict(config),
                "data_dir": str(config.data_dir),
                "output": str(config.output),
                "manifest": str(config.manifest) if config.manifest else None,
            },
            "history": history,
            "metrics": metrics,
        },
        config.output,
    )
    return {
        "model": str(config.output),
        "device": str(device),
        "history": history,
        "metrics": metrics,
    }


def train_jersey_sequences(config: SequenceTrainingConfig) -> dict[str, Any]:
    _, _, ConcatDataset, _, _ = _torch()
    synthetic_train = _synthetic_jersey_dataset(
        config.data_dir,
        train=True,
        samples=config.synthetic_train_samples,
        seed=config.seed,
    )
    synthetic_test = _synthetic_jersey_dataset(
        config.data_dir,
        train=False,
        samples=config.synthetic_test_samples,
        seed=config.seed,
    )
    train_sets = [synthetic_train]
    test_sets = [synthetic_test]
    if config.manifest is not None:
        train_items = _manifest_items(config.manifest, "scoring_event", "train")
        test_items = _manifest_items(config.manifest, "scoring_event", "test")
        if train_items:
            train_sets.append(_manifest_dataset(config.manifest, train_items, None))
        if test_items:
            test_sets.append(_manifest_dataset(config.manifest, test_items, None))
    return _train_model(
        JerseySequenceCNN.build(),
        ConcatDataset(train_sets),
        ConcatDataset(test_sets),
        config,
        task="jersey",
        labels=[str(value) for value in range(100)],
    )


def train_foul_symbols(config: SequenceTrainingConfig) -> dict[str, Any]:
    _, _, ConcatDataset, _, _ = _torch()
    train_sets = [
        _synthetic_foul_dataset(
            train=True,
            samples=config.synthetic_train_samples,
            seed=config.seed,
        )
    ]
    test_sets = [
        _synthetic_foul_dataset(
            train=False,
            samples=config.synthetic_test_samples,
            seed=config.seed,
        )
    ]
    if config.manifest is not None:
        train_items = _manifest_items(config.manifest, "foul_event", "train")
        test_items = _manifest_items(config.manifest, "foul_event", "test")
        if train_items:
            train_sets.append(
                _manifest_dataset(config.manifest, train_items, FOUL_LABELS)
            )
        if test_items:
            test_sets.append(_manifest_dataset(config.manifest, test_items, FOUL_LABELS))
    return _train_model(
        FoulSymbolCNN.build(),
        ConcatDataset(train_sets),
        ConcatDataset(test_sets),
        config,
        task="foul",
        labels=FOUL_LABELS,
    )


class TorchHandwritingRecognizer:
    def __init__(self, model_dir: str | Path) -> None:
        torch, _, _, _, _ = _torch()
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        root = Path(model_dir)
        self.model_fingerprints = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (root / "jersey.pt", root / "foul.pt")
            if path.exists()
        }
        self.jersey = self._load(root / "jersey.pt", "jersey")
        self.foul = self._load(root / "foul.pt", "foul")
        if self.jersey is None and self.foul is None:
            raise FileNotFoundError(f"Nenhum modelo jersey.pt ou foul.pt em {root}")

    def _load(self, path: Path, task: str):
        if not path.exists():
            return None
        checkpoint = self.torch.load(path, map_location=self.device, weights_only=False)
        model = (
            JerseySequenceCNN.build()
            if task == "jersey"
            else FoulSymbolCNN.build(len(checkpoint["labels"]))
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device).eval()
        return model, checkpoint

    def recognize(
        self,
        image: np.ndarray,
        *,
        field_type: str,
        allowed_labels: Sequence[str] | None = None,
        writer_id: str | None = None,
    ) -> RecognitionResult:
        del writer_id
        tensor = prepare_handwriting_image(image).unsqueeze(0).to(self.device)
        if field_type == "jersey" and self.jersey is not None:
            model, checkpoint = self.jersey
            with self.torch.no_grad():
                probabilities = model(tensor)[0].softmax(dim=1).cpu()
            labels = list(allowed_labels or checkpoint["labels"])
            scored = [
                RecognitionCandidate(label, _jersey_probability(probabilities, label))
                for label in labels
                if label.isdigit() and 1 <= len(label) <= 2
            ]
        elif field_type in {"foul_event", "foul_symbol"} and self.foul is not None:
            model, checkpoint = self.foul
            with self.torch.no_grad():
                probabilities = model(tensor)[0].softmax(dim=0).cpu()
            allowed = set(allowed_labels or checkpoint["labels"])
            scored = [
                RecognitionCandidate(label, float(probabilities[index]))
                for index, label in enumerate(checkpoint["labels"])
                if label in allowed
            ]
        else:
            return RecognitionResult(
                value=None,
                confidence=0.0,
                status=DecisionStatus.REVIEW,
            )
        total = sum(candidate.probability for candidate in scored)
        if total <= 0:
            return RecognitionResult(
                value=None,
                confidence=0.0,
                status=DecisionStatus.REVIEW,
            )
        candidates = sorted(
            (
                RecognitionCandidate(item.label, item.probability / total)
                for item in scored
            ),
            key=lambda item: item.probability,
            reverse=True,
        )
        best = candidates[0]
        checkpoint = self.jersey[1] if field_type == "jersey" else self.foul[1]
        threshold = float(checkpoint.get("threshold", 0.90))
        return RecognitionResult(
            value=best.label,
            confidence=round(best.probability, 4),
            candidates=candidates[:10],
            status=(
                DecisionStatus.ACCEPTED
                if best.probability >= threshold
                else DecisionStatus.REVIEW
            ),
        )
