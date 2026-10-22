"""Train and evaluate a CNN on the CNN HDF5 dataset.

The expected HDF5 layout is the one produced by
``CNN_Training_set_builder.py``:

    images: uint8, ``(samples, height, width, channels)``
    labels: uint8, ``(samples,)``

Example:

    .venv/bin/python CNN_test.py --dataset DataSets/cnn_training_test.h5 \
        --epochs 10 --batch-size 256
"""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class TrainingConfig:
    """Command-line configuration used for one training run."""

    dataset: Path = Path("DataSets/cnn_training.h5")
    model: Path = Path("models/CNN.pt")
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-3
    validation_split: float = 0.2
    seed: int = 42
    num_workers: int = 0


class HDF5ImageDataset(Dataset[tuple[Tensor, Tensor]]):
    """Read normalized CHW images and labels from HDF5 on demand."""

    def __init__(self, path: Path, indices: np.ndarray) -> None:
        self.path = path
        self.indices = np.asarray(indices, dtype=np.int64)
        self._file: h5py.File | None = None
        self._images: h5py.Dataset | None = None
        self._labels: h5py.Dataset | None = None

    def _open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
            self._images = self._file["images"]
            self._labels = self._file["labels"]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[Tensor, Tensor]:
        self._open()
        assert self._images is not None and self._labels is not None
        image = np.asarray(self._images[self.indices[position]], dtype=np.float32)
        label = float(self._labels[self.indices[position]])
        # The dataset stores NHWC images; PyTorch convolution expects NCHW.
        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)) / 255.0)
        return image_tensor, torch.tensor(label, dtype=torch.float32)

    def __del__(self) -> None:
        if self._file is not None:
            self._file.close()


class AssignmentCNN(nn.Module):
    """Small CNN suitable for the 40x40, three-channel assignment maps."""

    def __init__(self, input_channels: int, image_height: int, image_width: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            # Keep the spatial grid: the label depends on the task/agent
            # relationship, not just on which objects occur in the image.
            nn.AdaptiveAvgPool2d((image_height // 4, image_width // 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (image_height // 4) * (image_width // 4) + 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        self.input_shape = (input_channels, image_height, image_width)
        self.image_height = image_height
        self.image_width = image_width

    def forward(self, inputs: Tensor) -> Tensor:
        feature_map = self.features(inputs)
        task_positions = self._marker_positions(inputs[:, 1])
        agent_positions = self._marker_positions(inputs[:, 2])
        relation = torch.cat(
            (
                (task_positions - agent_positions)
                / inputs.new_tensor((self.image_width, self.image_height)),
                torch.linalg.vector_norm(
                    task_positions - agent_positions, dim=1, keepdim=True
                )
                / inputs.new_tensor(max(self.image_width, self.image_height)),
            ),
            dim=1,
        )
        classifier_input = torch.cat((feature_map.flatten(start_dim=1), relation), dim=1)
        return self.classifier(classifier_input).squeeze(1)

    @staticmethod
    def _marker_positions(channel: Tensor) -> Tensor:
        """Return normalized x/y coordinates of each channel's highlighted pixel."""
        height, width = channel.shape[-2:]
        indices = channel.flatten(start_dim=1).argmax(dim=1)
        y = torch.div(indices, width, rounding_mode="floor")
        x = indices.remainder(width)
        return torch.stack((x.float(), y.float()), dim=1)


def set_seed(seed: int) -> None:
    """Make data splitting and training reproducible where possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inspect_dataset(path: Path) -> tuple[int, int, int, np.ndarray]:
    """Validate the HDF5 layout and return its shape and labels."""
    with h5py.File(path, "r") as dataset:
        if "images" not in dataset or "labels" not in dataset:
            raise ValueError("Dataset must contain 'images' and 'labels' datasets")
        image_shape = dataset["images"].shape
        labels = np.asarray(dataset["labels"][:], dtype=np.uint8)

    if len(image_shape) != 4 or image_shape[-1] != 3:
        raise ValueError(
            f"Expected images with shape (samples, height, width, 3), got {image_shape}"
        )
    if image_shape[0] != len(labels) or len(labels) == 0:
        raise ValueError("The images and labels datasets must have the same non-zero length")
    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels.issubset({0, 1}) or len(unique_labels) < 2:
        raise ValueError("Labels must contain both binary classes 0 and 1")
    return image_shape[1], image_shape[2], image_shape[3], labels


def train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        total_samples += len(labels)
    return total_loss / total_samples


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray]:
    model.eval()
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            losses.append(loss_function(logits, labels).item() * len(labels))
            predictions.append((torch.sigmoid(logits) >= 0.5).cpu().numpy())
            targets.append(labels.cpu().numpy())

    predicted = np.concatenate(predictions)
    actual = np.concatenate(targets)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    return (
        sum(losses) / len(actual),
        accuracy_score(actual, predicted),
        f1_score(actual, predicted, zero_division=0),
        matrix,
    )


def save_checkpoint(
    path: Path,
    model: AssignmentCNN,
    config: TrainingConfig,
    input_shape: tuple[int, int, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "input_shape": input_shape,
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in asdict(config).items()},
    }
    torch.save(checkpoint, path)


def train(config: TrainingConfig) -> None:
    set_seed(config.seed)
    height, width, channels, labels = inspect_dataset(config.dataset)
    indices = np.arange(len(labels))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=config.validation_split,
        random_state=config.seed,
        stratify=labels,
    )
    train_labels = labels[train_indices]
    positive_weight = float((len(train_labels) - train_labels.sum()) / train_labels.sum())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        HDF5ImageDataset(config.dataset, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        HDF5ImageDataset(config.dataset, validation_indices),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = AssignmentCNN(channels, height, width).to(device)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print(f"Loading dataset from: {config.dataset}")
    print(f"Samples: {len(labels):,} | image shape: ({height}, {width}, {channels})")
    print(f"Training: {len(train_indices):,} | validation: {len(validation_indices):,}")
    print(f"Device: {device} | positive class weight: {positive_weight:.4f}")

    best_f1 = -1.0
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, loss_function, optimizer, device)
        validation_loss, accuracy, f1, matrix = evaluate(
            model, validation_loader, loss_function, device
        )
        print(
            f"Epoch {epoch:>2}/{config.epochs} | "
            f"train loss: {train_loss:.4f} | validation loss: {validation_loss:.4f} | "
            f"accuracy: {accuracy * 100:.2f}% | F1: {f1:.4f}"
        )
        if f1 > best_f1:
            best_f1 = f1
            save_checkpoint(config.model, model, config, (channels, height, width))

    print("Confusion Matrix (rows=true, columns=predicted):")
    print("              Predicted False  Predicted True")
    print(f"Actual False  {matrix[0, 0]:16d}  {matrix[0, 1]:14d}")
    print(f"Actual True   {matrix[1, 0]:16d}  {matrix[1, 1]:14d}")
    print(f"\nBest model checkpoint saved to: {config.model}")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainingConfig()
    parser.add_argument("--dataset", type=Path, default=defaults.dataset)
    parser.add_argument("--model", type=Path, default=defaults.model)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--validation-split", type=float, default=defaults.validation_split)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive")
    if not 0 < args.validation_split < 1:
        parser.error("validation-split must be between 0 and 1")
    if args.num_workers < 0:
        parser.error("num-workers cannot be negative")
    return TrainingConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())