"""Train CNN architectures and search a set of hyperparameters.

The training dataset is sampled without replacement at 25% per epoch. The
validation dataset is evaluated in full. HDF5 batches are read in sorted index
order, which is substantially faster than opening one compressed image at a
time.

Example:

    py -3.12 CNN_train.py --max-models 10 --epochs 16
"""

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import Tensor, nn


@dataclass(frozen=True)
class SearchConfig:
    train_dataset: Path = Path("DataSets/CNN_train.h5")
    validation_dataset: Path = Path("DataSets/CNN_validation.h5")
    model: Path = Path("models/CNN_best.pt")
    results_file: Path = Path("cnn_search_results.txt")
    epochs: int = 16
    sample_fraction: float = 0.25
    batch_size: int = 1024
    validation_batch_size: int = 2048
    max_models: int = 10
    early_stopping_patience: int = 4
    early_stopping_min_delta: float = 1e-4
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"


class HDF5BatchReader:
    """Load an HDF5 split once, then serve batches directly from RAM."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.images: np.ndarray | None = None
        self.labels: np.ndarray | None = None

    def __enter__(self) -> "HDF5BatchReader":
        with h5py.File(self.path, "r") as dataset:
            source_images = dataset["images"]
            self.images = np.empty(source_images.shape, dtype=np.uint8)
            # Avoid dataset[:] here: h5py would allocate a second multi-GiB
            # temporary array while decompressing the source dataset.
            load_chunk = 4096
            for start in range(0, source_images.shape[0], load_chunk):
                end = min(start + load_chunk, source_images.shape[0])
                self.images[start:end] = source_images[start:end]
            self.labels = np.asarray(dataset["labels"][:], dtype=np.float32)
        return self

    def __exit__(self, *_: object) -> None:
        self.images = None
        self.labels = None

    def read(self, indices: np.ndarray) -> tuple[Tensor, Tensor]:
        if self.images is None or self.labels is None:
            raise RuntimeError("HDF5BatchReader must be used as a context manager")
        images = self.images[indices]
        labels = self.labels[indices]
        return (
            torch.from_numpy(images),
            torch.from_numpy(labels),
        )


def _dataset_info(path: Path) -> tuple[tuple[int, int, int], np.ndarray]:
    with h5py.File(path, "r") as dataset:
        if "images" not in dataset or "labels" not in dataset:
            raise ValueError(f"{path} must contain images and labels")
        shape = dataset["images"].shape
        labels = np.asarray(dataset["labels"][:], dtype=np.uint8)
    if len(shape) != 4 or shape[-1] != 3 or shape[0] != len(labels):
        raise ValueError(f"{path} has invalid image/label shapes: {shape}, {labels.shape}")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError(f"{path} must contain both binary label classes")
    return (shape[1], shape[2], shape[3]), labels


def pooling_layer(pooling_type: str, adaptive_size: tuple[int, int] | None = None) -> nn.Module:
    if adaptive_size is not None:
        if pooling_type == "max":
            return nn.AdaptiveMaxPool2d(adaptive_size)
        if pooling_type == "average":
            return nn.AdaptiveAvgPool2d(adaptive_size)
    elif pooling_type == "max":
        return nn.MaxPool2d(2)
    elif pooling_type == "average":
        return nn.AvgPool2d(2)
    raise ValueError(f"Unsupported pooling type: {pooling_type}")


class AssignmentCNN(nn.Module):
    """Six branch CNN with max and average pooling for every input channel."""

    def __init__(
        self,
        input_channels: int,
        image_height: int,
        image_width: int,
        base_channels: int,
        convolutional_blocks: int,
        kernel_size: int,
        classifier_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_channels != 3:
            raise ValueError("AssignmentCNN requires exactly three input channels")
        self.branch_specs = tuple(
            (channel, pooling_type)
            for channel in range(input_channels)
            for pooling_type in ("max", "average")
        )
        self.pool_size = 4
        self.branches = nn.ModuleList(
            self._make_branch(
                base_channels,
                convolutional_blocks,
                kernel_size,
                pooling_type,
            )
            for _, pooling_type in self.branch_specs
        )
        self.image_width = image_width
        self.image_height = image_height
        final_channels = base_channels * 2 ** (convolutional_blocks - 1)
        feature_size = len(self.branch_specs) * final_channels * self.pool_size * self.pool_size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size + 3, classifier_width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_width, 1),
        )

    def _make_branch(
        self,
        base_channels: int,
        convolutional_blocks: int,
        kernel_size: int,
        pooling_type: str,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        input_size = 1
        for block in range(convolutional_blocks):
            output_size = base_channels * 2**block
            layers.extend(
                [
                    nn.Conv2d(input_size, output_size, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(),
                    pooling_layer(pooling_type),
                ]
            )
            input_size = output_size
        layers.append(pooling_layer(pooling_type, (self.pool_size, self.pool_size)))
        return nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        features = torch.cat(
            [
                branch(inputs[:, channel : channel + 1])
                for (channel, _), branch in zip(self.branch_specs, self.branches)
            ],
            dim=1,
        ).flatten(start_dim=1)
        task = self._marker_positions(inputs[:, 1])
        agent = self._marker_positions(inputs[:, 2])
        delta = task - agent
        scale = inputs.new_tensor((self.image_width, self.image_height))
        relation = torch.cat((delta / scale, torch.linalg.vector_norm(delta, dim=1, keepdim=True) / max(scale)), dim=1)
        return self.classifier(torch.cat((features, relation), dim=1)).squeeze(1)

    @staticmethod
    def _marker_positions(channel: Tensor) -> Tensor:
        height, width = channel.shape[-2:]
        index = channel.flatten(start_dim=1).argmax(dim=1)
        y = torch.div(index, width, rounding_mode="floor")
        x = index.remainder(width)
        return torch.stack((x.float(), y.float()), dim=1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batches(indices: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def train_epoch(
    model: AssignmentCNN,
    reader: HDF5BatchReader,
    indices: np.ndarray,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    batch_size: int,
    progress_label: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    total_batches = (len(indices) + batch_size - 1) // batch_size
    update_every = max(1, total_batches // 10)
    for batch_number, batch_indices in enumerate(batches(indices, batch_size), start=1):
        images, labels = reader.read(batch_indices)
        images = images.to(device, non_blocking=True).permute(0, 3, 1, 2)
        images = images.float().div_(255.0)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
        total_count += len(labels)
        if batch_number == total_batches or batch_number % update_every == 0:
            print(
                f"  {progress_label}: {batch_number}/{total_batches} batches",
                end="\r",
                flush=True,
            )
    print(" " * 60, end="\r", flush=True)
    return total_loss / total_count


def evaluate(
    model: AssignmentCNN,
    reader: HDF5BatchReader,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    progress_label: str,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    actual: list[np.ndarray] = []
    with torch.no_grad():
        total_batches = (len(indices) + batch_size - 1) // batch_size
        update_every = max(1, total_batches // 4)
        for batch_number, batch_indices in enumerate(
            batches(indices, batch_size), start=1
        ):
            images, labels = reader.read(batch_indices)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                images = images.to(device, non_blocking=True).permute(0, 3, 1, 2)
                logits = model(images.float().div_(255.0))
            predictions.append(torch.sigmoid(logits).cpu().numpy())
            actual.append(labels.numpy())
            if batch_number == total_batches or batch_number % update_every == 0:
                print(
                    f"  {progress_label}: {batch_number}/{total_batches} batches",
                    end="\r",
                    flush=True,
                )
        print(" " * 60, end="\r", flush=True)
    probabilities = np.concatenate(predictions)
    targets = np.concatenate(actual).astype(np.uint8)
    threshold = _best_threshold(targets, probabilities)
    predicted = probabilities >= threshold
    return float(threshold), float(f1_score(targets, predicted, zero_division=0)), confusion_matrix(targets, predicted, labels=[0, 1])


def _best_threshold(targets: np.ndarray, probabilities: np.ndarray) -> float:
    thresholds = np.linspace(0.10, 0.90, 33)
    scores = [f1_score(targets, probabilities >= threshold, zero_division=0) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def save_checkpoint(path: Path, model: AssignmentCNN, config: dict[str, object], input_shape: tuple[int, int, int], f1: float, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "input_shape": input_shape, "config": config, "validation_f1": f1, "threshold": threshold}, path)


def append_result(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as destination:
        destination.write(text + "\n")


def run_search(config: SearchConfig) -> None:
    set_seed(config.seed)
    (height, width, channels), train_labels = _dataset_info(config.train_dataset)
    validation_shape, validation_labels = _dataset_info(config.validation_dataset)
    if (height, width, channels) != validation_shape:
        raise ValueError("Training and validation image shapes must match")

    device_name = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    device = torch.device("cuda" if device_name == "cuda" else "cpu")
    positive_weight = (len(train_labels) - train_labels.sum()) / train_labels.sum()
    train_count = max(1, int(len(train_labels) * config.sample_fraction))
    all_validation = np.arange(len(validation_labels), dtype=np.int64)
    search_space = [
        {"base_channels": 16, "convolutional_blocks": 2, "kernel_size": 3, "classifier_width": 128, "dropout": 0.20, "lr": 1e-3, "weight_decay": 1e-4},
        {"base_channels": 24, "convolutional_blocks": 2, "kernel_size": 3, "classifier_width": 128, "dropout": 0.20, "lr": 1e-3, "weight_decay": 1e-4},
        {"base_channels": 16, "convolutional_blocks": 3, "kernel_size": 3, "classifier_width": 128, "dropout": 0.20, "lr": 1e-3, "weight_decay": 1e-4},
        {"base_channels": 24, "convolutional_blocks": 3, "kernel_size": 3, "classifier_width": 128, "dropout": 0.30, "lr": 5e-4, "weight_decay": 1e-4},
        {"base_channels": 32, "convolutional_blocks": 2, "kernel_size": 3, "classifier_width": 256, "dropout": 0.30, "lr": 5e-4, "weight_decay": 1e-4},
        {"base_channels": 16, "convolutional_blocks": 2, "kernel_size": 5, "classifier_width": 128, "dropout": 0.20, "lr": 1e-3, "weight_decay": 1e-4},
        {"base_channels": 24, "convolutional_blocks": 2, "kernel_size": 5, "classifier_width": 256, "dropout": 0.30, "lr": 5e-4, "weight_decay": 1e-3},
        {"base_channels": 32, "convolutional_blocks": 3, "kernel_size": 3, "classifier_width": 256, "dropout": 0.30, "lr": 5e-4, "weight_decay": 1e-4},
        {"base_channels": 16, "convolutional_blocks": 3, "kernel_size": 5, "classifier_width": 128, "dropout": 0.30, "lr": 5e-4, "weight_decay": 1e-3},
        {"base_channels": 24, "convolutional_blocks": 3, "kernel_size": 5, "classifier_width": 256, "dropout": 0.40, "lr": 3e-4, "weight_decay": 1e-3},
    ][: config.max_models]
    train_ram_gib = len(train_labels) * height * width * channels / 1024**3
    validation_ram_gib = len(validation_labels) * height * width * channels / 1024**3
    print(f"Device: {device} | training samples per epoch: {train_count:,} ({config.sample_fraction:.0%})")
    print(f"Positive class weight: {positive_weight:.4f} | validation samples: {len(validation_labels):,}")
    print(
        f"Loading datasets into RAM: {train_ram_gib:.2f} GiB train + "
        f"{validation_ram_gib:.2f} GiB validation"
    )

    config.results_file.parent.mkdir(parents=True, exist_ok=True)
    config.results_file.write_text("", encoding="utf-8")
    append_result(config.results_file, f"CNN hyperparameter search: {len(search_space)} models")
    best_f1 = -1.0
    with HDF5BatchReader(config.train_dataset) as train_reader, HDF5BatchReader(config.validation_dataset) as validation_reader:
        print("Datasets loaded into RAM; starting hyperparameter search", flush=True)
        for model_number, parameters in enumerate(search_space, start=1):
            set_seed(config.seed + model_number)
            architecture = (
                f"blocks={parameters['convolutional_blocks']}, "
                f"base_channels={parameters['base_channels']}, "
                f"kernel_size={parameters['kernel_size']}, "
                f"classifier_width={parameters['classifier_width']}, "
                "branches=map:(max,average),task:(max,average),agent:(max,average), "
                f"dropout={parameters['dropout']}"
            )
            append_result(
                config.results_file,
                f"\nModel {model_number}/{len(search_space)} started\n"
                f"Architecture: {architecture}\n"
                f"Optimizer: lr={parameters['lr']}, weight_decay={parameters['weight_decay']}",
            )
            print(f"Starting model {model_number}/{len(search_space)}: {architecture}")
            model = AssignmentCNN(
                channels,
                height,
                width,
                parameters["base_channels"],
                parameters["convolutional_blocks"],
                parameters["kernel_size"],
                parameters["classifier_width"],
                parameters["dropout"],
            ).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
            optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["lr"], weight_decay=parameters["weight_decay"])
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
            best_run_f1 = -1.0
            best_run_state: dict[str, Tensor] | None = None
            best_threshold = 0.5
            best_run_matrix: np.ndarray | None = None
            epochs_without_improvement = 0
            for epoch in range(1, config.epochs + 1):
                epoch_indices = np.random.default_rng(config.seed + model_number * 1000 + epoch).choice(len(train_labels), train_count, replace=False)
                train_loss = train_epoch(
                    model,
                    train_reader,
                    epoch_indices,
                    criterion,
                    optimizer,
                    device,
                    scaler,
                    config.batch_size,
                    f"model {model_number} epoch {epoch} training",
                )
                threshold, validation_f1, matrix = evaluate(
                    model,
                    validation_reader,
                    all_validation,
                    device,
                    config.validation_batch_size,
                    f"model {model_number} epoch {epoch} validation",
                )
                print(f"Model {model_number}/{len(search_space)} epoch {epoch}/{config.epochs} | loss={train_loss:.4f} | F1={validation_f1:.4f} | threshold={threshold:.2f}")
                if validation_f1 > best_run_f1 + config.early_stopping_min_delta:
                    best_run_f1 = validation_f1
                    best_threshold = threshold
                    best_run_state = copy.deepcopy(model.state_dict())
                    best_run_matrix = matrix.copy()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= config.early_stopping_patience:
                        print(
                            f"Early stopping model {model_number} after epoch {epoch}; "
                            f"no F1 improvement for {config.early_stopping_patience} epochs"
                        )
                        break
            assert best_run_state is not None
            assert best_run_matrix is not None
            if best_run_f1 > best_f1:
                best_f1 = best_run_f1
                model.load_state_dict(best_run_state)
                save_checkpoint(config.model, model, {**config.__dict__, **parameters}, (channels, height, width), best_f1, best_threshold)
                print(f"New best model: F1={best_f1:.4f}, saved to {config.model}")
                append_result(
                    config.results_file,
                    f"New global best: F1={best_f1:.4f}, threshold={best_threshold:.2f}\n"
                    f"Architecture: {architecture}\n"
                    f"Confusion matrix:\n{best_run_matrix}",
                )
            append_result(
                config.results_file,
                f"Model {model_number} completed: best_F1={best_run_f1:.4f}, "
                f"best_threshold={best_threshold:.2f}\n"
                f"Architecture: {architecture}\n"
                f"Confusion matrix:\n{best_run_matrix}",
            )
            print(f"Confusion matrix:\n{best_run_matrix}")


def parse_args() -> SearchConfig:
    defaults = SearchConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, default=defaults.train_dataset)
    parser.add_argument("--validation-dataset", type=Path, default=defaults.validation_dataset)
    parser.add_argument("--model", type=Path, default=defaults.model)
    parser.add_argument("--results-file", type=Path, default=defaults.results_file)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--sample-fraction", type=float, default=defaults.sample_fraction)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--validation-batch-size", type=int, default=defaults.validation_batch_size)
    parser.add_argument("--max-models", type=int, default=defaults.max_models)
    parser.add_argument("--early-stopping-patience", type=int, default=defaults.early_stopping_patience)
    parser.add_argument("--early-stopping-min-delta", type=float, default=defaults.early_stopping_min_delta)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=defaults.device)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.validation_batch_size < 1:
        parser.error("epochs and batch sizes must be positive")
    if args.early_stopping_patience < 1:
        parser.error("early-stopping-patience must be positive")
    if args.early_stopping_min_delta < 0:
        parser.error("early-stopping-min-delta must not be negative")
    if not 0 < args.sample_fraction <= 1:
        parser.error("sample-fraction must be between 0 and 1")
    if not 1 <= args.max_models <= 10:
        parser.error("max-models must be between 1 and 10")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but CUDA is unavailable")
    return SearchConfig(**vars(args))


if __name__ == "__main__":
    run_search(parse_args())
