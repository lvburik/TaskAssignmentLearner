"""Visualize samples from the CNN HDF5 dataset.

The generated figure shows the encoded RGB sample alongside its three input
channels. The channels are map, task markers, and agent markers.

Example:

    python CNN_dataset_inspector.py --dataset DataSets/cnn_training.h5 \
        --samples 12 --output cnn_samples.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _sample_indices(
    labels: np.ndarray, sample_count: int, seed: int
) -> np.ndarray:
    """Choose a reproducible, approximately class-balanced sample."""
    if sample_count > len(labels):
        raise ValueError(
            f"Requested {sample_count} samples, but the dataset contains only "
            f"{len(labels)}"
        )

    rng = np.random.default_rng(seed)
    classes = [label for label in (0, 1) if np.any(labels == label)]
    selected: list[int] = []
    for label in classes:
        class_indices = np.flatnonzero(labels == label)
        class_count = min(
            len(class_indices),
            (sample_count + len(classes) - 1) // len(classes),
        )
        selected.extend(rng.choice(class_indices, size=class_count, replace=False))

    if len(selected) < sample_count:
        remaining = np.setdiff1d(np.arange(len(labels)), selected)
        selected.extend(
            rng.choice(remaining, size=sample_count - len(selected), replace=False)
        )

    return rng.permutation(np.asarray(selected[:sample_count], dtype=np.int64))


def _read_dataset(
    path: Path, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Read and validate selected samples and their metadata."""
    with h5py.File(path, "r") as dataset:
        if "images" not in dataset or "labels" not in dataset:
            raise ValueError("Dataset must contain 'images' and 'labels'")

        read_order = np.argsort(indices)
        sorted_indices = indices[read_order]
        restore_order = np.argsort(read_order)
        images = np.asarray(dataset["images"][sorted_indices])[restore_order]
        labels = np.asarray(dataset["labels"][sorted_indices], dtype=np.uint8)[
            restore_order
        ]
        source_files = (
            [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in dataset["source_file"][sorted_indices][restore_order]
            ]
            if "source_file" in dataset
            else [""] * len(indices)
        )
        timesteps = (
            np.asarray(dataset["timestep"][sorted_indices])[restore_order]
            if "timestep" in dataset
            else np.full(len(indices), -1)
        )

    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(
            f"Expected images with shape (samples, height, width, 3), got {images.shape}"
        )
    return images, labels, source_files, timesteps


def plot_samples(
    images: np.ndarray,
    labels: np.ndarray,
    source_files: list[str],
    timesteps: np.ndarray,
    indices: np.ndarray,
    output: Path | None,
) -> None:
    """Plot selected samples as composite and per-channel images."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "matplotlib is required for visualization. Install the project "
            "requirements with `pip install -r requirements.txt`."
        ) from error

    rows = len(images)
    figure, axes = plt.subplots(rows, 4, figsize=(12, max(3, rows * 2.8)))
    axes = np.atleast_2d(axes)
    channel_names = ("Encoded sample", "Map", "Tasks", "Agents")

    for row, (image, label, source, timestep, index) in enumerate(
        zip(images, labels, source_files, timesteps, indices)
    ):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(
            f"index {index} | label {int(label)}\n"
            f"{source} t={int(timestep)}" if source else f"index {index} | label {int(label)}"
        )
        for channel in range(3):
            axes[row, channel + 1].imshow(image[:, :, channel], cmap="gray", vmin=0, vmax=255)
        for column, name in enumerate(channel_names):
            axes[row, column].set_ylabel(name if column == 0 else "", rotation=0, labelpad=35)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])

    figure.suptitle("CNN dataset samples", fontsize=14)
    figure.tight_layout()
    if output is None:
        plt.show()
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {output}")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("DataSets/cnn_training.h5"))
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Save a PNG instead of opening a window")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    with h5py.File(args.dataset, "r") as dataset:
        if "labels" not in dataset:
            raise ValueError("Dataset must contain 'labels'")
        labels = np.asarray(dataset["labels"][:], dtype=np.uint8)
    indices = _sample_indices(labels, args.samples, args.seed)
    images, selected_labels, source_files, timesteps = _read_dataset(args.dataset, indices)
    plot_samples(
        images, selected_labels, source_files, timesteps, indices, args.output
    )


if __name__ == "__main__":
    main()