"""Save three randomly selected CNN training samples as a PNG image.

Example:

    python CNN_sample_visualizer.py
    python CNN_sample_visualizer.py --seed 42 --output samples.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def save_random_samples(dataset_path: Path, output_path: Path, seed: int | None) -> None:
    """Choose three distinct samples and save full-image and channel figures."""
    with h5py.File(dataset_path, "r") as dataset:
        if "images" not in dataset or "labels" not in dataset:
            raise ValueError(f"{dataset_path} must contain images and labels")

        images_dataset = dataset["images"]
        labels_dataset = dataset["labels"]
        sample_count = images_dataset.shape[0]
        if sample_count < 3:
            raise ValueError(
                f"{dataset_path} contains {sample_count} samples; at least 3 are required"
            )
        if len(images_dataset.shape) != 4 or images_dataset.shape[-1] != 3:
            raise ValueError(
                f"Expected images with shape (samples, height, width, 3), "
                f"got {images_dataset.shape}"
            )
        if labels_dataset.shape != (sample_count,):
            raise ValueError(
                f"Expected one label per image, got {labels_dataset.shape} "
                f"for {sample_count} images"
            )

        rng = np.random.default_rng(seed)
        indices = rng.choice(sample_count, size=3, replace=False)
        read_order = np.argsort(indices)
        sorted_indices = indices[read_order]
        restore_order = np.argsort(read_order)
        images = np.asarray(images_dataset[sorted_indices])[restore_order]
        labels = np.asarray(labels_dataset[sorted_indices])[restore_order]

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "matplotlib is required to save sample images. "
            "Install the project dependencies with `pip install -r requirements.txt`."
        ) from error

    figure, axes = plt.subplots(1, 3, figsize=(10, 4))
    for axis, image, label, index in zip(axes, images, labels, indices):
        axis.imshow(image)
        axis.set_title(f"Sample {index} | label {int(label)}")
        axis.set_xticks([])
        axis.set_yticks([])

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved three random samples to {output_path}")

    channels_path = output_path.with_name(
        f"{output_path.stem}_channels{output_path.suffix or '.png'}"
    )
    figure, axes = plt.subplots(4, 3, figsize=(9, 12))
    row_labels = (
        "Full image",
        "R (map) channel",
        "G (tasks) channel",
        "B (agents) channel",
    )
    for row, name in enumerate(row_labels):
        axes[row, 0].set_ylabel(name)

    for column, (image, label, index) in enumerate(zip(images, labels, indices)):
        axes[0, column].set_title(f"Sample {index} | label {int(label)}")
        axes[0, column].imshow(image)
        for channel in range(3):
            axes[channel + 1, column].imshow(
                image[:, :, channel], cmap="gray", vmin=0, vmax=255
            )
        for axis in axes[:, column]:
            axis.set_xticks([])
            axis.set_yticks([])

    figure.tight_layout()
    figure.savefig(channels_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved samples with separate RGB channels to {channels_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("DataSets/CNN_train.h5")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("cnn_training_samples.png")
    )
    parser.add_argument(
        "--seed", type=int, help="Optional seed for reproducible sample selection"
    )
    args = parser.parse_args()
    save_random_samples(args.dataset, args.output, args.seed)


if __name__ == "__main__":
    main()
