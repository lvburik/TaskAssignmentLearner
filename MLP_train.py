"""Search, train, and evaluate MLP models on prepared HDF5 datasets."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def create_dataset(dataset_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load an HDF5 dataset produced by XGBoost_dataset_builder.py."""
    frame = pd.read_hdf(dataset_path, key="data")
    if "y" not in frame.columns:
        raise ValueError(f"{dataset_path} must contain a 'y' target column")
    return frame.drop(columns=["y"]), frame["y"].astype(int)


def print_dataset_info(
    name: str, features: pd.DataFrame, target: pd.Series
) -> None:
    positive_count = int(target.sum())
    print(
        f"{name}: {len(target):,} rows, {len(features.columns)} features, "
        f"{len(target) - positive_count:,} negative, {positive_count:,} positive "
        f"({positive_count / len(target):.2%})"
    )


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    parameters: dict[str, object],
    subset_fraction: float,
    epochs: int,
) -> tuple[StandardScaler, MLPClassifier]:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    target_values = target.to_numpy()
    subset_size = max(1, int(len(target_values) * subset_fraction))
    model = MLPClassifier(
        **parameters,
        batch_size=4096,
        max_iter=1,
        random_state=42,
        shuffle=True,
        verbose=False,
    )
    random = np.random.default_rng(42)
    for epoch in range(epochs):
        indices = random.choice(len(target_values), size=subset_size, replace=False)
        model.partial_fit(
            scaled_features[indices],
            target_values[indices],
            classes=np.array([0, 1]),
        )
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1}/{epochs}", end="\r")
    print(" " * 30, end="\r")
    return scaler, model


def predict(
    scaler: StandardScaler,
    model: MLPClassifier,
    features: pd.DataFrame,
) -> np.ndarray:
    return model.predict(scaler.transform(features))


def train_and_search(
    train_dataset: Path,
    validation_dataset: Path,
    model_path: Path,
    max_models: int,
    subset_fraction: float,
    epochs: int,
) -> None:
    x_train, y_train = create_dataset(train_dataset)
    x_validation, y_validation = create_dataset(validation_dataset)
    if list(x_train.columns) != list(x_validation.columns):
        raise ValueError("Training and validation datasets must have identical features")

    print_dataset_info("Training dataset", x_train, y_train)
    print_dataset_info("Validation dataset", x_validation, y_validation)

    search_space = [
        {"hidden_layer_sizes": (32,), "alpha": 0.0001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (64,), "alpha": 0.0001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (128,), "alpha": 0.0001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (64, 32), "alpha": 0.0001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (128, 64), "alpha": 0.0001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (128, 64), "alpha": 0.001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (128, 64), "alpha": 0.0001, "learning_rate_init": 0.0005},
        {"hidden_layer_sizes": (256, 128), "alpha": 0.0001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (256, 128), "alpha": 0.001, "learning_rate_init": 0.0005},
        {"hidden_layer_sizes": (256, 128, 64), "alpha": 0.001, "learning_rate_init": 0.0005},
    ][:max_models]

    results: list[
        tuple[StandardScaler, MLPClassifier, dict[str, float], dict[str, object]]
    ] = []
    print("\nValidation results (models ranked by F1):")
    for parameters in search_space:
        print(
            f"\nTraining {parameters} with "
            f"{subset_fraction:.0%} ({int(len(x_train) * subset_fraction):,}) "
            f"rows per epoch"
        )
        scaler, model = train_model(
            x_train, y_train, parameters, subset_fraction, epochs
        )
        validation_predictions = predict(scaler, model, x_validation)
        metrics = {
            "accuracy": accuracy_score(y_validation, validation_predictions),
            "balanced_accuracy": balanced_accuracy_score(
                y_validation, validation_predictions
            ),
            "precision": precision_score(
                y_validation, validation_predictions, zero_division=0
            ),
            "recall": recall_score(y_validation, validation_predictions, zero_division=0),
            "f1": f1_score(y_validation, validation_predictions, zero_division=0),
        }
        print(
            f"{parameters}: F1={metrics['f1']:.4f}, "
            f"balanced accuracy={metrics['balanced_accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}"
        )
        results.append((scaler, model, metrics, parameters))

    results.sort(key=lambda result: result[2]["f1"], reverse=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    for rank, (scaler, model, metrics, parameters) in enumerate(results[:2], start=1):
        print(f"\nRank {rank} parameters: {parameters}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 score: {metrics['f1']:.4f}")
        print("Confusion matrix (rows=true, columns=predicted):")
        print(
            confusion_matrix(
                y_validation,
                predict(scaler, model, x_validation),
                labels=[0, 1],
            )
        )

        ranked_model_path = model_path.with_name(
            f"{model_path.stem}_{rank}{model_path.suffix}"
        )
        with ranked_model_path.open("wb") as model_file:
            pickle.dump({"scaler": scaler, "model": model}, model_file)
        print(f"Saved rank {rank} model to {ranked_model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=Path("DataSets/XGBoost_train.h5"),
    )
    parser.add_argument(
        "--validation-dataset",
        type=Path,
        default=Path("DataSets/XGBoost_validation.h5"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/MLP_best.pkl"),
    )
    parser.add_argument("--max-models", type=int, default=10)
    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=0.10,
        help="Fraction of training rows sampled without replacement per epoch.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if args.max_models < 2 or args.max_models > 10:
        raise ValueError("--max-models must be between 2 and 10")
    if not 0 < args.subset_fraction <= 1:
        raise ValueError("--subset-fraction must be between 0 and 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    train_and_search(
        args.train_dataset,
        args.validation_dataset,
        args.model,
        args.max_models,
        args.subset_fraction,
        args.epochs,
    )


if __name__ == "__main__":
    main()
