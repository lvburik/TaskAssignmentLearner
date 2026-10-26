"""Search, train, and evaluate XGBoost models on prepared HDF5 datasets.

Example:

    .venv/bin/python XGBoost_train.py
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def load_dataset(dataset_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_hdf(dataset_path, key="data")
    if "y" not in frame.columns:
        raise ValueError(f"{dataset_path} must contain a 'y' target column")
    return frame.drop(columns=["y"]), frame["y"].astype(int)


def class_weight(target: pd.Series) -> float:
    positive_count = int(target.sum())
    negative_count = len(target) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Training data must contain both label classes")
    return negative_count / positive_count


def print_dataset_info(name: str, features: pd.DataFrame, target: pd.Series) -> None:
    positive_count = int(target.sum())
    print(
        f"{name}: {len(target):,} rows, {len(features.columns)} features, "
        f"{len(target) - positive_count:,} negative, {positive_count:,} positive "
        f"({positive_count / len(target):.2%})"
    )


def evaluate(
    model: xgb.XGBClassifier,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    predictions = model.predict(features)
    return {
        "accuracy": accuracy_score(target, predictions),
        "balanced_accuracy": balanced_accuracy_score(target, predictions),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
    }


def train_and_search(
    train_dataset: Path,
    validation_dataset: Path,
    model_path: Path,
    max_models: int,
) -> None:
    x_train, y_train = load_dataset(train_dataset)
    x_validation, y_validation = load_dataset(validation_dataset)
    if list(x_train.columns) != list(x_validation.columns):
        raise ValueError("Training and validation datasets must have identical features")

    print_dataset_info("Training dataset", x_train, y_train)
    print_dataset_info("Validation dataset", x_validation, y_validation)
    base_weight = class_weight(y_train)
    print(f"Training class weight (negative / positive): {base_weight:.4f}")

    search_space = [
        {"n_estimators": 128, "max_depth": 3, "learning_rate": 0.10, "subsample": 1.0},
        {"n_estimators": 256, "max_depth": 3, "learning_rate": 0.05, "subsample": 1.0},
        {"n_estimators": 256, "max_depth": 5, "learning_rate": 0.05, "subsample": 1.0},
        {"n_estimators": 512, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.9},
        {"n_estimators": 256, "max_depth": 7, "learning_rate": 0.05, "subsample": 0.9},
        {"n_estimators": 512, "max_depth": 7, "learning_rate": 0.03, "subsample": 0.9},
        {"n_estimators": 768, "max_depth": 7, "learning_rate": 0.03, "subsample": 0.9},
        {"n_estimators": 1024, "max_depth": 7, "learning_rate": 0.02, "subsample": 0.9},
        {"n_estimators": 768, "max_depth": 9, "learning_rate": 0.03, "subsample": 0.9},
        {"n_estimators": 1024, "max_depth": 9, "learning_rate": 0.02, "subsample": 0.85},
    ][:max_models]

    results: list[
        tuple[
            xgb.XGBClassifier,
            dict[str, float],
            dict[str, float | int],
        ]
    ] = []
    print("\nValidation results (models ranked by F1):")
    for parameters in search_space:
        model = xgb.XGBClassifier(
            **parameters,
            objective="binary:logistic",
            scale_pos_weight=base_weight,
            max_delta_step=1,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )
        model.fit(x_train, y_train, verbose=False)
        metrics = evaluate(model, x_validation, y_validation)
        print(
            f"{parameters}: F1={metrics['f1']:.4f}, "
            f"balanced accuracy={metrics['balanced_accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}"
        )
        results.append((model, metrics, parameters))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    results.sort(key=lambda result: result[1]["f1"], reverse=True)
    for rank, (model, metrics, parameters) in enumerate(results[:2], start=1):
        print(f"\nRank {rank} parameters: {parameters}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 score: {metrics['f1']:.4f}")
        print("Confusion matrix (rows=true, columns=predicted):")
        print(confusion_matrix(y_validation, model.predict(x_validation), labels=[0, 1]))

        ranked_model_path = model_path.with_name(
            f"{model_path.stem}_{rank}{model_path.suffix}"
        )
        with ranked_model_path.open("wb") as destination:
            pickle.dump(model, destination)
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
        default=Path("models/XGBoost_best.pkl"),
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=10,
        help="Number of configurations to evaluate from the built-in search space.",
    )
    args = parser.parse_args()
    if args.max_models < 2 or args.max_models > 10:
        raise ValueError("--max-models must be between 2 and 10")
    train_and_search(
        args.train_dataset,
        args.validation_dataset,
        args.model,
        args.max_models,
    )


if __name__ == "__main__":
    main()
