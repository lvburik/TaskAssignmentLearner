from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET_PATH = Path("DataSets/train1.h5")
MODEL_PATH = Path("models/MLP.pkl")


def create_dataset(dataset_path):
    """Load the pre-flattened HDF5 dataset."""
    df = pd.read_hdf(dataset_path, key="data")
    X = df.drop(columns=["y"])
    y = df["y"].astype(bool)

    return X, y


def train_model(X_train, y_train):
    """Train a scaled MLP classifier and print progress updates."""
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(100,100),
                    max_iter=1000,
                    early_stopping=True,
                    random_state=42,
                    verbose=True,
                ),
            ),
        ]
    )

    print("Starting model training...")
    model.fit(X_train, y_train)
    print("Training complete.")

    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=[False, True])

    print(f"\nClassification Accuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix (rows=true, columns=predicted):")
    print("              Predicted False  Predicted True")
    print(f"Actual False  {matrix[0, 0]:16d}  {matrix[0, 1]:14d}")
    print(f"Actual True   {matrix[1, 0]:16d}  {matrix[1, 1]:14d}")


def main():
    """Main workflow for dataset preparation, training, evaluation, and saving."""
    print(f"Loading dataset from: {DATASET_PATH}")
    X, y = create_dataset(DATASET_PATH)

    print("Using the full training dataset")
    X_train = X
    y_train = y

    X_test, y_test = create_dataset(Path("DataSets/test1.h5"))

    print("\nTraining label distribution:")
    print(y_train.value_counts())

    print("\nTesting label distribution:")
    print(y_test.value_counts())

    print("Training model")
    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
