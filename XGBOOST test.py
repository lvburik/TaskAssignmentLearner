from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


DATASET_PATH = Path("DataSets/set1.h5")
MODEL_PATH = Path("models/XGB100estimators.json")

def create_dataset(dataset_path):
    """Load the pre-flattened HDF5 dataset."""
    df = pd.read_hdf(dataset_path, key="data")
    X = df.drop(columns=["y"])
    y = df["y"].astype(bool)

    return X, y



def train_model(X_train, y_train):
    """Train the XGBoost classifier and print progress updates."""
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )

    print("Starting model training...")
    model.fit(X_train, y_train, verbose=True)
    print("Training complete.")

    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=[False, True])

    print(f"\n✅ Classification Accuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix (rows=true, columns=predicted):")
    print("              Predicted False  Predicted True")
    print(f"Actual False  {matrix[0, 0]:16d}  {matrix[0, 1]:14d}")
    print(f"Actual True   {matrix[1, 0]:16d}  {matrix[1, 1]:14d}")



def main():
    """Main workflow for dataset preparation, training, evaluation, and saving."""
    # 1) Prepare data
    print(f"Loading dataset from: {DATASET_PATH}")
    X, y = create_dataset(DATASET_PATH)

    print("Splitting data into test and training sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
    )

    print("\nTraining label distribution:")
    print(y_train.value_counts())

    print("\nTesting label distribution:")
    print(y_test.value_counts())

    # 2) Train model
    print("Training model")
    model = train_model(X_train, y_train)

    # 3) Evaluate model
    evaluate_model(model, X_test, y_test)

    # 4) Save the trained model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()