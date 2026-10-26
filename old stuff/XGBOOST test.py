from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

import pickle
number_of_estimators = 1028

DATASET_PATH = Path("DataSets/train1.h5")
MODEL_PATH = Path(f"models/XGB{number_of_estimators}estimators.json")



def create_dataset(dataset_path):
    """Load the pre-flattened HDF5 dataset."""
    df = pd.read_hdf(dataset_path, key="data")
    X = df.drop(columns=["y"])
    y = df["y"].astype(bool)

    return X, y



def train_model(X_train, y_train, pos_class_weight = 0.5, n_estimators = number_of_estimators):
    """Train the XGBoost classifier and print progress updates."""
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        objective='binary:logistic',
        scale_pos_weight=pos_class_weight,
        max_delta_step=1,
        random_state=42
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
    # X_train, X_test, y_train, y_test = train_test_split(
    #    X,
    #    y,
    #    test_size=0.0,
    #    random_state=42,
    #)
    X_train = X
    y_train = y

    X_test, y_test = create_dataset(Path("DataSets/test1.h5"))

    print("\nTraining label distribution:")
    print(y_train.value_counts())

    print("\nTesting label distribution:")
    print(y_test.value_counts())

    # 1.5) Calculate Class imbalance on the full dataset: 
    pos_class_weight = (len(y) - np.sum(y)) / np.sum(y)

    # 2) Train model
    print("Training model")
    model = train_model(X_train, y_train, pos_class_weight= pos_class_weight, n_estimators=number_of_estimators)

    # 3) Evaluate model
    evaluate_model(model, X_test, y_test)

    # 4) Save the trained model
    pickle.dump(model, open(MODEL_PATH, 'wb'))
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()