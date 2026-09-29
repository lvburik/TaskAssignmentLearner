import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb





def parse_sample(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def build_features(sample, max_team_size):
    agent_x, agent_y = sample['agent_loc']
    team_locations = sample['team_loc']

    features = {
        'task_location': sample['task_loc'],
        'agent_x': agent_x,
        'agent_y': agent_y,
        'team_count': len(team_locations),
    }

    for teammate_index in range(max_team_size):
        location = team_locations[teammate_index] if teammate_index < len(team_locations) else (0, 0)
        features[f'teammate_{teammate_index}_x'] = location[0]
        features[f'teammate_{teammate_index}_y'] = location[1]

    return features



def create_dataset(DATASET_DIR):
    df = pd.read_csv(DATASET_DIR)

    samples = df['X'].map(parse_sample)
    max_team_size = max(len(sample['team_loc']) for sample in samples)
    X = pd.DataFrame(
        [build_features(sample, max_team_size) for sample in samples]
    )
    y = df['y'].astype(bool)

    return X, y

X, y = create_dataset("Datasets/set1.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42)

print("Training labels:")
print(y_train.value_counts())

print("Testing labels:")
print(y_test.value_counts())


model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

if isinstance(model, xgb.XGBClassifier):
    accuracy = accuracy_score(y_test, predictions)
    print(f"✅ Classification Accuracy: {accuracy * 100:.2f}%")

