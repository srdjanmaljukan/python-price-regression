"""
Preprocessing pipeline for the California Housing dataset:
train/test split, feature scaling, and a reusable function to
load fully prepared data for training and evaluation.
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COL = "MedHouseVal"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw_data() -> pd.DataFrame:
    """Load the California Housing dataset into a single DataFrame."""
    housing = fetch_california_housing(as_frame=True)
    return housing.frame


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a few derived features that often help housing-price models:
    - RoomsPerHousehold: average rooms per occupant
    - BedroomsPerRoom: ratio of bedrooms to total rooms
    - PopulationPerHousehold: same as AveOccup, kept explicit for clarity
    """
    df = df.copy()
    df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"]
    df["RoomsPerPerson"] = df["AveRooms"] / df["AveOccup"]
    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def get_train_test_split():
    """
    Full pipeline: load data, engineer features, split into
    train/test sets, and scale features using StandardScaler
    (fit on train only, to avoid data leakage).

    Returns X_train, X_test, y_train, y_test, scaler, feature_names
    """
    df = load_raw_data()
    df = add_engineered_features(df)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    feature_names = X.columns.tolist()

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


if __name__ == "__main__":
    # Quick sanity check when running this file directly
    X_train, X_test, y_train, y_test, scaler, feature_names = get_train_test_split()
    print("Features:", feature_names)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)