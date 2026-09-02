"""
Shared feature-engineering logic used by both the training
pipeline (preprocessing.py) and the prediction interface
(predict.py), so the two never drift out of sync.
"""

import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to a DataFrame of raw California Housing
    columns:
    - BedroomsPerRoom: ratio of bedrooms to total rooms
    - RoomsPerPerson: average rooms per occupant
    """
    df = df.copy()
    df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"]
    df["RoomsPerPerson"] = df["AveRooms"] / df["AveOccup"]
    return df


def engineer_features_dict(raw: dict) -> dict:
    """
    Same logic as add_engineered_features, but for a single
    raw input dict (used by predict.py for one-off predictions).
    """
    raw = dict(raw)
    raw["BedroomsPerRoom"] = raw["AveBedrms"] / raw["AveRooms"]
    raw["RoomsPerPerson"] = raw["AveRooms"] / raw["AveOccup"]
    return raw