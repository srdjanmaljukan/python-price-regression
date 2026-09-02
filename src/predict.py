"""
Simple CLI for making predictions with the trained best model.

Usage:
    python predict.py --interactive
    python predict.py --sample
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from features import engineer_features_dict as engineer_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

ORIGINAL_FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]


def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))
    metadata = joblib.load(os.path.join(MODELS_DIR, "best_model_metadata.joblib"))
    return model, scaler, feature_names, metadata


def predict_price(raw_input: dict, model, scaler, feature_names) -> float:
    """
    raw_input: dict with the 8 original feature values (no engineered
    features needed — they're computed here automatically).
    """
    full_input = engineer_features(raw_input)
    df = pd.DataFrame([full_input])[feature_names]  # enforce correct column order
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]
    return prediction


def get_sample_input() -> dict:
    """A realistic example: mid-income area in the Bay Area."""
    return {
        "MedInc": 5.0,
        "HouseAge": 25.0,
        "AveRooms": 6.0,
        "AveBedrms": 1.0,
        "Population": 1200.0,
        "AveOccup": 3.0,
        "Latitude": 37.5,
        "Longitude": -122.0,
    }


def get_interactive_input() -> dict:
    print("Enter the following values:\n")
    raw = {}
    for feature in ORIGINAL_FEATURES:
        while True:
            try:
                value = float(input(f"  {feature}: "))
                raw[feature] = value
                break
            except ValueError:
                print("  Please enter a valid number.")
    return raw


def main():
    parser = argparse.ArgumentParser(description="Predict median house value.")
    parser.add_argument("--interactive", action="store_true", help="Enter values manually")
    parser.add_argument("--sample", action="store_true", help="Use a built-in sample input")
    args = parser.parse_args()

    model, scaler, feature_names, metadata = load_artifacts()
    print(f"Using model: {metadata['model_name']} (RMSE={metadata['RMSE']:.4f}, R2={metadata['R2']:.4f})\n")

    if args.interactive:
        raw_input = get_interactive_input()
    else:
        raw_input = get_sample_input()
        print("Using sample input:")
        for k, v in raw_input.items():
            print(f"  {k}: {v}")
        print()

    prediction = predict_price(raw_input, model, scaler, feature_names)
    print(f"\nPredicted median house value: {prediction:.4f} (in $100,000s) ≈ ${prediction * 100000:,.0f}")


if __name__ == "__main__":
    main()