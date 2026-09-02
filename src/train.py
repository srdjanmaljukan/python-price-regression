"""
Trains multiple regression models on the California Housing dataset
and saves each trained model to the models/ directory using joblib.

Models trained:
- Linear Regression (baseline)
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
"""

import os
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from preprocessing import get_train_test_split

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def get_models() -> dict:
    """Return a dictionary of model name -> unfitted model instance."""
    return {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0, random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }


def train_and_save_models(X_train, y_train) -> dict:
    """Train each model and save it to disk. Returns dict of fitted models."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    models = get_models()
    fitted_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"  Saved to {model_path}")

    return fitted_models


def main():
    X_train, X_test, y_train, y_test, scaler, feature_names = get_train_test_split()

    # Save the scaler and feature names too — needed later by predict.py
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.joblib"))

    train_and_save_models(X_train, y_train)
    print("\nAll models trained and saved.")


if __name__ == "__main__":
    main()  