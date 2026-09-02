"""
Loads all trained models, evaluates them on the test set, and
compares performance using MAE, RMSE, and R².
Also saves a plot comparing predicted vs. actual values for the
best-performing model.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import get_train_test_split

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

MODEL_FILES = {
    "linear_regression": "linear_regression.joblib",
    "ridge_regression": "ridge_regression.joblib",
    "random_forest": "random_forest.joblib",
    "gradient_boosting": "gradient_boosting.joblib",
}


def load_models() -> dict:
    models = {}
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, filename)
        models[name] = joblib.load(path)
    return models


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "y_pred": y_pred}


def plot_predictions_vs_actual(y_test, y_pred, model_name: str) -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual Median House Value")
    plt.ylabel("Predicted Median House Value")
    plt.title(f"Predicted vs. Actual — {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"predictions_vs_actual_{model_name}.png"))
    plt.close()


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, scaler, feature_names = get_train_test_split()
    models = load_models()

    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test)

    # Build comparison table
    summary = pd.DataFrame({
        name: {"MAE": r["MAE"], "RMSE": r["RMSE"], "R2": r["R2"]}
        for name, r in results.items()
    }).T.sort_values("RMSE")

    print("=" * 60)
    print("Model comparison (sorted by RMSE, lower is better):")
    print("=" * 60)
    print(summary.to_string(float_format="%.4f"))

    best_model_name = summary.index[0]
    print(f"\nBest model: {best_model_name}")

    plot_predictions_vs_actual(y_test, results[best_model_name]["y_pred"], best_model_name)
    print(f"Saved predictions-vs-actual plot for {best_model_name}")

    # Save summary table to disk for the README
    summary.to_csv(os.path.join(PLOTS_DIR, "..", "model_comparison.csv"))

    # --- Persist the best model separately, with metadata ---
    best_model = models[best_model_name]
    best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_model, best_model_path)

    metadata = {
        "model_name": best_model_name,
        "MAE": results[best_model_name]["MAE"],
        "RMSE": results[best_model_name]["RMSE"],
        "R2": results[best_model_name]["R2"],
    }
    joblib.dump(metadata, os.path.join(MODELS_DIR, "best_model_metadata.joblib"))

    print(f"\nBest model persisted as: {best_model_path}")
    print(f"Metadata: {metadata}")


if __name__ == "__main__":
    main()