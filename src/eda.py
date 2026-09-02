"""
Exploratory Data Analysis for the California Housing dataset.
Loads the data, prints summary statistics, and saves diagnostic
plots to the plots/ directory.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


def load_data() -> pd.DataFrame:
    """Load the California Housing dataset into a single DataFrame."""
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame  # features + target ("MedHouseVal") combined
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("Dataset shape:", df.shape)
    print("=" * 60)
    print("\nColumn info:\n")
    print(df.info())
    print("\nDescriptive statistics:\n")
    print(df.describe())
    print("\nMissing values per column:\n")
    print(df.isnull().sum())


def plot_target_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["MedHouseVal"], bins=50, kde=True)
    plt.title("Distribution of Median House Value")
    plt.xlabel("Median House Value ($100,000s)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "target_distribution.png"))
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"))
    plt.close()


def plot_feature_vs_target(df: pd.DataFrame) -> None:
    features = ["MedInc", "AveRooms", "HouseAge", "Latitude"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        axes[i].scatter(df[feature], df["MedHouseVal"], alpha=0.2, s=10)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("MedHouseVal")
        axes[i].set_title(f"{feature} vs. Median House Value")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_vs_target.png"))
    plt.close()


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_data()
    print_summary(df)

    plot_target_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_vs_target(df)

    print(f"\nPlots saved to: {os.path.abspath(PLOTS_DIR)}")


if __name__ == "__main__":
    main()