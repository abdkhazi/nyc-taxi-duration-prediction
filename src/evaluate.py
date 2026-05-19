import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from preprocess import load_data, remove_outliers, parse_dates
from feature_engineering import build_features

def plot_shap(model, X_test):
    print("Generating SHAP plot...")

    # Use a sample for SHAP — full 286k rows is too slow
    X_sample = X_test.sample(5000, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("SHAP plot saved as shap_summary.png")

def plot_predictions(y_test, y_pred):
    print("Generating prediction plot...")

    # Convert from log back to minutes
    y_test_mins = np.expm1(y_test) / 60
    y_pred_mins = np.expm1(y_pred) / 60

    # Sample 5000 points for plotting
    idx = np.random.choice(len(y_test_mins), 5000, replace=False)

    plt.figure(figsize=(8, 8))
    plt.scatter(
        y_test_mins.iloc[idx],
        y_pred_mins[idx],
        alpha=0.3,
        s=5,
        color="steelblue"
    )
    plt.plot([0, 120], [0, 120], "r--", linewidth=1.5, label="Perfect prediction")
    plt.xlabel("Actual duration (minutes)")
    plt.ylabel("Predicted duration (minutes)")
    plt.title("Predicted vs Actual Trip Duration")
    plt.xlim(0, 120)
    plt.ylim(0, 120)
    plt.legend()
    plt.tight_layout()
    plt.savefig("predictions_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Prediction plot saved as predictions_vs_actual.png")

def print_business_insights(model, X_test):
    X_sample = X_test.sample(5000, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    feature_importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance": abs(shap_values).mean(axis=0)
    }).sort_values("importance", ascending=False)

    print("\n=== TOP 10 TRIP DURATION DRIVERS ===")
    for _, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:<25} {row['importance']:.4f}")

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    df = remove_outliers(df)
    df = parse_dates(df)
    df = build_features(df)

    X = df.drop(columns=["trip_duration"])
    y = np.log1p(df["trip_duration"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Retrain XGBoost
    print("\nRetraining XGBoost for evaluation...")
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plot_shap(model, X_test)
    plot_predictions(y_test, y_pred)
    print_business_insights(model, X_test)