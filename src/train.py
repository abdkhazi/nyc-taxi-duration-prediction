import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

from preprocess import load_data, remove_outliers, parse_dates
from feature_engineering import build_features

def get_train_test(df):
    X = df.drop(columns=["trip_duration"])
    y = df["trip_duration"]

    # Log transform the target — fixes the right skew you saw in preprocess
    # Model predicts log(duration), we convert back at evaluation
    y_log = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    print(f"Train size: {X_train.shape}")
    print(f"Test size:  {X_test.shape}")

    return X_train, X_test, y_train, y_test

def rmsle(y_true, y_pred):
    """
    Root Mean Squared Log Error — the official competition metric.
    Lower is better.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1  # use all CPU cores — needed at 1.4M rows
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Primary metric — RMSLE (what Kaggle uses for this competition)
        score = rmsle(y_test, y_pred)

        # MAE in actual minutes for business interpretability
        # Convert log predictions back to seconds first
        y_pred_seconds = np.expm1(y_pred)
        y_test_seconds = np.expm1(y_test)
        mae_seconds = mean_absolute_error(y_test_seconds, y_pred_seconds)
        mae_minutes = mae_seconds / 60

        print(f"RMSLE: {score:.4f}")
        print(f"MAE:   {mae_minutes:.1f} minutes")

        results[name] = {
            "model": model,
            "rmsle": score,
            "mae_minutes": mae_minutes
        }

    return results

if __name__ == "__main__":
    print("Loading and preparing data...")
    df = load_data()
    df = remove_outliers(df)
    df = parse_dates(df)
    df = build_features(df)

    X_train, X_test, y_train, y_test = get_train_test(df)
    results = train_models(X_train, X_test, y_train, y_test)

    print("\n=== MODEL COMPARISON ===")
    for name, res in results.items():
        print(f"{name}: RMSLE = {res['rmsle']:.4f} | MAE = {res['mae_minutes']:.1f} mins")