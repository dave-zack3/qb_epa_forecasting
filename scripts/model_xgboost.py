"""
model_xgboost.py
----------------
XGBoost model for predicting next-season passing EPA.
Used as a point-estimate benchmark against the hierarchical Bayesian model.

The target is total passing EPA in season t+1. Volume (log_att) is included
as a feature so the model can learn the same volume/efficiency decomposition
as the Bayesian model, making the comparison fair.

Inputs
------
    data/processed/seasons_clean.csv

Outputs
-------
    data/processed/xgb_predictions.csv
    data/processed/xgb_feature_importance.csv

Usage
-----
    python3 scripts/model_xgboost.py
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Base features -- available for all seasons 1999-2023
BASE_FEATURES = [
    "log_att",
    "epa_per_att",
    "dakota",
    "age_c",
    "age_c2",
    "experience",
    "td_rate",
    "int_rate",
    "ypa",
    "cmp_pct",
    "pacr",
    "sack_rate",
    "epa_lag1",
    "log_att_lag1",
    "epa_per_att_lag1",
    "delta_epa",
    "delta_att",
]

# Game-log features -- only available 2018-2023
GAME_FEATURES = [
    "epa_std",
    "epa_per_att_std",
    "att_cv",
    "pct_games_positive_epa",
    "epa_min",
    "epa_max",
]


def load_data() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "seasons_clean.csv")
    return pd.read_csv(path)


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Target is next-season EPA per player."""
    df = df.sort_values(["gsis_id", "season"]).copy()
    df["epa_next"] = df.groupby("gsis_id")["epa"].shift(-1)
    df["att_next"] = df.groupby("gsis_id")["att"].shift(-1)
    return df.dropna(subset=["epa_next"]).copy()


def build_feature_matrix(df: pd.DataFrame):
    available_game = [f for f in GAME_FEATURES if f in df.columns]
    features       = [f for f in BASE_FEATURES + available_game if f in df.columns]

    # Drop rows missing any base feature
    df_model = df.dropna(subset=[f for f in BASE_FEATURES if f in df.columns]).copy()
    X        = df_model[features].fillna(df_model[features].median())
    y        = df_model["epa_next"]

    return X, y, df_model, features


def train_and_evaluate(X, y):
    model = xgb.XGBRegressor(
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 5,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        random_state     = 42,
        n_jobs           = -1,
    )

    tscv   = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y,
                             cv=tscv, scoring="neg_mean_absolute_error")
    print(f"  CV MAE (5-fold time series): {-scores.mean():.2f} +/- {scores.std():.2f}")

    model.fit(X, y)
    return model


def main():
    print("Loading processed data...")
    df_raw = load_data()
    print(f"  {len(df_raw):,} player-seasons")

    print("\nBuilding target (next-season EPA)...")
    df = build_target(df_raw)
    print(f"  {len(df):,} rows with valid target")

    print("\nBuilding feature matrix...")
    X, y, df_model, features = build_feature_matrix(df)
    print(f"  {len(X):,} rows after dropping missing features")
    print(f"  {len(features)} features: {features}")

    print("\nTraining XGBoost with time-series CV...")
    model = train_and_evaluate(X, y)

    # Predictions and residuals
    df_model = df_model.copy()
    df_model["xgb_pred"] = model.predict(X)
    df_model["xgb_resid"] = df_model["epa_next"] - df_model["xgb_pred"]

    mae = mean_absolute_error(y, df_model["xgb_pred"])
    print(f"\n  In-sample MAE:  {mae:.2f}")
    print(f"  Baseline MAE (predict mean): "
          f"{mean_absolute_error(y, np.full(len(y), y.mean())):.2f}")

    # Feature importance
    importance = pd.DataFrame({
        "feature":    features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(f"\n  Feature importance (top 10):")
    print(importance.head(10).to_string(index=False))

    # 2024/2025 forecasts: use 2023 season as base (from original data, before target drop)
    # Minimum attempt filter removes small-sample backup QBs with inflated efficiency
    MIN_ATT_FORECAST = 300
    base_2023 = df_raw[(df_raw["season"] == 2023) & (df_raw["att"] >= MIN_ATT_FORECAST)].copy()
    base_2023_X = base_2023[
        [f for f in features if f in base_2023.columns]
    ].fillna(base_2023[[f for f in features if f in base_2023.columns]].median())

    base_2023["xgb_forecast_2024"] = model.predict(base_2023_X)

    print(f"\n  Top 15 QB forecasts for 2024 (XGBoost):")
    top = (
        base_2023[["display_name", "season", "att", "epa", "xgb_forecast_2024"]]
        .sort_values("xgb_forecast_2024", ascending=False)
        .head(15)
    )
    print(top.round(1).to_string(index=False))

    # Save
    out_cols = ["gsis_id", "display_name", "season", "att",
                "epa", "epa_next", "xgb_pred", "xgb_resid"]
    pred_path = os.path.join(PROCESSED_DIR, "xgb_predictions.csv")
    imp_path  = os.path.join(PROCESSED_DIR, "xgb_feature_importance.csv")
    fc_path   = os.path.join(PROCESSED_DIR, "xgb_forecasts_2024.csv")

    df_model[out_cols].to_csv(pred_path, index=False)
    importance.to_csv(imp_path, index=False)
    base_2023[["gsis_id", "display_name", "season", "att",
               "epa", "xgb_forecast_2024"]].to_csv(fc_path, index=False)

    print(f"\nSaved:")
    print(f"  {pred_path}")
    print(f"  {imp_path}")
    print(f"  {fc_path}")


if __name__ == "__main__":
    main()
