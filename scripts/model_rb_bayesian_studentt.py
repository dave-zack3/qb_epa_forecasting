"""
model_rb_bayesian_studentt.py
------------------------------
Hierarchical Bayesian model for RB rushing EPA forecasting.
Uses Student-t likelihood with learned degrees of freedom (nu).

Target: rushing_epa_per_att (efficiency)
Forecast: epa_per_att_proj × carries_proj = total rushing EPA

Rationale: total rushing EPA = epa_per_att × carries (multiplicative).
A linear model of log_carries + ryoe_per_att → total EPA is misspecified
(beta_vol went negative; all forecasts were negative). Modeling efficiency
directly and multiplying by projected carries is the correct two-stage approach.

Predictors:
    - ryoe_per_att    (sole population-level predictor; beta_age and beta_box
                       both showed near-zero posteriors and were dropped)

Forecast random walks (player-specific sigma):
    - ryoe_per_att projected forward → feeds efficiency model
    - carries projected forward      → multiplied by efficiency to get total EPA

Inputs
------
    data/processed/rb_seasons_clean.csv

Outputs
-------
    data/processed/rb_bayesian_t_forecasts.csv
    data/processed/rb_bayesian_t_convergence_summary.csv
    data/processed/rb_bayesian_t_trace.nc

Usage
-----
    python3 scripts/model_rb_bayesian_studentt.py
"""

import os
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIGURES_DIR   = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

MIN_CARRIES          = 75    # minimum carries for model training
MIN_CARRIES_FORECAST = 150   # minimum carries in final season for forecast base
FORECAST_SEASONS     = [2024, 2025]

DRAWS         = 1000
TUNE          = 1000
CHAINS        = 4
TARGET        = 0.9
MIN_DELTA_OBS = 2


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "rb_seasons_clean.csv")
    df   = pd.read_csv(path)
    df   = df[df["carries"] >= MIN_CARRIES].copy()
    print(f"  {len(df):,} player-seasons, {df['gsis_id'].nunique()} players")
    print(f"  has_ngs: {df['has_ngs'].sum()} real / {(df['has_ngs']==0).sum()} imputed")
    return df


def compute_player_volatility(df: pd.DataFrame) -> pd.DataFrame:
    def robust_std(x):
        vals = x.dropna()
        return vals.std() if len(vals) >= MIN_DELTA_OBS else np.nan

    vol = (
        df.groupby("gsis_id")
        .agg(
            sigma_carries_player  = ("delta_carries",      robust_std),
            sigma_ryoe_player     = ("delta_ryoe_per_att", robust_std),  # rate, not total EPA
            n_seasons             = ("season",             "count"),
        )
        .reset_index()
    )

    pop_sigma_carries = vol["sigma_carries_player"].median()
    pop_sigma_ryoe    = vol["sigma_ryoe_player"].median()

    vol["sigma_carries_player"] = vol["sigma_carries_player"].fillna(pop_sigma_carries)
    vol["sigma_ryoe_player"]    = vol["sigma_ryoe_player"].fillna(pop_sigma_ryoe)

    # Cap at 75th percentile to prevent starter-to-backup swings inflating IQRs
    carries_cap = vol["sigma_carries_player"].quantile(0.75)
    ryoe_cap    = vol["sigma_ryoe_player"].quantile(0.75)

    vol["sigma_carries_player"] = vol["sigma_carries_player"].clip(lower=5.0,   upper=carries_cap)
    vol["sigma_ryoe_player"]    = vol["sigma_ryoe_player"].clip(lower=0.001, upper=ryoe_cap)

    print(f"  Pop median sigma_carries: {pop_sigma_carries:.1f}  (cap={carries_cap:.1f})")
    print(f"  Pop median sigma_ryoe:    {pop_sigma_ryoe:.4f}  (cap={ryoe_cap:.4f})")
    return vol


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(df: pd.DataFrame):
    players    = df["gsis_id"].unique()
    player_map = {p: i for i, p in enumerate(players)}
    player_idx = df["gsis_id"].map(player_map).values

    coords = {"player": players, "obs": np.arange(len(df))}

    with pm.Model(coords=coords) as model:
        # Player intercepts (non-centered)
        sigma_player = pm.HalfNormal("sigma_player", sigma=0.1)
        z_player     = pm.Normal("z_player", mu=0, sigma=1, dims="player")
        alpha_player = pm.Deterministic(
            "alpha_player", z_player * sigma_player, dims="player"
        )

        # Population-level coefficients
        beta_ryoe = pm.Normal("beta_ryoe", mu=0, sigma=1)  # ryoe_per_att

        mu = (
            alpha_player[player_idx]
            + beta_ryoe * df["ryoe_per_att"].values
        )

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=2)

        # Student-t likelihood with learned nu
        nu = pm.Gamma("nu", alpha=2, beta=0.1)
        pm.StudentT("epa_per_att_obs", mu=mu, sigma=sigma_obs, nu=nu,
                    observed=df["rushing_epa_per_att"].values, dims="obs")

    return model, player_map


def sample_model(model) -> az.InferenceData:
    with model:
        trace = pm.sample(
            draws               = DRAWS,
            tune                = TUNE,
            chains              = CHAINS,
            target_accept       = TARGET,
            return_inferencedata = True,
        )
    return trace


def check_convergence(trace: az.InferenceData) -> pd.DataFrame:
    var_names = [
        "nu", "sigma_player",
        "beta_ryoe",
        "sigma_obs",
    ]
    summary = az.summary(trace, var_names=var_names)
    print("\nConvergence summary (R-hat should be ~1.0):")
    print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat"]].to_string())

    nu_mean = summary.loc["nu", "mean"]
    print(f"\n  nu = {nu_mean:.2f} -- "
          + ("fat tails confirmed." if nu_mean < 10 else
             "moderate tails." if nu_mean < 30 else
             "tails near Normal."))

    return summary


# ── Forecast ──────────────────────────────────────────────────────────────────

def forecast_players(df: pd.DataFrame, trace: az.InferenceData,
                     vol: pd.DataFrame, player_map: dict) -> pd.DataFrame:
    base = df[(df["season"] == 2023) & (df["carries"] >= MIN_CARRIES_FORECAST)].copy()
    base = base.merge(vol, on="gsis_id", how="left")
    print(f"\n  Forecast base: {len(base)} players (carries >= {MIN_CARRIES_FORECAST} in 2023)")

    posterior = trace.posterior
    n_samples = DRAWS * CHAINS
    nu_samples = posterior["nu"].values.flatten()

    records = []
    for _, row in base.iterrows():
        if row["gsis_id"] not in player_map:
            continue

        p_idx     = player_map[row["gsis_id"]]
        alpha_s   = posterior["alpha_player"].values[:, :, p_idx].flatten()
        bryoe_s   = posterior["beta_ryoe"].values.flatten()
        sigma_s   = posterior["sigma_obs"].values.flatten()

        carries_0     = row["carries"]
        ryoe_0        = row["ryoe_per_att"]
        sigma_carries = row["sigma_carries_player"]
        sigma_ryoe    = row["sigma_ryoe_player"]
        ngs_0         = row["has_ngs"]

        for i, future_season in enumerate(FORECAST_SEASONS):
            years_ahead = i + 1

            # Random walk: carries (volume) and ryoe_per_att (efficiency signal)
            carries_proj = np.maximum(
                carries_0 + np.random.normal(
                    0, sigma_carries * np.sqrt(years_ahead), n_samples
                ),
                1
            )
            ryoe_proj = ryoe_0 + np.random.normal(
                0, sigma_ryoe * np.sqrt(years_ahead), n_samples
            )

            # Stage 1: project rushing_epa_per_att (efficiency)
            mu_proj = (
                alpha_s
                + bryoe_s * ryoe_proj
            )
            epa_per_att_samples = sp_stats.t.rvs(
                df=nu_samples, loc=mu_proj, scale=sigma_s
            )

            # Stage 2: total rushing EPA = epa_per_att × carries
            epa_total_samples = epa_per_att_samples * carries_proj

            records.append({
                "gsis_id":              row["gsis_id"],
                "player":               row["display_name"],
                "forecast_year":        future_season,
                "carries_2023":         carries_0,
                "rushing_epa_2023":     row["rushing_epa"],
                "epa_per_att_2023":     row["rushing_epa_per_att"],
                "ryoe_per_att_2023":    ryoe_0,
                "has_ngs":              ngs_0,
                # Efficiency (epa/att)
                "epa_per_att_mean":     epa_per_att_samples.mean(),
                "epa_per_att_q25":      np.quantile(epa_per_att_samples, 0.25),
                "epa_per_att_q75":      np.quantile(epa_per_att_samples, 0.75),
                # Total EPA
                "mean":                 epa_total_samples.mean(),
                "median":               np.median(epa_total_samples),
                "q05":                  np.quantile(epa_total_samples, 0.05),
                "q25":                  np.quantile(epa_total_samples, 0.25),
                "q75":                  np.quantile(epa_total_samples, 0.75),
                "q95":                  np.quantile(epa_total_samples, 0.95),
                "iqr":                  np.quantile(epa_total_samples, 0.75) - np.quantile(epa_total_samples, 0.25),
            })

    return pd.DataFrame(records)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = load_data()

    print("\nComputing player-specific volatility...")
    vol = compute_player_volatility(df)

    print("\nBuilding RB Bayesian Student-t model...")
    model, player_map = build_model(df)
    print(model)

    print("\nSampling posterior...")
    trace = sample_model(model)

    print("\nChecking convergence...")
    summary = check_convergence(trace)

    print("\nGenerating forecasts (Student-t draws)...")
    forecasts = forecast_players(df, trace, vol, player_map)

    print("\n2024 top 15 by forecast mean rushing EPA:")
    top = (
        forecasts[forecasts["forecast_year"] == 2024]
        .sort_values("mean", ascending=False)
        .head(15)[["player", "carries_2023", "rushing_epa_2023",
                   "ryoe_per_att_2023", "epa_per_att_mean", "has_ngs",
                   "mean", "q25", "q75", "iqr"]]
    )
    print(top.round(2).to_string(index=False))

    # Save
    fc_path      = os.path.join(PROCESSED_DIR, "rb_bayesian_t_forecasts.csv")
    summary_path = os.path.join(PROCESSED_DIR, "rb_bayesian_t_convergence_summary.csv")
    trace_path   = os.path.join(PROCESSED_DIR, "rb_bayesian_t_trace.nc")

    forecasts.to_csv(fc_path, index=False)
    summary.to_csv(summary_path)
    trace.to_netcdf(trace_path)

    print(f"\nSaved:")
    print(f"  {fc_path}")
    print(f"  {summary_path}")
    print(f"  {trace_path}")


if __name__ == "__main__":
    main()
