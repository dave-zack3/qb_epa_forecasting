"""
model_bayesian.py
-----------------
Hierarchical Bayesian model for QB passing EPA forecasting.

Key improvements over the Raiders WAR model
--------------------------------------------
1. Player-specific random walk variance: each player's EPA and attempt
   projections use their own historical year-over-year volatility rather
   than a single population-level SD. Consistent starters get tighter
   forecast distributions; erratic or low-data players get wider ones.
2. Public data (nflverse) -- fully reproducible.
3. Total passing EPA as target -- volume-controlled via log(att).

Model specification
-------------------
    EPA_it ~ Normal(mu_it, sigma_obs)

    mu_it = alpha_player[p]
          + beta_vol     * log(att_it)
          + beta_epa_att * epa_per_att_it
          + beta_age     * age_c_it
          + beta_age2    * age_c2_it
          + beta_exp     * experience_it

    alpha_player[p] ~ Normal(0, sigma_player)   [non-centered]
    sigma_player    ~ HalfNormal(2)
    beta_*          ~ Normal(0, 1)
    sigma_obs       ~ HalfNormal(2)

Forecast
--------
    For each player with a 2023 season, project 2024 and 2025 EPA by:
      - Advancing age and experience
      - Sampling next-year att and epa_per_att as random walks using
        PLAYER-SPECIFIC sigma estimated from their own delta history
      - Propagating both posterior parameter uncertainty and predictor
        uncertainty through the linear model

Inputs
------
    data/processed/seasons_clean.csv

Outputs
-------
    data/processed/bayesian_forecasts.csv
    data/processed/bayesian_trace.nc
    data/processed/convergence_summary.csv

Usage
-----
    python3 scripts/model_bayesian.py
"""

import os
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

MIN_ATT                = 100
AGING_CURVE_EXCLUDE    = ["Tom Brady", "Drew Brees"]
FORECAST_SEASONS       = [2024, 2025]

DRAWS   = 1000
TUNE    = 1000
CHAINS  = 4
TARGET  = 0.9

# Population fallback for players with fewer than 2 delta observations
MIN_DELTA_OBS = 2


def load_data() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "seasons_clean.csv")
    df   = pd.read_csv(path)
    return df[df["att"] >= MIN_ATT].copy()


# ── Player-specific volatility ────────────────────────────────────────────────

def compute_player_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate player-specific year-over-year volatility in attempts and
    EPA per attempt from historical deltas.
    Players with fewer than MIN_DELTA_OBS observations fall back to
    the population median.
    """
    def robust_std(x):
        vals = x.dropna()
        return vals.std() if len(vals) >= MIN_DELTA_OBS else np.nan

    vol = (
        df.groupby("gsis_id")
        .agg(
            sigma_att_player     = ("delta_att",     robust_std),
            sigma_epa_att_player = ("delta_epa",     robust_std),
            n_seasons            = ("season",        "count"),
        )
        .reset_index()
    )

    pop_sigma_att     = vol["sigma_att_player"].median()
    pop_sigma_epa_att = vol["sigma_epa_att_player"].median()

    vol["sigma_att_player"]     = vol["sigma_att_player"].fillna(pop_sigma_att)
    vol["sigma_epa_att_player"] = vol["sigma_epa_att_player"].fillna(pop_sigma_epa_att)

    # Floor to avoid degenerate zero variance; cap at 75th percentile to prevent
    # starter-to-backup swings from producing absurd forecast IQRs
    att_cap     = vol["sigma_att_player"].quantile(0.75)
    epa_att_cap = vol["sigma_epa_att_player"].quantile(0.75)

    vol["sigma_att_player"]     = vol["sigma_att_player"].clip(lower=5.0,   upper=att_cap)
    vol["sigma_epa_att_player"] = vol["sigma_epa_att_player"].clip(lower=0.001, upper=epa_att_cap)

    return vol


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(df: pd.DataFrame):
    players    = df["gsis_id"].unique()
    player_map = {p: i for i, p in enumerate(players)}
    player_idx = df["gsis_id"].map(player_map).values

    coords = {"player": players, "obs": np.arange(len(df))}

    with pm.Model(coords=coords) as model:
        # Hyperprior
        sigma_player = pm.HalfNormal("sigma_player", sigma=2)

        # Per-player intercept (non-centered parameterization)
        z_player     = pm.Normal("z_player", mu=0, sigma=1, dims="player")
        alpha_player = pm.Deterministic(
            "alpha_player", z_player * sigma_player, dims="player"
        )

        # Population-level coefficients
        beta_vol     = pm.Normal("beta_vol",     mu=0, sigma=1)
        beta_epa_att = pm.Normal("beta_epa_att", mu=0, sigma=1)
        beta_dakota  = pm.Normal("beta_dakota",  mu=0, sigma=1)
        beta_age     = pm.Normal("beta_age",     mu=0, sigma=1)
        beta_age2    = pm.Normal("beta_age2",    mu=0, sigma=1)
        beta_pacr    = pm.Normal("beta_pacr",    mu=0, sigma=1)

        # Linear predictor
        mu = (
            alpha_player[player_idx]
            + beta_vol     * df["log_att"].values
            + beta_epa_att * df["epa_per_att"].values
            + beta_dakota  * df["dakota"].values
            + beta_age     * df["age_c"].values
            + beta_age2    * df["age_c2"].values
            + beta_pacr    * df["pacr"].values
        )

        # Likelihood
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=2)
        pm.Normal("epa_obs", mu=mu, sigma=sigma_obs,
                  observed=df["epa"].values, dims="obs")

    return model, coords, player_map


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
        "sigma_player", "beta_vol", "beta_epa_att", "beta_dakota",
        "beta_age", "beta_age2", "beta_pacr", "sigma_obs"
    ]
    summary = az.summary(trace, var_names=var_names)
    print("\nConvergence summary (R-hat should be ~1.0):")
    print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat"]].to_string())
    return summary


# ── Forecast ──────────────────────────────────────────────────────────────────

def forecast_players(df: pd.DataFrame, trace: az.InferenceData,
                     vol: pd.DataFrame, player_map: dict) -> pd.DataFrame:
    """
    Generate 2024 and 2025 EPA forecasts using player-specific random walk
    variance for att and epa_per_att projections.
    """
    MIN_ATT_FORECAST = 300
    base = df[(df["season"] == 2023) & (df["att"] >= MIN_ATT_FORECAST)].copy()
    base = base.merge(vol, on="gsis_id", how="left")

    age_mean  = df["age"].mean()
    posterior = trace.posterior
    n_samples = DRAWS * CHAINS

    records = []
    for _, row in base.iterrows():
        if row["gsis_id"] not in player_map:
            continue

        p_idx       = player_map[row["gsis_id"]]
        alpha_s     = posterior["alpha_player"].values[:, :, p_idx].flatten()
        bvol_s      = posterior["beta_vol"].values.flatten()
        bepa_att_s  = posterior["beta_epa_att"].values.flatten()
        bdakota_s   = posterior["beta_dakota"].values.flatten()
        bage_s      = posterior["beta_age"].values.flatten()
        bage2_s     = posterior["beta_age2"].values.flatten()
        bpacr_s     = posterior["beta_pacr"].values.flatten()
        sigma_s     = posterior["sigma_obs"].values.flatten()

        att_0       = row["att"]
        epa_att_0   = row["epa_per_att"]
        age_0       = row["age"]
        sigma_att   = row["sigma_att_player"]
        sigma_epa   = row["sigma_epa_att_player"]

        # Fixed covariates held at 2023 values for forecast
        dakota_0 = row["dakota"]
        pacr_0   = row["pacr"]

        for i, future_season in enumerate(FORECAST_SEASONS):
            years_ahead = i + 1

            # Player-specific random walk for volume and efficiency
            att_proj = np.maximum(
                att_0 + np.random.normal(
                    0, sigma_att * np.sqrt(years_ahead), n_samples
                ),
                1
            )
            epa_att_proj = epa_att_0 + np.random.normal(
                0, sigma_epa * np.sqrt(years_ahead), n_samples
            )

            age_proj  = age_0 + years_ahead
            age_c_p   = age_proj - age_mean
            age_c2_p  = age_c_p ** 2
            log_att_p = np.log(att_proj)

            mu_proj = (
                alpha_s
                + bvol_s     * log_att_p
                + bepa_att_s * epa_att_proj
                + bdakota_s  * dakota_0
                + bage_s     * age_c_p
                + bage2_s    * age_c2_p
                + bpacr_s    * pacr_0
            )

            epa_samples = np.random.normal(mu_proj, sigma_s)

            records.append({
                "gsis_id":       row["gsis_id"],
                "player":        row["display_name"],
                "forecast_year": future_season,
                "att_2023":      att_0,
                "epa_2023":      row["epa"],
                "mean":          epa_samples.mean(),
                "median":        np.median(epa_samples),
                "q05":           np.quantile(epa_samples, 0.05),
                "q25":           np.quantile(epa_samples, 0.25),
                "q75":           np.quantile(epa_samples, 0.75),
                "q95":           np.quantile(epa_samples, 0.95),
                "iqr":           np.quantile(epa_samples, 0.75) - np.quantile(epa_samples, 0.25),
                "sigma_att":     sigma_att,
                "sigma_epa_att": sigma_epa,
            })

    return pd.DataFrame(records)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading processed data...")
    df = load_data()
    print(f"  {len(df):,} player-seasons (att >= {MIN_ATT})")
    print(f"  {df['gsis_id'].nunique()} unique players")

    print("\nComputing player-specific volatility...")
    vol = compute_player_volatility(df)
    print(f"  Population median sigma_att:     {vol['sigma_att_player'].median():.1f}")
    print(f"  Population median sigma_epa_att: {vol['sigma_epa_att_player'].median():.4f}")

    print("\nBuilding model...")
    model, coords, player_map = build_model(df)
    print(model)

    print("\nSampling posterior (this takes ~10-15 min)...")
    trace = sample_model(model)

    print("\nChecking convergence...")
    summary = check_convergence(trace)

    print("\nGenerating forecasts...")
    forecasts = forecast_players(df, trace, vol, player_map)

    print("\n2024 top 15 by forecast mean EPA:")
    top = (
        forecasts[forecasts["forecast_year"] == 2024]
        .sort_values("mean", ascending=False)
        .head(15)[["player", "att_2023", "epa_2023", "mean", "q25", "q75", "iqr"]]
    )
    print(top.round(1).to_string(index=False))

    # Save
    fc_path      = os.path.join(PROCESSED_DIR, "bayesian_forecasts.csv")
    trace_path   = os.path.join(PROCESSED_DIR, "bayesian_trace.nc")
    summary_path = os.path.join(PROCESSED_DIR, "convergence_summary.csv")

    forecasts.to_csv(fc_path, index=False)
    trace.to_netcdf(trace_path)
    summary.to_csv(summary_path)

    print(f"\nSaved:")
    print(f"  {fc_path}")
    print(f"  {trace_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
