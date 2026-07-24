"""
model_bayesian_studentt.py
--------------------------
Identical to model_bayesian.py with one change: the Normal likelihood is
replaced with a Student-t likelihood with learned degrees of freedom (nu).

MOTIVATION
----------
EDA on our dataset shows:
    Total EPA skewness:       +0.75  (right-skewed, not left as naive intuition suggests)
    Residual excess kurtosis:  1.93  (fatter tails than Normal)

The fat tails are the relevant finding. Outlier seasons (injuries, scheme
changes, breakout years) occur more often than a Normal predicts. A Student-t
with low nu (typically 3-10) accommodates this without assuming a specific
skew direction.

WHAT CHANGES vs model_bayesian.py
----------------------------------
    1. `nu = pm.Gamma("nu", alpha=2, beta=0.1)` added -- nu is learned from
       data; prior is weakly informative, placing mass on 3-30 (fat to moderate
       tails). If the data truly need fat tails, nu will be pulled toward low
       values; if Normal is fine, nu will be large.

    2. Likelihood: pm.Normal -> pm.StudentT with mu, sigma, nu.

    3. Forecast sampling: np.random.normal -> scipy.stats.t (Student-t draws)
       so that forecast intervals also reflect the fat-tail assumption.

    4. Output files use suffix _t to avoid overwriting Normal results:
       bayesian_t_forecasts.csv, bayesian_t_convergence_summary.csv

All other structure (predictors, non-centered parameterization, player-specific
volatility, MIN_ATT_FORECAST filter) is identical.

Inputs
------
    data/processed/seasons_clean.csv

Outputs
-------
    data/processed/bayesian_t_forecasts.csv
    data/processed/bayesian_t_trace.nc
    data/processed/bayesian_t_convergence_summary.csv

Usage
-----
    python3 scripts/model_bayesian_studentt.py
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

MIN_ATT          = 100
MIN_ATT_FORECAST = 300
FORECAST_SEASONS = [2024, 2025]

DRAWS         = 1000
TUNE          = 1000
CHAINS        = 4
TARGET        = 0.9
MIN_DELTA_OBS = 2


def load_data() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "seasons_clean.csv")
    df   = pd.read_csv(path)
    return df[df["att"] >= MIN_ATT].copy()


def compute_player_volatility(df: pd.DataFrame) -> pd.DataFrame:
    def robust_std(x):
        vals = x.dropna()
        return vals.std() if len(vals) >= MIN_DELTA_OBS else np.nan

    vol = (
        df.groupby("gsis_id")
        .agg(
            sigma_att_player     = ("delta_att", robust_std),
            sigma_epa_att_player = ("delta_epa", robust_std),
            n_seasons            = ("season",    "count"),
        )
        .reset_index()
    )

    pop_sigma_att     = vol["sigma_att_player"].median()
    pop_sigma_epa_att = vol["sigma_epa_att_player"].median()

    vol["sigma_att_player"]     = vol["sigma_att_player"].fillna(pop_sigma_att)
    vol["sigma_epa_att_player"] = vol["sigma_epa_att_player"].fillna(pop_sigma_epa_att)

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

        # Per-player intercept (non-centered)
        z_player     = pm.Normal("z_player", mu=0, sigma=1, dims="player")
        alpha_player = pm.Deterministic(
            "alpha_player", z_player * sigma_player, dims="player"
        )

        # Population-level coefficients (identical to Normal model)
        beta_vol     = pm.Normal("beta_vol",     mu=0, sigma=1)
        beta_epa_att = pm.Normal("beta_epa_att", mu=0, sigma=1)
        beta_dakota  = pm.Normal("beta_dakota",  mu=0, sigma=1)
        beta_age     = pm.Normal("beta_age",     mu=0, sigma=1)
        beta_age2    = pm.Normal("beta_age2",    mu=0, sigma=1)
        beta_pacr    = pm.Normal("beta_pacr",    mu=0, sigma=1)

        mu = (
            alpha_player[player_idx]
            + beta_vol     * df["log_att"].values
            + beta_epa_att * df["epa_per_att"].values
            + beta_dakota  * df["dakota"].values
            + beta_age     * df["age_c"].values
            + beta_age2    * df["age_c2"].values
            + beta_pacr    * df["pacr"].values
        )

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=2)

        # ── KEY CHANGE: Student-t likelihood ──────────────────────────────────
        # nu (degrees of freedom) is learned from data.
        # Prior: Gamma(2, 0.1) -- mean=20, but data can pull it toward 3-6
        # if tails are genuinely fat. Large nu recovers the Normal.
        nu = pm.Gamma("nu", alpha=2, beta=0.1)

        pm.StudentT("epa_obs", mu=mu, sigma=sigma_obs, nu=nu,
                    observed=df["epa"].values, dims="obs")
        # ─────────────────────────────────────────────────────────────────────

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
        "nu",   # <- new: what did the data say about tail thickness?
        "sigma_player", "beta_vol", "beta_epa_att", "beta_dakota",
        "beta_age", "beta_age2", "beta_pacr", "sigma_obs"
    ]
    summary = az.summary(trace, var_names=var_names)
    print("\nConvergence summary (R-hat should be ~1.0):")
    print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat"]].to_string())

    nu_mean = summary.loc["nu", "mean"]
    if nu_mean < 10:
        print(f"\n  nu = {nu_mean:.1f} -- data show meaningfully fat tails; "
              f"Normal would underestimate outlier season probability.")
    elif nu_mean < 30:
        print(f"\n  nu = {nu_mean:.1f} -- moderate tails; "
              f"Normal is a reasonable but imperfect approximation.")
    else:
        print(f"\n  nu = {nu_mean:.1f} -- tails are close to Normal; "
              f"Student-t is conservative but harmless.")

    return summary


# ── Forecast (Student-t draws) ─────────────────────────────────────────────────

def forecast_players(df: pd.DataFrame, trace: az.InferenceData,
                     vol: pd.DataFrame, player_map: dict) -> pd.DataFrame:
    base = df[(df["season"] == 2023) & (df["att"] >= MIN_ATT_FORECAST)].copy()
    base = base.merge(vol, on="gsis_id", how="left")

    age_mean  = df["age"].mean()
    posterior = trace.posterior
    n_samples = DRAWS * CHAINS

    # Posterior draws for nu -- used in Student-t forecast sampling
    nu_samples = posterior["nu"].values.flatten()

    records = []
    for _, row in base.iterrows():
        if row["gsis_id"] not in player_map:
            continue

        p_idx      = player_map[row["gsis_id"]]
        alpha_s    = posterior["alpha_player"].values[:, :, p_idx].flatten()
        bvol_s     = posterior["beta_vol"].values.flatten()
        bepa_att_s = posterior["beta_epa_att"].values.flatten()
        bdakota_s  = posterior["beta_dakota"].values.flatten()
        bage_s     = posterior["beta_age"].values.flatten()
        bage2_s    = posterior["beta_age2"].values.flatten()
        bpacr_s    = posterior["beta_pacr"].values.flatten()
        sigma_s    = posterior["sigma_obs"].values.flatten()

        att_0     = row["att"]
        epa_att_0 = row["epa_per_att"]
        age_0     = row["age"]
        sigma_att = row["sigma_att_player"]
        sigma_epa = row["sigma_epa_att_player"]
        dakota_0  = row["dakota"]
        pacr_0    = row["pacr"]

        for i, future_season in enumerate(FORECAST_SEASONS):
            years_ahead = i + 1

            att_proj = np.maximum(
                att_0 + np.random.normal(
                    0, sigma_att * np.sqrt(years_ahead), n_samples
                ),
                1
            )
            epa_att_proj = epa_att_0 + np.random.normal(
                0, sigma_epa * np.sqrt(years_ahead), n_samples
            )

            age_proj = age_0 + years_ahead
            age_c_p  = age_proj - age_mean
            age_c2_p = age_c_p ** 2

            mu_proj = (
                alpha_s
                + bvol_s     * np.log(att_proj)
                + bepa_att_s * epa_att_proj
                + bdakota_s  * dakota_0
                + bage_s     * age_c_p
                + bage2_s    * age_c2_p
                + bpacr_s    * pacr_0
            )

            # ── KEY CHANGE: Student-t forecast draws ─────────────────────────
            # Draw from t(nu, mu_proj, sigma_s) rather than Normal(mu_proj, sigma_s).
            # This propagates the fat-tail assumption into the forecast intervals.
            epa_samples = sp_stats.t.rvs(
                df=nu_samples, loc=mu_proj, scale=sigma_s
            )
            # ─────────────────────────────────────────────────────────────────

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

    print("\nBuilding Student-t model...")
    model, coords, player_map = build_model(df)
    print(model)

    print("\nSampling posterior...")
    trace = sample_model(model)

    print("\nChecking convergence...")
    summary = check_convergence(trace)

    print("\nGenerating forecasts (Student-t draws)...")
    forecasts = forecast_players(df, trace, vol, player_map)

    print("\n2024 top 15 by forecast mean EPA:")
    top = (
        forecasts[forecasts["forecast_year"] == 2024]
        .sort_values("mean", ascending=False)
        .head(15)[["player", "att_2023", "epa_2023", "mean", "q25", "q75", "iqr"]]
    )
    print(top.round(1).to_string(index=False))

    # Compare mean forecasts to Normal model
    normal_path = os.path.join(PROCESSED_DIR, "bayesian_forecasts.csv")
    if os.path.exists(normal_path):
        normal_fc = pd.read_csv(normal_path)
        n24 = normal_fc[normal_fc["forecast_year"]==2024][["player","mean","iqr"]].rename(
            columns={"mean":"normal_mean","iqr":"normal_iqr"})
        t24 = forecasts[forecasts["forecast_year"]==2024][["player","mean","iqr"]].rename(
            columns={"mean":"t_mean","iqr":"t_iqr"})
        comp = n24.merge(t24, on="player").sort_values("t_mean", ascending=False).head(15)
        comp["iqr_delta"] = (comp["t_iqr"] - comp["normal_iqr"]).round(1)
        print("\nNormal vs Student-t (top 15 by t-model mean):")
        print(comp[["player","normal_mean","t_mean","normal_iqr","t_iqr","iqr_delta"]].round(1).to_string(index=False))

    # Save
    fc_path      = os.path.join(PROCESSED_DIR, "bayesian_t_forecasts.csv")
    trace_path   = os.path.join(PROCESSED_DIR, "bayesian_t_trace.nc")
    summary_path = os.path.join(PROCESSED_DIR, "bayesian_t_convergence_summary.csv")

    forecasts.to_csv(fc_path, index=False)
    trace.to_netcdf(trace_path)
    summary.to_csv(summary_path)

    print(f"\nSaved:")
    print(f"  {fc_path}")
    print(f"  {trace_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
