"""
model_gp.py
-----------
Gaussian Process model for QB passing EPA forecasting.

Key difference from model_bayesian.py
--------------------------------------
The polynomial aging curve (age_c + age_c^2) is replaced with a Hilbert
Space Gaussian Process (HSGP) over age. This lets the model learn a
flexible population-level career arc rather than forcing a quadratic shape.

A QB who peaks at 29 and declines fast vs one who plateaus at 32 will be
captured correctly by the GP, whereas the polynomial model imposes the same
symmetric arc for everyone.

Model specification
-------------------
    EPA_it ~ Normal(mu_it, sigma_obs)

    mu_it = alpha_player[p]
          + f(age_it)              [GP career curve]
          + beta_vol * log(att_it)
          + beta_epa_att * epa_per_att_it
          + beta_exp * experience_it

    f(age) ~ GP(0, eta^2 * ExpQuad(ell))   [HSGP approximation]
    eta          ~ HalfNormal(10)
    ell          ~ Gamma(4, 1)              [mean ~4 years]
    alpha_player ~ Normal(0, sigma_player)  [non-centered]
    sigma_player ~ HalfNormal(2)
    beta_*       ~ Normal(0, 1)
    sigma_obs    ~ HalfNormal(2)

Inputs
------
    data/processed/seasons_clean.csv

Outputs
-------
    data/processed/gp_forecasts.csv
    data/processed/gp_convergence_summary.csv
    data/processed/gp_trace.nc
    figures/gp_career_arc.png

Usage
-----
    python3 scripts/model_gp.py
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
FIGURES_DIR   = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

MIN_ATT          = 100
MIN_ATT_FORECAST = 300
FORECAST_SEASONS = [2024, 2025]
HSGP_M           = 20      # basis functions for HSGP approximation
DRAWS            = 1000
TUNE             = 1000
CHAINS           = 4
TARGET           = 0.9
MIN_DELTA_OBS    = 2


# ── Data ──────────────────────────────────────────────────────────────────────

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

def build_gp_model(df: pd.DataFrame):
    players    = df["gsis_id"].unique()
    player_map = {p: i for i, p in enumerate(players)}
    player_idx = df["gsis_id"].map(player_map).values.astype(int)

    age_c_obs = df["age_c"].values[:, None]
    L = float(np.abs(df["age_c"]).max() * 1.5)

    with pm.Model() as model:
        # --- GP hyperparameters ---
        ell      = pm.Gamma("ell", alpha=4, beta=1)
        eta      = pm.HalfNormal("eta", sigma=10)
        cov_func = eta**2 * pm.gp.cov.ExpQuad(1, ls=ell)
        gp       = pm.gp.HSGP(m=[HSGP_M], L=[L], cov_func=cov_func)
        f_age    = gp.prior("f_age", X=age_c_obs)

        # --- Player intercepts (non-centered) ---
        sigma_player = pm.HalfNormal("sigma_player", sigma=2)
        z_player     = pm.Normal("z_player", mu=0, sigma=1, shape=len(players))
        alpha_player = pm.Deterministic("alpha_player", z_player * sigma_player)

        # --- Population-level covariates ---
        beta_vol     = pm.Normal("beta_vol",     mu=0, sigma=1)
        beta_epa_att = pm.Normal("beta_epa_att", mu=0, sigma=1)
        beta_dakota  = pm.Normal("beta_dakota",  mu=0, sigma=1)
        beta_pacr    = pm.Normal("beta_pacr",    mu=0, sigma=1)

        mu = (
            alpha_player[player_idx]
            + f_age
            + beta_vol     * df["log_att"].values
            + beta_epa_att * df["epa_per_att"].values
            + beta_dakota  * df["dakota"].values
            + beta_pacr    * df["pacr"].values
        )

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=2)
        pm.Normal("epa_obs", mu=mu, sigma=sigma_obs,
                  observed=df["epa"].values)

    return model, gp, player_map, L


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
        "ell", "eta", "sigma_player",
        "beta_vol", "beta_epa_att", "beta_dakota", "beta_pacr", "sigma_obs"
    ]
    summary = az.summary(trace, var_names=var_names)
    print("\nConvergence summary (R-hat should be ~1.0):")
    print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat"]].to_string())
    return summary


# ── Career arc plot ───────────────────────────────────────────────────────────

def plot_career_curve(trace: az.InferenceData, df: pd.DataFrame,
                      model, gp) -> None:
    age_mean = df["age"].mean()
    age_grid = np.linspace(df["age_c"].min() - 0.5, df["age_c"].max() + 2, 80)

    print("  Sampling GP posterior at forecast ages (career arc)...")
    with model:
        f_new = gp.conditional("f_new", Xnew=age_grid[:, None])
        ppc   = pm.sample_posterior_predictive(
            trace, var_names=["f_new"], progressbar=False
        )

    f_samples = ppc.posterior_predictive["f_new"].values.reshape(-1, len(age_grid))
    age_axis  = age_grid + age_mean

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(age_axis,
                    np.quantile(f_samples, 0.05, axis=0),
                    np.quantile(f_samples, 0.95, axis=0),
                    alpha=0.2, color="steelblue", label="90% HDI")
    ax.fill_between(age_axis,
                    np.quantile(f_samples, 0.25, axis=0),
                    np.quantile(f_samples, 0.75, axis=0),
                    alpha=0.4, color="steelblue", label="50% HDI")
    ax.plot(age_axis, np.median(f_samples, axis=0),
            color="steelblue", lw=2, label="Posterior median")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Age")
    ax.set_ylabel("GP contribution to EPA")
    ax.set_title("Population-Level QB Career Arc (Gaussian Process)")
    ax.legend()
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, "gp_career_arc.png")
    fig.savefig(out, dpi=150)
    print(f"  Career arc plot saved: {out}")


# ── Forecast ──────────────────────────────────────────────────────────────────

def forecast_players(df: pd.DataFrame, trace: az.InferenceData,
                     vol: pd.DataFrame, player_map: dict,
                     model, gp) -> pd.DataFrame:

    base = df[(df["season"] == 2023) & (df["att"] >= MIN_ATT_FORECAST)].copy()
    base = base.merge(vol, on="gsis_id", how="left")

    # Unique forecast ages for GP conditional
    fc_ages_c = np.unique([
        row["age_c"] + (i + 1)
        for _, row in base.iterrows()
        for i in range(len(FORECAST_SEASONS))
    ])

    print(f"  Sampling GP conditional at {len(fc_ages_c)} forecast ages...")
    with model:
        f_fc   = gp.conditional("f_fc", Xnew=fc_ages_c[:, None])
        ppc_fc = pm.sample_posterior_predictive(
            trace, var_names=["f_fc"], progressbar=False
        )

    f_fc_samples = ppc_fc.posterior_predictive["f_fc"].values.reshape(
        -1, len(fc_ages_c)
    )

    posterior = trace.posterior
    n_samples = DRAWS * CHAINS

    records = []
    for _, row in base.iterrows():
        if row["gsis_id"] not in player_map:
            continue

        p_idx      = player_map[row["gsis_id"]]
        alpha_s    = posterior["alpha_player"].values[:, :, p_idx].flatten()
        bvol_s     = posterior["beta_vol"].values.flatten()
        bepa_att_s = posterior["beta_epa_att"].values.flatten()
        bdakota_s  = posterior["beta_dakota"].values.flatten()
        bpacr_s    = posterior["beta_pacr"].values.flatten()
        sigma_s    = posterior["sigma_obs"].values.flatten()

        att_0       = row["att"]
        epa_att_0   = row["epa_per_att"]
        sigma_att   = row["sigma_att_player"]
        sigma_epa   = row["sigma_epa_att_player"]

        # Fixed covariates held at 2023 values
        dakota_0 = row["dakota"]
        pacr_0   = row["pacr"]

        for i, future_season in enumerate(FORECAST_SEASONS):
            years_ahead = i + 1
            target_age_c = row["age_c"] + years_ahead
            fc_idx = int(np.argmin(np.abs(fc_ages_c - target_age_c)))
            f_s = f_fc_samples[:, fc_idx]

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

            mu_proj = (
                alpha_s
                + f_s
                + bvol_s     * np.log(att_proj)
                + bepa_att_s * epa_att_proj
                + bdakota_s  * dakota_0
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
            })

    return pd.DataFrame(records)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = load_data()
    print(f"  {len(df):,} player-seasons, {df['gsis_id'].nunique()} players")

    print("\nComputing player-specific volatility...")
    vol = compute_player_volatility(df)

    print("\nBuilding GP model (HSGP)...")
    model, gp, player_map, L = build_gp_model(df)
    print(f"  HSGP: m={HSGP_M} basis functions, L={L:.2f}")

    print("\nSampling posterior (15-25 min)...")
    trace = sample_model(model)

    print("\nChecking convergence...")
    summary = check_convergence(trace)

    print("\nPlotting population career arc...")
    plot_career_curve(trace, df, model, gp)

    print("\nGenerating forecasts...")
    forecasts = forecast_players(df, trace, vol, player_map, model, gp)

    print("\n2024 top 15 (GP model):")
    top = (
        forecasts[forecasts["forecast_year"] == 2024]
        .sort_values("mean", ascending=False)
        .head(15)[["player", "att_2023", "epa_2023", "mean", "q25", "q75", "iqr"]]
    )
    print(top.round(1).to_string(index=False))

    # Side-by-side comparison with Bayesian model
    bayes_path = os.path.join(PROCESSED_DIR, "bayesian_forecasts.csv")
    if os.path.exists(bayes_path):
        bayes_2024 = pd.read_csv(bayes_path)
        bayes_2024 = bayes_2024[bayes_2024["forecast_year"] == 2024][
            ["player", "mean"]
        ].rename(columns={"mean": "bayes_mean"})

        compare = (
            forecasts[forecasts["forecast_year"] == 2024][["player", "mean"]]
            .rename(columns={"mean": "gp_mean"})
            .merge(bayes_2024, on="player", how="inner")
            .sort_values("gp_mean", ascending=False)
            .head(15)
        )
        print("\nGP vs Bayesian 2024 (top 15 by GP mean):")
        print(compare.round(1).to_string(index=False))

    # Save
    fc_path      = os.path.join(PROCESSED_DIR, "gp_forecasts.csv")
    summary_path = os.path.join(PROCESSED_DIR, "gp_convergence_summary.csv")
    trace_path   = os.path.join(PROCESSED_DIR, "gp_trace.nc")

    forecasts.to_csv(fc_path, index=False)
    summary.to_csv(summary_path)
    trace.to_netcdf(trace_path)

    print(f"\nSaved:")
    print(f"  {fc_path}")
    print(f"  {summary_path}")
    print(f"  {trace_path}")


if __name__ == "__main__":
    main()
