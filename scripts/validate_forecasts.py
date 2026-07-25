"""
validate_forecasts.py
---------------------
Compare 2024 Bayesian Student-t forecasts to actual 2024 results
for both QB (passing EPA) and RB (rushing EPA).

Inputs
------
    data/processed/bayesian_t_forecasts.csv     -- QB forecasts
    data/processed/rb_bayesian_t_forecasts.csv  -- RB forecasts
    nflverse (via nfl_data_py): 2024 seasonal data

Outputs
-------
    data/processed/qb_forecast_validation_2024.csv
    data/processed/rb_forecast_validation_2024.csv
    Prints calibration summary to stdout

Usage
-----
    python3 scripts/validate_forecasts.py
"""

import os
import numpy as np
import pandas as pd
import nfl_data_py as nfl

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MIN_ATT_QB    = 150   # minimum 2024 pass attempts for QB validation
MIN_CARRIES_RB = 50   # minimum 2024 carries for RB validation


# ── Fetch 2024 actuals ────────────────────────────────────────────────────────

def load_actuals_2024() -> pd.DataFrame:
    print("Fetching 2024 seasonal data from nflverse...")
    df = nfl.import_seasonal_data([2024])
    df = df[df["season_type"] == "REG"].copy() if "season_type" in df.columns else df
    df = df.rename(columns={"player_id": "gsis_id"})

    meta = (
        nfl.import_players()[["gsis_id", "position", "display_name"]]
        .drop_duplicates("gsis_id")
    )
    df = df.merge(meta, on="gsis_id", how="left")
    print(f"  {len(df):,} total player-seasons fetched")
    return df


# ── Calibration metrics ───────────────────────────────────────────────────────

def calibration_report(val: pd.DataFrame, actual_col: str, label: str) -> dict:
    val = val.copy()
    val["error"]     = val[actual_col] - val["mean"]
    val["abs_error"] = val["error"].abs()
    val["in_50"]     = (val[actual_col] >= val["q25"]) & (val[actual_col] <= val["q75"])
    val["in_90"]     = (val[actual_col] >= val["q05"]) & (val[actual_col] <= val["q95"])

    mae   = val["abs_error"].mean()
    bias  = val["error"].mean()
    corr  = val[["mean", actual_col]].corr().iloc[0, 1]
    cov50 = val["in_50"].mean()
    cov90 = val["in_90"].mean()

    print(f"\n── {label} calibration ──────────────────────────────")
    print(f"  Players evaluated:     {len(val)}")
    print(f"  MAE:                   {mae:.2f} EPA")
    print(f"  Bias (actual - pred):  {bias:.2f}  (+ = under-predicted)")
    print(f"  Correlation (r):       {corr:.3f}")
    print(f"  50% interval coverage: {cov50:.1%}  (ideal: 50%)")
    print(f"  90% interval coverage: {cov90:.1%}  (ideal: 90%)")

    return {"mae": mae, "bias": bias, "corr": corr, "cov50": cov50, "cov90": cov90,
            "n": len(val), "val": val}


# ── QB validation ─────────────────────────────────────────────────────────────

def validate_qb(actuals: pd.DataFrame) -> pd.DataFrame:
    fc = pd.read_csv(os.path.join(PROCESSED_DIR, "bayesian_t_forecasts.csv"))
    fc = fc[fc["forecast_year"] == 2024].copy()
    print(f"\nQB: {len(fc)} players forecast for 2024")

    qb_act = actuals[
        (actuals["position"] == "QB") & (actuals["attempts"] >= MIN_ATT_QB)
    ][["gsis_id", "attempts", "passing_epa"]].copy()
    print(f"     {len(qb_act)} QBs with >={MIN_ATT_QB} attempts in 2024 actuals")

    val = fc.merge(qb_act, on="gsis_id", how="inner")
    print(f"     {len(val)} matched")

    result = calibration_report(val, "passing_epa", "QB passing EPA")

    # Player table
    display_cols = ["player", "att_2023", "epa_2023", "mean", "q25", "q75",
                    "passing_epa", "error"]
    out = result["val"][display_cols].sort_values("passing_epa", ascending=False)
    out.columns = ["player", "att_2023", "epa_2023", "pred_mean", "pred_q25",
                   "pred_q75", "actual_epa_2024", "error"]

    print("\n── QB player comparison (sorted by actual 2024 EPA) ─")
    print(out.round(1).to_string(index=False))

    print("\n── QB biggest misses ────────────────────────────────")
    misses = result["val"].nlargest(5, "abs_error")[
        ["player", "mean", "q25", "q75", "passing_epa", "error"]
    ]
    misses.columns = ["player", "pred_mean", "pred_q25", "pred_q75", "actual_epa", "error"]
    print(misses.round(1).to_string(index=False))

    out_path = os.path.join(PROCESSED_DIR, "qb_forecast_validation_2024.csv")
    result["val"].to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return result["val"]


# ── RB validation ─────────────────────────────────────────────────────────────

def validate_rb(actuals: pd.DataFrame) -> pd.DataFrame:
    fc = pd.read_csv(os.path.join(PROCESSED_DIR, "rb_bayesian_t_forecasts.csv"))
    fc = fc[fc["forecast_year"] == 2024].copy()
    print(f"\nRB: {len(fc)} players forecast for 2024")

    rb_act = actuals[
        (actuals["position"] == "RB") & (actuals["carries"] >= MIN_CARRIES_RB)
    ][["gsis_id", "carries", "rushing_epa"]].copy()
    print(f"     {len(rb_act)} RBs with >={MIN_CARRIES_RB} carries in 2024 actuals")

    val = fc.merge(rb_act, on="gsis_id", how="inner")
    print(f"     {len(val)} matched")

    result = calibration_report(val, "rushing_epa", "RB rushing EPA")

    # Player table
    display_cols = ["player", "carries_2023", "ryoe_per_att_2023", "mean",
                    "q25", "q75", "rushing_epa", "error"]
    out = result["val"][display_cols].sort_values("rushing_epa", ascending=False)
    out.columns = ["player", "carries_2023", "ryoe_2023", "pred_mean", "pred_q25",
                   "pred_q75", "actual_epa_2024", "error"]

    print("\n── RB player comparison (sorted by actual 2024 EPA) ─")
    print(out.round(1).to_string(index=False))

    print("\n── RB biggest misses ────────────────────────────────")
    misses = result["val"].nlargest(5, "abs_error")[
        ["player", "mean", "q25", "q75", "rushing_epa", "error"]
    ]
    misses.columns = ["player", "pred_mean", "pred_q25", "pred_q75", "actual_epa", "error"]
    print(misses.round(1).to_string(index=False))

    out_path = os.path.join(PROCESSED_DIR, "rb_forecast_validation_2024.csv")
    result["val"].to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return result["val"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    actuals = load_actuals_2024()
    validate_qb(actuals)
    validate_rb(actuals)


if __name__ == "__main__":
    main()
