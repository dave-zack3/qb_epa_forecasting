"""
process_data_rb.py
------------------
Fetches and processes RB rushing data from nflverse for modeling.

Data sources (all via nfl_data_py, no proprietary data):
    - import_seasonal_data(['RB'])  -- seasonal rushing stats (carries, rushing_epa, etc.)
    - import_ngs_data('rushing')    -- NGS metrics (RYOE, stacked box rate) 2016+
    - import_players()              -- player metadata (birth date, draft info)

Modeling target: total rushing EPA per season
Primary efficiency predictor: rush_yards_over_expected_per_att (RYOE/att) from NGS
Volume control: log(carries)
Context control: pct_box_8plus (stacked box rate from NGS)

Outputs
-------
    data/processed/rb_seasons_clean.csv

Usage
-----
    python3 scripts/process_data_rb.py
"""

import os
import numpy as np
import pandas as pd
import nfl_data_py as nfl

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

SEASONS     = list(range(2016, 2024))
MIN_CARRIES = 75           # minimum carries to include a player-season in modeling


# ── Fetch ─────────────────────────────────────────────────────────────────────

def load_seasonal(seasons: list) -> pd.DataFrame:
    """
    Seasonal rushing stats from nflverse. Filters to RB position and
    regular season only.
    """
    print("  Fetching seasonal data (nfl_data_py.import_seasonal_data)...")
    df = nfl.import_seasonal_data(seasons)

    # Filter to RBs in regular season
    df = df[df["position"].isin(["RB"])].copy()
    df = df[df["season_type"] == "REG"].copy() if "season_type" in df.columns else df

    # Standardize carry column name (nflverse uses 'carries')
    carry_col = "carries" if "carries" in df.columns else "rushing_attempts"
    df = df.rename(columns={carry_col: "carries"})

    keep = [
        "player_id", "player_display_name", "position", "recent_team",
        "season", "carries", "rushing_yards", "rushing_tds",
        "rushing_epa", "rushing_first_downs",
        "rushing_fumbles", "rushing_fumbles_lost",
    ]
    # Only keep columns that actually exist
    keep = [c for c in keep if c in df.columns]
    df = df[keep].rename(columns={
        "player_id":           "gsis_id",
        "player_display_name": "display_name",
        "recent_team":         "team",
    })

    return df


def load_ngs(seasons: list) -> pd.DataFrame:
    """
    NGS rushing data: RYOE, stacked box rate, efficiency. Filtered to
    week==0 (season-level summaries).
    """
    print("  Fetching NGS rushing data (nfl_data_py.import_ngs_data)...")
    ngs = nfl.import_ngs_data("rushing", seasons=seasons)

    # week == 0 is the full-season summary in NGS
    ngs = ngs[ngs["week"] == 0].copy()

    keep = [
        "player_gsis_id", "season",
        "rush_attempts",
        "rush_yards_over_expected",
        "rush_yards_over_expected_per_att",
        "rush_pct_over_expected",
        "percent_attempts_gte_eight_defenders",
        "avg_time_to_los",
        "efficiency",
    ]
    keep = [c for c in keep if c in ngs.columns]
    ngs = ngs[keep].rename(columns={
        "player_gsis_id":                     "gsis_id",
        "rush_yards_over_expected_per_att":   "ryoe_per_att",
        "rush_yards_over_expected":           "ryoe_total",
        "rush_pct_over_expected":             "rush_pct_oe",
        "percent_attempts_gte_eight_defenders": "pct_box_8plus",
        "avg_time_to_los":                    "avg_time_to_los",
        "efficiency":                         "ngs_efficiency",
    })

    return ngs


def load_metadata() -> pd.DataFrame:
    """
    Player metadata for age calculation. Falls back to existing
    player_metadata.csv if nfl_data_py.import_players() fails.
    """
    print("  Fetching player metadata...")
    try:
        meta = nfl.import_players()
        meta = meta.rename(columns={
            "gsis_id":       "gsis_id",
            "display_name":  "display_name",
            "birth_date":    "birth_date",
            "entry_year":    "rookie_season",
            "draft_year":    "draft_year",
            "draft_round":   "draft_round",
            "draft_number":  "draft_pick",
            "college_name":  "college_name",
        })
        meta["birth_date"] = pd.to_datetime(meta["birth_date"], errors="coerce")
        return meta[["gsis_id", "birth_date", "rookie_season",
                     "draft_year", "draft_round", "draft_pick",
                     "college_name"]].drop_duplicates(subset="gsis_id")
    except Exception as e:
        print(f"    import_players() failed ({e}), trying player_metadata.csv...")
        meta_path = os.path.join(PROCESSED_DIR, "..", "data", "raw", "player_metadata.csv")
        meta = pd.read_csv(meta_path, parse_dates=["birth_date"])
        return meta[["gsis_id", "birth_date", "rookie_season"]].drop_duplicates(subset="gsis_id")


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all derived features for the RB season model.
    """
    df = df.copy()

    # Age at season start (Sept 1)
    df["season_start"] = pd.to_datetime(df["season"].astype(str) + "-09-01")
    df["age"]          = (df["season_start"] - df["birth_date"]).dt.days / 365.25
    df["age_c"]        = df["age"] - df["age"].mean()
    df["age_c2"]       = df["age_c"] ** 2
    df = df.drop(columns=["season_start"])

    # Experience
    df["experience"] = (df["season"] - df["rookie_season"]).clip(lower=0)

    # Volume features
    df["log_carries"]         = np.log(df["carries"].clip(lower=1))
    df["carries_per_game"]    = df["carries"] / 17.0   # approx; could use games col if available
    df["rushing_epa_per_att"] = df["rushing_epa"] / df["carries"].clip(lower=1)

    # Lag features (prior season per player)
    df = df.sort_values(["gsis_id", "season"])
    for col in ["rushing_epa", "carries", "log_carries", "ryoe_per_att",
                "rushing_epa_per_att", "pct_box_8plus"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby("gsis_id")[col].shift(1)

    # Year-over-year deltas (used to estimate player-specific forecast volatility)
    df["delta_rushing_epa"] = df["rushing_epa"] - df["rushing_epa_lag1"]
    df["delta_carries"]     = df["carries"]     - df["carries_lag1"]

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading nflverse data...")
    seasonal = load_seasonal(SEASONS)
    ngs      = load_ngs(SEASONS)
    meta     = load_metadata()

    print(f"\n  Seasonal RB rows (pre-filter): {len(seasonal):,}")
    print(f"  NGS rushing rows (week==0):    {len(ngs):,}")

    # Merge seasonal + NGS
    print("\nMerging seasonal + NGS data...")
    df = seasonal.merge(ngs, on=["gsis_id", "season"], how="left")
    print(f"  {df['ryoe_per_att'].notna().sum()} / {len(df)} rows have NGS data")

    # Merge player metadata
    df = df.merge(meta, on="gsis_id", how="left")
    print(f"  Missing birth_date: {df['birth_date'].isna().sum()}")

    # Minimum carries filter
    df = df[df["carries"] >= MIN_CARRIES].copy()
    print(f"\n  After carry filter (>={MIN_CARRIES}): {len(df):,} player-seasons")
    print(f"  Players: {df['gsis_id'].nunique()}")
    print(f"  Seasons: {sorted(df['season'].unique())}")

    # Feature engineering
    print("\nBuilding features...")
    df = build_features(df)

    # Drop rows with no age (missing birth_date)
    missing_age = df["age"].isna().sum()
    if missing_age:
        print(f"  Dropping {missing_age} rows with missing age")
        df = df.dropna(subset=["age"])

    # Summary stats
    print(f"\nFinal dataset: {len(df):,} player-seasons, {df['gsis_id'].nunique()} players")
    print(f"  rushing_epa range:  {df['rushing_epa'].min():.1f} to {df['rushing_epa'].max():.1f}")
    print(f"  ryoe_per_att range: {df['ryoe_per_att'].min():.3f} to {df['ryoe_per_att'].max():.3f}")
    print(f"  age range:          {df['age'].min():.1f} to {df['age'].max():.1f}")
    print(f"\nTop 10 by rushing EPA (any season):")
    print(df.nlargest(10, "rushing_epa")[
        ["display_name", "season", "carries", "rushing_epa", "ryoe_per_att"]
    ].to_string(index=False))

    # Save
    out = os.path.join(PROCESSED_DIR, "rb_seasons_clean.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(df):,} rows, {len(df.columns)} cols)")
    print("\nColumns:")
    for col in df.columns:
        print(f"  {col}")


if __name__ == "__main__":
    main()
