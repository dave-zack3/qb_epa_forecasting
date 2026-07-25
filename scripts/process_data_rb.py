"""
process_data_rb.py
------------------
Fetches and processes RB rushing data from nflverse for modeling.

Data sources (all via nfl_data_py, no proprietary data):
    - import_seasonal_data()   -- seasonal stats (carries, rushing_epa, etc.) 2016+
    - import_ngs_data()        -- NGS metrics (RYOE, stacked box rate) 2016+
    - import_players()         -- player metadata (position, birth date, draft info)

Note: import_seasonal_data returns all positions without a position column.
      Position filtering is done after merging with import_players() metadata.

Modeling target: total rushing EPA per season
Primary efficiency predictor: ryoe_per_att (RYOE/att from NGS)
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
MIN_CARRIES = 75   # minimum carries to include a player-season in modeling


# ── Fetch ─────────────────────────────────────────────────────────────────────

def load_seasonal(seasons: list) -> pd.DataFrame:
    """
    Seasonal stats from nflverse. No position column is returned here --
    position filtering happens after merging with player metadata.
    """
    print("  Fetching seasonal data...")
    df = nfl.import_seasonal_data(seasons)
    df = df[df["season_type"] == "REG"].copy() if "season_type" in df.columns else df

    keep = [
        "player_id", "season",
        "carries", "rushing_yards", "rushing_tds",
        "rushing_epa", "rushing_first_downs",
        "rushing_fumbles", "rushing_fumbles_lost",
        "games",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].rename(columns={"player_id": "gsis_id"})
    print(f"    {len(df):,} player-seasons (all positions, pre-filter)")
    return df


def load_ngs(seasons: list) -> pd.DataFrame:
    """
    NGS rushing data: RYOE, stacked box rate, efficiency.
    Filtered to week==0 which is the full-season summary.
    """
    print("  Fetching NGS rushing data...")
    # import_ngs_data signature: stat_type, years (positional)
    try:
        ngs = nfl.import_ngs_data("rushing", seasons)
    except TypeError:
        ngs = nfl.import_ngs_data("rushing")
        ngs = ngs[ngs["season"].isin(seasons)] if "season" in ngs.columns else ngs
    ngs = ngs[ngs["week"] == 0].copy()

    keep = [
        "player_gsis_id", "season",
        "rush_yards_over_expected",
        "rush_yards_over_expected_per_att",
        "rush_pct_over_expected",
        "percent_attempts_gte_eight_defenders",
        "avg_time_to_los",
        "efficiency",
    ]
    keep = [c for c in keep if c in ngs.columns]
    ngs = ngs[keep].rename(columns={
        "player_gsis_id":                       "gsis_id",
        "rush_yards_over_expected_per_att":     "ryoe_per_att",
        "rush_yards_over_expected":             "ryoe_total",
        "rush_pct_over_expected":               "rush_pct_oe",
        "percent_attempts_gte_eight_defenders": "pct_box_8plus",
        "efficiency":                           "ngs_efficiency",
    })
    print(f"    {len(ngs):,} player-seasons in NGS (week==0)")
    return ngs


def load_metadata() -> pd.DataFrame:
    """
    Player metadata: position, display name, birth date, draft info.
    Used to filter to RBs and compute age.
    """
    print("  Fetching player metadata...")
    meta = nfl.import_players()

    # Map known column name variants
    rename = {}
    for target, candidates in {
        "display_name":  ["display_name", "player_display_name", "short_name"],
        "position":      ["position", "position_group"],
        "birth_date":    ["birth_date"],
        "rookie_season": ["entry_year", "rookie_year", "rookie_season"],
        "draft_round":   ["draft_round"],
        "draft_pick":    ["draft_number", "draft_pick"],
    }.items():
        for c in candidates:
            if c in meta.columns and c != target:
                rename[c] = target
                break
    meta = meta.rename(columns=rename)

    # Drop duplicate columns that arose from renaming both 'position' and 'position_group'
    meta = meta.loc[:, ~meta.columns.duplicated(keep="first")]

    meta["birth_date"] = pd.to_datetime(meta["birth_date"], errors="coerce")
    keep = [c for c in ["gsis_id", "display_name", "position", "birth_date",
                         "rookie_season", "draft_round", "draft_pick"]
            if c in meta.columns]
    meta = meta[keep].drop_duplicates(subset="gsis_id")
    print(f"    {len(meta):,} players total")
    print(f"    RBs in metadata: {(meta['position'] == 'RB').sum()}")
    return meta


# ── Feature engineering ───────────────────────────────────────────────────────

def impute_ryoe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing ryoe_per_att using each player's own mean across seasons
    where NGS data exists. Falls back to population mean for players with
    zero NGS seasons. Adds has_ngs flag (1 = real NGS data, 0 = imputed).
    """
    df = df.copy()
    df["has_ngs"] = df["ryoe_per_att"].notna().astype(int)

    # Player-level mean from real NGS seasons
    player_mean = (
        df[df["has_ngs"] == 1]
        .groupby("gsis_id")["ryoe_per_att"]
        .mean()
        .rename("ryoe_player_mean")
    )
    df = df.merge(player_mean, on="gsis_id", how="left")

    # Population mean as final fallback
    pop_mean = df.loc[df["has_ngs"] == 1, "ryoe_per_att"].mean()

    df["ryoe_per_att"] = df["ryoe_per_att"].fillna(
        df["ryoe_player_mean"].fillna(pop_mean)
    )
    df = df.drop(columns=["ryoe_player_mean"])

    n_imputed = (df["has_ngs"] == 0).sum()
    n_player_imputed = df[df["has_ngs"] == 0]["ryoe_player_mean"].notna().sum() if "ryoe_player_mean" in df.columns else None
    print(f"    ryoe_per_att: {(df['has_ngs']==1).sum()} real NGS, "
          f"{n_imputed} imputed (player mean where available, else pop mean={pop_mean:.3f})")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Impute missing NGS columns before computing lags
    df = impute_ryoe(df)

    # pct_box_8plus: same NGS coverage gap, impute with player mean then pop mean
    pop_box = df.loc[df["has_ngs"] == 1, "pct_box_8plus"].mean()
    player_box = (
        df[df["has_ngs"] == 1]
        .groupby("gsis_id")["pct_box_8plus"]
        .mean()
        .rename("box_player_mean")
    )
    df = df.merge(player_box, on="gsis_id", how="left")
    df["pct_box_8plus"] = df["pct_box_8plus"].fillna(
        df["box_player_mean"].fillna(pop_box)
    )
    df = df.drop(columns=["box_player_mean"])

    # Age at season start (Sept 1)
    df["season_start"] = pd.to_datetime(df["season"].astype(str) + "-09-01")
    df["age"]          = (df["season_start"] - df["birth_date"]).dt.days / 365.25
    df["age_c"]        = df["age"] - df["age"].mean()
    df["age_c2"]       = df["age_c"] ** 2
    df = df.drop(columns=["season_start"])

    # Experience
    if "rookie_season" in df.columns:
        df["experience"] = (df["season"] - df["rookie_season"]).clip(lower=0)

    # Volume
    df["log_carries"]         = np.log(df["carries"].clip(lower=1))
    df["rushing_epa_per_att"] = df["rushing_epa"] / df["carries"].clip(lower=1)

    # Lag features per player
    df = df.sort_values(["gsis_id", "season"])
    for col in ["rushing_epa", "carries", "log_carries",
                "ryoe_per_att", "rushing_epa_per_att", "pct_box_8plus"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby("gsis_id")[col].shift(1)

    # YoY deltas for player-specific volatility estimation
    df["delta_rushing_epa"] = df["rushing_epa"]   - df["rushing_epa_lag1"]
    df["delta_carries"]     = df["carries"]        - df["carries_lag1"]
    df["delta_ryoe_per_att"] = df["ryoe_per_att"] - df["ryoe_per_att_lag1"]

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading nflverse data...")
    seasonal = load_seasonal(SEASONS)
    ngs      = load_ngs(SEASONS)
    meta     = load_metadata()

    # Filter metadata to RBs and merge into seasonal
    rb_ids = meta[meta["position"] == "RB"]["gsis_id"].unique()
    seasonal = seasonal[seasonal["gsis_id"].isin(rb_ids)].copy()
    print(f"\n  RB player-seasons (pre-carry filter): {len(seasonal):,}")

    # Merge seasonal + NGS + metadata
    df = seasonal.merge(ngs, on=["gsis_id", "season"], how="left")
    df = df.merge(meta, on="gsis_id", how="left")

    ngs_fill = df["ryoe_per_att"].notna().sum()
    print(f"  NGS merge fill rate: {ngs_fill}/{len(df)} ({ngs_fill/len(df):.1%})")

    # Carry filter
    df = df[df["carries"] >= MIN_CARRIES].copy()
    print(f"  After carry filter (>={MIN_CARRIES}): {len(df):,} player-seasons, "
          f"{df['gsis_id'].nunique()} players")

    # Drop rows missing birth_date (can't compute age)
    missing_age = df["birth_date"].isna().sum()
    if missing_age:
        print(f"  Dropping {missing_age} rows with missing birth_date")
        df = df.dropna(subset=["birth_date"])

    # Feature engineering
    print("\nBuilding features...")
    df = build_features(df)

    # Summary
    print(f"\nFinal dataset: {len(df):,} player-seasons, {df['gsis_id'].nunique()} players")
    print(f"  Seasons: {sorted(df['season'].unique())}")
    print(f"  Age range: {df['age'].min():.1f} to {df['age'].max():.1f}")
    print(f"  rushing_epa range: {df['rushing_epa'].min():.1f} to {df['rushing_epa'].max():.1f}")
    if df["ryoe_per_att"].notna().any():
        print(f"  ryoe_per_att range: {df['ryoe_per_att'].min():.3f} to {df['ryoe_per_att'].max():.3f}")

    print(f"\nTop 10 seasons by rushing EPA:")
    print(df.nlargest(10, "rushing_epa")[
        ["display_name", "season", "carries", "rushing_epa", "ryoe_per_att"]
    ].round(2).to_string(index=False))

    # Save
    out = os.path.join(PROCESSED_DIR, "rb_seasons_clean.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(df):,} rows, {len(df.columns)} cols)")
    print("\nColumns:")
    for col in df.columns:
        print(f"  {col}")


if __name__ == "__main__":
    main()
