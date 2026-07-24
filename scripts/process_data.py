"""
process_data.py
---------------
Cleans and merges raw nflverse CSVs into analysis-ready datasets.

Inputs  (data/raw/)
-------
    passing_seasons.csv    -- one row per player-season (1999-2023)
    passing_game_logs.csv  -- one row per player-game (2018-2023)
    player_metadata.csv    -- one row per player

Outputs (data/processed/)
--------
    seasons_clean.csv      -- season-level modeling dataset
    games_clean.csv        -- game-level dataset
    player_features.csv    -- per-player consistency features from game logs

Usage
-----
    python3 scripts/process_data.py
"""

import os
import numpy as np
import pandas as pd

RAW_DIR       = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# Minimum attempts to include in season-level modeling
MIN_ATT = 100


# ── Load raw data ─────────────────────────────────────────────────────────────

def load_raw():
    seasons = pd.read_csv(os.path.join(RAW_DIR, "passing_seasons.csv"))
    games   = pd.read_csv(os.path.join(RAW_DIR, "passing_game_logs.csv"))
    meta    = pd.read_csv(os.path.join(RAW_DIR, "player_metadata.csv"),
                          parse_dates=["birth_date"])
    return seasons, games, meta


# ── Season-level cleaning ─────────────────────────────────────────────────────

def clean_seasons(seasons: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    df = seasons.copy()

    # Merge player metadata on gsis_id
    meta_slim = meta[[
        "gsis_id", "display_name", "first_name", "last_name",
        "position", "birth_date", "rookie_season",
        "draft_year", "draft_round", "draft_pick", "college_name"
    ]].drop_duplicates(subset="gsis_id")

    df = df.merge(meta_slim, on="gsis_id", how="left")

    # Age at season start (Sept 1)
    df["birth_date"]   = pd.to_datetime(df["birth_date"], errors="coerce")
    df["season_start"] = pd.to_datetime(df["season"].astype(str) + "-09-01")
    df["age"]          = (df["season_start"] - df["birth_date"]).dt.days / 365.25
    df["age_c"]        = df["age"] - df["age"].mean()
    df["age_c2"]       = df["age_c"] ** 2

    # Experience: seasons since rookie year
    df["experience"] = (df["season"] - df["rookie_season"]).clip(lower=0)

    # Volume features
    df["log_att"]      = np.log(df["att"].clip(lower=1))
    df["att_per_game"] = df["att"] / df["games"].clip(lower=1)

    # Pocket / pressure proxy
    df["sack_rate"] = df["sacks"] / (df["att"] + df["sacks"]).clip(lower=1)

    # Efficiency rates
    df["epa_per_att"]  = df["epa"] / df["att"].clip(lower=1)
    df["epa_per_game"] = df["epa"] / df["games"].clip(lower=1)
    df["cmp_pct"]      = df["cmp"] / df["att"].clip(lower=1)
    df["td_rate"]      = df["td"]  / df["att"].clip(lower=1)
    df["int_rate"]     = df["int"] / df["att"].clip(lower=1)
    df["ypa"]          = df["yds"] / df["att"].clip(lower=1)

    # Lag features (prior season values per player)
    df = df.sort_values(["gsis_id", "season"])
    for col in ["epa", "att", "log_att", "epa_per_att", "dakota"]:
        df[f"{col}_lag1"] = df.groupby("gsis_id")[col].shift(1)

    # Year-over-year deltas (used to estimate player-specific volatility)
    df["delta_epa"] = df["epa"] - df["epa_lag1"]
    df["delta_att"] = df["att"] - df["att_lag1"]

    # Filter to meaningful seasons
    df = df[df["att"] >= MIN_ATT].copy()
    df = df.drop(columns=["season_start"], errors="ignore")

    return df


# ── Game-level cleaning ───────────────────────────────────────────────────────

def clean_games(games: pd.DataFrame) -> pd.DataFrame:
    df = games.copy()

    # Efficiency at game level
    df["epa_per_att"] = df["epa"] / df["att"].clip(lower=1)
    df["cmp_pct"]     = df["cmp"] / df["att"].clip(lower=1)
    df["td_rate"]     = df["td"]  / df["att"].clip(lower=1)
    df["int_rate"]    = df["int"] / df["att"].clip(lower=1)
    df["ypa"]         = df["yds"] / df["att"].clip(lower=1)

    return df


# ── Player consistency features from game logs ────────────────────────────────

def build_player_features(games: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate game-level stats to player-season level to produce
    consistency and volatility features for use in the season model.
    """
    agg = (
        games.groupby(["gsis_id", "season"])
        .agg(
            games_played      = ("att",         "count"),
            att_mean          = ("att",         "mean"),
            att_std           = ("att",         "std"),
            epa_mean          = ("epa",         "mean"),
            epa_std           = ("epa",         "std"),
            epa_per_att_mean  = ("epa_per_att", "mean"),
            epa_per_att_std   = ("epa_per_att", "std"),
            ypa_mean          = ("ypa",         "mean"),
            ypa_std           = ("ypa",         "std"),
            cmp_pct_mean      = ("cmp_pct",     "mean"),
            td_rate_mean      = ("td_rate",     "mean"),
            int_rate_mean     = ("int_rate",    "mean"),
            epa_min           = ("epa",         "min"),
            epa_max           = ("epa",         "max"),
        )
        .reset_index()
    )

    # Coefficient of variation: normalized volatility
    agg["att_cv"]         = agg["att_std"]         / agg["att_mean"].clip(lower=1)
    agg["epa_cv"]         = agg["epa_std"]         / agg["epa_mean"].abs().clip(lower=0.1)
    agg["epa_per_att_cv"] = agg["epa_per_att_std"] / agg["epa_per_att_mean"].abs().clip(lower=0.01)

    # Fraction of games with positive EPA
    positive_epa = (
        games[games["epa"] > 0]
        .groupby(["gsis_id", "season"])
        .size()
        .rename("games_positive_epa")
        .reset_index()
    )
    agg = agg.merge(positive_epa, on=["gsis_id", "season"], how="left")
    agg["games_positive_epa"] = agg["games_positive_epa"].fillna(0)
    agg["pct_games_positive_epa"] = (
        agg["games_positive_epa"] / agg["games_played"].clip(lower=1)
    )

    return agg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading raw data...")
    seasons, games, meta = load_raw()
    print(f"  seasons: {len(seasons):,} rows")
    print(f"  games:   {len(games):,} rows")
    print(f"  players: {len(meta):,} rows")

    print(f"\nCleaning season data (att >= {MIN_ATT})...")
    df_seasons = clean_seasons(seasons, meta)
    print(f"  {len(df_seasons):,} player-seasons after cleaning")
    print(f"  Seasons covered: {sorted(df_seasons['season'].unique())}")
    print(f"  Null age: {df_seasons['age'].isna().sum()} "
          f"| Null experience: {df_seasons['experience'].isna().sum()}")

    print("\nCleaning game log data...")
    df_games = clean_games(games)
    print(f"  {len(df_games):,} player-games after cleaning")

    print("\nBuilding player consistency features from game logs...")
    df_features = build_player_features(df_games)
    print(f"  {len(df_features):,} player-seasons with game-level features")

    # Merge game-level features into season dataset
    df_seasons = df_seasons.merge(
        df_features, on=["gsis_id", "season"], how="left"
    )
    print(f"  {df_seasons['epa_std'].notna().sum()} seasons with game-level features joined")

    # Save
    seasons_out  = os.path.join(PROCESSED_DIR, "seasons_clean.csv")
    games_out    = os.path.join(PROCESSED_DIR, "games_clean.csv")
    features_out = os.path.join(PROCESSED_DIR, "player_features.csv")

    df_seasons.to_csv(seasons_out, index=False)
    df_games.to_csv(games_out, index=False)
    df_features.to_csv(features_out, index=False)

    print(f"\nSaved:")
    print(f"  {seasons_out}  ({len(df_seasons):,} rows, {len(df_seasons.columns)} cols)")
    print(f"  {games_out}  ({len(df_games):,} rows)")
    print(f"  {features_out}  ({len(df_features):,} rows)")
    print(f"\nColumns in seasons_clean:")
    for col in df_seasons.columns:
        print(f"  {col}")


if __name__ == "__main__":
    main()
