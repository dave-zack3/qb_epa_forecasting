"""
scrape_pfr.py
-------------
Pulls QB passing data from nfl_data_py (nflverse) instead of scraping
Pro Football Reference directly.

Outputs
-------
data/raw/passing_seasons.csv   -- one row per player-season (1999-2023)
data/raw/passing_game_logs.csv -- one row per player-game (2018-2023)
data/raw/player_metadata.csv   -- one row per player (DOB, draft info)

Usage
-----
    python3 scripts/scrape_pfr.py
"""

import os
import nfl_data_py as nfl
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

SEASON_START  = 1999
SEASON_END    = 2023
WEEKLY_START  = 2018   # weekly data availability
MIN_ATT       = 1      # minimum attempts to keep a row


# ── Step 1: Season-level data ─────────────────────────────────────────────────

def pull_seasonal() -> pd.DataFrame:
    print(f"  Pulling seasonal data {SEASON_START}-{SEASON_END}...")
    df = nfl.import_seasonal_data(list(range(SEASON_START, SEASON_END + 1)))

    # Keep only players with pass attempts
    df = df[df["attempts"] >= MIN_ATT].copy()

    # Rename for consistency with rest of pipeline
    df = df.rename(columns={
        "player_id":    "gsis_id",
        "attempts":     "att",
        "completions":  "cmp",
        "passing_yards": "yds",
        "passing_tds":  "td",
        "interceptions": "int",
        "passing_epa":  "epa",
    })

    keep_cols = [
        "gsis_id", "season", "games", "att", "cmp", "yds", "td", "int",
        "epa", "passing_air_yards", "passing_yards_after_catch",
        "passing_first_downs", "pacr", "dakota",
        "sacks", "sack_yards", "passing_2pt_conversions",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    print(f"  {len(df):,} player-seasons")
    return df


# ── Step 2: Weekly (game-level) data ─────────────────────────────────────────

def pull_weekly() -> pd.DataFrame:
    print(f"  Pulling weekly data {WEEKLY_START}-{SEASON_END}...")
    df = nfl.import_weekly_data(list(range(WEEKLY_START, SEASON_END + 1)))

    # Keep only players with pass attempts
    df = df[df["attempts"] >= MIN_ATT].copy()

    df = df.rename(columns={
        "player_id":      "gsis_id",
        "attempts":       "att",
        "completions":    "cmp",
        "passing_yards":  "yds",
        "passing_tds":    "td",
        "interceptions":  "int",
        "passing_epa":    "epa",
        "recent_team":    "team",
    })

    keep_cols = [
        "gsis_id", "player_display_name", "season", "week", "team",
        "opponent_team", "att", "cmp", "yds", "td", "int",
        "epa", "passing_air_yards", "passing_yards_after_catch",
        "passing_first_downs", "pacr", "dakota",
        "sacks", "sack_yards",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    print(f"  {len(df):,} player-games")
    return df


# ── Step 3: Player metadata ───────────────────────────────────────────────────

def pull_players() -> pd.DataFrame:
    print("  Pulling player metadata...")
    df = nfl.import_players()

    df = df.rename(columns={"gsis_id": "gsis_id"})

    keep_cols = [
        "gsis_id", "display_name", "first_name", "last_name",
        "position", "position_group", "birth_date",
        "rookie_season", "draft_year", "draft_round", "draft_pick",
        "college_name", "height", "weight", "pfr_id",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")

    print(f"  {len(df):,} players")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 1: Seasonal passing data")
    print("=" * 60)
    df_seasons = pull_seasonal()
    seasons_path = os.path.join(RAW_DIR, "passing_seasons.csv")
    df_seasons.to_csv(seasons_path, index=False)
    print(f"  Saved to {seasons_path}")

    print("\n" + "=" * 60)
    print("Step 2: Weekly (game-level) passing data")
    print("=" * 60)
    df_weekly = pull_weekly()
    weekly_path = os.path.join(RAW_DIR, "passing_game_logs.csv")
    df_weekly.to_csv(weekly_path, index=False)
    print(f"  Saved to {weekly_path}")

    print("\n" + "=" * 60)
    print("Step 3: Player metadata")
    print("=" * 60)
    df_players = pull_players()
    players_path = os.path.join(RAW_DIR, "player_metadata.csv")
    df_players.to_csv(players_path, index=False)
    print(f"  Saved to {players_path}")

    print("\nAll done. Run process_data.py next.")


if __name__ == "__main__":
    main()
