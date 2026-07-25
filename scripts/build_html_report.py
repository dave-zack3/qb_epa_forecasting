"""
build_html_report.py
--------------------
Generates docs/index.html: a self-contained report covering QB and RB
EPA forecasting models, validation against 2024 actuals, and player
spotlights comparing high-sample (veteran) vs low-sample (newer) players.

All figures are embedded as base64 so the page is a single portable file.

Usage
-----
    python3 scripts/build_html_report.py
"""

import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

PROCESSED = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
DOCS      = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(DOCS, exist_ok=True)

GITHUB = "https://github.com/dave-zack3/football_player_forecasting"

# ── Colour palette ─────────────────────────────────────────────────────────────
C_BAYES  = "#2563EB"   # blue
C_GP     = "#059669"   # green
C_XGB    = "#D97706"   # amber
C_ACTUAL = "#DC2626"   # red
C_BG     = "#F8FAFC"
C_GRID   = "#E2E8F0"


# ── Helpers ────────────────────────────────────────────────────────────────────

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def card(title: str, value: str, note: str = "") -> str:
    return f"""
    <div class="card">
      <div class="card-title">{title}</div>
      <div class="card-value">{value}</div>
      {"<div class='card-note'>" + note + "</div>" if note else ""}
    </div>"""


# ── Load data ──────────────────────────────────────────────────────────────────

def load_data():
    bayes  = pd.read_csv(f"{PROCESSED}/bayesian_t_forecasts.csv")
    gp     = pd.read_csv(f"{PROCESSED}/gp_t_forecasts.csv")
    xgb    = pd.read_csv(f"{PROCESSED}/xgb_forecasts_2024.csv")
    qb_val = pd.read_csv(f"{PROCESSED}/qb_forecast_validation_2024.csv")

    rb_bayes = pd.read_csv(f"{PROCESSED}/rb_bayesian_t_forecasts.csv")
    rb_val   = pd.read_csv(f"{PROCESSED}/rb_forecast_validation_2024.csv")

    bayes24 = bayes[bayes["forecast_year"] == 2024].copy()
    gp24    = gp[gp["forecast_year"] == 2024].copy()
    rb24    = rb_bayes[rb_bayes["forecast_year"] == 2024].copy()

    # Merge XGB into QB validation
    xgb_slim = xgb[["gsis_id", "xgb_forecast_2024"]].copy()
    qb_val   = qb_val.merge(xgb_slim, on="gsis_id", how="left")

    # Merge GP into QB validation
    gp_slim = gp24[["gsis_id", "mean", "q05", "q25", "q75", "q95"]].rename(
        columns={c: f"gp_{c}" for c in ["mean", "q05", "q25", "q75", "q95"]}
    )
    qb_val = qb_val.merge(gp_slim, on="gsis_id", how="left")

    # Merge Bayesian back in (qb_val only has the fixed version)
    b_slim = bayes24[["gsis_id", "mean", "q05", "q25", "q75", "q95"]].rename(
        columns={c: f"b_{c}" for c in ["mean", "q05", "q25", "q75", "q95"]}
    )
    qb_val = qb_val.merge(b_slim, on="gsis_id", how="left")

    return bayes24, gp24, xgb, qb_val, rb24, rb_val


# ── QB Forest Plot ─────────────────────────────────────────────────────────────

def plot_qb_forest(qb_val: pd.DataFrame) -> str:
    df = qb_val.sort_values("passing_epa").reset_index(drop=True)
    n  = len(df)
    y  = np.arange(n)

    fig, ax = plt.subplots(figsize=(11, 8), facecolor=C_BG)
    ax.set_facecolor(C_BG)

    for i, row in df.iterrows():
        # Bayesian 90% interval
        ax.plot([row["q05"], row["q95"]], [i, i],
                color=C_BAYES, lw=1.2, alpha=0.4, solid_capstyle="round")
        # Bayesian 50% interval
        ax.plot([row["q25"], row["q75"]], [i, i],
                color=C_BAYES, lw=3.5, alpha=0.7, solid_capstyle="round")
        # Bayesian mean
        ax.scatter(row["b_mean"], i, color=C_BAYES, s=30, zorder=4)
        # XGBoost point estimate
        if not pd.isna(row.get("xgb_forecast_2024")):
            ax.scatter(row["xgb_forecast_2024"], i,
                       color=C_XGB, marker="D", s=28, zorder=5)
        # Actual 2024
        ax.scatter(row["passing_epa"], i,
                   color=C_ACTUAL, marker="|", s=120, lw=2, zorder=6)

    ax.set_yticks(y)
    ax.set_yticklabels(df["player"], fontsize=7.5)
    ax.axvline(0, color="#94A3B8", lw=0.8, ls="--")
    ax.set_xlabel("Total Passing EPA", fontsize=9)
    ax.set_title("QB 2024 Forecast vs Actual — All Validated Players",
                 fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="x", color=C_GRID, lw=0.6)
    ax.spines[["top","right","left"]].set_visible(False)

    legend_els = [
        Line2D([0],[0], color=C_BAYES, lw=3.5, label="Bayesian 50% interval"),
        Line2D([0],[0], color=C_BAYES, lw=1.2, alpha=0.5, label="Bayesian 90% interval"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_BAYES,
               markersize=6, label="Bayesian mean"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor=C_XGB,
               markersize=6, label="XGBoost forecast"),
        Line2D([0],[0], marker="|", color=C_ACTUAL, markersize=10,
               lw=2, label="Actual 2024 EPA"),
    ]
    ax.legend(handles=legend_els, fontsize=7.5, loc="lower right",
              framealpha=0.9, edgecolor=C_GRID)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── QB Spotlight (high / low sample) ──────────────────────────────────────────

def plot_qb_spotlight(players: list, qb_val: pd.DataFrame, title: str) -> str:
    df = qb_val[qb_val["player"].isin(players)].set_index("player")
    n  = len(players)

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5),
                             sharey=False, facecolor=C_BG)
    if n == 1:
        axes = [axes]

    for ax, player in zip(axes, players):
        ax.set_facecolor(C_BG)
        if player not in df.index:
            ax.set_title(player, fontsize=9)
            continue
        row = df.loc[player]

        models = []

        # Bayesian
        b_mid  = (row["b_q25"] + row["b_q75"]) / 2
        b_h50  = (row["b_q75"] - row["b_q25"]) / 2
        b_h90  = (row["b_q95"] - row["b_q05"]) / 2
        ax.barh(0, 2*b_h90, left=row["b_q05"],
                height=0.3, color=C_BAYES, alpha=0.25, label="_")
        ax.barh(0, 2*b_h50, left=row["b_q25"],
                height=0.3, color=C_BAYES, alpha=0.65, label="_")
        ax.scatter(row["b_mean"], 0, color=C_BAYES, s=60, zorder=5,
                   label=f"Bayesian ({row['b_mean']:.0f})")

        # GP
        if not pd.isna(row.get("gp_mean")):
            gp_h50 = (row["gp_q75"] - row["gp_q25"]) / 2
            gp_h90 = (row["gp_q95"] - row["gp_q05"]) / 2
            ax.barh(-0.45, 2*gp_h90, left=row["gp_q05"],
                    height=0.3, color=C_GP, alpha=0.25, label="_")
            ax.barh(-0.45, 2*gp_h50, left=row["gp_q25"],
                    height=0.3, color=C_GP, alpha=0.65, label="_")
            ax.scatter(row["gp_mean"], -0.45, color=C_GP, s=60, zorder=5,
                       label=f"GP ({row['gp_mean']:.0f})*")

        # XGBoost
        if not pd.isna(row.get("xgb_forecast_2024")):
            ax.scatter(row["xgb_forecast_2024"], -0.9, color=C_XGB,
                       marker="D", s=80, zorder=5,
                       label=f"XGBoost ({row['xgb_forecast_2024']:.0f})")

        # Actual
        ax.axvline(row["passing_epa"], color=C_ACTUAL, lw=2, ls="--",
                   label=f"Actual ({row['passing_epa']:.0f})")

        ax.set_yticks([0, -0.45, -0.9])
        ax.set_yticklabels(["Bayesian", "GP*", "XGBoost"], fontsize=8)
        ax.set_xlabel("Passing EPA", fontsize=8)
        ax.set_title(player, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9,
                  edgecolor=C_GRID)
        ax.grid(axis="x", color=C_GRID, lw=0.6)
        ax.spines[["top","right","left"]].set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── RB Forest Plot ─────────────────────────────────────────────────────────────

def plot_rb_forest(rb_val: pd.DataFrame) -> str:
    df = rb_val.sort_values("rushing_epa").reset_index(drop=True)
    n  = len(df)
    y  = np.arange(n)

    fig, ax = plt.subplots(figsize=(11, 9), facecolor=C_BG)
    ax.set_facecolor(C_BG)

    for i, row in df.iterrows():
        ax.plot([row["q05"], row["q95"]], [i, i],
                color=C_BAYES, lw=1.2, alpha=0.4, solid_capstyle="round")
        ax.plot([row["q25"], row["q75"]], [i, i],
                color=C_BAYES, lw=3.5, alpha=0.7, solid_capstyle="round")
        ax.scatter(row["mean"], i, color=C_BAYES, s=30, zorder=4)
        ax.scatter(row["rushing_epa"], i,
                   color=C_ACTUAL, marker="|", s=120, lw=2, zorder=6)

    ax.set_yticks(y)
    ax.set_yticklabels(df["player"], fontsize=7.5)
    ax.axvline(0, color="#94A3B8", lw=0.8, ls="--")
    ax.set_xlabel("Total Rushing EPA", fontsize=9)
    ax.set_title("RB 2024 Forecast vs Actual — All Validated Players",
                 fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="x", color=C_GRID, lw=0.6)
    ax.spines[["top","right","left"]].set_visible(False)

    legend_els = [
        Line2D([0],[0], color=C_BAYES, lw=3.5, label="Bayesian 50% interval"),
        Line2D([0],[0], color=C_BAYES, lw=1.2, alpha=0.5, label="Bayesian 90% interval"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_BAYES,
               markersize=6, label="Bayesian mean"),
        Line2D([0],[0], marker="|", color=C_ACTUAL, markersize=10,
               lw=2, label="Actual 2024 EPA"),
    ]
    ax.legend(handles=legend_els, fontsize=7.5, loc="lower right",
              framealpha=0.9, edgecolor=C_GRID)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── RB Spotlight ───────────────────────────────────────────────────────────────

def plot_rb_spotlight(players: list, rb_val: pd.DataFrame, title: str) -> str:
    df = rb_val[rb_val["player"].isin(players)].set_index("player")
    n  = len(players)

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.5),
                             facecolor=C_BG)
    if n == 1:
        axes = [axes]

    for ax, player in zip(axes, players):
        ax.set_facecolor(C_BG)
        if player not in df.index:
            ax.set_title(player, fontsize=9)
            continue
        row = df.loc[player]

        h50 = (row["q75"] - row["q25"]) / 2
        h90 = (row["q95"] - row["q05"]) / 2

        ax.barh(0, 2*h90, left=row["q05"],
                height=0.35, color=C_BAYES, alpha=0.25)
        ax.barh(0, 2*h50, left=row["q25"],
                height=0.35, color=C_BAYES, alpha=0.65)
        ax.scatter(row["mean"], 0, color=C_BAYES, s=70, zorder=5,
                   label=f"Pred mean ({row['mean']:.0f})")
        ax.axvline(row["rushing_epa"], color=C_ACTUAL, lw=2, ls="--",
                   label=f"Actual ({row['rushing_epa']:.0f})")

        carries_note = f"{int(row['carries_2023'])} carries | RYOE/att {row['ryoe_per_att_2023']:.2f}"
        ax.set_xlabel("Rushing EPA", fontsize=8)
        ax.set_title(f"{player}\n{carries_note}", fontsize=8.5, fontweight="bold")
        ax.set_yticks([0])
        ax.set_yticklabels(["Bayesian"], fontsize=8)
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9,
                  edgecolor=C_GRID)
        ax.grid(axis="x", color=C_GRID, lw=0.6)
        ax.spines[["top","right","left"]].set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.05)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── Forecast tables ────────────────────────────────────────────────────────────

def df_to_html_table(df: pd.DataFrame, id: str = "") -> str:
    id_attr = f' id="{id}"' if id else ""
    rows = ""
    for _, r in df.iterrows():
        rows += "<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>"
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"""
    <div class="table-wrap">
      <table{id_attr}>
        <thead><tr>{headers}</tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


# ── HTML assembly ──────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #F1F5F9; color: #1E293B; font-size: 14px; }
header { background: #1E3A5F; color: white; padding: 32px 48px; }
header h1 { font-size: 1.8rem; font-weight: 700; }
header p  { margin-top: 6px; opacity: 0.8; font-size: 0.95rem; }
header a  { color: #93C5FD; text-decoration: none; }
header a:hover { text-decoration: underline; }
nav { background: #162D4A; display: flex; gap: 4px; padding: 0 48px; }
nav a { color: #CBD5E1; padding: 10px 16px; font-size: 0.85rem;
        text-decoration: none; display: block; border-bottom: 3px solid transparent; }
nav a:hover, nav a.active { color: white; border-color: #3B82F6; }
main { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }
h2 { font-size: 1.3rem; font-weight: 700; color: #1E3A5F;
     border-left: 4px solid #2563EB; padding-left: 12px; margin: 32px 0 16px; }
h3 { font-size: 1rem; font-weight: 600; color: #334155; margin: 24px 0 10px; }
p  { line-height: 1.65; color: #475569; margin-bottom: 12px; }
.cards { display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0 24px; }
.card { background: white; border-radius: 10px; padding: 18px 22px;
        box-shadow: 0 1px 4px rgba(0,0,0,.08); min-width: 140px; flex: 1; }
.card-title { font-size: 0.75rem; text-transform: uppercase;
              letter-spacing: .05em; color: #64748B; }
.card-value { font-size: 1.6rem; font-weight: 700; color: #1E3A5F; margin-top: 4px; }
.card-note  { font-size: 0.72rem; color: #94A3B8; margin-top: 4px; }
.fig-wrap { background: white; border-radius: 10px; padding: 16px;
            box-shadow: 0 1px 4px rgba(0,0,0,.08); margin: 16px 0; }
.fig-wrap img { width: 100%; height: auto; display: block; }
.fig-caption { font-size: 0.78rem; color: #64748B; margin-top: 8px;
               line-height: 1.5; }
.table-wrap { overflow-x: auto; background: white; border-radius: 10px;
              box-shadow: 0 1px 4px rgba(0,0,0,.08); margin: 16px 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.8rem; }
thead th { background: #1E3A5F; color: white; padding: 10px 12px;
           text-align: left; font-weight: 600; white-space: nowrap; }
tbody tr:nth-child(even) { background: #F8FAFC; }
tbody td { padding: 8px 12px; border-bottom: 1px solid #E2E8F0; white-space: nowrap; }
.note { background: #FEF3C7; border-left: 4px solid #D97706;
        padding: 12px 16px; border-radius: 4px; margin: 12px 0;
        font-size: 0.82rem; color: #92400E; line-height: 1.55; }
.section-divider { border: none; border-top: 2px solid #E2E8F0; margin: 40px 0; }
footer { text-align: center; padding: 24px; color: #94A3B8; font-size: 0.8rem; }
"""


def build_html(figs: dict, qb_val: pd.DataFrame, rb_val: pd.DataFrame,
               rb24: pd.DataFrame) -> str:

    # ── QB calibration cards
    qb_corr = qb_val[["b_mean","passing_epa"]].corr().iloc[0,1]
    qb_mae  = (qb_val["passing_epa"] - qb_val["b_mean"]).abs().mean()
    qb_bias = (qb_val["passing_epa"] - qb_val["b_mean"]).mean()
    qb_50   = ((qb_val["passing_epa"] >= qb_val["q25"]) &
               (qb_val["passing_epa"] <= qb_val["q75"])).mean()
    qb_90   = ((qb_val["passing_epa"] >= qb_val["q05"]) &
               (qb_val["passing_epa"] <= qb_val["q95"])).mean()

    # ── RB calibration cards
    rb_corr = rb_val[["mean","rushing_epa"]].corr().iloc[0,1]
    rb_mae  = (rb_val["rushing_epa"] - rb_val["mean"]).abs().mean()
    rb_bias = (rb_val["rushing_epa"] - rb_val["mean"]).mean()
    rb_50   = ((rb_val["rushing_epa"] >= rb_val["q25"]) &
               (rb_val["rushing_epa"] <= rb_val["q75"])).mean()
    rb_90   = ((rb_val["rushing_epa"] >= rb_val["q05"]) &
               (rb_val["rushing_epa"] <= rb_val["q95"])).mean()

    # ── QB forecast table
    qb_tbl = qb_val[["player","b_mean","q25","q75","xgb_forecast_2024",
                      "passing_epa"]].copy()
    qb_tbl.columns = ["Player","Bayes Mean","Bayes Q25","Bayes Q75",
                       "XGBoost","Actual 2024 EPA"]
    qb_tbl = qb_tbl.sort_values("Actual 2024 EPA", ascending=False).round(1)

    # ── RB forecast table
    rb_tbl = rb_val[["player","carries_2023","ryoe_per_att_2023",
                      "mean","q25","q75","rushing_epa"]].copy()
    rb_tbl.columns = ["Player","2023 Carries","RYOE/att",
                       "Pred Mean","Q25","Q75","Actual 2024 EPA"]
    rb_tbl = rb_tbl.sort_values("Actual 2024 EPA", ascending=False).round(2)

    def img(key, caption=""):
        return f"""
        <div class="fig-wrap">
          <img src="data:image/png;base64,{figs[key]}" alt="{key}">
          {"<p class='fig-caption'>" + caption + "</p>" if caption else ""}
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Football Player EPA Forecasting</title>
  <style>{CSS}</style>
</head>
<body>
<header>
  <h1>Football Player EPA Forecasting</h1>
  <p>Hierarchical Bayesian and Gaussian Process models for NFL player value forecasting
     using publicly available nflverse data &mdash;
     <a href="{GITHUB}" target="_blank">view on GitHub</a>
  </p>
</header>
<nav>
  <a href="#overview">Overview</a>
  <a href="#qb">Quarterback (QB)</a>
  <a href="#rb">Running Back (RB)</a>
  <a href="#methodology">Methodology</a>
</nav>

<main>

<!-- ── OVERVIEW ── -->
<h2 id="overview">Overview</h2>
<p>This project forecasts NFL player Expected Points Added (EPA) using hierarchical
Bayesian models with player-specific volatility, Student-t likelihoods, and
position-adapted feature engineering. All data is sourced from
<a href="https://github.com/nflverse/nflverse-data" target="_blank">nflverse</a>
via <code>nfl_data_py</code>. Models are trained through 2023 and validated
against 2024 actuals.</p>

<p>The framework is designed to be position-agnostic: the core model structure
(non-centered parameterization, partial pooling, player-specific random walks)
is the same across positions. What changes is the latent talent estimator &mdash;
EPA/att and dakota for QBs, RYOE/att for RBs &mdash; and the forecast architecture
when EPA is multiplicative (efficiency &times; volume) rather than additive.</p>

<hr class="section-divider">

<!-- ── QB ── -->
<h2 id="qb">Quarterback (QB) &mdash; Passing EPA</h2>
<p>Three models are compared for QB passing EPA forecasting. Training data covers
1999&ndash;2023 QB seasons with a minimum of 100 pass attempts. Forecasts are
generated for 2024 and 2025.</p>

<h3>2024 Validation Metrics &mdash; Bayesian Student-t</h3>
<div class="cards">
  {card("Players", "24")}
  {card("MAE", f"{qb_mae:.1f} EPA")}
  {card("Bias", f"{qb_bias:+.1f} EPA", "actual − predicted")}
  {card("Correlation", f"{qb_corr:.2f}")}
  {card("50% Coverage", f"{qb_50:.0%}", "ideal: 50%")}
  {card("90% Coverage", f"{qb_90:.0%}", "ideal: 90%")}
</div>

{img("qb_forest",
     "Each row is one QB. The thick bar is the 50% interval; the thin bar is the 90% interval. "
     "The blue dot is the Bayesian posterior mean; the amber diamond is the XGBoost point estimate; "
     "the red tick is the actual 2024 passing EPA.")}

<h3>Player Spotlight &mdash; High-Sample (Veterans)</h3>
<p>These QBs had 7+ seasons of NFL data through 2023, giving the model a well-established
prior. Tighter intervals reflect more stable year-over-year volatility estimates.</p>

{img("qb_high",
     "Bayesian 50%/90% intervals (blue), GP 50%/90% intervals (green, note: GP interval fix "
     "pending &mdash; these are wider than the corrected Bayesian estimates), XGBoost point "
     "estimate (amber diamond), and actual 2024 EPA (red dashed line).")}

<h3>Player Spotlight &mdash; Low-Sample (Newer Starters)</h3>
<p>These QBs had 1&ndash;3 seasons of full starter data through 2023. Wider intervals
reflect genuine uncertainty; the model has less data to estimate their true volatility
and career trajectory.</p>

{img("qb_low",
     "Same format as above. The wider intervals for C.J. Stroud and Bryce Young are expected: "
     "with a single starter season the model relies more on population-level priors.")}

<h3>Full 2024 QB Forecasts</h3>
{df_to_html_table(qb_tbl, "qb-table")}

<div class="note">
  <strong>GP model note:</strong> The GP Student-t model uses the same volatility
  estimation approach as the original Bayesian Normal model, which had a unit mismatch:
  &sigma;<sub>epa_att</sub> was computed from total EPA deltas (~47 EPA/yr) rather than
  EPA/att deltas (~0.18/yr). This inflates GP forecast intervals significantly. The fix
  has been applied to the Bayesian Student-t model only; a GP re-run is planned.
</div>

<hr class="section-divider">

<!-- ── RB ── -->
<h2 id="rb">Running Back (RB) &mdash; Rushing EPA</h2>
<p>RB rushing EPA is modeled via a two-stage forecast: the Bayesian model predicts
<strong>EPA per carry</strong> (efficiency) using RYOE/att as the primary signal,
then multiplies projected efficiency by projected carries to get total rushing EPA.
This structure is necessary because total EPA = epa/carry &times; carries is
multiplicative &mdash; a direct linear model of log(carries) + RYOE predicting
total EPA produced a negative volume coefficient. Training data covers 2016&ndash;2023
(NGS availability) with a minimum of 75 carries.</p>

<h3>2024 Validation Metrics &mdash; Bayesian Student-t</h3>
<div class="cards">
  {card("Players", "34")}
  {card("MAE", f"{rb_mae:.1f} EPA")}
  {card("Bias", f"{rb_bias:+.1f} EPA", "actual − predicted")}
  {card("Correlation", f"{rb_corr:.2f}")}
  {card("50% Coverage", f"{rb_50:.0%}", "ideal: 50%")}
  {card("90% Coverage", f"{rb_90:.0%}", "ideal: 90%")}
</div>

{img("rb_forest",
     "Each row is one RB. Same format as the QB forest plot. Near-zero bias and "
     "well-calibrated 50% intervals reflect an unbiased model; low rank correlation "
     "reflects genuine RB unpredictability driven by opportunity (team/scheme changes).")}

<h3>Player Spotlight &mdash; High-Sample (Established Backs)</h3>
<p>These RBs had 4+ seasons through 2023 with sufficient carries each year to contribute
to the volatility estimate. The model has a more informed prior on their efficiency range.</p>

{img("rb_high",
     "CMC's negative 2024 actual reflects a significant injury-shortened season. "
     "Jonathan Taylor's miss reflects a scheme and opportunity change after leaving "
     "Indianapolis. Neither is capturable from public data alone.")}

<h3>Player Spotlight &mdash; Low-Sample (Newer Backs)</h3>
<p>These RBs had 1&ndash;2 qualifying seasons through 2023. Wider intervals reflect
the model's limited player-specific history and increased reliance on population priors.</p>

{img("rb_low",
     "Bijan Robinson's actual 2024 (+17 EPA) fell within the forecast interval. "
     "Jahmyr Gibbs (+35 EPA) significantly outperformed: his exceptional 2023 RYOE "
     "was the main signal, but 2024 opportunity expanded well beyond what 2023 carries implied.")}

<h3>Full 2024 RB Forecasts</h3>
{df_to_html_table(rb_tbl, "rb-table")}

<hr class="section-divider">

<!-- ── METHODOLOGY ── -->
<h2 id="methodology">Methodology</h2>

<h3>Model Architecture</h3>
<p>Both the QB and RB models use a non-centered hierarchical Bayesian parameterization
with player intercepts drawn from a half-Normal hyperprior. Player-specific forecast
uncertainty is estimated from each player's own year-over-year volatility history
(with population median as fallback for players with insufficient data), rather than
a single population sigma.</p>

<p>The likelihood is Student-t with learned degrees of freedom &nu;. For QBs, residual
excess kurtosis of 1.93 confirmed fat tails; both the Bayesian and GP models independently
estimated &nu; near 3&ndash;4. Switching from Normal to Student-t dropped &sigma;<sub>obs</sub>
from 33 to 24 EPA.</p>

<h3>QB Predictors</h3>
<p>epa_per_att (r=0.91 with total EPA), dakota (CPOE + EPA/play composite, r=0.86),
pacr (Passing Air Conversion Ratio), log(attempts) for volume, and an aging curve
(polynomial in Bayesian, GP kernel in the GP model).</p>

<h3>RB Forecast Architecture</h3>
<p>A direct linear model predicting total rushing EPA with log(carries) as a predictor
produced a negative volume coefficient (&beta;<sub>vol</sub> = &minus;1.7). The reason:
total EPA = epa/carry &times; carries is multiplicative, not additive. Including both
epa/carry and carries in a linear predictor of total EPA is quasi-circular.</p>
<p>The two-stage fix: (1) predict EPA/carry from RYOE/att using the Bayesian model;
(2) multiply posterior draws of EPA/carry by a carries random walk. This correctly
separates efficiency signal from volume projection.</p>

<h3>Interval Calibration Fixes</h3>
<p>Initial QB forecast intervals had IQR ~300 EPA (100% coverage at both 50% and 90%
thresholds). Two unit-mismatch bugs were identified and corrected:</p>
<p>
(1) &sigma;<sub>epa_att</sub> was computed from year-over-year <em>total</em> EPA deltas
(~47 EPA/yr) but applied as noise on <em>epa_per_att</em> (range ~0.1&ndash;0.5).
Fixed by computing from delta_epa_per_att (~0.18/yr).</p>
<p>
(2) &sigma;<sub>att</sub> was inflated to ~169 attempts/yr by backup and injury seasons
(e.g. a QB going from 100 attempts to 550 in consecutive seasons).
Fixed by restricting delta_att computation to season pairs where both years had
&ge;300 attempts, yielding a realistic starter-level &sigma;<sub>att</sub> ~88.</p>
<p>After correction: IQR ~43 EPA, 50% coverage 54% (near ideal), 90% coverage 71%.</p>

</main>
<footer>Built by Dave Zack &mdash; <a href="{GITHUB}">GitHub</a></footer>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    bayes24, gp24, xgb, qb_val, rb24, rb_val = load_data()

    figs = {}

    print("Plotting QB forest...")
    figs["qb_forest"] = plot_qb_forest(qb_val)

    print("Plotting QB high-sample spotlight...")
    qb_high = ["Patrick Mahomes", "Matthew Stafford", "Jared Goff"]
    figs["qb_high"] = plot_qb_spotlight(
        qb_high, qb_val,
        "QB Spotlight — High Sample (7+ Starter Seasons Through 2023)"
    )

    print("Plotting QB low-sample spotlight...")
    qb_low = ["Jordan Love", "C.J. Stroud", "Bryce Young"]
    figs["qb_low"] = plot_qb_spotlight(
        qb_low, qb_val,
        "QB Spotlight — Low Sample (1–3 Starter Seasons Through 2023)"
    )

    print("Plotting RB forest...")
    figs["rb_forest"] = plot_rb_forest(rb_val)

    print("Plotting RB high-sample spotlight...")
    rb_high = ["Christian McCaffrey", "Jonathan Taylor", "Alvin Kamara"]
    figs["rb_high"] = plot_rb_spotlight(
        rb_high, rb_val,
        "RB Spotlight — High Sample (4+ Qualifying Seasons Through 2023)"
    )

    print("Plotting RB low-sample spotlight...")
    rb_low = ["Jahmyr Gibbs", "Bijan Robinson", "James Cook"]
    figs["rb_low"] = plot_rb_spotlight(
        rb_low, rb_val,
        "RB Spotlight — Low Sample (1–2 Qualifying Seasons Through 2023)"
    )

    print("Assembling HTML...")
    html = build_html(figs, qb_val, rb_val, rb24)

    out = os.path.join(DOCS, "index.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"Saved: {out}  ({os.path.getsize(out)//1024} KB)")


if __name__ == "__main__":
    main()
