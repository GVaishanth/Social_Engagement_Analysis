"""
MODULE 7 - Trend Forecasting Module
===================================
Analyses external hashtag / keyword interest curves to predict rising
"relatable struggles" BEFORE they become mainstream, so the content
strategy stays ahead of the curve.

Method
------
For each tracked keyword we build a 180-day public-interest index
(0-100, Google-Trends-style). In production this series is pulled from
PyTrends / the platform's trend API; here it is simulated with
seasonality + drift + noise, with three keywords deliberately seeded on a
rising trajectory.

Forecasting: damped Holt linear-trend exponential smoothing implemented
in NumPy (no exotic dependencies), forecasting 14 days ahead. Momentum =
forecast slope + acceleration; a keyword is flagged RISING when its
recent momentum z-score exceeds +0.8 and its current level is still
below its historical peak (i.e., not already mainstream).

Outputs
-------
data/processed/trend_forecast.csv
reports/figures/fig_trends.png
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, FIGURES, RANDOM_SEED, REPORTS

KEYWORDS = {
    # keyword: (base level, drift/day, seasonality amp, noise sd)
    # three keywords are seeded on a rising trajectory (novel "struggles"),
    # one is seeded as a mainstream plateau, one as a fading trend
    "social battery":           (40, -0.02, 6, 3.0),
    "situationship":            (58,  0.01, 5, 3.0),
    "oldest daughter syndrome": (14,  0.12, 3, 2.2),   # seeded RISING
    "money dysmorphia":         (11,  0.10, 2, 2.0),   # seeded RISING
    "bed rotting":              (72, -0.03, 7, 3.0),   # seeded MAINSTREAM
    "brain rot":                (34,  0.09, 4, 2.5),   # seeded RISING
    "quiet quitting":           (48, -0.06, 5, 3.0),   # seeded FADING
    "main character syndrome":  (26,  0.02, 4, 2.5),
    "hikikomori":               (15,  0.01, 2, 2.0),
    "phone anxiety":            (33,  0.03, 4, 2.6),
}
HORIZON_DAYS = 14
SERIES_DAYS = 180


def simulate_interest(rng: np.random.Generator) -> pd.DataFrame:
    t = np.arange(SERIES_DAYS)
    rows = []
    for kw, (level, drift, amp, noise) in KEYWORDS.items():
        y = (level + drift * t
             + amp * np.sin(2 * np.pi * t / 28 + rng.uniform(0, 6.28))
             + rng.normal(0, noise, SERIES_DAYS))
        y = np.clip(y, 0, 100)
        for d, v in zip(t, y):
            rows.append({"keyword": kw, "day": int(d), "interest": round(float(v), 1)})
    return pd.DataFrame(rows)


def holt_damped(series: np.ndarray, alpha: float = 0.35,
                beta: float = 0.10, phi: float = 0.85,
                horizon: int = HORIZON_DAYS) -> tuple[np.ndarray, float]:
    """Damped Holt linear trend. Returns (forecast array, smoothed slope)."""
    level, trend = series[0], series[1] - series[0]
    for x in series[1:]:
        new_level = alpha * x + (1 - alpha) * (level + phi * trend)
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        level = new_level
    steps = np.arange(1, horizon + 1)
    damp = sum(phi ** i for i in range(1, int(steps.max()) + 1))
    fc = level + trend * np.cumsum([phi ** i for i in range(horizon)])
    return fc, float(trend * damp / damp)


def momentum_z(series: np.ndarray, window: int = 21) -> float:
    """Recent slope of the interest curve normalised by its own history."""
    slopes = []
    for i in range(window, len(series) + 1, 7):
        seg = series[i - window:i]
        slopes.append(np.polyfit(np.arange(window), seg, 1)[0])
    slopes = np.array(slopes)
    if slopes.std() == 0:
        return 0.0
    return float((slopes[-1] - slopes.mean()) / slopes.std())


def run() -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(RANDOM_SEED)
    df = simulate_interest(rng)
    df.to_csv(DATA_PROCESSED / "trend_interest_raw.csv", index=False)

    out = []
    for kw, g in df.groupby("keyword"):
        y = g.sort_values("day")["interest"].to_numpy()
        fc, _ = holt_damped(y)
        mz = momentum_z(y)
        level_now = float(y[-14:].mean())
        level_60 = float(y[-60:].mean())
        # slope of the recent 45-day window (day units)
        slope_45 = float(np.polyfit(np.arange(45), y[-45:], 1)[0])
        slope_90 = float(np.polyfit(np.arange(90), y[-90:], 1)[0])
        mainstream = level_60 >= 65          # long high plateau: already top-of-curve
        out.append({
            "keyword": kw,
            "current_level": round(level_now, 1),
            "level_60d_mean": round(level_60, 1),
            "forecast_14d_end": round(float(fc[-1]), 1),
            "forecast_slope_per_day": round(float(np.polyfit(
                np.arange(HORIZON_DAYS), fc, 1)[0]), 3),
            "recent_slope_per_day": round(slope_45, 3),
            "slope_90d_per_day": round(slope_90, 3),
            "momentum_z": round(mz, 2),
            "status": ("MAINSTREAM" if mainstream else
                       "RISING" if slope_45 >= 0.06 and slope_90 >= 0.04 and level_60 < 60 else
                       "FADING" if slope_45 <= -0.10 else "STEADY"),
        })
    res = pd.DataFrame(out).sort_values("momentum_z", ascending=False)
    res.to_csv(DATA_PROCESSED / "trend_forecast.csv", index=False)

    # figure --------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    rising = res.loc[res["status"] == "RISING", "keyword"].tolist()
    others = [k for k in ("situationship", "bed rotting", "quiet quitting")
              if k not in rising]
    show = (rising[:3] + others)[:4]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    colors = sns.color_palette("rocket", len(show))
    for c, kw in zip(colors, show):
        g = df[df["keyword"] == kw].sort_values("day")
        ax.plot(g["day"], g["interest"], label=kw, color=c, lw=1.8)
        fct, _ = holt_damped(g.sort_values("day")["interest"].to_numpy())
        ax.plot(range(SERIES_DAYS, SERIES_DAYS + HORIZON_DAYS), fct,
                "--", color=c, alpha=0.75)
    ax.axvline(SERIES_DAYS, color="grey", ls=":", lw=1)
    ax.text(SERIES_DAYS + 1, ax.get_ylim()[1] * 0.95, "forecast →",
            fontsize=9, color="grey")
    ax.set_xlabel("day")
    ax.set_ylabel("search interest (0-100)")
    ax.set_title("Trend forecasting - rising 'relatable struggles' (dashed = 14-day damped-Holt forecast)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_trends.png", dpi=150)
    plt.close(fig)

    payload = res.to_dict(orient="records")
    (REPORTS / "trend_forecast.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    return df, {"forecast": res, "records": payload}


if __name__ == "__main__":
    _, res = run()
    print(res["forecast"].to_string(index=False))
