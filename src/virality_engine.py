"""
MODULE 2 - Virality Prediction Engine
=====================================
1. Viral Coefficient (K)
   A transparent statistical measure computed for every post:

        K = (w_shares * shares + w_saves * saves + w_comments * comments) / reach

   Shares and saves (high-value, effortful actions) dominate the index,
   while likes are intentionally excluded (passive, low-signal).
   K > 1.0 would mean every viewer recruits more than one new viewer
   (classic epidemic threshold); for feed content, our study threshold
   for "viral-grade" distribution is K >= 0.035 (see config).

2. Prediction model
   A Random Forest regressor is trained on *creative features only*
   (topic, format, hook, duration bucket, caption style, posting hour/day)
   to predict the log-viral-coefficient of a post BEFORE it is published.
   Because the target is skewed, we evaluate on log-space R2 and MAE.

Outputs
-------
data/processed/posts_scored.csv          (K, virality score, breakout flag)
nlp_model/../reports/virality_model.json (holdout metrics + feature importance)
reports/figures/fig_virality_*.png
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from config import (DATA_PROCESSED, FIGURES, MODEL_DIR, REPORTS,
                    VIRAL_K_THRESHOLD, VIRAL_WEIGHTS)


def compute_viral_coefficient(df: pd.DataFrame) -> pd.DataFrame:
    w = VIRAL_WEIGHTS
    k = (w["shares"] * df["shares"]
         + w["saves"] * df["saves"]
         + w["comments"] * df["comments"]) / df["reach"].clip(lower=1)
    out = df.copy()
    out["viral_coefficient"] = k.round(5)
    # 0-100 index for dashboards (99th percentile capped)
    cap = k.quantile(0.99)
    out["virality_score"] = ((k / cap).clip(upper=1.0) * 100).round(2)
    out["is_viral"] = (k >= VIRAL_K_THRESHOLD).astype(int)
    return out


def _duration_bucket(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return pd.cut(s, bins=[0, 20, 30, 45, 200],
                  labels=["ultra_short", "short", "medium", "long"])


def train_predictor(posts: pd.DataFrame) -> tuple[Pipeline, dict]:
    df = posts.copy()
    df["duration_bucket"] = _duration_bucket(df["duration_s"])
    df["time_slot"] = pd.cut(df["posting_hour"], bins=[0, 11, 15, 23],
                             labels=["morning", "midday", "evening"])
    df["log_k"] = np.log1p(df["viral_coefficient"] * 100)  # stable target

    cat_features = ["topic", "format", "hook_type", "caption_style",
                    "duration_bucket", "time_slot"]
    num_features = ["day"]

    X = df[cat_features + num_features]
    y = df["log_k"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", "passthrough", num_features),
    ])
    model = Pipeline([
        ("prep", pre),
        ("rf", RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                     random_state=42)),
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {
        "holdout_r2_logspace": round(float(r2_score(y_te, pred)), 3),
        "holdout_mae_logspace": round(float(mean_absolute_error(y_te, pred)), 3),
        "cv_r2_logspace_5fold": round(
            float(cross_val_score(model, X, y, cv=5,
                                  scoring="r2").mean()), 3),
    }

    # feature importance aggregated back to parent feature names
    names = np.array(model.named_steps["prep"].get_feature_names_out())
    importances = model.named_steps["rf"].feature_importances_
    parent: dict[str, float] = {}
    for f in cat_features:
        mask = np.array([n.startswith(f"cat__{f}_") for n in names])
        parent[f] = float(np.sum(importances[mask]))
    for f in num_features:
        parent[f] = float(importances[np.array(names == f"num__{f}")][0])
    metrics["feature_importance"] = dict(
        sorted(parent.items(), key=lambda kv: -kv[1]))

    import joblib
    joblib.dump(model, MODEL_DIR / "virality_predictor.joblib")

    (REPORTS / "virality_model.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")

    return model, metrics


def run(posts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    scored = compute_viral_coefficient(posts)
    scored.to_csv(DATA_PROCESSED / "posts_scored.csv", index=False)
    _, metrics = train_predictor(scored)

    # topic-level statistical summary for the strategy layer
    topic_summary = (scored.groupby("topic")
                     .agg(posts=("post_id", "count"),
                          mean_k=("viral_coefficient", "mean"),
                          median_k=("viral_coefficient", "median"),
                          viral_rate=("is_viral", "mean"),
                          mean_saves=("saves", "mean"),
                          mean_shares=("shares", "mean"))
                     .round(4)
                     .sort_values("mean_k", ascending=False)
                     .reset_index())
    topic_summary.to_csv(DATA_PROCESSED / "virality_by_topic.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    sns.boxplot(data=scored, x="topic", y="viral_coefficient",
                hue="topic", palette="viridis", legend=False, ax=ax)
    ax.tick_params(axis="x", rotation=18)
    ax.set_title("Viral coefficient distribution by struggle topic")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_k_by_topic.png", dpi=150)
    plt.close(fig)

    return scored, {"model_metrics": metrics, "topic_summary": topic_summary}


if __name__ == "__main__":
    posts = pd.read_csv(DATA_PROCESSED / "posts.csv")
    scored, res = run(posts)
    print(json.dumps(res["model_metrics"], indent=2))
    print(res["topic_summary"].to_string(index=False))
