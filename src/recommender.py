"""
MODULE 5 - Engagement Optimization Recommender
==============================================
Prescriptive analytics engine: converts the descriptive and inferential
results (Modules 2-4) into next-week content strategy. For every topic,
format, length bucket, time slot and caption style it estimates the
expected Viral Coefficient and engagement rate from historical evidence,
applies shrinkage toward the global mean for thin evidence cells
(empirical-Bayes style), and emits the highest-probability plays.

Outputs
-------
data/processed/recommendations.csv      ranked playbook for next week
reports/recommendations.json            machine-readable version
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, REPORTS

SHRINKAGE_PRIOR_N = 3.0  # cells with n < this are shrunk toward the mean


def _shrunk_mean(values: pd.Series, global_mean: float,
                 prior_n: float = SHRINKAGE_PRIOR_N) -> float:
    n = len(values)
    return (values.sum() + prior_n * global_mean) / (n + prior_n)


def build_plays(scored: pd.DataFrame) -> pd.DataFrame:
    df = scored.copy()
    df["engagement_rate"] = ((df["likes"] + df["comments"] + df["shares"]
                              + df["saves"]) / df["reach"])
    df["duration_bucket"] = np.select(
        [pd.to_numeric(df["duration_s"], errors="coerce") <= 20,
         pd.to_numeric(df["duration_s"], errors="coerce") <= 30,
         pd.to_numeric(df["duration_s"], errors="coerce") > 30],
        ["<=20s", "21-30s", ">30s"], default="n/a")
    df["time_slot"] = np.where(df["posting_hour"] >= 18, "evening",
                        np.where(df["posting_hour"] >= 12, "midday", "morning"))
    g_mean_k = df["viral_coefficient"].mean()
    g_mean_er = df["engagement_rate"].mean()

    plays = []
    for topic in df["topic"].unique():
        t_df = df[df["topic"] == topic]
        for fmt in t_df["format"].unique():
            f_df = t_df[t_df["format"] == fmt]
            for bucket in f_df["duration_bucket"].unique():
                cell = f_df[f_df["duration_bucket"] == bucket]
                for slot in ["evening", "midday", "morning"]:
                    sub = cell[cell["time_slot"] == slot]
                    k_hat = _shrunk_mean(sub["viral_coefficient"], g_mean_k)
                    er_hat = _shrunk_mean(sub["engagement_rate"], g_mean_er)
                    plays.append({
                        "topic": topic, "format": fmt,
                        "duration_bucket": bucket, "time_slot": slot,
                        "n_posts": len(sub),
                        "expected_k": round(float(k_hat), 5),
                        "expected_engagement_rate": round(float(er_hat), 5),
                    })
    plays = pd.DataFrame(plays)
    plays["success_index"] = (plays["expected_k"] / g_mean_k
                              * 0.7 + plays["expected_engagement_rate"]
                              / g_mean_er * 0.3).round(3)
    return plays.sort_values("success_index", ascending=False)


def top_recommendations(plays: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    """Diverse top-k: at most 2 plays per topic so the week stays varied."""
    picked, per_topic = [], {}
    for _, row in plays.iterrows():
        t = row["topic"]
        if per_topic.get(t, 0) < 2:
            picked.append(row)
            per_topic[t] = per_topic.get(t, 0) + 1
        if len(picked) == k:
            break
    return pd.DataFrame(picked)


def caption_guidance(scored: pd.DataFrame) -> dict:
    """Evidence-backed micro-guidance for the content team."""
    df = scored.copy()
    guidance: dict = {"best_hook_by_format": {}}
    for fmt, g in df.groupby("format"):
        means = g.groupby("hook_type")["viral_coefficient"].mean()
        guidance["best_hook_by_format"][fmt] = str(means.idxmax())
    guidance["best_caption_style"] = str(
        df.groupby("caption_style")["viral_coefficient"].mean().idxmax())
    slot = df.groupby(df["posting_hour"].map(
        lambda h: "morning" if h <= 11 else "midday" if h <= 15 else "evening"))["viral_coefficient"].mean()
    guidance["best_time_slot"] = str(slot.idxmax())
    guidance["median_k_by_topic"] = (
        df.groupby("topic")["viral_coefficient"].median().round(4).to_dict())
    return guidance


def run(scored: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    plays = build_plays(scored)
    plays.to_csv(DATA_PROCESSED / "recommendations.csv", index=False)
    topk = top_recommendations(plays)
    guidance = caption_guidance(scored)
    payload = {"top_plays": topk.to_dict(orient="records"),
               "guidance": guidance}
    (REPORTS / "recommendations.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return plays, payload


if __name__ == "__main__":
    scored = pd.read_csv(DATA_PROCESSED / "posts_scored.csv")
    plays, payload = run(scored)
    print(top_recommendations(plays).to_string(index=False))
    print(json.dumps(guidance := payload["guidance"], indent=2, default=str))
