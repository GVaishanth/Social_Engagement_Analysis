"""
MODULE 4 - A/B Testing Framework
================================
Runs controlled comparative experiments on the content variables recorded
in the Content Series and reports statistically significant factors
driving engagement.

Experiments
-----------
E1  Format          : Reel  vs  Carousel                  (engagement rate)
E2  Hook            : Visual vs  Text                     (engagement rate)
E3  Reel length     : short (<=30s) vs long (>30s)        (retention rate)
E4  Posting time    : evening (18-22) vs day (9-14)       (engagement rate)
E5  Caption style   : Storytelling vs Listicle vs Question (K, ANOVA)
E6  Topic awareness : proportion of Relatable comments
                      vs topic (chi-square test of independence)

Method notes
------------
* Engagement rate = (likes + comments + shares + saves) / reach.
* We run Welch's t-test (unequal variances) AND the non-parametric
  Mann-Whitney U test; we report Cohen's d for effect size and the
  relative lift. With n=66 posts we expect limited power, so effects are
  interpreted with both p-values and effect sizes (never p alone).
* E5 uses one-way ANOVA (+ Kruskal-Wallis as robustness check).
* E6 uses a chi-square test on the comment-level tagged corpus.

Outputs
-------
data/processed/ab_test_results.csv
reports/ab_tests.json
reports/figures/fig_ab_tests.png
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from config import ALPHA, DATA_PROCESSED, REPORTS


def engagement_rate(df: pd.DataFrame) -> pd.Series:
    return (df["likes"] + df["comments"] + df["shares"] + df["saves"]) / df["reach"]


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                     / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled else 0.0


def _two_group_test(a: pd.Series, b: pd.Series) -> dict:
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    t, p_t = stats.ttest_ind(a, b, equal_var=False)
    u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
    d = _cohens_d(a.to_numpy(), b.to_numpy())
    lift = (a.mean() - b.mean()) / b.mean() if b.mean() else np.nan
    return {
        "n_a": int(len(a)), "mean_a": round(float(a.mean()), 5),
        "n_b": int(len(b)), "mean_b": round(float(b.mean()), 5),
        "rel_lift": round(float(lift), 4),
        "welch_t": round(float(t), 3), "p_welch": round(float(p_t), 5),
        "mannwhitney_p": round(float(p_u), 5),
        "cohens_d": round(d, 3),
        "significant": bool(p_t < ALPHA or p_u < ALPHA),
    }


def _multi_metric(df: pd.DataFrame, mask_a: pd.Series, mask_b: pd.Series,
                  metrics: list[str], name_a: str, name_b: str) -> dict:
    """Run the two-group test for several metrics; the first metric is
    treated as primary (reported in the summary table)."""
    out: dict = {"variant_a": name_a, "variant_b": name_b, "metric": metrics[0]}
    for i, m in enumerate(metrics):
        r = _two_group_test(df.loc[mask_a, m], df.loc[mask_b, m])
        r["rel_lift"] = round(r["rel_lift"], 4)
        if i == 0:
            out.update({k: r[k] for k in
                        ("n_a", "mean_a", "n_b", "mean_b",
                         "rel_lift", "welch_t", "p_welch", "mannwhitney_p",
                         "cohens_d", "significant")})
        else:
            out[f"also_{m}"] = r
    return out


def run_all(posts_scored: pd.DataFrame, comments_tagged: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = posts_scored.copy()
    df["engagement_rate"] = engagement_rate(df)
    df["save_rate"] = df["saves"] / df["reach"]
    df["share_rate"] = df["shares"] / df["reach"]
    df["log_reach"] = np.log(df["reach"])
    df["time_slot"] = np.where(df["posting_hour"] >= 18, "evening", "day")
    results: dict[str, dict] = {}

    # E1 format - reach AND engagement rate AND save rate (Reels win reach,
    # Carousels win saves: the objective decides the format)
    results["E1_Format_Reel_vs_Carousel"] = _multi_metric(
        df, df["format"] == "Reel", df["format"] == "Carousel",
        ["log_reach", "engagement_rate", "save_rate", "share_rate"],
        "Reel", "Carousel")
    results["E1_Format_Reel_vs_Carousel"]["metric"] = "log_reach"
    results["E1_Format_Reel_vs_Carousel"]["note"] = (
        "Reels drive distribution (reach); Carousels drive save-rate. "
        "Engagement-rate-only A/B tests hide this split.")

    # E2 hook
    results["E2_Hook_Visual_vs_Text"] = {
        "metric": "engagement_rate", "variant_a": "Visual Hook", "variant_b": "Text Hook",
        **_two_group_test(df.loc[df["hook_type"] == "Visual Hook", "engagement_rate"],
                          df.loc[df["hook_type"] == "Text Hook", "engagement_rate"])}

    # E3 reel length (retention, Reels only)
    d = df[df["format"] == "Reel"].copy()
    d["dur"] = pd.to_numeric(d["duration_s"], errors="coerce")
    results["E3_ReelLength_Short_vs_Long"] = {
        "metric": "retention_rate", "variant_a": "short <=30s", "variant_b": "long >30s",
        **_two_group_test(d.loc[d["dur"] <= 30, "retention_rate"],
                          d.loc[d["dur"] > 30, "retention_rate"])}

    # E4 posting time - reach is the primary metric here: time slots change
    # distribution (algorithmic supply), which then cascades into absolute actions
    results["E4_Time_Evening_vs_Day"] = _multi_metric(
        df, df["time_slot"] == "evening", df["time_slot"] == "day",
        ["log_reach", "engagement_rate", "followers_gained"],
        "evening 18-22", "day 9-14")
    results["E4_Time_Evening_vs_Day"]["metric"] = "log_reach"

    # E5 caption style (ANOVA + Kruskal-Wallis on viral coefficient)
    groups = [g["viral_coefficient"].to_numpy()
              for _, g in df.groupby("caption_style")]
    f, p_f = stats.f_oneway(*groups)
    h, p_h = stats.kruskal(*groups)
    means = df.groupby("caption_style")["viral_coefficient"].mean().round(5)
    results["E5_CaptionStyle_ANOVA"] = {
        "metric": "viral_coefficient",
        "group_means": means.to_dict(),
        "f_stat": round(float(f), 3), "p_anova": round(float(p_f), 5),
        "kruskal_p": round(float(p_h), 5),
        "significant": bool(p_f < ALPHA),
    }

    # E6 topic x relatable-comment proportion (chi-square)
    ct = pd.crosstab(comments_tagged["topic"], comments_tagged["relatable_tag"])
    chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
    prop = (ct["Relatable"] / ct.sum(axis=1)).sort_values(ascending=False)
    results["E6_Topic_Relatable_ChiSquare"] = {
        "metric": "relatable_comment_share",
        "chi2": round(float(chi2), 2), "dof": int(dof),
        "p_value": float(f"{p_chi:.3e}"),
        "share_by_topic": prop.round(4).to_dict(),
        "significant": bool(p_chi < ALPHA),
    }

    rows = []
    for name, r in results.items():
        rows.append({"experiment": name,
                     "metric": r.get("metric", ""),
                     "comparison": r.get("variant_a", "") + " vs " + r.get("variant_b", "")
                     if "variant_a" in r else "multi-group",
                     "p_value": r.get("p_welch", r.get("p_anova", r.get("p_value"))),
                     "effect": r.get("cohens_d", r.get("f_stat", r.get("chi2"))),
                     "significant": r.get("significant")})
    table = pd.DataFrame(rows)
    table.to_csv(DATA_PROCESSED / "ab_test_results.csv", index=False)
    (REPORTS / "ab_tests.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")

    return table, results


if __name__ == "__main__":
    scored = pd.read_csv(DATA_PROCESSED / "posts_scored.csv")
    tagged = pd.read_csv(DATA_PROCESSED / "comments_tagged.csv")
    table, res = run_all(scored, tagged)
    print(table.to_string(index=False))
