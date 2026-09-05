"""
END-TO-END PIPELINE for The Data-Driven Social Engagement Initiative
====================================================================
Runs every module in dependency order and prints a run report.

    python run_pipeline.py

Steps
-----
1. Content Series catalogue            (Deliverable 5)
2. Extraction engine + simulation      (Module 1, Deliverable 2)
3. NLP sentiment model                 (Module 3, Deliverable 3)
4. Virality prediction engine          (Module 2)
5. A/B testing framework               (Module 4)
6. Optimization recommender            (Module 5)
7. Trend forecasting module            (Module 7)
8. Relational database load            (tech-stack layer)
"""
from __future__ import annotations

import time

from src import (ab_testing, content_series, database, extraction_engine,
                 recommender, sentiment_analyzer, trend_forecaster,
                 virality_engine)


def main() -> None:
    t0 = time.time()
    print("=" * 72)
    print("THE DATA-DRIVEN SOCIAL ENGAGEMENT INITIATIVE - pipeline run")
    print("researcher: G.Vaishanth  ·  batch: Data Science June Batch")
    print("=" * 72)

    print("\n[1/8] Content Series catalogue (Deliverable 5)")
    content_series.main()

    print("\n[2/8] Content Performance Tracker - extraction + simulation (Module 1)")
    data = extraction_engine.run()
    s = data["summary"]
    print(f"      {s['posts']} posts · {s['comments']} comments · "
          f"reach {s['total_reach']:,} · {s['final_followers']:,} followers")

    print("\n[3/8] Audience Sentiment Analyzer - training NLP model (Module 3)")
    _, sent = sentiment_analyzer.run(data["comments"])
    m = sent["metrics"]
    print(f"      accuracy {m['accuracy']:.1%} · F1(relatable) {m['f1_relatable']:.1%}")

    print("\n[4/8] Virality Prediction Engine (Module 2)")
    scored, vir = virality_engine.run(data["posts"])
    print(f"      holdout R2 (log K): {vir['model_metrics']['holdout_r2_logspace']}")

    print("\n[5/8] A/B Testing Framework (Module 4)")
    import pandas as pd
    ab_table, _ = ab_testing.run_all(scored, pd.read_csv("data/processed/comments_tagged.csv"))
    print(ab_table.to_string(index=False))

    print("\n[6/8] Engagement Optimization Recommender (Module 5)")
    _, recs = recommender.run(scored)
    print(f"      {len(recs['top_plays'])} plays queued for next week")

    print("\n[7/8] Trend Forecasting Module (Module 7)")
    _, trends = trend_forecaster.run()
    rising = [r["keyword"] for r in trends["records"] if r["status"] == "RISING"]
    print(f"      rising struggles detected: {', '.join(rising) or 'none'}")

    print("\n[8/8] Relational database load")
    counts = database.load_all()
    print(f"      tables: {counts}")

    print("\n" + "=" * 72)
    print(f"Pipeline finished in {time.time() - t0:.1f}s")
    print("Next: streamlit run app.py   →  open the analytics dashboard")


if __name__ == "__main__":
    main()
