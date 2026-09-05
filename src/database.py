"""
DATABASE LAYER - MySQL / PostgreSQL compatible
==============================================
Loads every processed dataset into a relational schema. Runs on SQLite
out of the box (zero setup) and switches to MySQL/PostgreSQL by changing
DB_URL in config.py - the DDL below is standard SQL that both engines
accept.

Schema
------
posts(post_id PK, handle, topic, format, hook_type, duration_s,
      caption_style, posted_on, posting_hour, impressions, reach, likes,
      comments, shares, saves, retention_rate, followers_gained, viral_coefficient)
comments(comment_id PK, post_id FK, topic, comment_text, relatable_tag,
         relatable_probability)
followers(day PK, date, followers)
ab_results(experiment PK, metric, comparison, p_value, effect, significant)

Usage:  python -m src.database
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text

from config import DATA_PROCESSED, DB_URL


def get_engine(url: str = DB_URL):
    return create_engine(url, echo=False)


DDL = [
    """CREATE TABLE IF NOT EXISTS posts (
        post_id TEXT PRIMARY KEY, handle TEXT, topic TEXT, format TEXT,
        hook_type TEXT, duration_s TEXT, caption_style TEXT,
        posted_on DATE, posting_hour INTEGER, impressions INTEGER,
        reach INTEGER, likes INTEGER, comments INTEGER, shares INTEGER,
        saves INTEGER, retention_rate REAL, followers_gained INTEGER,
        viral_coefficient REAL)""",
    """CREATE TABLE IF NOT EXISTS comments (
        comment_id TEXT PRIMARY KEY, post_id TEXT REFERENCES posts(post_id),
        topic TEXT, comment_text TEXT, relatable_tag TEXT,
        relatable_probability REAL)""",
    """CREATE TABLE IF NOT EXISTS followers (
        day INTEGER PRIMARY KEY, date DATE, followers INTEGER)""",
    """CREATE TABLE IF NOT EXISTS ab_results (
        experiment TEXT PRIMARY KEY, metric TEXT, comparison TEXT,
        p_value REAL, effect REAL, significant BOOLEAN)""",
]


def load_all(url: str = DB_URL) -> dict[str, int]:
    engine = get_engine(url)
    with engine.begin() as conn:
        for stmt in DDL:
            conn.execute(text(stmt))

    posts = pd.read_csv(DATA_PROCESSED / "posts_scored.csv")
    comments = pd.read_csv(DATA_PROCESSED / "comments_tagged.csv")[
        ["comment_id", "post_id", "topic", "comment_text",
         "relatable_tag", "relatable_probability"]]
    followers = pd.read_csv(DATA_PROCESSED / "followers.csv")
    ab = pd.read_csv(DATA_PROCESSED / "ab_test_results.csv")

    frames = {"posts": posts, "comments": comments,
              "followers": followers, "ab_results": ab}
    counts = {}
    for name, frame in frames.items():
        frame.to_sql(name, engine, if_exists="replace", index=False)
        counts[name] = len(frame)
    return counts


def sanity_query(url: str = DB_URL) -> pd.DataFrame:
    """Example analytical SQL: mean K by topic straight from the DB."""
    engine = get_engine(url)
    q = """
        SELECT topic,
               COUNT(*)                AS posts,
               ROUND(AVG(viral_coefficient), 5) AS mean_k,
               SUM(followers_gained)   AS followers_gained
        FROM posts
        GROUP BY topic
        ORDER BY mean_k DESC
    """
    with engine.connect() as conn:
        return pd.read_sql(text(q), conn)


if __name__ == "__main__":
    counts = load_all()
    print(f"loaded tables: {counts}")
    print("\nSQL sanity check - mean viral coefficient by topic:")
    print(sanity_query().to_string(index=False))
