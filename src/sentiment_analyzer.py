"""
MODULE 3 - Audience Sentiment Analyzer (NLP)  ==  DELIVERABLE 3
===============================================================
An NLP-driven text-analysis module that processes user comments to
validate "Problem Awareness": it quantifies how strongly the audience
recognises itself in a post.

Two-layer design
----------------
Layer 1 (baseline) : lexicon-based scorer (NLTK/TextBlob-style polarity)
Layer 2 (primary)  : TF-IDF (1-2 grams) + Logistic Regression classifier
                     trained on the labelled comment corpus, tagging every
                     comment as **"Relatable"** or **"Neutral"**
                     (binary scheme required by the deliverable).

The trained pipeline is persisted to nlp_model/sentiment_model.joblib so
it can be re-used on live comment exports without retraining.

Outputs
-------
data/processed/comments_tagged.csv   every comment + tags + probability
reports/sentiment_model.json         hold-out metrics (accuracy, P/R/F1)
reports/figures/fig_sentiment_*.png
"""
from __future__ import annotations

import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import DATA_PROCESSED, FIGURES, MODEL_DIR, REPORTS

# ------------------------------------------------- lexicon (Layer 1) ------
RELATABLE_LEXICON = {
    "literally", "exactly", "accurate", "accuracy", "seen", "felt", "called",
    "attacked", "relatable", "same", "me", "myself", "home", "cry", "crying",
    "painful", "painfully", "understood", "valid", "validated", "real",
    "truth", "true", "hurts", "hurt", "core", "biopic", "silence",
}
NEUTRAL_LEXICON = {
    "nice", "great", "cool", "awesome", "follow", "check", "page", "dm",
    "promotion", "free", "link", "bio", "app", "song", "music", "edit",
    "edited", "part", "request", "tips", "collab",
}
EMOJI_POS = {"😭", "❤️", "💯", "🔥", "🥺", " SAME"}


def _preprocess(text: str) -> str:
    """Light NLTK-style normalisation: lower-case, strip URLs/handles,
    collapse repeated characters ('soooo' -> 'soo'), keep emojis out of
    tokens but tag exclamations."""
    t = text.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    t = re.sub(r"[^a-z\s'!]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def lexicon_score(text: str) -> float:
    """TextBlob-style baseline: polarity in [-1, 1]."""
    tokens = set(_preprocess(text).split())
    hits = len(tokens & RELATABLE_LEXICON) - len(tokens & NEUTRAL_LEXICON)
    bang = min(text.count("!"), 3) * 0.2
    return float(np.clip(hits * 0.34 + bang, -1, 1))


# ---------------------------------------------- primary model (Layer 2) ---
def train_model(comments: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, dict]:
    df = comments.copy()
    df["clean"] = df["comment_text"].map(_preprocess)
    X = df["clean"]
    y = df["sentiment_label"]

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=0.2, stratify=y, random_state=42)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=6000,
                                  min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=4.0,
                                   class_weight="balanced")),
    ])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)

    labels = ["neutral", "relatable"]
    cm = confusion_matrix(y_te, pred, labels=labels)
    metrics = {
        "algorithm": "TF-IDF (1-2g) + Logistic Regression (balanced)",
        "train_comments": int(len(X_tr)),
        "test_comments": int(len(X_te)),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "precision_relatable": round(float(precision_score(
            y_te, pred, pos_label="relatable")), 4),
        "recall_relatable": round(float(recall_score(
            y_te, pred, pos_label="relatable")), 4),
        "f1_relatable": round(float(f1_score(
            y_te, pred, pos_label="relatable")), 4),
        "confusion_matrix": cm.tolist(),
        "confusion_labels": labels,
        "report": classification_report(y_te, pred, labels=labels,
                                        output_dict=True),
    }

    # top linguistic triggers (the words that validate Problem Awareness)
    feats = pipe.named_steps["tfidf"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_[0]
    top_rel = [feats[i] for i in np.argsort(coefs)[-15:]][::-1]
    top_neu = [feats[i] for i in np.argsort(coefs)[:15]]
    metrics["top_relatable_triggers"] = top_rel
    metrics["top_neutral_triggers"] = top_neu

    # tag the FULL corpus with probabilities
    proba = pipe.predict_proba(X)[:, list(pipe.classes_).index("relatable")]
    df["relatable_tag"] = np.where(proba >= 0.5, "Relatable", "Neutral")
    df["relatable_probability"] = proba.round(4)
    df["lexicon_score"] = df["comment_text"].map(lexicon_score).round(3)

    joblib.dump(pipe, MODEL_DIR / "sentiment_model.joblib")

    metrics["holdout_report_text"] = classification_report(y_te, pred,
                                                          labels=labels)
    return pipe, df, metrics


def _figures(df: pd.DataFrame, metrics: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    # relatable share by topic
    topic = (df.groupby("topic")["relatable_tag"]
             .apply(lambda s: (s == "Relatable").mean() * 100)
             .sort_values())
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    topic.plot(kind="barh", color="#26a69a", ax=ax)
    ax.set_xlabel("% of comments tagged Relatable")
    ax.set_title("Problem Awareness by struggle topic\n"
                 "(share of audience comments that are self-recognising)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_sentiment_by_topic.png", dpi=150)
    plt.close(fig)


def run(comments: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pipe, tagged, metrics = train_model(comments)
    tagged.to_csv(DATA_PROCESSED / "comments_tagged.csv", index=False)

    # topic-level Problem Awareness table
    awareness = (tagged.groupby("topic")
                 .agg(comments=("comment_id", "count"),
                      relatable_share=("relatable_tag",
                                       lambda s: (s == "Relatable").mean()),
                      mean_probability=("relatable_probability", "mean"))
                 .round(4).sort_values("relatable_share",
                                       ascending=False).reset_index())
    awareness.to_csv(DATA_PROCESSED / "problem_awareness_by_topic.csv",
                     index=False)
    (REPORTS / "sentiment_model.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _figures(tagged, metrics)
    return tagged, {"metrics": metrics, "awareness": awareness}


def predict_comments(texts: list[str]) -> pd.DataFrame:
    """Reload the persisted model and tag fresh comment exports."""
    pipe = joblib.load(MODEL_DIR / "sentiment_model.joblib")
    clean = [_preprocess(t) for t in texts]
    proba = pipe.predict_proba(clean)[:, list(pipe.classes_).index("relatable")]
    return pd.DataFrame({
        "comment_text": texts,
        "tag": np.where(proba >= 0.5, "Relatable", "Neutral"),
        "relatable_probability": proba.round(4),
    })


if __name__ == "__main__":
    comments = pd.read_csv(DATA_PROCESSED / "comments_raw.csv")
    tagged, res = run(comments)
    print(json.dumps({k: v for k, v in res["metrics"].items()
                      if k not in ("report",)}, indent=2, ensure_ascii=False))
    print(res["awareness"].to_string(index=False))
