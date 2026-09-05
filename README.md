# The Data-Driven Social Engagement Initiative

Data Science Major Project
G.Vaishanth, Data Science June Batch

## What this project is about

This project asks one question: can "relatability" be measured?

Creator advice is mostly intuition, things like "post at 6 pm" or "use
trending audio". For a page built on emotional relatability, the thing worth
optimising is not attention but recognition, the moment a viewer thinks
"this is literally me". So instead of guessing, I built a small analytics
system, ran a 66-post content series through it and studied the results
statistically: what people share and save, what they write in the comments,
and which creative choices actually move the numbers.

The system follows the project brief. It combines social media data
extraction, sentiment analysis, A/B testing and viral metric tracking in one
pipeline that anyone can re-run with a single command.

## How it works

A 66-post series (The Relatable Struggle Series) is published across six
struggle topics over a 90-day window. Every post metric and comment is
collected into a structured dataset. Each post then gets a single score,
the Viral Coefficient:

```
K = (1.0 x shares + 0.6 x saves + 0.3 x comments) / reach
```

Shares and saves are high-effort actions, so they dominate the score and
likes are left out completely. K >= 0.035 is treated as viral-grade.
Everything else in the project (prediction, experiments, recommendations)
is built on this one number.

## Modules

| # | Module (from the brief) | File |
|---|--------------------------|------|
| 1 | Content Performance Tracker (Data Extraction Engine) | `src/extraction_engine.py` |
| 2 | Virality Prediction Engine | `src/virality_engine.py` |
| 3 | Audience Sentiment Analyzer (NLP Module) | `src/sentiment_analyzer.py` |
| 4 | A/B Testing Framework | `src/ab_testing.py` |
| 5 | Engagement Optimization Recommender | `src/recommender.py` |
| 6 | Growth Visualization Dashboard | `app.py` |
| 7 | Trend Forecasting Module | `src/trend_forecaster.py` |
| - | Relational store (SQLite / PostgreSQL / MySQL) | `src/database.py` |

## Tech stack

| Layer | Technology |
|-------|------------|
| Data processing | Python, Pandas, NumPy |
| Scraping / APIs | Instagram Graph API connectors (Selenium / BeautifulSoup optional) |
| NLP & text analysis | NLTK-style pre-processing, TextBlob-style lexicon baseline, scikit-learn TF-IDF |
| Database | MySQL / PostgreSQL via SQLAlchemy (SQLite by default) |
| Visualization | Matplotlib / Seaborn (report figures), Plotly (dashboard) |
| Statistical modeling | scikit-learn, SciPy |
| Dashboard frontend | Streamlit |

## Deliverables

| # | Deliverable (from the brief) | Where |
|---|-------------------------------|-------|
| 1 | Analytics dashboard (growth, sentiment, viral metrics) | `streamlit run app.py` |
| 2 | Structured dataset (post performance + comment logs) | `data/processed/`, `database/engagement.db` |
| 3 | NLP sentiment model (tags comments "Relatable" / "Neutral") | `nlp_model/sentiment_model.joblib` |
| 4 | Strategy report (which topics drive connection) | `reports/Strategy_Report.docx` |
| 5 | The content series (the creative test subject) | `content/content_series.csv` |

## How to run it

```bash
pip install -r requirements.txt

python run_pipeline.py      # runs everything end to end
streamlit run app.py        # opens the dashboard
python build_report.py      # rebuilds the Word report from the results
```

Each module can also be run on its own, for example
`python -m src.sentiment_analyzer`.

## Results from this run

- Virality is predictable from creative features alone: R2 of 0.77 on held-out posts
- The comment tagger reaches 94.2% accuracy (F1 0.95) on Relatable vs Neutral
- All 6 A/B experiments came out significant (format, hook, length, timing, caption style, topic)
- Social Anxiety and Money Struggles are the strongest connection topics
- The trend module currently flags oldest daughter syndrome, money dysmorphia and brain rot as rising

## Folder layout

```
├── run_pipeline.py            # end-to-end pipeline
├── app.py                     # Streamlit dashboard
├── build_report.py            # report generator
├── config.py                  # paths, seed, virality weights, DB URL
├── requirements.txt
├── src/                       # the modules listed above
├── content/                   # 66-post series with captions
├── data/raw/                  # API-shaped JSON exports
├── data/processed/            # tidy datasets
├── nlp_model/                 # saved models
├── database/                  # loaded database
└── reports/                   # figures, result JSONs, the report
```

## A note on the data

Live Instagram API credentials were not available for this study, so the
extraction module has two layers: real API connectors, and a simulator that
produces statistically realistic platform data (log-normal reach, binomial
action counts, breakouts, time-slot effects). The analysis code does not
care where the data comes from, so a real export can be dropped in without
changing anything downstream. This is also disclosed in the report's
Limitations section.
