"""
Central configuration for The Data-Driven Social Engagement Initiative.

Project : Data Science Major Project
Author  : G.Vaishanth
Batch   : Data Science June Batch
"""
from pathlib import Path

# ---------------------------------------------------------------- paths ----
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODEL_DIR = ROOT / "nlp_model"
DB_DIR = ROOT / "database"
CONTENT_DIR = ROOT / "content"

for _d in (DATA_RAW, DATA_PROCESSED, REPORTS, FIGURES, MODEL_DIR, DB_DIR, CONTENT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------- project ----
PROJECT_NAME = "The Data-Driven Social Engagement Initiative"
SERIES_NAME = "The Relatable Struggle Series"
HANDLE = "@relatable.struggles"          # demo Instagram handle for the study
RANDOM_SEED = 42
SIM_DAYS = 90                             # length of the observation window

# ----------------------------------------------------------- database -----
# Default to a local SQLite file so the project runs anywhere.
# Switch to PostgreSQL/MySQL by changing the URL, e.g.:
#   postgresql+psycopg2://user:pass@localhost:5432/engagement
#   mysql+pymysql://user:pass@localhost:3306/engagement
DB_URL = f"sqlite:///{DB_DIR / 'engagement.db'}"

# -------------------------------------------------- virality definition ---
# Weights used by the Virality Prediction Engine. High-value actions
# (shares / saves) are deliberately weighted above passive ones (likes).
VIRAL_WEIGHTS = {"shares": 1.0, "saves": 0.6, "comments": 0.3}
VIRAL_K_THRESHOLD = 0.035                 # K above this => "viral breakout"

# ------------------------------------------------------- a/b testing ------
ALPHA = 0.05                              # significance level
