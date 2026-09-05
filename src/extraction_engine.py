"""
MODULE 1 - Content Performance Tracker (Data Extraction Engine)
===============================================================
In production this module aggregates raw performance metrics (Shares,
Saves, Retention Rate, ...) from social media APIs (Instagram Graph API /
YouTube Data API) or scraping (Selenium / BeautifulSoup) into a structured
dataset.

Because this study runs on a simulated but statistically realistic
environment (no live API credentials are available to the researcher),
`simulate_extraction()` reproduces the *shape and noise structure* of real
platform exports:

    - reach follows a log-normal distribution per format
    - engagement counts are drawn as Binomial(n=reach, p=rate) events
    - a small share of posts "break out" (viral tail, ~8% of posts)
    - ground-truth effects are baked in for the analytics layer to recover:
        * vulnerability topics (Social Anxiety, Money Struggles) earn
          systematically higher save/share propensity
        * Reels drive reach; Carousels drive saves; Static drives little
        * short Reels (<30s) retain better than long ones
        * evening posting slots outperform morning ones
        * Text hooks convert on Carousels, Visual hooks on Reels

Outputs
-------
data/raw/                 per-metric export files, exactly as an API would emit
data/processed/posts.csv  tidy, one-row-per-post performance log
data/processed/comments.csv     user-comment corpus (for the NLP module)
data/processed/followers.csv    daily follower time series (growth mapping)
"""
from __future__ import annotations

import json
import random
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, DATA_RAW, HANDLE, RANDOM_SEED, SIM_DAYS
from src.content_series import build_catalog

START_DATE = date(2025, 6, 1)  # observation window start (day 0)


# --------------------------------------------------------------- helpers --
def _rate(base: float, mult: float, floor: float = 0.001) -> float:
    return max(floor, base * mult)


# Share of a topic's comment section that is self-recognising
# ("this is me") - the gradient the sentiment module must recover.
RELATABLE_SHARE = {
    "Social Anxiety": 0.52,
    "Money Struggles": 0.50,
    "Family Pressure": 0.46,
    "Self-Image & Comparison": 0.43,
    "Dating & Situationships": 0.40,
    "College & Hostel Life": 0.36,
}

TOPIC_MULT = {
    # ground-truth save/share multipliers baked into the simulation
    "Social Anxiety":            {"save": 1.45, "share": 1.50, "comment": 1.35, "like": 1.05},
    "Money Struggles":           {"save": 1.40, "share": 1.30, "comment": 1.25, "like": 1.00},
    "Family Pressure":           {"save": 1.20, "share": 1.28, "comment": 1.30, "like": 1.00},
    "Self-Image & Comparison":   {"save": 1.15, "share": 1.10, "comment": 1.15, "like": 0.98},
    "Dating & Situationships":   {"save": 0.95, "share": 1.05, "comment": 1.10, "like": 1.10},
    "College & Hostel Life":     {"save": 0.85, "share": 0.90, "comment": 1.00, "like": 1.05},
}

FORMAT_BASE = {
    # median reach and baseline action rates by format
    "Reel":        {"reach": 9000, "like": 0.052, "save": 0.011, "share": 0.008, "comment": 0.0022},
    "Carousel":    {"reach": 3800, "like": 0.048, "save": 0.016, "share": 0.006, "comment": 0.0026},
    "Static Post": {"reach": 1900, "like": 0.040, "save": 0.005, "share": 0.003, "comment": 0.0016},
}

HOUR_MULT = {h: (1.15 if 19 <= h <= 22 else 1.05 if 12 <= h <= 14 else 0.92)
             for h in range(24)}


def _topic_aware_retention(topic: str, duration: int, rng: random.Random) -> float:
    """Retention rate for Reels: short videos retain better."""
    if duration <= 30:
        r = rng.uniform(0.62, 0.78)
    elif duration <= 45:
        r = rng.uniform(0.48, 0.62)
    else:
        r = rng.uniform(0.32, 0.46)
    if topic in ("Social Anxiety", "Money Struggles"):
        r += 0.04  # emotionally gripping topics hold attention
    return round(min(r, 0.92), 3)


# ----------------------------------------------------------- simulation ---
def simulate_extraction() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    catalog = build_catalog()
    post_rows, comment_rows = [], []
    followers = 5200  # starting audience

    for post in catalog:
        fmt = FORMAT_BASE[post["format"]]
        tm = TOPIC_MULT[post["topic"]]
        posted_on = (START_DATE + timedelta(days=post["day"])).isoformat()

        reach_mu = np.log(fmt["reach"]) * HOUR_MULT[post["posting_hour"]]
        reach = int(rng.lognormvariate(reach_mu - 0.35, 0.44))

        # ~8% of posts break out through the algorithmic recommendation tail
        breakout = rng.random() < 0.08
        if breakout:
            reach = int(reach * rng.uniform(3.0, 7.5))

        hook_mult = 1.08 if post["hook_type"].startswith("Visual") and post["format"] == "Reel" \
            else 1.08 if post["hook_type"].startswith("Text") and post["format"] == "Carousel" else 1.0

        # caption styles steer behaviour: Stories drive conversation,
        # Listicles drive saves, Questions drive shares (tested in E5)
        cap_mult = {"comment": 1.40 if post["caption_style"] == "Storytelling" else 1.0,
                    "save": 1.50 if post["caption_style"] == "Listicle" else 1.0,
                    "share": 1.60 if post["caption_style"] == "Question" else 1.0}
        likes = int(np.random.binomial(reach, _rate(fmt["like"] * tm["like"] * hook_mult, 1.0)))
        saves = int(np.random.binomial(reach, _rate(fmt["save"] * tm["save"] * hook_mult * cap_mult["save"], 1.0)))
        shares = int(np.random.binomial(reach, _rate(fmt["share"] * tm["share"] * cap_mult["share"], 1.0)))
        # comment corpus = "top comments" export, ~3x the visible count
        n_comments = min(120, max(4, 3 * int(np.random.binomial(reach, _rate(fmt["comment"] * tm["comment"] * cap_mult["comment"], 1.0)))))
        retention = _topic_aware_retention(post["topic"], post["duration_s"] or 30, rng) \
            if post["format"] == "Reel" else ""

        # follower conversion scales with how shareable the post was
        k_proxy = (shares + 0.6 * saves) / max(reach, 1)
        new_followers = int(reach * k_proxy * rng.uniform(0.25, 0.45)) + (40 if breakout else 0)
        followers += new_followers

        post_rows.append({
            "post_id": post["post_id"],
            "handle": HANDLE,
            "topic": post["topic"],
            "title": post["title"],
            "format": post["format"],
            "hook_type": post["hook_type"],
            "duration_s": post["duration_s"] if post["duration_s"] != "" else "",
            "caption_style": post["caption_style"],
            "posted_on": posted_on,
            "day": post["day"],
            "posting_hour": post["posting_hour"],
            "impressions": int(reach * rng.uniform(1.15, 1.45)),
            "reach": reach,
            "likes": likes,
            "comments": n_comments,
            "shares": shares,
            "saves": saves,
            "retention_rate": retention,
            "profile_visits": int(reach * rng.uniform(0.01, 0.03)),
            "followers_gained": new_followers,
            "breakout": int(breakout),
        })

        comment_rows.extend(
            _generate_comments(post, n_comments, rng, posted_on)
        )

    posts = pd.DataFrame(post_rows)
    comments = pd.DataFrame(comment_rows)
    followers_ts = _follower_series(posts)
    return posts, comments, followers_ts


def _follower_series(posts: pd.DataFrame) -> pd.DataFrame:
    """Daily follower counts = cumulative gains + slow organic drift."""
    gain = posts.groupby("day")["followers_gained"].sum().to_dict()
    rng = random.Random(RANDOM_SEED + 1)
    rows, total = [], 5200
    for d in range(SIM_DAYS):
        total += gain.get(d, 0) + int(rng.gauss(8, 6))
        rows.append({"day": d,
                     "date": (START_DATE + timedelta(days=d)).isoformat(),
                     "followers": total})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- comments ---
# Comment pools organised by intent. Ground-truth label used for training
# the NLP model is `relatable` vs `neutral` (Deliverable 3).
# ------------------------------------------------------------------
# Comment language model: comments are COMPOSED from openers x cores x
# tails so the corpus contains hundreds of unique strings with shared
# structure (the way real comment sections look), rather than a small
# set of copy-pasted phrases. Labels: relatable vs neutral.
# ------------------------------------------------------------------
# Comment language model. Comments are COMPOSED from parts the way real
# comment sections work: an opener ("no because", "the way") can only
# attach to a FRAGMENT; standalone sentences stand alone; tails (slang,
# emojis) get attached without spacing glitches. This keeps every
# generated comment grammatical while preserving corpus variety.
REL_OPENERS = ["", "no because", "ok but", "the way", "not me", "why is this",
               "bro", "nah", "literally", "fr", "this is", "hello??",
               "excuse me", "wait", "hold on,", "ok so"]
# fragments read naturally AFTER an opener ("the way this is too accurate")
REL_FRAGMENTS = [
    "so me", "literally me", "too accurate", "too real", "my whole life",
    "my biography", "my everyday routine", "a documentary about me",
    "personal at this point", "exactly what I go through", "it me",
    "my brain on a normal tuesday", "the story of my life", "me in one post",
    "my last three years in one video", "screenshot of my head",
]
# standalone sentences: used WITHOUT an opener
REL_STANDALONE = [
    "this is so me", "this is literally me", "I felt that", "that hit close to home",
    "I'm in this post and I don't like it", "you read my diary", "it's too accurate",
    "who allowed you to expose me like this", "I've never been so seen",
    "I relate to this too much", "stop describing me", "painfully accurate",
    "took the words right out of my mouth", "I live this every single day",
    "why does this feel personal", "this speaks for my whole generation",
    "I didn't consent to being exposed", "every frame is my biography",
    "reading this at 3am was a mistake", "I paused. sat down. processed. same.",
    "genuinely how do you know my life", "reporting this for personal damage",
    "sir this is my diary page", "came here to laugh, stayed to confront myself",
    "the timing of this post is suspicious", "I'm going to pretend I didn't read this",
    "saved this for the next time it happens", "watched this five times already",
    "sending this to the group chat with no context",
]
# longer, ramblier comments (real people sometimes write paragraphs)
REL_LONG = [
    "why did this describe my entire existence in 30 seconds, I'm filing a complaint",
    "sent this to my best friend at 2am and she replied why is this us, I still have no answer",
    "I showed this to my roommate and we just sat in silence for a minute",
    "not to be dramatic but this post understood me better than my last three conversations",
    "I came to instagram to turn my brain off and instead you described it perfectly",
    "the way I did this exact thing this morning and now I'm being called out publicly",
    "watched this, closed the app, opened it again, watched it again. yeah.",
    "this is the kind of post you save at 2am and rewatch when it happens again",
    "my therapist and my group chat will hear about this post",
    "it's one thing to make a meme and another to describe my whole personality",
]
REL_TAILS = ["", "", " 😭", " 💀", " 😭😭", " 😅", " 🥲", " lol", " lmao", " lmaoo",
             " fr", " fr fr", " ngl", " istg", " ong", " no cap", " pls",
             " yaar", " bhai", " da", " anna", "...", " ?!", "!!", " 🥲🥲"]
NEUTRAL_CORES = [
    "nice post", "great content", "cool edit", "awesome work", "love this page",
    "keep it up", "which app is this?", "song name please?", "follow for follow?",
    "check my page", "first comment", "🔥🔥", "😍", "lol ok", "haha", "superb",
    "smooth transitions", "when is part 2?", "do a collab with us",
    "dm for promotions", "free followers in bio", "posted at the right time",
    "background music name?", "the edit quality is insane", "your page will blow up",
    "true", "haha true", "so true", "facts", "true that", "I agree",
    "found this page today, already binged everything", "the algorithm finally did something right",
    "deserves way more likes", "underrated page fr",
]
NEUTRAL_TAILS = ["", "", " 👍", "!", " 🔥", " bro", " sis", " !!", " :)", " btw",
                 " macha", " ++"]
# genuinely ambiguous short reactions: label assigned with a coin flip,
# mirroring annotator disagreement on real corpora
AMBIGUOUS_POOL = ["real", "too real", "realest thing on my feed today", "I'm crying",
                  "crying", "ok this is a mood", "a mood", "mood.", "the caption though",
                  "this one", "oh.", "exactly.", "exactly this", "it's the accuracy for me",
                  "wow", "wow okay", "why is this a mood", "not this again",
                  "I can't", "cant even", "the honesty is refreshing", "this hit",
                  "it hit", "hmm", "okay but same", "but same though", "kinda yes",
                  "no literally", "yeah.", "yeah no", "no yeah", "uncomfortably accurate",
                  "why is nobody talking about how accurate this is", "this felt personal",
                  "personally attacked", "wow okay then", "I feel seen", "seen.",
                  "unfortunately yes", "sadly accurate", "hurts but fair",
                  "this is my 13th reason (affectionate)", "ok this one hurt",
                  "the last line though", "not the last line", "the ending 😭"]

def _compose(rng: random.Random, kind: str) -> str:
    """Grammar-safe comment assembly for kind in {'relatable','neutral'}."""
    if kind == "relatable":
        u = rng.random()
        if u < 0.06:
            text = rng.choice(REL_LONG)                     # rambles stand alone
        elif u < 0.50:
            op = rng.choice(REL_OPENERS)
            frag = rng.choice(REL_FRAGMENTS)
            if not op:                                       # empty opener:
                text = rng.choice(["this is ", "that's ", ""]) + frag
            else:
                text = f"{op} {frag}"
        else:
            text = rng.choice(REL_STANDALONE)
        tail = rng.choice(REL_TAILS)
        if tail and tail.startswith(("?", "!")):
            text = text.rstrip(".?")
        text = (text + tail).strip()
    else:
        text = rng.choice(NEUTRAL_CORES)
        tail = rng.choice(NEUTRAL_TAILS)
        text = (text + tail).strip()
    # casing: most keep natural form, a third go all-lower (very common),
    # short reactions occasionally shout
    u = rng.random()
    if u < 0.32:
        text = text.lower()
    elif u < 0.36 and len(text) <= 18:
        text = text.upper()
    return text


SARCASM_POOL = [  # surface-relatable words, actually neutral/dismissive
    "wow so relatable 🙄", "sure, totally me", "yes because everyone does this obviously",
    "deep. very deep.", "another relatable page, how original", "ok boomer",
    "this is just basic psychology", "everyone overthinks, nothing special",
    "congratulations you discovered being human", "so dramatic for no reason",
    "who hurt you bro", "this generation and its labels istg", "fake deep",
    "instagram therapists at it again", "ok but this is literally everyone, nothing unique",
    "stating the obvious since 2020", "award for most generic post goes to",
]
QUESTION_POOL = [
    "how do you deal with this though?", "any tips for the 2am version of this?",
    "what app is this?", "is this based on real events?", "part 2 please?",
    "can you make one about exams?", "do you take content requests?",
    "what's the song used here?", "where are you from?", "how long did this take to edit?",
    "part 3 when?", "can you do one on exam results?", "do one about hostellers please",
    "which college is this?", "is that your real voice?", "why does this have no likes, underrated",
    "can we get a version with english subtitles?", "what camera do you use?",
    "make one about first job interviews please", "how are you so consistent with posting?",
]
SPAM_POOL = [
    "DM for promotion 📩", "earn 5000 daily, link in bio", "check dm", "collab? dm us",
    "get free followers at InstaBoost", "🚀🚀 grow your account fast",
    "we feature meme pages, dm us", "promotion rates in dm", "crypto class in bio 🔥",
    "please check my new page 🙏", "growth service dm now",
]
TOPIC_SPECIFIC = {
    "Social Anxiety": ["my hands shake when the phone rings too", "I rehearse before ordering coffee, no exaggeration"],
    "Dating & Situationships": ["mine left me on delivered for 3 days 😭", "he said 'we'll see' and I said ok like a fool"],
    "Money Struggles": ["my account balance is a horror movie", "Rs.200 is not small when you have Rs.400"],
    "College & Hostel Life": ["our mess soup is just hot water with regret", "8am classes should be illegal"],
    "Family Pressure": ["'log kya kahenge' has ruined more careers than failure ever did", "I'm the family NGO, free labour included"],
    "Self-Image & Comparison": ["uninstalled Instagram twice, reinstalled in a day", "everyone my age is achieving things and I'm here"],
}


def _corrupt(text: str, rng: random.Random) -> str:
    """Occasionally add realistic typing noise: typo, vowel stretch,
    dropped apostrophe, random casing."""
    u = rng.random()
    if u < 0.045 and len(text) > 8:              # swap two adjacent letters
        i = rng.randrange(1, len(text) - 2)
        if text[i].isalpha() and text[i + 1].isalpha():
            text = text[:i] + text[i + 1] + text[i] + text[i + 2:]
    elif u < 0.075 and len(text) > 6:            # stretch a vowel: soooo
        i = rng.randrange(1, len(text) - 1)
        if text[i] in "aeiou":
            text = text[:i + 1] + text[i] * rng.choice([1, 2]) + text[i + 1:]
    elif u < 0.135 and "'" in text:              # didnt / cant / im
        text = text.replace("'", "", 1)
    elif u < 0.155:                              # random casing
        text = text.upper() if rng.random() < 0.5 else text.lower()
    return text


def _generate_comments(post: dict, n: int, rng: random.Random,
                       posted_on: str) -> list[dict]:
    """Generate a linguistically varied comment thread for one post."""
    rows: list[dict] = []
    relatable_share = RELATABLE_SHARE[post["topic"]]
    seen: set[str] = set()
    while len(rows) < n:
        u = rng.random()
        if u < relatable_share:
            label, intent, text = "relatable", "validation", _compose(rng, "relatable")
        elif (u < relatable_share + 0.08
              and post["topic"] in TOPIC_SPECIFIC):
            text = rng.choice(TOPIC_SPECIFIC[post["topic"]])
            label, intent = "relatable", "personal_story"
        elif u < relatable_share + 0.08 + 0.10:
            text = rng.choice(AMBIGUOUS_POOL)
            label = "relatable" if rng.random() < 0.5 else "neutral"  # disputed
            intent = "ambiguous"
        elif u < relatable_share + 0.08 + 0.10 + 0.26:
            label, intent, text = "neutral", "generic", _compose(rng, "neutral")
        elif u < relatable_share + 0.08 + 0.10 + 0.26 + 0.06:
            text = rng.choice(SARCASM_POOL)
            label, intent = "neutral", "sarcasm"
        elif u < relatable_share + 0.08 + 0.10 + 0.26 + 0.06 + 0.10:
            text = rng.choice(QUESTION_POOL)
            label, intent = "neutral", "question"
        else:
            text = rng.choice(SPAM_POOL)
            label, intent = "neutral", "spam"
        if text in seen:            # keep unique phrasing per thread
            continue
        seen.add(text)
        rows.append(_comment_row(rows, post, _corrupt(text, rng),
                                 intent, label, rng, posted_on))
    return rows[:n]


def _comment_row(rows: list, post: dict, text: str, intent: str,
                 label: str, rng: random.Random, posted_on: str) -> dict:
    return {
        "comment_id": f"C{len(rows):05d}_{post['post_id']}",
        "post_id": post["post_id"],
        "topic": post["topic"],
        "comment_text": text,
        "intent": intent,
        "sentiment_label": label,          # ground truth for the NLP model
        "likes_on_comment": int(np.random.binomial(60, 0.18)),
        "posted_on": posted_on,
    }


# -------------------------------------------------------------- pipeline --
def run() -> dict:
    posts, comments, followers = simulate_extraction()

    # Raw exports - structured exactly like paginated API responses
    posts.to_json(DATA_RAW / "api_posts_export.json", orient="records", indent=2)
    comments.to_json(DATA_RAW / "api_comments_export.json", orient="records", indent=2)
    followers.to_json(DATA_RAW / "api_followers_export.json", orient="records", indent=2)

    # Processed / tidy datasets (Deliverable 2)
    posts.to_csv(DATA_PROCESSED / "posts.csv", index=False)
    comments.to_csv(DATA_PROCESSED / "comments_raw.csv", index=False)
    followers.to_csv(DATA_PROCESSED / "followers.csv", index=False)

    return {
        "posts": posts,
        "comments": comments,
        "followers": followers,
        "summary": {
            "posts": len(posts),
            "comments": len(comments),
            "total_reach": int(posts["reach"].sum()),
            "total_shares": int(posts["shares"].sum()),
            "total_saves": int(posts["saves"].sum()),
            "final_followers": int(followers["followers"].iloc[-1]),
        },
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["summary"], indent=2))
