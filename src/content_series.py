"""
DELIVERABLE 5 - The Content Series
==================================
"The Relatable Struggle Series" is the creative output used as the test
subject for every analytical module in this project.

66 posts across 6 "struggle topics" were planned and published over a
90-day observation window. Each record captures the creative variables
that the A/B Testing Framework later manipulates statistically:

    topic          - which relatable struggle the post speaks to
    format         - Reel / Carousel / Static Post
    hook_type      - Visual hook vs. Text hook (first-frame treatment)
    duration_s     - length of the video (Reels only)
    caption_style  - Storytelling / Listicle / Question
    posting_hour   - 24h clock time of publishing

Running this file as a script writes content/content_series.csv.
"""
from __future__ import annotations

import csv
import random

from config import CONTENT_DIR, RANDOM_SEED, SIM_DAYS

# ----------------------------------------------------------- topic bank ---
TOPICS = {
    "Social Anxiety": [
        ("Rehearsing a conversation that was never going to happen", "Reel", "Text", 24),
        ("Leaving the party without telling anyone: a documentary", "Reel", "Visual", 31),
        ("The 3-hour spiral after 'seen' but no reply", "Carousel", "Text", None),
        ("Ordering food like you're defusing a bomb", "Reel", "Visual", 18),
        ("'You've changed' - no, I just stopped performing", "Static", "Text", None),
        ("Phone anxiety: drafting the text for 40 minutes", "Reel", "Text", 27),
        ("Small talk is my final boss", "Reel", "Visual", 52),
        ("A survival guide for people who rehearse orders in their head", "Carousel", "Text", None),
        ("When the teacher says 'pair up'", "Reel", "Visual", 15),
        ("Being called 'quiet' like it's an accusation", "Static", "Text", None),
        ("The walk of shame from the doorbell back to your room", "Reel", "Text", 19),
    ],
    "Dating & Situationships": [
        ("Situationship: 99% girlfriend, 0% definition", "Reel", "Text", 29),
        ("Analyzing his 'gn' like it's ancient scripture", "Reel", "Visual", 22),
        ("Red flags I ignored, a Pinterest board", "Carousel", "Text", None),
        ("'We should hang out sometime' - sometime never arrives", "Static", "Text", None),
        ("The ick is real and it arrived mid-date", "Reel", "Visual", 26),
        ("Being the therapist friend and the problem", "Carousel", "Visual", None),
        ("Soft launching a person like a startup", "Reel", "Text", 20),
        ("Left on read by someone who posts stories hourly", "Reel", "Visual", 24),
        ("'You deserve better' - from the person making me settle", "Carousel", "Text", None),
        ("Falling for the 'let's see where this goes'", "Static", "Text", None),
        ("Reading the chat at 2am with a forensic lens", "Reel", "Text", 48),
    ],
    "Money Struggles": [
        ("My bank account after the 1st of the month", "Reel", "Visual", 17),
        ("Price of a coffee vs. my self-worth", "Carousel", "Text", None),
        ("Budgeting: a fantasy genre", "Reel", "Text", 15),
        ("'It's just Rs.200' said 30 times a month", "Carousel", "Visual", None),
        ("Checking the bill with one eye closed", "Reel", "Visual", 21),
        ("Rich until the UPI fails publicly", "Static", "Text", None),
        ("The math of 'save more' with nothing left to save", "Carousel", "Text", None),
        ("Adding to cart as self-care, never checking out", "Reel", "Text", 23),
        ("Salary day is a 24-hour festival", "Reel", "Visual", 19),
        ("Splitting the bill like a courtroom drama", "Reel", "Text", 55),
        ("Financial plan: win a lottery I never bought", "Static", "Text", None),
    ],
    "College & Hostel Life": [
        ("Attendance: a horror story in 3 acts", "Reel", "Text", 25),
        ("Hostel food vs. my gut: the ongoing war", "Carousel", "Visual", None),
        ("Studying everything the night before, a tradition", "Reel", "Visual", 46),
        ("The roommate who snores like a jet engine", "Reel", "Text", 22),
        ("Backlog season is a personality trait", "Static", "Text", None),
        ("'Tomorrow I start' - a lie told since 2019", "Carousel", "Text", None),
        ("Mess food roulette: today's mystery curry", "Reel", "Visual", 16),
        ("Group project: doing everything, sharing nothing", "Carousel", "Text", None),
        ("That one 8am class that ruined a whole GPA", "Reel", "Text", 20),
        ("Laundry day: day 23 of the same hoodie", "Static", "Visual", None),
        ("Result day heartbeat > any workout", "Reel", "Visual", 18),
    ],
    "Family Pressure": [
        ("'Log kya kahenge' - the national question", "Reel", "Text", 27),
        ("Comparing me to Sharma ji's son since birth", "Carousel", "Text", None),
        ("Career counseling by relatives who failed at theirs", "Reel", "Visual", 24),
        ("I said 'no' once and became the villain", "Static", "Text", None),
        ("The 'sab kuch shi ho jayega' parenting style", "Carousel", "Visual", None),
        ("Hiding marksheets like state secrets", "Reel", "Text", 21),
        ("Being the family's free tech support, forever", "Reel", "Visual", 17),
        ("Love in this house is served with conditions", "Carousel", "Text", None),
        ("'Beta, shaadi ka plan?' - I'm 22", "Reel", "Text", 44),
        ("Explaining my degree for the 47th time at dinner", "Static", "Text", None),
        ("Growing up before I was ready", "Carousel", "Text", None),
    ],
    "Self-Image & Comparison": [
        ("Everyone's life is a highlight reel, mine is bloopers", "Reel", "Text", 26),
        ("Scrolling LinkedIn with chest pain", "Reel", "Visual", 20),
        ("Body checking in every reflective surface", "Carousel", "Text", None),
        ("'Be yourself' - but not like that", "Static", "Text", None),
        ("The 2am glow-up I never start", "Reel", "Text", 23),
        ("My skin has more drama than my life", "Carousel", "Visual", None),
        ("Unfollowing people to feel better, it never works", "Reel", "Text", 18),
        ("Gym day 1 vs. day 2: the sequel nobody makes", "Reel", "Visual", 21),
        ("Smiling in photos I hate myself in", "Carousel", "Text", None),
        ("Aesthetic is a cage I built myself", "Static", "Text", None),
        ("Losing myself to look 'put together'", "Reel", "Text", 51),
    ],
}

CAPTION_TEMPLATES = {
    "Storytelling": (
        "{title}. If you know, you know.\n\n"
        "I made this because I lived it - every single frame of it. "
        "Somewhere between pretending we're fine and actually being fine, "
        "there's a whole generation figuring itself out.\n\n"
        "If this hit home, send it to the friend who needs it today. "
        "And tell me your version in the comments - I read every one.\n\n"
        "{hashtags}"
    ),
    "Listicle": (
        "{title} - the unofficial guide:\n\n"
        "1. Pretend it's fine.\n"
        "2. Overthink that it's not fine.\n"
        "3. Make a meme about it instead of processing it.\n"
        "4. Repeat.\n\n"
        "Save this for the next time it happens (we both know it will). "
        "Which step are you stuck on? Number in the comments.\n\n"
        "{hashtags}"
    ),
    "Question": (
        "{title}.\n\n"
        "Real question: when did this become normal for us? "
        "Drop your story below - the weirder the better. "
        "Sharing this might be easier than explaining it out loud, "
        "so if it resonates, pass it on quietly.\n\n"
        "{hashtags}"
    ),
}

TOPIC_HASHTAGS = {
    "Social Anxiety": "#socialanxiety #overthinking #introvertlife #mentalhealthmatters #relatable",
    "Dating & Situationships": "#situationship #datinglife #modernlove #redflags #relatable",
    "Money Struggles": "#brokelife #moneystruggles #studentlife #budgeting #relatable",
    "College & Hostel Life": "#hostellife #collegelife #exams #messfood #relatable",
    "Family Pressure": "#familypressure #desiparents #logkyakahenge #brownkidproblems #relatable",
    "Self-Image & Comparison": "#selfimage #comparisontrap #highlightreel #selfworth #relatable",
}

CAPTION_STYLES = list(CAPTION_TEMPLATES.keys())
# Posting hours clustered into 3 strategic slots used by the A/B framework:
# morning (9-11), midday (12-14), evening (18-22)
POSTING_HOURS = [9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22]


def build_catalog() -> list[dict]:
    """Build the full 66-post catalogue of the series."""
    rng = random.Random(RANDOM_SEED)
    rows: list[dict] = []
    # Interleave topics so the publishing calendar alternates themes,
    # then spread posts roughly evenly across the 90-day window.
    ordered: list[tuple[str, tuple]] = []
    max_len = max(len(v) for v in TOPICS.values())
    for i in range(max_len):
        for topic, posts in TOPICS.items():
            if i < len(posts):
                ordered.append((topic, posts[i]))

    n = len(ordered)
    topic_index = {t: k for k, t in enumerate(TOPICS)}
    per_topic_count: dict[str, int] = {}
    for idx, (topic, (title, fmt, hook, duration)) in enumerate(ordered):
        fmt = "Static Post" if fmt == "Static" else fmt
        day = round(idx * (SIM_DAYS - 1) / (n - 1))
        hour = rng.choice(POSTING_HOURS)
        # caption styles are BALANCED within each topic (an A/B design
        # requirement - otherwise style is confounded with topic)
        k = per_topic_count.get(topic, 0)
        per_topic_count[topic] = k + 1
        style = CAPTION_STYLES[(k + topic_index[topic]) % 3]
        rows.append(
            {
                "post_id": f"P{idx + 1:03d}",
                "episode": idx + 1,
                "topic": topic,
                "title": title,
                "format": fmt,
                "hook_type": f"{hook} Hook",
                "duration_s": duration if fmt == "Reel" else "",
                "caption_style": style,
                "day": day,
                "posting_hour": hour,
                "caption": CAPTION_TEMPLATES[style].format(
                    title=title, hashtags=TOPIC_HASHTAGS[topic]
                ),
            }
        )
    return rows


def main() -> None:
    rows = build_catalog()
    out = CONTENT_DIR / "content_series.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[content_series] wrote {len(rows)} posts -> {out.relative_to(CONTENT_DIR.parents[1])}")


if __name__ == "__main__":
    main()
