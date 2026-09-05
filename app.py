"""
DELIVERABLE 1 - Growth Visualization Dashboard
==============================================
Analytics dashboard for The Data-Driven Social Engagement Initiative.
Growth, sentiment and viral metrics for the 66-post Content Series,
plus a live demo of the comment tagger and downloadable datasets.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from config import (DATA_PROCESSED, MODEL_DIR, PROJECT_NAME, REPORTS,
                    SERIES_NAME)

ACCENT = "#1a73e8"          # one accent, used sparingly

st.set_page_config(page_title="Social engagement analytics",
                   layout="wide", initial_sidebar_state="expanded")


# ------------------------------------------------------------------ data --
@st.cache_data(ttl=0)
def load_data():
    posts = pd.read_csv(DATA_PROCESSED / "posts_scored.csv")
    comments = pd.read_csv(DATA_PROCESSED / "comments_tagged.csv")
    followers = pd.read_csv(DATA_PROCESSED / "followers.csv")
    ab = pd.read_csv(DATA_PROCESSED / "ab_test_results.csv")
    recs = pd.read_csv(DATA_PROCESSED / "recommendations.csv")
    trends = pd.read_csv(DATA_PROCESSED / "trend_forecast.csv")
    trend_raw = pd.read_csv(DATA_PROCESSED / "trend_interest_raw.csv")
    ab_detail = json.loads((REPORTS / "ab_tests.json").read_text())
    sent = json.loads((REPORTS / "sentiment_model.json").read_text())
    viral = json.loads((REPORTS / "virality_model.json").read_text())
    rec_payload = json.loads((REPORTS / "recommendations.json").read_text())
    catalog = pd.read_csv(DATA_PROCESSED.parent.parent / "content" / "content_series.csv")
    posts["posted_on"] = pd.to_datetime(posts["posted_on"])
    posts["engagement_rate"] = ((posts["likes"] + posts["comments"]
                                 + posts["shares"] + posts["saves"])
                                / posts["reach"])
    return dict(posts=posts, comments=comments, followers=followers, ab=ab,
                recs=recs, trends=trends, trend_raw=trend_raw,
                ab_detail=ab_detail, sent=sent, viral=viral,
                rec_payload=rec_payload, catalog=catalog)


D = load_data()
posts, comments, followers = D["posts"], D["comments"], D["followers"]


# ----------------------------------------------------------------- theme --
def make_theme(dark: bool) -> dict:
    if dark:
        return dict(bg="#0e1117", panel="#181c26", text="#e8eaed",
                    sub="#9aa3b2", muted="#414a5c", grid="#272d3a",
                    line="#3d4452", template="plotly_dark")
    return dict(bg="#ffffff", panel="#f6f8fa", text="#33383f",
                sub="#6b7280", muted="#c9cfd8", grid="#eceff3",
                line="#d5dae2", template="plotly_white")


SECTIONS = ["Overview", "Virality & growth", "Comment sentiment",
            "A/B tests", "Next week's plan", "Trend radar", "Data"]

# On a fresh session, honour page/theme shortcuts passed in the URL
# (the keyboard handler works by rewriting these params and reloading).
if "nav_init" not in st.session_state:
    st.session_state.nav_init = True
    qp_page = st.query_params.get("page")
    if qp_page and qp_page.isdigit() and 1 <= int(qp_page) <= len(SECTIONS):
        st.session_state.page = SECTIONS[int(qp_page) - 1]
if "dark" not in st.session_state:
    st.session_state.dark = st.query_params.get("theme") == "dark"

if "page" not in st.session_state:
    st.session_state.page = "Overview"

with st.sidebar:
    dark = st.toggle("Dark theme", key="dark")
    st.markdown("")
    st.markdown(f"**{PROJECT_NAME}**")
    st.caption("G.Vaishanth · Data Science June Batch")

T = make_theme(dark)
st.markdown(
    f"""
    <style>
      .stApp {{ background: {T['bg']}; color-scheme: {'dark' if dark else 'light'}; }}
      [data-testid="stSidebar"] {{ background: {T['panel']}; }}
      [data-testid="stHeader"] {{
          background: {T['bg']}; border-bottom: 1px solid {T['line']};
          position: relative;
      }}
      [data-testid="stHeader"]::after {{
          content: "Social Engagement Analysis";
          position: absolute; left: 50%; top: 50%;
          transform: translate(-50%, -50%);
          font-family: 'Times New Roman', Times, serif;
          font-size: 36px; font-weight: 700; letter-spacing: .5px;
          color: {T['text']}; white-space: nowrap;
      }}
      h1, h2, h3,
      [data-testid="stMarkdownContainer"] h1,
      [data-testid="stMarkdownContainer"] h2,
      [data-testid="stMarkdownContainer"] h3,
      .stMarkdown h1, .stMarkdown h2, .stMarkdown h3
          {{ color: {T['text']} !important; }}
      .stMarkdown p {{ color: {T['text']}; }}
      [data-testid="stMetricValue"] {{ color: {T['text']}; }}
      [data-testid="stMetricLabel"] p {{ color: {T['sub']}; }}
      [data-testid="stDataFrame"] {{ color-scheme: {'dark' if dark else 'light'}; }}
      /* section tiles, sidebar only */
      [data-testid="stSidebar"] button[kind="secondary"] {{
          background: {T['bg']}; border: 1px solid {T['line']};
          border-radius: 8px; padding: 9px 14px; margin-bottom: 2px;
      }}
      [data-testid="stSidebar"] button[kind="secondary"] p
          {{ color: {T['text']}; font-weight: 500; }}
      [data-testid="stSidebar"] button[kind="secondary"]:hover
          {{ border-color: {ACCENT}; }}
      [data-testid="stSidebar"] button[kind="secondary"]:hover p
          {{ color: {ACCENT}; }}
      [data-testid="stSidebar"] button[kind="primary"] {{
          background: {ACCENT}; border: 1px solid {ACCENT};
          border-radius: 8px; padding: 9px 14px; margin-bottom: 2px;
      }}
      [data-testid="stSidebar"] button[kind="primary"] p
          {{ color: #ffffff; font-weight: 600; }}
    </style>
    """, unsafe_allow_html=True)

# --------------------------------------------------- sidebar tile nav -----
with st.sidebar:
    st.markdown("")
    for name in SECTIONS:
        st.button(name, key=f"nav_{name}", use_container_width=True,
                  type="primary" if st.session_state.page == name else "secondary",
                  on_click=lambda n=name: setattr(st.session_state, "page", n))
    st.caption("Shortcuts: 1-7 = sections, Tab = theme")
    st.caption("(click the page once first; blocked in embedded previews)")

page = st.session_state.page


def page_header(title: str) -> None:
    """Section title with credit, identical on every page."""
    st.header(title)
    st.caption("by G.Vaishanth")


# ------------------------------------------------ keyboard shortcuts ------
# A zero-height frame installs a keydown listener on the app document.
# Keys are translated into real clicks on the sidebar tiles (1-7) and the
# theme toggle (Tab), so everything goes through normal Streamlit reruns
# with no page reloads. (Embedded sandboxed previews block cross-frame
# access, so shortcuts need a full browser tab.)
components.html(
    """
    <script>
    (function () {
      var win = null, target = null;
      try { win = window.parent; win.document; target = win.document; } catch (e) {}
      if (!target) {
        try { win = window.top; win.document; target = win.document; } catch (e) {}
      }
      if (!target) { win = window; target = document; }
      if (target.__dseKeys) return;
      target.__dseKeys = true;

      var editing = function () {
        var a = null;
        try { a = win.document.activeElement; } catch (err) {}
        a = a || document.activeElement || {};
        var t = (a.tagName || "").toLowerCase();
        return t === "input" || t === "textarea" || a.isContentEditable;
      };

      var handler = function (e) {
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        if (editing()) return;
        if (/^[1-7]$/.test(e.key)) {
          e.preventDefault();
          var idx = parseInt(e.key, 10) - 1;
          // only the nav tiles: primary (active) + secondary (inactive),
          // excluding the sidebar collapse control
          var btns = target.querySelectorAll(
            '[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"],' +
            '[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]');
          if (btns.length > idx) { btns[idx].click(); }
        } else if (e.key === "Tab") {
          e.preventDefault();
          var boxes = target.querySelectorAll('[data-testid="stCheckbox"]');
          for (var i = 0; i < boxes.length; i++) {
            if (boxes[i].textContent.indexOf("Dark theme") !== -1) {
              var inp = boxes[i].querySelector("input");
              if (inp) { inp.click(); }
              break;
            }
          }
        }
      };
      target.addEventListener("keydown", handler, true);
    })();
    </script>
    """,
    height=0,
)


# ------------------------------------------------------- chart utilities --
def clean(fig, height=330, legend=False):
    """Strip chart junk so the data does the talking."""
    fig.update_layout(template=T["template"], height=height, showlegend=legend,
                      margin=dict(l=5, r=15, t=20, b=5),
                      font=dict(size=12.5, color=T["text"]),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      coloraxis_showscale=False)
    # explicit colors so no template can dim the labels
    fig.update_xaxes(gridcolor=T["grid"], linecolor=T["line"],
                     tickfont=dict(color=T["text"]),
                     title_font=dict(color=T["text"]))
    fig.update_yaxes(gridcolor=T["grid"], linecolor=T["line"],
                     tickfont=dict(color=T["text"]),
                     title_font=dict(color=T["text"]))
    fig.update_layout(legend=dict(font=dict(color=T["text"])))
    return fig


def plot(fig, **kw):
    """Render with the figure's own theme (theme=None), so the explicit
    per-theme colors set in clean() are not overridden by Streamlit's
    built-in light palette."""
    kw.setdefault("use_container_width", True)
    st.plotly_chart(fig, theme=None, **kw)


def highlight_bars(df, x, y, top_by, top_n=1, horizontal=True):
    """Grey bars, accent for the ones worth looking at."""
    order = df.sort_values(top_by, ascending=not horizontal).head(top_n)[x]
    colors = [ACCENT if v in set(order) else T["muted"] for v in df[x]]
    fig = px.bar(df, x=x, y=y, orientation="h" if horizontal else "v")
    fig.update_traces(marker_color=colors)
    return fig


def note(text):
    st.markdown(f"<span style='color:{T['sub']};font-size:14px'>{text}</span>",
                unsafe_allow_html=True)


# ============================================================== OVERVIEW ==
if page == "Overview":
    page_header("Overview")
    c = st.columns(5)
    c[0].metric("Posts", len(posts))
    c[1].metric("Reach", f"{posts['reach'].sum()/1000:,.0f}k")
    c[2].metric("Followers", f"{followers['followers'].iloc[-1]:,}",
                f"+{followers['followers'].iloc[-1] - followers['followers'].iloc[0]:,}")
    c[3].metric("Avg viral coefficient", f"{posts['viral_coefficient'].mean():.4f}")
    c[4].metric("Relatable comments", f"{(comments['relatable_tag']=='Relatable').mean():.0%}")

    st.write("")
    st.subheader("Follower count, day by day")
    fig = px.area(followers, x="date", y="followers")
    fig.update_traces(line_color=ACCENT,
                      fillcolor="rgba(26,115,232,.08)")
    plot(clean(fig, 290), use_container_width=True)

    jumps = posts.nlargest(3, "followers_gained")[["post_id", "topic", "followers_gained"]]
    jump_txt = ", ".join(f"{r.post_id} ({r.topic}, +{r.followers_gained:,})"
                         for r in jumps.itertuples())
    note(f"The flat stretches are ordinary posts. Almost all of the growth sits "
         f"in a few episodes: {jump_txt}. That concentration is what the rest "
         f"of this dashboard tries to explain.")

    left, right = st.columns((3, 2), gap="large")
    with left:
        st.subheader("Reach by topic")
        by_topic = (posts.groupby("topic")["reach"].sum()
                    .reset_index().sort_values("reach"))
        fig = highlight_bars(by_topic, "reach", "topic", "reach")
        fig.update_layout(height=280)
        plot(clean(fig, 280), use_container_width=True)
    with right:
        st.subheader("Ten most spread posts")
        top = posts.nlargest(10, "viral_coefficient")[
            ["post_id", "topic", "format", "reach", "shares", "saves",
             "viral_coefficient"]].copy()
        top["viral_coefficient"] = top["viral_coefficient"].round(4)
        st.dataframe(top, use_container_width=True, hide_index=True, height=280)

# ==================================================== VIRALITY & GROWTH ===
elif page == "Virality & growth":
    page_header("Virality & growth")
    note("K, the viral coefficient, is the study's core index: "
         "K = (shares + 0.6·saves + 0.3·comments) ÷ reach. Shares and saves are "
         "effortful actions, so they dominate; likes are ignored.")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("K by topic")
        fig = px.box(posts, x="viral_coefficient", y="topic", orientation="h")
        fig.update_traces(marker_color=ACCENT, line_color=ACCENT,
                          fillcolor="rgba(26,115,232,.12)")
        plot(clean(fig, 360), use_container_width=True)
    with right:
        st.subheader("Reach vs K, every episode")
        fig = px.scatter(posts, x="reach", y="viral_coefficient",
                         hover_data=["post_id", "topic", "title"])
        fig.update_traces(marker=dict(color=ACCENT, size=9, opacity=.65))
        best = posts.loc[posts["viral_coefficient"].idxmax()]
        fig.add_annotation(x=best["reach"], y=best["viral_coefficient"],
                           text=f"{best['post_id']} · {best['topic']}",
                           showarrow=True, arrowhead=2, ay=-40,
                           font=dict(size=11, color=T["text"]))
        fig.update_xaxes(type="log")
        plot(clean(fig, 360), use_container_width=True)

    med_reel = posts.loc[posts["format"] == "Reel", "reach"].median()
    med_car = posts.loc[posts["format"] == "Carousel", "reach"].median()
    note(f"Reels median reach {med_reel:,.0f} vs {med_car:,.0f} for carousels, "
         f"roughly a {med_reel/max(med_car,1):.1f}x gap. The K "
         f"ranking is a different story: it rewards topics people privately "
         f"save, not just formats the algorithm pushes.")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Save-to-share ratio")
        sts = (posts.assign(r=posts["saves"] / posts["shares"].clip(lower=1))
               .groupby("topic")["r"].mean().reset_index().sort_values("r"))
        fig = highlight_bars(sts, "r", "topic", "r")
        fig.update_layout(height=300)
        plot(clean(fig, 300), use_container_width=True)
        note("High ratio = people kept this for themselves. Low ratio = they "
             "sent it to someone. Family-pressure and social-anxiety posts "
             "are the ones audiences bookmark.")
    with right:
        st.subheader("What predicts K before publishing")
        imp = pd.Series(D["viral"]["feature_importance"]).sort_values()
        fig = px.bar(imp, orientation="h")
        fig.update_traces(marker_color=[ACCENT if v == imp.max() else T["muted"]
                                        for v in imp])
        fig.update_layout(height=300)
        plot(clean(fig, 300), use_container_width=True)
        m = D["viral"]
        note(f"Random forest on creative features only: hold-out R² "
             f"{m['holdout_r2_logspace']}, 5-fold CV {m['cv_r2_logspace_5fold']}. "
             f"Format matters most; topic and caption style carry real weight too.")

    st.subheader("Followers gained by topic")
    growth = (posts.groupby("topic")["followers_gained"].sum()
              .reset_index().sort_values("followers_gained"))
    fig = highlight_bars(growth, "followers_gained", "topic", "followers_gained")
    plot(clean(fig, 280), use_container_width=True)

# ========================================================== SENTIMENT =====
elif page == "Comment sentiment":
    page_header("Comment sentiment")
    m = D["sent"]
    c = st.columns(4)
    c[0].metric("Accuracy", f"{m['accuracy']:.1%}")
    c[1].metric("Precision", f"{m['precision_relatable']:.1%}")
    c[2].metric("Recall", f"{m['recall_relatable']:.1%}")
    c[3].metric("F1 (relatable)", f"{m['f1_relatable']:.1%}")
    note(f"TF-IDF + logistic regression, trained on {m['train_comments']:,} "
         f"labelled comments, tested on {m['test_comments']:,} held out. The "
         f"job: separate comments where the audience recognises itself "
         f"(\"relatable\") from everything else (\"neutral\").")

    left, right = st.columns((2, 3), gap="large")
    with left:
        st.subheader("Confusion matrix (hold-out)")
        cm = np.array(m["confusion_matrix"])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Neutral", "Relatable"], y=["Neutral", "Relatable"])
        plot(clean(fig, 300), use_container_width=True)

        st.subheader("Try it on a comment")
        text = st.text_input("Type a comment", placeholder="this is literally me")
        if text.strip():
            pipe = joblib.load(MODEL_DIR / "sentiment_model.joblib")
            p = float(pipe.predict_proba([text.lower()])[0][
                list(pipe.classes_).index("relatable")])
            verdict = "Relatable" if p >= 0.5 else "Neutral"
            st.markdown(f"**{verdict}** · p = {p:.2f}")
    with right:
        st.subheader("Share of comments that self-recognise, by topic")
        aw = (comments.groupby("topic")["relatable_tag"]
              .apply(lambda s: (s == "Relatable").mean() * 100)
              .reset_index(name="pct").sort_values("pct"))
        fig = highlight_bars(aw, "pct", "topic", "pct")
        fig.update_xaxes(ticksuffix="%")
        plot(clean(fig, 320), use_container_width=True)
        note("Same ranking as the virality chart, measured a completely "
             "different way: from language instead of numbers. That "
             "agreement is the project's central finding.")
        st.subheader("Words that carry \"feeling seen\"")
        st.markdown(", ".join(f"**{t}**" for t in m["top_relatable_triggers"][:10]))

# =============================================================== A/B ======
elif page == "A/B tests":
    page_header("A/B tests")
    note("Six planned comparisons on the 66 posts. Two-sample tests use "
         "Welch's t with Mann-Whitney as a check; caption style uses ANOVA; "
         "the topic-language link uses χ². α = 0.05.")

    ab = D["ab"].copy()
    ab["label"] = (ab["experiment"].str.replace(r"^E\d_", "", regex=True)
                   .str.replace("_", " "))
    ab["p"] = ab["p_value"].map(lambda v: f"{v:.4f}" if v >= 1e-3 else f"{v:.1e}")
    ab["verdict"] = ab["significant"].map(lambda b: "significant" if b else "ns")
    show = ab[["label", "comparison", "metric", "p", "verdict"]]
    show = show.rename(columns={"label": "experiment", "p": "p-value"})
    st.dataframe(show, use_container_width=True, hide_index=True, height=230)

    e = D["ab_detail"]
    d = posts[posts["format"] == "Reel"].copy()
    d["dur"] = pd.to_numeric(d["duration_s"], errors="coerce")
    ret_short = d.loc[d["dur"] <= 30, "retention_rate"].mean()
    ret_long = d.loc[d["dur"] > 30, "retention_rate"].mean()

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Format → reach")
        fig = px.box(posts, x="format", y="reach")
        fig.update_traces(marker_color=T["muted"], line_color=ACCENT,
                          fillcolor="rgba(26,115,232,.10)")
        plot(clean(fig, 300), use_container_width=True)
        note(f"p = {e['E1_Format_Reel_vs_Carousel']['p_welch']:.1e}. Reels for "
             f"distribution; carousels for saves. The objective picks the format.")
        st.subheader("Caption style → K")
        fig = px.box(posts, x="caption_style", y="viral_coefficient")
        fig.update_traces(marker_color=T["muted"], line_color=ACCENT,
                          fillcolor="rgba(26,115,232,.10)")
        plot(clean(fig, 300), use_container_width=True)
        note(f"p = {e['E5_CaptionStyle_ANOVA']['p_anova']:.3f}. Questions pull "
             f"shares, listicles pull saves, stories pull comments.")
    with right:
        st.subheader("Time slot → reach")
        slot = posts.copy()
        slot["slot"] = np.where(slot["posting_hour"] >= 18, "evening",
                         np.where(slot["posting_hour"] >= 12, "midday", "morning"))
        fig = px.box(slot, x="slot", y="reach",
                     category_orders={"slot": ["morning", "midday", "evening"]})
        fig.update_traces(marker_color=T["muted"], line_color=ACCENT,
                          fillcolor="rgba(26,115,232,.10)")
        plot(clean(fig, 300), use_container_width=True)
        note(f"p = {e['E4_Time_Evening_vs_Day']['p_welch']:.3f} for the evening "
             f"reach lift.")
        st.subheader("Reel length → retention")
        fig = px.scatter(d, x="dur", y="retention_rate",
                         hover_data=["post_id", "topic"])
        fig.update_traces(marker=dict(color=ACCENT, size=9, opacity=.7))
        coef = np.polyfit(d["dur"].dropna(), d["retention_rate"].dropna(), 1)
        xs = np.linspace(d["dur"].min(), d["dur"].max(), 40)
        fig.add_scatter(x=xs, y=np.polyval(coef, xs), mode="lines",
                        line=dict(color=T["muted"], width=2, dash="dash"))
        plot(clean(fig, 300), use_container_width=True)
        note(f"Retention averages {ret_short:.0%} under 30s and falls to "
             f"{ret_long:.0%} beyond it "
             f"(p = {e['E3_ReelLength_Short_vs_Long']['p_welch']:.1e}).")

# ========================================================= RECOMMENDER ====
elif page == "Next week's plan":
    page_header("Next week's plan")
    note("Expected performance for every topic × format × length × slot cell, "
         "estimated from the 90-day record with shrinkage toward the mean "
         "where evidence is thin. Ranked 70/30 on K and engagement rate.")

    plays = D["recs"].sort_values("success_index", ascending=False).head(10)
    plays["play"] = plays["topic"] + " · " + plays["format"] + " · " + \
        plays["time_slot"]
    fig = px.bar(plays, x="success_index", y="play", orientation="h")
    fig.update_traces(marker_color=[ACCENT if i == 0 else T["muted"]
                                    for i in range(len(plays))])
    plot(clean(fig, 330), use_container_width=True)

    st.dataframe(
        plays[["topic", "format", "duration_bucket", "time_slot",
               "expected_k", "expected_engagement_rate", "success_index"]]
        .rename(columns={"expected_k": "exp. K",
                         "expected_engagement_rate": "exp. eng. rate"}),
        use_container_width=True, hide_index=True, height=340)

    g = D["rec_payload"]["guidance"]
    st.subheader("Rules the numbers support")
    hooks = "; ".join(f"{k} → {v.lower()}" for k, v in g["best_hook_by_format"].items())
    st.markdown(
        f"- **Hooks.** {hooks}.\n"
        f"- **Captions.** {g['best_caption_style']} captions carried the highest K overall.\n"
        f"- **Timing.** Evening slots buy reach; {g['best_time_slot']} slots were "
        f"slightly more K-efficient. Volume vs quality, pick per goal.\n"
        f"- **Topics.** " + ", ".join(
            f"{k} ({v})" for k, v in sorted(g["median_k_by_topic"].items(),
                                            key=lambda kv: -kv[1])[:3]) +
        " lead the median-K table.")

# ================================================================ TRENDS ==
elif page == "Trend radar":
    page_header("Trend radar")
    note("Search-interest curves for ten candidate \"struggle\" keywords, "
         "projected 14 days with damped-Holt smoothing. The flag is for "
         "climbers that haven't saturated yet, worth an episode before "
         "everyone covers them.")

    trends = D["trends"]
    rising = trends.loc[trends["status"] == "RISING", "keyword"].tolist()
    st.dataframe(trends[["keyword", "status", "current_level",
                         "recent_slope_per_day", "forecast_14d_end"]],
                 use_container_width=True, hide_index=True, height=330)
    if rising:
        note(f"Climbing now: {', '.join(rising)}.")

    raw = D["trend_raw"]
    default = (rising + [k for k in ("situationship", "bed rotting")
                         if k not in rising])[:4]
    show = st.multiselect("Keywords shown", sorted(raw["keyword"].unique()),
                          default=default)
    fig = go.Figure()
    for kw in show:
        g = raw[raw["keyword"] == kw].sort_values("day")
        col = ACCENT if kw in rising else T["muted"]
        fig.add_trace(go.Scatter(x=g["day"], y=g["interest"], name=kw,
                                 line=dict(color=col, width=2)))
        end = trends.set_index("keyword").loc[kw, "forecast_14d_end"]
        fig.add_trace(go.Scatter(x=[180, 194], y=[g["interest"].iloc[-14:].mean(), end],
                                 name=f"{kw} fcst", line=dict(color=col, width=2,
                                                              dash="dot"),
                                 opacity=.8))
    fig.add_vline(x=180, line_width=1, line_color=T["line"], line_dash="dash")
    plot(clean(fig, 400, legend=True), use_container_width=True)
    note("Solid lines: observed interest. Dotted: 14-day forecast.")

# ========================================================= DATA EXPLORER ==
else:
    page_header("Data")
    t1, t2, t3 = st.tabs(["Posts", "Comments", "Followers"])
    with t1:
        topic_f = st.multiselect("Topic", sorted(posts["topic"].unique()),
                                 default=list(posts["topic"].unique()))
        fmt_f = st.multiselect("Format", sorted(posts["format"].unique()),
                               default=list(posts["format"].unique()))
        view = posts[posts["topic"].isin(topic_f) & posts["format"].isin(fmt_f)]
        st.dataframe(view.drop(columns={"title", "handle"}),
                     use_container_width=True, hide_index=True, height=420)
        st.download_button("posts_scored.csv", view.to_csv(index=False),
                           "posts_scored.csv", "text/csv")
    with t2:
        tag_f = st.radio("Tag", ["all", "Relatable", "Neutral"],
                         horizontal=True)
        view = comments if tag_f == "all" else comments[comments["relatable_tag"] == tag_f]
        st.dataframe(view[["comment_id", "post_id", "topic", "comment_text",
                           "relatable_tag", "relatable_probability"]],
                     use_container_width=True, hide_index=True, height=420)
        st.download_button("comments_tagged.csv", view.to_csv(index=False),
                           "comments_tagged.csv", "text/csv")
    with t3:
        st.dataframe(followers, use_container_width=True, hide_index=True,
                     height=420)
        st.download_button("followers.csv", followers.to_csv(index=False),
                           "followers.csv", "text/csv")
