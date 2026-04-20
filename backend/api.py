"""
api.py
------
Flask REST API for AI4Sustain.

Endpoints:
  GET  /api/health                         — liveness check
  GET  /api/articles?theme=&region=&limit= — list articles
  GET  /api/trends?theme=&region=          — time-series counts + avg sentiment
  POST /api/query   body: {theme, region, time_window, nl_query}
                          → RAG summary + top articles
  GET  /api/stats                          — aggregate dashboard numbers

Run:
    python backend/api.py
"""

import json, math, os, sqlite3, time
from datetime import datetime, timedelta
from functools import lru_cache

from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI

# ── config ────────────────────────────────────────────────────────────────────
DB_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "articles.db")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

EMBED_MODEL  = "text-embedding-3-small"
SUMMARY_MODEL = "gpt-3.5-turbo"       # cheap; gpt-4o-mini also fine

# ── init ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
client = OpenAI(api_key=OPENAI_KEY)


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── helpers ───────────────────────────────────────────────────────────────────

def cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


def time_window_cutoff(tw: str) -> str:
    """Return ISO datetime string for the start of the window."""
    days = {"7d": 7, "30d": 30, "90d": 90, "1y": 365}.get(tw, 30)
    return (datetime.utcnow() - timedelta(days=days)).isoformat()


def sentiment_label(score: float) -> str:
    if score > 0.2:  return "Positive"
    if score < -0.2: return "Negative"
    return "Neutral"


def row_to_dict(r) -> dict:
    d = dict(r)
    d["sentiment_label"] = sentiment_label(d.get("sentiment", 0))
    d.pop("embedding", None)   # never send raw vector to frontend
    return d


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    embedded = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    return jsonify({"status": "ok", "articles": count, "embedded": embedded})


@app.get("/api/articles")
def list_articles():
    theme  = request.args.get("theme")
    region = request.args.get("region")
    limit  = int(request.args.get("limit", 20))
    tw     = request.args.get("time_window", "30d")
    cutoff = time_window_cutoff(tw)

    q  = "SELECT id,title,abstract,url,source,theme,region,published,sentiment FROM articles WHERE published >= ?"
    params: list = [cutoff]

    if theme  and theme  != "all": q += " AND theme  = ?"; params.append(theme)
    if region and region != "global": q += " AND region = ?"; params.append(region)

    q += " ORDER BY published DESC LIMIT ?"
    params.append(limit)

    conn = get_conn()
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return jsonify([row_to_dict(r) for r in rows])


@app.get("/api/trends")
def trends():
    """Return weekly article counts + avg sentiment for the last 6 weeks."""
    theme  = request.args.get("theme")
    region = request.args.get("region")

    conn = get_conn()
    cutoff = (datetime.utcnow() - timedelta(weeks=6)).isoformat()

    q = "SELECT published, sentiment FROM articles WHERE published >= ?"
    params: list = [cutoff]
    if theme  and theme  != "all": q += " AND theme  = ?"; params.append(theme)
    if region and region != "global": q += " AND region = ?"; params.append(region)

    rows = conn.execute(q, params).fetchall()
    conn.close()

    # bucket into 6 weekly slots
    buckets: dict[str, dict] = {}
    now = datetime.utcnow()
    for i in range(5, -1, -1):
        wstart = now - timedelta(weeks=i+1)
        label  = wstart.strftime("%-d %b")
        buckets[label] = {"count": 0, "sentiment_sum": 0.0}

    for r in rows:
        try:
            pub = datetime.fromisoformat(r["published"])
        except Exception:
            continue
        for i in range(5, -1, -1):
            wstart = now - timedelta(weeks=i+1)
            wend   = now - timedelta(weeks=i)
            if wstart <= pub < wend:
                label = wstart.strftime("%-d %b")
                buckets[label]["count"] += 1
                buckets[label]["sentiment_sum"] += r["sentiment"]
                break

    labels, counts, sentiment_avgs = [], [], []
    for label, data in buckets.items():
        labels.append(label)
        counts.append(data["count"])
        avg = (data["sentiment_sum"] / data["count"]) if data["count"] else 0.0
        sentiment_avgs.append(round(avg, 3))

    return jsonify({"labels": labels, "counts": counts, "sentiment": sentiment_avgs})


@app.get("/api/stats")
def stats():
    conn = get_conn()
    total     = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    embedded  = conn.execute("SELECT COUNT(*) FROM articles WHERE embedding IS NOT NULL").fetchone()[0]
    by_theme  = dict(conn.execute(
        "SELECT theme, COUNT(*) FROM articles GROUP BY theme"
    ).fetchall())
    avg_sent  = conn.execute("SELECT AVG(sentiment) FROM articles").fetchone()[0] or 0
    conn.close()
    return jsonify({
        "total": total,
        "embedded": embedded,
        "by_theme": by_theme,
        "avg_sentiment": round(avg_sent, 3),
    })


# ─────────────────────────────────────────────────────────────────────────────
# RAG QUERY
# ─────────────────────────────────────────────────────────────────────────────

RAG_SYSTEM = """You are AI4Sustain, an environmental insight assistant.
You receive a set of news article abstracts and synthesise a concise, factual
trend summary (3-4 sentences). Always ground your answer in the provided articles.
End with one concrete, actionable implication for policymakers or researchers.
Never fabricate statistics not present in the abstracts."""

RAG_USER = """Theme: {theme} | Region: {region} | Window: {time_window}
{nl_query_line}

Top relevant articles (title + abstract):
{context}

Write a trend summary."""


@app.post("/api/query")
def rag_query():
    body       = request.get_json(force=True)
    theme      = body.get("theme", "renewable")
    region     = body.get("region", "global")
    time_window = body.get("time_window", "30d")
    nl_query   = body.get("nl_query", "").strip()

    cutoff = time_window_cutoff(time_window)
    conn   = get_conn()

    # ── candidate articles ────────────────────────────────────────────────────
    q = "SELECT id,title,abstract,url,source,theme,region,published,sentiment,embedding FROM articles WHERE published >= ?"
    params: list = [cutoff]
    if theme  and theme  != "all": q += " AND theme  = ?"; params.append(theme)
    if region and region != "global": q += " AND region = ?"; params.append(region)
    q += " ORDER BY published DESC LIMIT 50"

    candidates = conn.execute(q, params).fetchall()
    conn.close()

    if not candidates:
        return jsonify({"summary": "No articles found for the selected filters.", "articles": []})

    # ── hybrid retrieval ──────────────────────────────────────────────────────
    # If nl_query provided and embeddings exist → dense retrieval
    if nl_query and any(r["embedding"] for r in candidates):
        try:
            q_resp = client.embeddings.create(model=EMBED_MODEL, input=[nl_query])
            q_vec  = q_resp.data[0].embedding
            scored = []
            for r in candidates:
                if r["embedding"]:
                    vec   = json.loads(r["embedding"])
                    score = cosine(q_vec, vec)
                    scored.append((score, r))
                else:
                    scored.append((0.0, r))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = [r for _, r in scored[:8]]
        except Exception:
            top = list(candidates[:8])
    else:
        top = list(candidates[:8])

    # ── build context for LLM ─────────────────────────────────────────────────
    context_parts = []
    for i, r in enumerate(top, 1):
        context_parts.append(
            f"{i}. [{r['source']}] {r['title']}\n   {r['abstract'][:400]}"
        )
    context = "\n\n".join(context_parts)

    nl_line = f"User query: {nl_query}" if nl_query else ""

    theme_labels = {
        "renewable": "Renewable Energy", "emissions": "Emissions & Carbon",
        "biodiversity": "Biodiversity", "water": "Water & Oceans", "policy": "Climate Policy"
    }

    # ── RAG summarisation ─────────────────────────────────────────────────────
    try:
        resp = client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": RAG_SYSTEM},
                {"role": "user",   "content": RAG_USER.format(
                    theme=theme_labels.get(theme, theme),
                    region=region,
                    time_window=time_window,
                    nl_query_line=nl_line,
                    context=context
                )}
            ],
            temperature=0.4,
            max_tokens=220,
        )
        summary = resp.choices[0].message.content.strip()
    except Exception as e:
        summary = f"[Summary unavailable: {e}]"

    return jsonify({
        "summary":  summary,
        "articles": [row_to_dict(r) for r in top],
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
