"""
pipeline.py
-----------
Processes articles already in the DB:
  1. Sentiment scoring  — GPT-3.5-turbo (cheap, batched)
  2. Embedding          — text-embedding-3-small (cheapest OpenAI embedder)

Only processes articles that haven't been embedded yet.

Run after fetch_articles.py:
    python backend/pipeline.py

Cost estimate for 100 articles (abstracts ≤ 600 chars ≈ ~100 tokens each):
  Embedding  : 100 × 100 tokens @ $0.02/1M  ≈ $0.0002
  Sentiment  : 100 × ~200 tokens @ $0.50/1M ≈ $0.01
  Total      : well under $0.05
"""

import sqlite3, json, time, os, sys
from openai import OpenAI

DB_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "articles.db")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

EMBED_MODEL     = "text-embedding-3-small"   # $0.02 / 1M tokens
SENTIMENT_MODEL = "gpt-3.5-turbo"            # $0.50 / 1M input tokens

# ── batch sizes (keep well under rate limits on $5 plan) ──────────────────
EMBED_BATCH     = 20   # embed 20 at a time
SENTIMENT_BATCH =  5   # classify 5 at a time (one chat call per batch)
SLEEP_BETWEEN   =  1.5 # seconds between API calls


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# 1. EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

def embed_articles(client: OpenAI):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, title, abstract FROM articles WHERE embedding IS NULL"
    ).fetchall()

    if not rows:
        print("✅ All articles already embedded.")
        conn.close()
        return

    print(f"\n📐 Embedding {len(rows)} articles in batches of {EMBED_BATCH}…")
    updated = 0

    for i in range(0, len(rows), EMBED_BATCH):
        batch = rows[i : i + EMBED_BATCH]
        texts = [f"{r['title']}. {r['abstract']}" for r in batch]

        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            for j, r in enumerate(batch):
                vec = resp.data[j].embedding          # list[float] len=1536
                conn.execute(
                    "UPDATE articles SET embedding = ? WHERE id = ?",
                    (json.dumps(vec), r["id"])
                )
            conn.commit()
            updated += len(batch)
            print(f"  Embedded {updated}/{len(rows)}", end="\r")
            time.sleep(SLEEP_BETWEEN)
        except Exception as e:
            print(f"\n  ⚠ Embedding error on batch {i}: {e}")
            time.sleep(5)

    conn.close()
    print(f"\n✅ Embedded {updated} articles.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SENTIMENT
# ─────────────────────────────────────────────────────────────────────────────

SENTIMENT_PROMPT = """You are a sustainability news sentiment classifier.
For each article abstract below, output a JSON array of numbers.
Each number is a sentiment score from -1.0 (very negative) to +1.0 (very positive).
Return ONLY the JSON array, no explanation.

Abstracts:
{abstracts}
"""


def score_sentiment(client: OpenAI):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, title, abstract FROM articles WHERE sentiment = 0.0"
    ).fetchall()

    if not rows:
        print("✅ Sentiment already scored for all articles.")
        conn.close()
        return

    print(f"\n🎭 Scoring sentiment for {len(rows)} articles in batches of {SENTIMENT_BATCH}…")
    updated = 0

    for i in range(0, len(rows), SENTIMENT_BATCH):
        batch = rows[i : i + SENTIMENT_BATCH]
        abstracts_text = "\n".join(
            f"{j+1}. {r['title']}: {r['abstract'][:300]}"
            for j, r in enumerate(batch)
        )

        try:
            resp = client.chat.completions.create(
                model=SENTIMENT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise JSON-only classifier."},
                    {"role": "user",   "content": SENTIMENT_PROMPT.format(abstracts=abstracts_text)}
                ],
                temperature=0,
                max_tokens=80,
            )
            raw = resp.choices[0].message.content.strip()
            scores = json.loads(raw)

            for j, r in enumerate(batch):
                score = float(scores[j]) if j < len(scores) else 0.0
                score = max(-1.0, min(1.0, score))          # clamp
                conn.execute(
                    "UPDATE articles SET sentiment = ? WHERE id = ?",
                    (score, r["id"])
                )
            conn.commit()
            updated += len(batch)
            print(f"  Scored {updated}/{len(rows)}", end="\r")
            time.sleep(SLEEP_BETWEEN)

        except json.JSONDecodeError:
            print(f"\n  ⚠ JSON parse error on batch {i}, skipping")
        except Exception as e:
            print(f"\n  ⚠ Sentiment error on batch {i}: {e}")
            time.sleep(5)

    conn.close()
    print(f"\n✅ Sentiment scored for {updated} articles.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not OPENAI_KEY.startswith("sk-"):
        sys.exit("❌  Set OPENAI_API_KEY env var or update OPENAI_KEY in pipeline.py")

    client = OpenAI(api_key=OPENAI_KEY)

    embed_articles(client)
    score_sentiment(client)
    print("\n🎉 Pipeline complete.")
