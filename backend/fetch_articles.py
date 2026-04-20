"""
fetch_articles.py
-----------------
Fetches ~100 sustainability articles from the GDELT 2.0 Doc API,
extracts title + snippet (used as abstract), and stores them in
a local SQLite database (data/articles.db).

Run once to populate the database:
    python backend/fetch_articles.py
"""

import sqlite3, requests, time, json, os, hashlib, re
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "articles.db")

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

THEME_QUERIES = {
    "renewable":     "solar wind renewable energy",
    "emissions":     "carbon emissions CO2 climate",
    "biodiversity":  "biodiversity deforestation species",
    "water":         "water scarcity ocean sea level",
    "policy":        "climate policy COP Paris Agreement",
}

REGION_MAP = {
    "Europe":       ["BBC", "Guardian", "DW", "euronews"],
    "Asia-Pacific": ["scmp", "straitstimes", "indiatimes"],
    "Americas":     ["nytimes", "washingtonpost", "latimes"],
    "Africa":       ["allafrica", "ewn", "dailynation"],
    "Global":       [],  # no filter
}


def init_db(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            abstract    TEXT NOT NULL,
            url         TEXT,
            source      TEXT,
            theme       TEXT,
            region      TEXT,
            published   TEXT,
            sentiment   REAL DEFAULT 0.0,
            embedding   TEXT,          -- JSON list stored as text
            fetched_at  TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_theme  ON articles(theme)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_region ON articles(region)")
    conn.commit()


def _make_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16]


def _infer_region(domain: str) -> str:
    domain = domain.lower()
    for region, keywords in REGION_MAP.items():
        if any(k in domain for k in keywords):
            return region
    return "Global"


def fetch_gdelt(query: str, max_records: int = 25) -> list[dict]:
    """Call GDELT Doc API and return a list of article dicts."""
    params = {
        "query":      query + " sourcelang:english",
        "mode":       "artlist",
        "maxrecords": max_records,
        "format":     "json",
        "timespan":   "7d",          # last 7 days
        "sort":       "datedesc",
    }
    try:
        resp = requests.get(GDELT_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return data.get("articles", [])
    except Exception as e:
        print(f"  ⚠ GDELT fetch error for '{query}': {e}")
        return []


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:600]  # cap abstract length to keep token costs low


def run_fetch(target: int = 100):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    total_new = 0
    per_theme = max(1, target // len(THEME_QUERIES))  # ~20 per theme

    for theme, query in THEME_QUERIES.items():
        print(f"\n🔍 Fetching theme: {theme} (query: '{query}')")
        articles = fetch_gdelt(query, max_records=per_theme + 5)  # +5 buffer for duplicates

        saved = 0
        for art in articles:
            url   = art.get("url", "")
            title = clean_text(art.get("title", ""))
            abstract = clean_text(art.get("seendescription") or art.get("socialimage", "") or title)

            if not title or not url:
                continue

            art_id  = _make_id(url)
            domain  = url.split("/")[2] if "/" in url else url
            source  = art.get("domain", domain)
            region  = _infer_region(source)
            pub_dt  = art.get("seendate", datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
            # normalise to ISO
            try:
                dt = datetime.strptime(pub_dt, "%Y%m%dT%H%M%SZ")
                published = dt.isoformat()
            except Exception:
                published = datetime.utcnow().isoformat()

            try:
                conn.execute("""
                    INSERT OR IGNORE INTO articles
                      (id, title, abstract, url, source, theme, region, published, fetched_at)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (art_id, title, abstract, url, source, theme, region,
                      published, datetime.utcnow().isoformat()))
                if conn.execute("SELECT changes()").fetchone()[0]:
                    saved += 1
                    total_new += 1
            except sqlite3.Error as e:
                print(f"  DB error: {e}")

            if saved >= per_theme:
                break

        conn.commit()
        print(f"  ✅ Saved {saved} new articles for '{theme}'")
        time.sleep(1.2)  # be polite to GDELT

    total_in_db = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    conn.close()
    print(f"\n🎉 Done. Added {total_new} new articles. Total in DB: {total_in_db}")


if __name__ == "__main__":
    run_fetch(target=100)
