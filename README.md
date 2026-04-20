# 🌿 AI4Sustain
**Environmental Insight & Trend Analysis Agent**  
*LLM Spring 2026 · TeamX*

> Turning unstructured sustainability data into actionable intelligence — powered by real GDELT articles, OpenAI embeddings, and RAG summarisation.

---

## Architecture

```
GDELT Doc API ──► fetch_articles.py ──► data/articles.db (SQLite)
                                              │
                                    pipeline.py (OpenAI)
                                    ├── text-embedding-3-small  (embeddings)
                                    └── gpt-3.5-turbo           (sentiment)
                                              │
                                         api.py  (Flask)
                                    ├── GET  /api/health
                                    ├── GET  /api/articles
                                    ├── GET  /api/trends
                                    ├── GET  /api/stats
                                    └── POST /api/query  ← RAG endpoint
                                              │
                                   frontend/index.html  (vanilla JS)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI key (optional — already embedded in scripts)
```bash
export OPENAI_API_KEY="sk-proj-..."
```

### 3. Fetch 100 articles from GDELT
```bash
python backend/fetch_articles.py
```
This populates `data/articles.db` with ~100 real articles across 5 themes.  
No API key needed for GDELT — it's free and public.

### 4. Run the ML pipeline (embeddings + sentiment)
```bash
python backend/pipeline.py
```
**Estimated cost: < $0.05** for 100 articles using abstracts only (~100 tokens each).

| Step       | Model                    | Cost/1M tokens | Est. cost (100 articles) |
|------------|--------------------------|----------------|--------------------------|
| Embedding  | text-embedding-3-small   | $0.020         | ~$0.0002                 |
| Sentiment  | gpt-3.5-turbo            | $0.500 input   | ~$0.010                  |
| RAG query  | gpt-3.5-turbo            | $0.500 input   | ~$0.005 per query        |

### 5. Start the API server
```bash
python backend/api.py
```
Server starts at `http://localhost:5000`

### 6. Open the frontend
```bash
open frontend/index.html
# or just drag it into your browser
```

---

## Project Structure

```
ai4sustain/
├── README.md
├── requirements.txt
├── data/
│   └── articles.db          # auto-created by fetch_articles.py
├── backend/
│   ├── fetch_articles.py    # GDELT ingestion → SQLite
│   ├── pipeline.py          # OpenAI embeddings + sentiment
│   └── api.py               # Flask REST API + RAG
└── frontend/
    └── index.html           # Single-file frontend
```

---

## API Reference

### `GET /api/health`
Returns DB article count and embedding status.

### `GET /api/articles?theme=renewable&region=global&time_window=30d&limit=20`
Returns paginated article list with sentiment labels.

**Theme values:** `renewable` · `emissions` · `biodiversity` · `water` · `policy`  
**Region values:** `global` · `Europe` · `Asia-Pacific` · `Americas` · `Africa`  
**Time window:** `7d` · `30d` · `90d` · `1y`

### `GET /api/trends?theme=renewable&region=global`
Returns weekly article counts and average sentiment for the last 6 weeks.

### `GET /api/stats`
Returns aggregate counts by theme, total articles, and mean sentiment.

### `POST /api/query`
RAG endpoint. Body:
```json
{
  "theme": "renewable",
  "region": "global",
  "time_window": "30d",
  "nl_query": "solar investment trends"
}
```
Response:
```json
{
  "summary": "GPT-generated trend summary grounded in retrieved articles...",
  "articles": [{ "title": "...", "abstract": "...", "source": "...", ... }]
}
```

---

## RAG Pipeline Detail

1. **Candidate retrieval** — filter DB by theme + region + time window (up to 50 articles)
2. **Dense retrieval** — if `nl_query` provided and embeddings exist, embed the query with `text-embedding-3-small` and rank by cosine similarity
3. **Top-k** — take top 8 articles
4. **Summarisation** — send titles + abstracts (≤400 chars each) to GPT-3.5-turbo with a grounding-focused system prompt
5. **Response** — summary + ranked article list with sentiment labels

---

## Budget Management (for $5 OpenAI plan)

- **Abstract-only**: Each article is capped at 600 chars (~100 tokens). Full articles are never sent.
- **Batching**: Embeddings in batches of 20; sentiment in batches of 5 with 1.5s sleep.
- **Rate limiting**: Conservative sleeps between API calls avoid hitting tier-1 limits.
- **Per-query cost**: ~$0.005 (8 abstracts × ~150 tokens + 220 output tokens)
- **Full 100-article pipeline**: < $0.05 total

---

## Research Questions

| # | Question | Target |
|---|----------|--------|
| RQ1 | Fine-tuned DeBERTa-v3 vs zero-shot classification on GDELT data | F1 ≥ 0.80 |
| RQ2 | RAG summaries: coherence and grounding quality | G-Eval ≥ 4.0/5.0 |
| RQ3 | Hybrid (dense + BM25) vs dense-only retrieval | MRR & P@5 improvement |

---

## Team

| Name | Role |
|------|------|
| Aditi Phadke | RAG Systems |
| Sudhiksha Mhatre | NLP Classification |
| Duhita Pradhan | Data Ingestion |
| Jahanvi Joshi | Vector Retrieval |
| Pranati Sukh | Evaluation & UI |

---

## Deployment on GitHub Pages

The `frontend/index.html` is fully self-contained (no build step).  
To publish:
1. Push the repo to GitHub
2. Go to **Settings → Pages → Source: main branch / `frontend/` folder**
3. The frontend will be live at `https://<user>.github.io/<repo>/`

> Note: For GitHub Pages deployment, the Flask API must be hosted separately (e.g. Railway, Render, or a university server). Update the `const API = "..."` line in `index.html` to point to your deployed API URL.
