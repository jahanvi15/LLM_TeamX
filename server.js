require('dotenv').config();
const express = require('express');
const fetch = require('node-fetch');
const OpenAI = require('openai');
const cors = require('cors');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(__dirname));

// Broad single-word/short keywords — GDELT full-text search is strict;
// multi-word phrases often return 0 results.
const THEME_KEYWORDS = {
  renewable:    'energy',
  emissions:    'carbon',
  biodiversity: 'environment',
  water:        'water',
  policy:       'climate'
};

const REGION_TERMS = {
  global:   '',
  europe:   ' Europe',
  asia:     ' Asia',
  americas: ' Americas',
  africa:   ' Africa'
};

// Always fetch the last 90 days from GDELT regardless of user selection;
// chart bucketing handles the user's chosen window client-side.
const GDELT_FETCH_TIMESPAN = '90d';
const WINDOW_DAYS = { '7d':7, '30d':30, '90d':90, '1y':365 };

const FALLBACK_ARTICLES = [
  { title: 'Global renewable energy capacity surpasses 3,500 GW milestone', source: 'Reuters', url: '#', date: null, snippet: 'Solar and wind power additions broke records for the third consecutive year as countries accelerate clean energy transitions.', from: 'fallback' },
  { title: 'UN climate report warns of accelerating ice sheet loss', source: 'BBC', url: '#', date: null, snippet: 'New satellite data shows Greenland and Antarctic ice sheets losing mass at rates that exceed earlier IPCC projections.', from: 'fallback' },
  { title: 'Carbon markets hit record trading volumes amid new regulations', source: 'Financial Times', url: '#', date: null, snippet: 'EU Emissions Trading System prices rose sharply as stricter caps took effect, driving compliance buying across heavy industries.', from: 'fallback' },
  { title: 'Deforestation in the Amazon fell 50% in 2024, Brazil reports', source: 'Guardian', url: '#', date: null, snippet: 'Enforcement operations and satellite monitoring contributed to a significant drop in illegal forest clearing activity.', from: 'fallback' },
  { title: 'COP30 host Brazil unveils ambitious national climate pledge', source: 'AP News', url: '#', date: null, snippet: 'Brazil committed to ending illegal deforestation and cutting methane emissions 30% by 2030 ahead of the November conference.', from: 'fallback' },
  { title: 'Ocean temperatures reach new highs, threatening coral reefs globally', source: 'Nature', url: '#', date: null, snippet: 'Marine heatwaves now affect more than 40% of the world\'s oceans, putting reef ecosystems under sustained thermal stress.', from: 'fallback' },
];

app.post('/api/analyze', async (req, res) => {
  const { theme = 'renewable', region = 'global', timeWindow = '30d' } = req.body;

  const keyword   = THEME_KEYWORDS[theme] || 'climate';
  const regionStr = REGION_TERMS[region] || '';
  const query     = (keyword + regionStr).trim();
  const days      = WINDOW_DAYS[timeWindow] || 30;

  let gdeltArticles = [];
  let newsArticles  = [];

  // ── GDELT (free, no key) ──────────────────────────────────────────────
  // Always fetch 90 days so short windows still get results; chart buckets filter to user window.
  gdeltArticles = await fetchGdelt(query, GDELT_FETCH_TIMESPAN);

  // First attempt returned nothing — retry with the bare fallback keyword "climate change"
  if (gdeltArticles.length === 0) {
    console.log('[GDELT] 0 results for primary query, retrying with "climate change"');
    gdeltArticles = await fetchGdelt('climate change', GDELT_FETCH_TIMESPAN);
  }

  // ── NewsAPI ───────────────────────────────────────────────────────────
  if (process.env.NEWS_API_KEY) {
    try {
      const fromDate = new Date(Date.now() - days * 86400000).toISOString().split('T')[0];
      const newsUrl  =
        `https://newsapi.org/v2/everything` +
        `?q=${encodeURIComponent(query)}` +
        `&pageSize=10&language=en&sortBy=publishedAt` +
        `&from=${fromDate}&apiKey=${process.env.NEWS_API_KEY}`;
      const newsRes  = await fetch(newsUrl, { timeout: 10000 });
      const newsData = await newsRes.json();
      newsArticles = (newsData.articles || [])
        .filter(a => a.title && a.title !== '[Removed]')
        .map(a => ({
          title:   a.title,
          source:  a.source?.name || 'NewsAPI',
          url:     a.url  || '#',
          date:    a.publishedAt ? a.publishedAt.substring(0, 10).replace(/-/g, '') : null,
          snippet: a.description || '',
          from:    'newsapi'
        }));
    } catch (e) {
      console.warn('NewsAPI fetch failed:', e.message);
    }
  }

  let allArticles = [...gdeltArticles, ...newsArticles];

  // ── Hardcoded fallback so the UI never shows empty ────────────────────
  if (allArticles.length === 0) {
    console.log('[FALLBACK] All sources empty — using hardcoded stubs');
    allArticles = FALLBACK_ARTICLES.map(a => ({ ...a, sentiment: keywordSentiment(a.title) }));
  }

  // ── Chart data (article counts bucketed by time) ──────────────────────
  const chartData = buildChartData(allArticles, timeWindow);

  // ── GPT-4o-mini RAG summary ───────────────────────────────────────────
  let summary        = '';
  let sentimentLine  = chartData.labels.map(() => 3.0);

  if (process.env.OPENAI_API_KEY && allArticles.length > 0) {
    try {
      const articleTexts = allArticles.slice(0, 25)
        .map((a, i) => {
          const snip = a.snippet ? ' — ' + a.snippet.substring(0, 120) : '';
          return `${i + 1}. "${a.title}"${snip} [${a.source}]`;
        })
        .join('\n');

      const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
      const completion = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [{
          role: 'user',
          content:
            `You are an environmental news analyst. Analyze these recent articles ` +
            `about "${keyword}"${regionStr ? ' in ' + regionStr.trim() : ''}.\n\n` +
            `Articles:\n${articleTexts}\n\n` +
            `Return ONLY valid JSON (no markdown fences):\n` +
            `{\n` +
            `  "summary": "3-4 sentence RAG-style trend summary grounded in the articles",\n` +
            `  "sentiment": [array of exactly ${chartData.labels.length} floats 1.0–5.0 representing sentiment trend]\n` +
            `}`
        }],
        max_tokens: 450,
        temperature: 0.3
      });

      const raw = completion.choices[0].message.content.trim();
      const parsed = JSON.parse(raw);
      summary = parsed.summary || '';
      if (Array.isArray(parsed.sentiment) && parsed.sentiment.length > 0) {
        sentimentLine = padOrTrim(parsed.sentiment, chartData.labels.length);
      }
    } catch (e) {
      console.warn('OpenAI call failed:', e.message);
    }
  }

  // ── Per-article keyword sentiment ─────────────────────────────────────
  const articlesOut = allArticles.map(a => ({
    ...a,
    sentiment: keywordSentiment(a.title + ' ' + a.snippet)
  }));

  res.json({
    articles:  articlesOut,
    summary,
    chartData: { labels: chartData.labels, values: chartData.values, sentiment: sentimentLine }
  });
});

// ── Helpers ───────────────────────────────────────────────────────────────

// TLDs whose content is predominantly non-English
const BLOCKED_TLDS = new Set(['.cn','.ru','.de','.fr','.es','.it','.pt','.jp','.kr','.ar','.br','.pl','.nl','.tr','.ua','.ro','.hu','.cz','.sk','.bg','.hr','.rs']);

// Returns false if the domain ends in a blocked TLD
function isEnglishDomain(domain) {
  if (!domain) return true;
  const d = domain.toLowerCase();
  for (const tld of BLOCKED_TLDS) {
    if (d === tld.slice(1) || d.endsWith(tld)) return false;
  }
  return true;
}

// Returns false if the title contains Cyrillic, Arabic, CJK, Hangul, Devanagari, or Thai
function isLatinTitle(title) {
  return !/[\u0400-\u04FF\u0600-\u06FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF\u0900-\u097F\u0E00-\u0E7F]/.test(title);
}

async function fetchGdelt(query, timespan) {
  const url =
    `https://api.gdeltproject.org/api/v2/doc/doc` +
    `?query=${encodeURIComponent(query)}` +
    `&mode=artlist&maxrecords=20&format=json` +
    `&timespan=${timespan}&sourcelang=english`;
  console.log(`[GDELT] GET ${url}`);
  try {
    const res  = await fetch(url, { timeout: 12000 });
    const data = await res.json();
    const raw = (data.articles || []).map(a => ({
      title:   a.title  || '',
      source:  a.domain || 'GDELT',
      url:     a.url    || '#',
      date:    parseGdeltDate(a.seendate),
      snippet: '',
      from:    'gdelt'
    }));
    const articles = raw
      .filter(a => a.title)
      .filter(a => isEnglishDomain(a.source))
      .filter(a => isLatinTitle(a.title));
    const dropped = raw.length - articles.length;
    console.log(`[GDELT] ${articles.length} articles returned for query="${query}" (${dropped} non-English filtered)`);
    return articles;
  } catch (e) {
    console.warn(`[GDELT] fetch failed for query="${query}":`, e.message);
    return [];
  }
}

function parseGdeltDate(seendate) {
  // GDELT format: "20240115T120000Z" or "20240115120000"
  if (!seendate) return null;
  return seendate.replace(/[T\-:Z]/g, '').substring(0, 8);
}

function buildChartData(articles, timeWindow) {
  const days    = WINDOW_DAYS[timeWindow] || 30;
  const buckets = days <= 7 ? 7 : 6;
  const now     = Date.now();
  const start   = now - days * 86400000;
  const bucketMs = (days * 86400000) / buckets;

  const counts = new Array(buckets).fill(0);
  const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const DAYS   = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];

  const labels = Array.from({ length: buckets }, (_, i) => {
    const t = new Date(start + (i + 0.5) * bucketMs);
    if (days <= 7)  return DAYS[t.getDay()];
    if (days <= 90) return MONTHS[t.getMonth()] + ' ' + t.getDate();
    return MONTHS[t.getMonth()];
  });

  articles.forEach(a => {
    if (!a.date || a.date.length < 8) return;
    const y = parseInt(a.date.substring(0, 4));
    const m = parseInt(a.date.substring(4, 6)) - 1;
    const d = parseInt(a.date.substring(6, 8));
    if (isNaN(y) || isNaN(m) || isNaN(d)) return;
    const t   = new Date(y, m, d).getTime();
    const idx = Math.floor((t - start) / bucketMs);
    if (idx >= 0 && idx < buckets) counts[idx]++;
  });

  return { labels, values: counts };
}

function keywordSentiment(text) {
  const t = (text || '').toLowerCase();
  const pos = ['surge','record','boost','milestone','breakthrough','success',
               'achieve','improve','expand','gain','increase','rise','advance',
               'agreement','progress','renewable','clean','invest'];
  const neg = ['fail','crisis','threat','decline','loss','damage','disaster',
               'warn','shortage','conflict','drop','fall','bleach','flood',
               'drought','pollution','collapse','deforest'];
  let score = 0;
  pos.forEach(w => { if (t.includes(w)) score++; });
  neg.forEach(w => { if (t.includes(w)) score--; });
  if (score > 0) return 'Positive';
  if (score < 0) return 'Negative';
  return 'Neutral';
}

function padOrTrim(arr, length) {
  const out = arr.slice(0, length).map(Number);
  while (out.length < length) out.push(out[out.length - 1] ?? 3.0);
  return out;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Evaluation ────────────────────────────────────────────────────────────

const CATEGORIES = ['renewable', 'emissions', 'biodiversity', 'water', 'policy'];

const EVAL_KEYWORDS = {
  renewable:    ['solar','wind','renewable','clean energy','battery','turbine','geothermal','hydrogen','nuclear','photovoltaic','ev','electric vehicle'],
  emissions:    ['carbon','emission','co2','methane','greenhouse','fossil fuel','net zero','decarbonize','carbon capture','ghg','coal'],
  biodiversity: ['species','forest','wildlife','ecosystem','biodiversity','deforestation','extinction','habitat','coral','reef','mammal','bird','fish','plant'],
  water:        ['ocean','water','flood','drought','river','sea level','groundwater','rainfall','aquifer','monsoon','glacier','ice'],
  policy:       ['policy','agreement','cop','law','regulation','government','treaty','pledge','summit','legislation','fund','subsidy','mandate'],
};

function classifyKeyword(title, snippet) {
  const text = (title + ' ' + (snippet || '')).toLowerCase();
  let best = null, bestCount = 0;
  for (const cat of CATEGORIES) {
    const count = EVAL_KEYWORDS[cat].filter(kw => text.includes(kw)).length;
    if (count > bestCount) { bestCount = count; best = cat; }
  }
  return best || 'renewable'; // default to first category on all-zero tie
}

function computeMetrics(labels, preds) {
  const total   = labels.length;
  const correct = labels.filter((l, i) => l === preds[i]).length;
  const accuracy = total > 0 ? correct / total : 0;

  const perClass = {};
  for (const cat of CATEGORIES) {
    let tp = 0, fp = 0, fn = 0;
    for (let i = 0; i < total; i++) {
      const isGold = labels[i] === cat;
      const isPred = preds[i]  === cat;
      if (isGold && isPred)  tp++;
      else if (!isGold && isPred) fp++;
      else if (isGold && !isPred) fn++;
    }
    const support   = labels.filter(l => l === cat).length;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall    = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1        = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
    perClass[cat]   = { precision, recall, f1, support };
  }

  const macroF1 = CATEGORIES.reduce((s, c) => s + perClass[c].f1, 0) / CATEGORIES.length;
  return { accuracy, macroF1, perClass };
}

// POST /api/evaluate — runs full evaluation and returns JSON
app.post('/api/evaluate', async (req, res) => {
  if (!process.env.OPENAI_API_KEY) {
    return res.status(400).json({ error: 'OPENAI_API_KEY not set' });
  }

  let labeled;
  try {
    const raw = require('fs').readFileSync(path.join(__dirname, 'labeled_articles.json'), 'utf8');
    labeled = JSON.parse(raw);
  } catch (e) {
    return res.status(500).json({ error: `Cannot read labeled_articles.json: ${e.message}` });
  }

  const total = labeled.length;
  console.log(`[EVAL] Starting evaluation on ${total} articles`);

  const goldLabels   = labeled.map(a => a.label);
  const keywordPreds = labeled.map(a => classifyKeyword(a.title, a.snippet));

  // GPT zero-shot — batches of 10 with 1 s delay
  const openai    = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const gptPreds  = new Array(total);
  const BATCH     = 10;

  for (let start = 0; start < total; start += BATCH) {
    const batch = labeled.slice(start, start + BATCH);
    await Promise.all(batch.map(async (article, bi) => {
      const idx = start + bi;
      console.log(`[EVAL] Evaluating article ${idx + 1} of ${total}...`);
      try {
        const completion = await openai.chat.completions.create({
          model: 'gpt-4o-mini',
          messages: [{
            role: 'user',
            content:
              `Classify this sustainability news article into exactly one of these categories:\n` +
              `renewable, emissions, biodiversity, water, policy\n\n` +
              `Title: ${article.title}\n` +
              `Snippet: ${article.snippet || ''}\n\n` +
              `Reply with just the single category word in lowercase, nothing else.`
          }],
          max_tokens: 10,
          temperature: 0,
        });
        const pred = completion.choices[0].message.content.trim().toLowerCase();
        gptPreds[idx] = CATEGORIES.includes(pred) ? pred : 'renewable';
      } catch (e) {
        console.warn(`[EVAL] GPT failed for article ${idx + 1}:`, e.message);
        gptPreds[idx] = 'renewable';
      }
    }));

    if (start + BATCH < total) await sleep(1000);
  }

  console.log('[EVAL] Computing classification metrics...');
  const gptMetrics     = computeMetrics(goldLabels, gptPreds);
  const keywordMetrics = computeMetrics(goldLabels, keywordPreds);

  // ── G-Eval: score RAG summary quality on a 5-article sample ─────────
  let geval = null;
  try {
    const sample     = labeled.slice(0, 5);
    const sampleText = sample.map((a, i) =>
      `${i + 1}. "${a.title}" — ${(a.snippet || '').substring(0, 120)}`
    ).join('\n');

    console.log('[EVAL] Running G-Eval summary scoring...');
    const summaryRes = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content:
        `Summarize these sustainability news articles in 3-4 sentences:\n${sampleText}`
      }],
      max_tokens: 200,
      temperature: 0.3,
    });
    const generatedSummary = summaryRes.choices[0].message.content.trim();

    const scoreRes = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content:
        `You are an evaluation judge. Score the following summary against the source articles.\n\n` +
        `Source articles:\n${sampleText}\n\n` +
        `Summary:\n${generatedSummary}\n\n` +
        `Rate each dimension 1.0–5.0. Return ONLY valid JSON (no markdown):\n` +
        `{"relevance":X,"coherence":X,"grounding":X}`
      }],
      max_tokens: 40,
      temperature: 0,
    });
    geval = JSON.parse(scoreRes.choices[0].message.content.trim());
    console.log('[EVAL] G-Eval scores:', geval);
  } catch (e) {
    console.warn('[EVAL] G-Eval scoring failed:', e.message);
  }

  // Round all floats to 3 dp for readability
  const fmt = obj => {
    const out = {};
    for (const [k, v] of Object.entries(obj)) {
      out[k] = typeof v === 'number' ? Math.round(v * 1000) / 1000 : v;
    }
    return out;
  };
  const fmtMetrics = m => ({
    accuracy: Math.round(m.accuracy * 1000) / 1000,
    macroF1:  Math.round(m.macroF1  * 1000) / 1000,
    perClass: Object.fromEntries(
      CATEGORIES.map(c => [c, fmt(m.perClass[c])])
    ),
  });

  const result = {
    total,
    gpt:     fmtMetrics(gptMetrics),
    keyword: fmtMetrics(keywordMetrics),
    geval,
  };

  // Cache to disk so GET /api/eval-results can serve it instantly
  try {
    require('fs').writeFileSync(
      path.join(__dirname, 'eval_results.json'),
      JSON.stringify(result, null, 2)
    );
    console.log('[EVAL] Results saved to eval_results.json');
  } catch (e) {
    console.warn('[EVAL] Could not save eval_results.json:', e.message);
  }

  res.json(result);
});

// GET /api/eval-results — serves cached eval_results.json (written by POST /api/evaluate)
app.get('/api/eval-results', (req, res) => {
  try {
    const raw = require('fs').readFileSync(path.join(__dirname, 'eval_results.json'), 'utf8');
    res.json(JSON.parse(raw));
  } catch (e) {
    res.status(404).json({ error: 'No cached results yet — run POST /api/evaluate first.' });
  }
});

// GET /api/evaluate — HTML shell that triggers the POST and renders results
app.get('/api/evaluate', (req, res) => {
  res.send(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AI4Sustain — Classifier Evaluation</title>
<style>
  :root{--forest:#1a3d2b;--emerald:#2d6a4f;--sage:#52b788;--mint:#b7e4c7;
        --cream:#f8f5ef;--paper:#fff;--sand:#ede8dc;--ink:#1c2b22;--muted:#6b7c70;--border:#d4e8da}
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'DM Sans',system-ui,sans-serif;background:var(--cream);color:var(--ink);padding:2rem}
  h1{font-family:Georgia,serif;font-size:1.8rem;color:var(--forest);margin-bottom:.4rem}
  .subtitle{color:var(--muted);font-size:.9rem;margin-bottom:2rem}
  .note{background:#fff8e1;border-left:4px solid #c4933f;border-radius:6px;
        padding:.8rem 1.1rem;font-size:.85rem;color:#5a4000;margin-bottom:2rem}
  button{background:var(--emerald);color:#fff;border:none;border-radius:8px;
         padding:.65rem 1.6rem;font-size:.95rem;font-weight:600;cursor:pointer;transition:background .2s}
  button:hover{background:var(--forest)}
  button:disabled{opacity:.5;cursor:not-allowed}
  .spinner{display:none;margin-top:1.5rem;color:var(--muted);font-size:.9rem}
  .spinner.on{display:block}
  .results{display:none;margin-top:2rem}
  .results.on{display:block}
  .summary-row{display:flex;gap:1.5rem;margin-bottom:2rem;flex-wrap:wrap}
  .summary-card{background:var(--paper);border:1px solid var(--border);border-radius:12px;
                padding:1.4rem 1.8rem;min-width:180px;box-shadow:0 2px 12px rgba(26,61,43,.07)}
  .summary-card .label{font-size:.75rem;font-weight:700;color:var(--muted);
                        text-transform:uppercase;letter-spacing:.06em;margin-bottom:.3rem}
  .summary-card .value{font-family:Georgia,serif;font-size:2rem;color:var(--emerald)}
  .summary-card .sub{font-size:.78rem;color:var(--muted);margin-top:.15rem}
  h2{font-family:Georgia,serif;font-size:1.2rem;color:var(--forest);margin-bottom:1rem}
  table{width:100%;border-collapse:collapse;background:var(--paper);
        border:1px solid var(--border);border-radius:12px;overflow:hidden;
        box-shadow:0 2px 12px rgba(26,61,43,.07);margin-bottom:2rem}
  th{background:var(--forest);color:#fff;font-size:.78rem;font-weight:600;
     text-transform:uppercase;letter-spacing:.05em;padding:.7rem 1rem;text-align:left}
  th.num,td.num{text-align:right}
  td{padding:.65rem 1rem;font-size:.85rem;border-top:1px solid var(--border)}
  tr:nth-child(even) td{background:#fafdf9}
  .cat{font-weight:600;color:var(--forest);text-transform:capitalize}
  .hi{color:#155724;font-weight:700}
  .lo{color:#721c24}
  .col-gpt{background:rgba(45,106,79,.06)}
  .col-kw{background:rgba(196,147,63,.06)}
  .error{color:#721c24;background:#f8d7da;border:1px solid #f5c6cb;
         border-radius:8px;padding:1rem;margin-top:1rem;font-size:.88rem}
  footer{margin-top:3rem;font-size:.75rem;color:var(--muted);text-align:center}
</style>
</head>
<body>
<h1>🌿 AI4Sustain — Classifier Evaluation</h1>
<p class="subtitle">GPT-4o-mini zero-shot vs keyword baseline · computed from <code>labeled_articles.json</code></p>
<div class="note">
  ⏱️ <strong>This evaluation takes 2–3 minutes</strong> — GPT-4o-mini is called once per article in batches of 10.
  Do not close this tab while it runs.
</div>
<button id="runBtn" onclick="runEval()">▶ Run Evaluation</button>
<div class="spinner" id="spinner">⟳ Running… <span id="progress"></span></div>
<div class="results" id="results"></div>

<script>
async function runEval() {
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  document.getElementById('spinner').classList.add('on');
  document.getElementById('results').classList.remove('on');

  try {
    const res  = await fetch('/api/evaluate', { method: 'POST', headers: {'Content-Type':'application/json'}, body: '{}' });
    if (!res.ok) { const e = await res.json(); throw new Error(e.error || res.status); }
    const data = await res.json();
    document.getElementById('spinner').classList.remove('on');
    document.getElementById('results').innerHTML = buildHTML(data);
    document.getElementById('results').classList.add('on');
  } catch(e) {
    document.getElementById('spinner').classList.remove('on');
    document.getElementById('results').innerHTML = '<div class="error">Error: ' + e.message + '</div>';
    document.getElementById('results').classList.add('on');
  }
  btn.disabled = false;
}

function pct(v){ return (v*100).toFixed(1)+'%'; }
function cls(v){ return v >= 0.75 ? 'hi' : v < 0.50 ? 'lo' : ''; }

function buildHTML(d) {
  const cats = ['renewable','emissions','biodiversity','water','policy'];
  const rows = cats.map(cat => {
    const g = d.gpt.perClass[cat];
    const k = d.keyword.perClass[cat];
    return \`<tr>
      <td class="cat">\${cat}</td>
      <td class="num col-gpt \${cls(g.precision)}">\${pct(g.precision)}</td>
      <td class="num col-gpt \${cls(g.recall)}">\${pct(g.recall)}</td>
      <td class="num col-gpt \${cls(g.f1)}">\${pct(g.f1)}</td>
      <td class="num">\${g.support}</td>
      <td class="num col-kw \${cls(k.precision)}">\${pct(k.precision)}</td>
      <td class="num col-kw \${cls(k.recall)}">\${pct(k.recall)}</td>
      <td class="num col-kw \${cls(k.f1)}">\${pct(k.f1)}</td>
    </tr>\`;
  }).join('');

  return \`
  <div class="summary-row">
    <div class="summary-card">
      <div class="label">Articles evaluated</div>
      <div class="value">\${d.total}</div>
    </div>
    <div class="summary-card">
      <div class="label">GPT-4o-mini accuracy</div>
      <div class="value \${cls(d.gpt.accuracy)}">\${pct(d.gpt.accuracy)}</div>
      <div class="sub">Macro F1: \${pct(d.gpt.macroF1)}</div>
    </div>
    <div class="summary-card">
      <div class="label">Keyword baseline accuracy</div>
      <div class="value \${cls(d.keyword.accuracy)}">\${pct(d.keyword.accuracy)}</div>
      <div class="sub">Macro F1: \${pct(d.keyword.macroF1)}</div>
    </div>
  </div>
  <h2>Per-class results</h2>
  <table>
    <thead>
      <tr>
        <th>Category</th>
        <th class="num col-gpt">GPT Prec</th>
        <th class="num col-gpt">GPT Rec</th>
        <th class="num col-gpt">GPT F1</th>
        <th class="num">Support</th>
        <th class="num col-kw">KW Prec</th>
        <th class="num col-kw">KW Rec</th>
        <th class="num col-kw">KW F1</th>
      </tr>
    </thead>
    <tbody>\${rows}</tbody>
    <tfoot>
      <tr style="background:var(--sand)">
        <td class="cat">Macro avg</td>
        <td class="num col-gpt" colspan="2"></td>
        <td class="num col-gpt \${cls(d.gpt.macroF1)}"><strong>\${pct(d.gpt.macroF1)}</strong></td>
        <td class="num">\${d.total}</td>
        <td class="num col-kw" colspan="2"></td>
        <td class="num col-kw \${cls(d.keyword.macroF1)}"><strong>\${pct(d.keyword.macroF1)}</strong></td>
      </tr>
      <tr style="background:var(--sand)">
        <td class="cat">Accuracy</td>
        <td class="num col-gpt \${cls(d.gpt.accuracy)}" colspan="3"><strong>\${pct(d.gpt.accuracy)}</strong></td>
        <td></td>
        <td class="num col-kw \${cls(d.keyword.accuracy)}" colspan="3"><strong>\${pct(d.keyword.accuracy)}</strong></td>
      </tr>
    </tfoot>
  </table>\`;
}
</script>
<footer>AI4Sustain · LLM Spring 2026 · TeamX</footer>
</body>
</html>`);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`AI4Sustain running → http://localhost:${PORT}`));
