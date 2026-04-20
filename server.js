require('dotenv').config();
const express = require('express');
const fetch   = require('node-fetch');
const OpenAI  = require('openai');
const cors    = require('cors');
const path    = require('path');
const fs      = require('fs');

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(__dirname));

const EVAL_RESULTS_PATH = path.join(__dirname, 'eval_results.json');

// ── Per-theme queries ─────────────────────────────────────────────────────
// Specific two-to-three-word phrases joined with OR.
// Region is appended as a separate AND term (not mixed into the OR chain).
const THEME_QUERIES = {
  renewable:    '"solar energy" OR "wind power" OR "renewable energy" OR "clean energy" OR "offshore wind" OR "battery storage"',
  emissions:    '"carbon emissions" OR "greenhouse gas" OR "net zero" OR "carbon capture" OR "methane emissions" OR "fossil fuel"',
  biodiversity: '"coral reef" OR "deforestation" OR "endangered species" OR "biodiversity loss" OR "wildlife habitat" OR "species extinction"',
  water:        '"sea level rise" OR "water scarcity" OR "ocean warming" OR "glacier melt" OR "drought" OR "water crisis"',
  policy:       '"climate policy" OR "Paris Agreement" OR "carbon tax" OR "climate legislation" OR "green new deal" OR "COP30"',
};

const REGION_TERMS = {
  global:   '',
  europe:   'Europe',
  asia:     'Asia',
  americas: 'Americas',
  africa:   'Africa',
};

// Map user time-window to GDELT timespan — use the real window so chart data
// matches, and sort=DateDesc ensures freshest articles come first.
const GDELT_TIMESPANS = { '7d':'7d', '30d':'30d', '90d':'90d', '1y':'180d' };
const WINDOW_DAYS     = { '7d':7,   '30d':30,   '90d':90,   '1y':365 };

const FALLBACK_ARTICLES = [
  { title:'Global renewable energy capacity surpasses 3,500 GW milestone',  source:'Reuters',        url:'#', date:null, snippet:'Solar and wind power additions broke records for the third consecutive year.',           from:'fallback' },
  { title:'UN climate report warns of accelerating ice sheet loss',          source:'BBC',            url:'#', date:null, snippet:'New satellite data shows ice sheets losing mass at rates exceeding IPCC projections.',  from:'fallback' },
  { title:'Carbon markets hit record trading volumes amid new regulations',   source:'Financial Times',url:'#', date:null, snippet:'EU ETS prices rose sharply as stricter caps took effect across heavy industries.',      from:'fallback' },
  { title:'Amazon deforestation fell 50% in 2024, Brazil reports',           source:'Guardian',       url:'#', date:null, snippet:'Enforcement and satellite monitoring contributed to the significant decline.',          from:'fallback' },
  { title:'COP30 host Brazil unveils ambitious national climate pledge',      source:'AP News',        url:'#', date:null, snippet:'Brazil committed to ending illegal deforestation and cutting methane 30% by 2030.',    from:'fallback' },
  { title:'Ocean temperatures reach new highs, threatening coral reefs',     source:'Nature',         url:'#', date:null, snippet:'Marine heatwaves now affect over 40% of the world\'s oceans.',                        from:'fallback' },
];

// ── POST /api/analyze ─────────────────────────────────────────────────────
app.post('/api/analyze', async (req, res) => {
  const { theme = 'renewable', region = 'global', timeWindow = '30d' } = req.body;

  const baseQuery  = THEME_QUERIES[theme] || '"climate change"';
  const regionWord = REGION_TERMS[region] || '';
  // Wrap OR chain in parens so region is a separate AND term, not part of the OR
  const query   = regionWord ? `(${baseQuery}) ${regionWord}` : baseQuery;
  const timespan = GDELT_TIMESPANS[timeWindow] || '30d';
  const days     = WINDOW_DAYS[timeWindow]     || 30;

  // ── GDELT ───────────────────────────────────────────────────────────────
  let gdeltArticles = await fetchGdelt(query, timespan);
  if (gdeltArticles.length === 0) {
    console.log('[GDELT] 0 results — retrying with bare theme query');
    const bareQuery = regionWord ? `"${theme}" ${regionWord}` : `"${theme}"`;
    gdeltArticles = await fetchGdelt(bareQuery, timespan);
  }
  if (gdeltArticles.length === 0) {
    console.log('[GDELT] still 0 — falling back to "climate change"');
    gdeltArticles = await fetchGdelt('climate change', timespan);
  }

  // ── NewsAPI ─────────────────────────────────────────────────────────────
  let newsArticles = [];
  if (process.env.NEWS_API_KEY) {
    try {
      const fromDate = new Date(Date.now() - days * 86400000).toISOString().split('T')[0];
      // NewsAPI query: first OR-term of the theme + optional region
      const newsQ = (theme === 'renewable' ? 'renewable energy OR solar OR wind'
                   : theme === 'emissions'  ? 'carbon emissions OR greenhouse gas OR net zero'
                   : theme === 'biodiversity' ? 'deforestation OR biodiversity OR coral reef'
                   : theme === 'water'      ? 'water scarcity OR drought OR sea level rise'
                   : 'climate policy OR Paris Agreement OR carbon tax')
                   + (regionWord ? ` ${regionWord}` : '');
      const newsUrl =
        `https://newsapi.org/v2/everything` +
        `?q=${encodeURIComponent(newsQ)}&pageSize=10&language=en&sortBy=publishedAt` +
        `&from=${fromDate}&apiKey=${process.env.NEWS_API_KEY}`;
      const newsData = await (await fetch(newsUrl, { timeout: 10000 })).json();
      newsArticles = (newsData.articles || [])
        .filter(a => a.title && a.title !== '[Removed]')
        .map(a => ({
          title:   a.title,
          source:  a.source?.name || 'NewsAPI',
          url:     a.url  || '#',
          date:    a.publishedAt ? a.publishedAt.substring(0,10).replace(/-/g,'') : null,
          snippet: a.description || '',
          from:    'newsapi',
        }));
    } catch (e) { console.warn('[NewsAPI] fetch failed:', e.message); }
  }

  let allArticles = [...gdeltArticles, ...newsArticles];
  if (allArticles.length === 0) {
    console.log('[FALLBACK] using hardcoded stubs');
    allArticles = FALLBACK_ARTICLES.map(a => ({ ...a }));
  }

  const chartData    = buildChartData(allArticles, timeWindow);
  let   summary      = '';
  let   sentimentLine = chartData.labels.map(() => 3.0);

  if (process.env.OPENAI_API_KEY && allArticles.length > 0) {
    try {
      const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
      const articleTexts = allArticles.slice(0, 20).map((a, i) => {
        const snip = a.snippet ? ' — ' + a.snippet.substring(0, 100) : '';
        return `${i+1}. "${a.title}"${snip} [${a.source}]`;
      }).join('\n');

      const completion = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [{ role:'user', content:
          `You are an environmental news analyst. Analyze these articles about "${theme}"` +
          `${regionWord ? ' in ' + regionWord : ''}.\n\nArticles:\n${articleTexts}\n\n` +
          `Return ONLY valid JSON (no markdown):\n` +
          `{"summary":"3-4 sentence RAG trend summary","sentiment":[${chartData.labels.length} floats 1.0-5.0]}`
        }],
        max_tokens: 450,
        temperature: 0.3,
      });
      const parsed = JSON.parse(completion.choices[0].message.content.trim());
      summary = parsed.summary || '';
      if (Array.isArray(parsed.sentiment) && parsed.sentiment.length > 0)
        sentimentLine = padOrTrim(parsed.sentiment, chartData.labels.length);
    } catch (e) { console.warn('[OpenAI] analyze failed:', e.message); }
  }

  const articlesOut = allArticles.map(a => ({
    ...a,
    sentiment: keywordSentiment(a.title + ' ' + a.snippet),
  }));

  res.json({ articles: articlesOut, summary,
    chartData: { labels: chartData.labels, values: chartData.values, sentiment: sentimentLine } });

  // ── Background: compute G-Eval + live keyword precision, save to disk ──
  // Fires after response is sent — never blocks the user.
  if (summary && process.env.OPENAI_API_KEY) {
    saveEvalMetrics(summary, articlesOut, theme).catch(e =>
      console.warn('[EVAL] background save failed:', e.message));
  }
});

// ── GET /api/eval-results ─────────────────────────────────────────────────
app.get('/api/eval-results', (req, res) => {
  try {
    res.json(JSON.parse(fs.readFileSync(EVAL_RESULTS_PATH, 'utf8')));
  } catch (e) {
    res.status(404).json({ error: 'No results yet — run an analysis first.' });
  }
});

// ── POST /api/evaluate ────────────────────────────────────────────────────
// Full labeled evaluation (runs once against labeled_articles.json)
app.post('/api/evaluate', async (req, res) => {
  if (!process.env.OPENAI_API_KEY)
    return res.status(400).json({ error: 'OPENAI_API_KEY not set' });

  let labeled;
  try {
    labeled = JSON.parse(fs.readFileSync(path.join(__dirname, 'labeled_articles.json'), 'utf8'));
  } catch (e) {
    return res.status(500).json({ error: `Cannot read labeled_articles.json: ${e.message}` });
  }

  const total       = labeled.length;
  const goldLabels  = labeled.map(a => a.label);
  const kwPreds     = labeled.map(a => classifyKeyword(a.title, a.snippet));
  const openai      = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const gptPreds    = new Array(total);
  const BATCH       = 10;

  console.log(`[EVAL] Starting evaluation on ${total} articles`);
  for (let start = 0; start < total; start += BATCH) {
    await Promise.all(labeled.slice(start, start + BATCH).map(async (article, bi) => {
      const idx = start + bi;
      console.log(`[EVAL] Evaluating article ${idx + 1} of ${total}...`);
      try {
        const c = await openai.chat.completions.create({
          model: 'gpt-4o-mini',
          messages: [{ role: 'user', content:
            `Classify this sustainability article into exactly one of: renewable, emissions, biodiversity, water, policy\n\n` +
            `Title: ${article.title}\nSnippet: ${article.snippet || ''}\n\n` +
            `Reply with just the single category word in lowercase.`
          }],
          max_tokens: 10, temperature: 0,
        });
        const pred = c.choices[0].message.content.trim().toLowerCase();
        gptPreds[idx] = CATEGORIES.includes(pred) ? pred : 'renewable';
      } catch (e) {
        console.warn(`[EVAL] GPT failed article ${idx+1}:`, e.message);
        gptPreds[idx] = 'renewable';
      }
    }));
    if (start + BATCH < total) await sleep(1000);
  }

  console.log('[EVAL] Computing metrics…');
  const gptMetrics = computeMetrics(goldLabels, gptPreds);
  const kwMetrics  = computeMetrics(goldLabels, kwPreds);

  // G-Eval on 5-article sample
  let geval = null;
  try {
    const sample = labeled.slice(0, 5);
    const sampleText = sample.map((a,i) =>
      `${i+1}. "${a.title}" — ${(a.snippet||'').substring(0,100)}`).join('\n');
    const sumRes = await openai.chat.completions.create({
      model:'gpt-4o-mini',
      messages:[{role:'user',content:`Summarize in 3-4 sentences:\n${sampleText}`}],
      max_tokens:200, temperature:0.3,
    });
    const genSummary = sumRes.choices[0].message.content.trim();
    const scoreRes = await openai.chat.completions.create({
      model:'gpt-4o-mini',
      messages:[{role:'user',content:
        `Rate this summary against the source articles. Return ONLY JSON {"relevance":X,"coherence":X,"grounding":X} (1.0-5.0).\n\n` +
        `Sources:\n${sampleText}\n\nSummary: ${genSummary}`
      }],
      max_tokens:40, temperature:0,
    });
    geval = JSON.parse(scoreRes.choices[0].message.content.trim());
  } catch(e) { console.warn('[EVAL] G-Eval failed:', e.message); }

  const fmt = v => Math.round(v * 1000) / 1000;
  const fmtM = m => ({
    accuracy: fmt(m.accuracy), macroF1: fmt(m.macroF1),
    perClass: Object.fromEntries(CATEGORIES.map(c => [c, {
      precision: fmt(m.perClass[c].precision),
      recall:    fmt(m.perClass[c].recall),
      f1:        fmt(m.perClass[c].f1),
      support:   m.perClass[c].support,
    }])),
  });

  const result = { total, gpt: fmtM(gptMetrics), keyword: fmtM(kwMetrics), geval };

  try {
    // Merge with any existing geval from live analysis
    let existing = {};
    try { existing = JSON.parse(fs.readFileSync(EVAL_RESULTS_PATH,'utf8')); } catch(e) {}
    fs.writeFileSync(EVAL_RESULTS_PATH, JSON.stringify({ ...existing, ...result }, null, 2));
    console.log('[EVAL] Results saved to eval_results.json');
  } catch(e) { console.warn('[EVAL] Could not save:', e.message); }

  res.json(result);
});

// ── GET /api/evaluate ─────────────────────────────────────────────────────
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
  .note{background:#fff8e1;border-left:4px solid #c4933f;border-radius:6px;padding:.8rem 1.1rem;font-size:.85rem;color:#5a4000;margin-bottom:2rem}
  button{background:var(--emerald);color:#fff;border:none;border-radius:8px;padding:.65rem 1.6rem;font-size:.95rem;font-weight:600;cursor:pointer;transition:background .2s}
  button:hover{background:var(--forest)} button:disabled{opacity:.5;cursor:not-allowed}
  .spinner{display:none;margin-top:1.5rem;color:var(--muted);font-size:.9rem} .spinner.on{display:block}
  .results{display:none;margin-top:2rem} .results.on{display:block}
  .summary-row{display:flex;gap:1.5rem;margin-bottom:2rem;flex-wrap:wrap}
  .summary-card{background:var(--paper);border:1px solid var(--border);border-radius:12px;padding:1.4rem 1.8rem;min-width:180px;box-shadow:0 2px 12px rgba(26,61,43,.07)}
  .summary-card .label{font-size:.75rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:.3rem}
  .summary-card .value{font-family:Georgia,serif;font-size:2rem;color:var(--emerald)}
  .summary-card .sub{font-size:.78rem;color:var(--muted);margin-top:.15rem}
  h2{font-family:Georgia,serif;font-size:1.2rem;color:var(--forest);margin-bottom:1rem}
  table{width:100%;border-collapse:collapse;background:var(--paper);border:1px solid var(--border);border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(26,61,43,.07);margin-bottom:2rem}
  th{background:var(--forest);color:#fff;font-size:.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em;padding:.7rem 1rem;text-align:left}
  th.num,td.num{text-align:right} td{padding:.65rem 1rem;font-size:.85rem;border-top:1px solid var(--border)}
  tr:nth-child(even) td{background:#fafdf9} .cat{font-weight:600;color:var(--forest);text-transform:capitalize}
  .hi{color:#155724;font-weight:700} .lo{color:#721c24}
  .col-gpt{background:rgba(45,106,79,.06)} .col-kw{background:rgba(196,147,63,.06)}
  .error{color:#721c24;background:#f8d7da;border:1px solid #f5c6cb;border-radius:8px;padding:1rem;margin-top:1rem;font-size:.88rem}
  footer{margin-top:3rem;font-size:.75rem;color:var(--muted);text-align:center}
</style>
</head>
<body>
<h1>🌿 AI4Sustain — Classifier Evaluation</h1>
<p class="subtitle">GPT-4o-mini zero-shot vs keyword baseline · computed from <code>labeled_articles.json</code></p>
<div class="note">⏱️ <strong>This takes 2–3 minutes</strong> — GPT-4o-mini is called once per article in batches of 10. Do not close this tab.</div>
<button id="runBtn" onclick="runEval()">▶ Run Evaluation</button>
<div class="spinner" id="spinner">⟳ Running…</div>
<div class="results" id="results"></div>
<script>
async function runEval(){
  const btn=document.getElementById('runBtn');
  btn.disabled=true;
  document.getElementById('spinner').classList.add('on');
  document.getElementById('results').classList.remove('on');
  try{
    const res=await fetch('/api/evaluate',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    if(!res.ok){const e=await res.json();throw new Error(e.error||res.status);}
    const data=await res.json();
    document.getElementById('spinner').classList.remove('on');
    document.getElementById('results').innerHTML=buildHTML(data);
    document.getElementById('results').classList.add('on');
  }catch(e){
    document.getElementById('spinner').classList.remove('on');
    document.getElementById('results').innerHTML='<div class="error">Error: '+e.message+'</div>';
    document.getElementById('results').classList.add('on');
  }
  btn.disabled=false;
}
function pct(v){return(v*100).toFixed(1)+'%';}
function cls(v){return v>=0.75?'hi':v<0.50?'lo':'';}
function buildHTML(d){
  const cats=['renewable','emissions','biodiversity','water','policy'];
  const rows=cats.map(cat=>{
    const g=d.gpt.perClass[cat],k=d.keyword.perClass[cat];
    return \`<tr><td class="cat">\${cat}</td>
      <td class="num col-gpt \${cls(g.precision)}">\${pct(g.precision)}</td>
      <td class="num col-gpt \${cls(g.recall)}">\${pct(g.recall)}</td>
      <td class="num col-gpt \${cls(g.f1)}">\${pct(g.f1)}</td>
      <td class="num">\${g.support}</td>
      <td class="num col-kw \${cls(k.precision)}">\${pct(k.precision)}</td>
      <td class="num col-kw \${cls(k.recall)}">\${pct(k.recall)}</td>
      <td class="num col-kw \${cls(k.f1)}">\${pct(k.f1)}</td></tr>\`;
  }).join('');
  return \`<div class="summary-row">
    <div class="summary-card"><div class="label">Articles</div><div class="value">\${d.total}</div></div>
    <div class="summary-card"><div class="label">GPT-4o-mini accuracy</div><div class="value \${cls(d.gpt.accuracy)}">\${pct(d.gpt.accuracy)}</div><div class="sub">Macro F1: \${pct(d.gpt.macroF1)}</div></div>
    <div class="summary-card"><div class="label">Keyword accuracy</div><div class="value \${cls(d.keyword.accuracy)}">\${pct(d.keyword.accuracy)}</div><div class="sub">Macro F1: \${pct(d.keyword.macroF1)}</div></div>
    \${d.geval?'<div class="summary-card"><div class="label">G-Eval coherence</div><div class="value">'+d.geval.coherence.toFixed(1)+'<span style="font-size:1.2rem">/5</span></div></div>':''}
  </div>
  <h2>Per-class results</h2>
  <table><thead><tr>
    <th>Category</th>
    <th class="num col-gpt">GPT Prec</th><th class="num col-gpt">GPT Rec</th><th class="num col-gpt">GPT F1</th>
    <th class="num">Support</th>
    <th class="num col-kw">KW Prec</th><th class="num col-kw">KW Rec</th><th class="num col-kw">KW F1</th>
  </tr></thead><tbody>\${rows}</tbody>
  <tfoot>
    <tr style="background:var(--sand)"><td class="cat">Macro avg</td><td class="num col-gpt" colspan="2"></td><td class="num col-gpt \${cls(d.gpt.macroF1)}"><strong>\${pct(d.gpt.macroF1)}</strong></td><td class="num">\${d.total}</td><td class="num col-kw" colspan="2"></td><td class="num col-kw \${cls(d.keyword.macroF1)}"><strong>\${pct(d.keyword.macroF1)}</strong></td></tr>
    <tr style="background:var(--sand)"><td class="cat">Accuracy</td><td class="num col-gpt \${cls(d.gpt.accuracy)}" colspan="3"><strong>\${pct(d.gpt.accuracy)}</strong></td><td></td><td class="num col-kw \${cls(d.keyword.accuracy)}" colspan="3"><strong>\${pct(d.keyword.accuracy)}</strong></td></tr>
  </tfoot></table>\`;
}
</script>
<footer>AI4Sustain · LLM Spring 2026 · TeamX</footer>
</body></html>`);
});

// ── Helpers ───────────────────────────────────────────────────────────────

// Returns false if the title contains non-Latin script characters
function isLatinTitle(title) {
  return !/[\u0400-\u04FF\u0600-\u06FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF\u0900-\u097F\u0E00-\u0E7F]/.test(title);
}

async function fetchGdelt(query, timespan) {
  const url =
    `https://api.gdeltproject.org/api/v2/doc/doc` +
    `?query=${encodeURIComponent(query)}` +
    `&mode=artlist&maxrecords=25&format=json` +
    `&timespan=${timespan}&sourcelang=english&sort=DateDesc`;
  console.log(`[GDELT] GET ${url}`);
  try {
    const res  = await fetch(url, { timeout: 12000 });
    const data = await res.json();
    const raw  = (data.articles || []).map(a => ({
      title:    a.title    || '',
      source:   a.domain   || 'GDELT',
      url:      a.url      || '#',
      date:     parseGdeltDate(a.seendate),
      snippet:  '',
      from:     'gdelt',
      language: (a.language || '').toLowerCase(),
    }));
    const articles = raw
      .filter(a => a.title)
      // Use GDELT's own language field as the primary English filter
      .filter(a => !a.language || a.language === 'english')
      // Regex backup: reject titles with non-Latin script
      .filter(a => isLatinTitle(a.title));
    console.log(`[GDELT] ${articles.length}/${raw.length} English articles for query="${query.substring(0,60)}…"`);
    return articles;
  } catch (e) {
    console.warn(`[GDELT] failed for "${query.substring(0,40)}…":`, e.message);
    return [];
  }
}

function parseGdeltDate(seendate) {
  if (!seendate) return null;
  return seendate.replace(/[T\-:Z]/g, '').substring(0, 8);
}

function buildChartData(articles, timeWindow) {
  const days     = WINDOW_DAYS[timeWindow] || 30;
  const buckets  = days <= 7 ? 7 : 6;
  const now      = Date.now();
  const start    = now - days * 86400000;
  const bucketMs = (days * 86400000) / buckets;
  const MONTHS   = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const DNAMES   = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];

  const labels = Array.from({ length: buckets }, (_, i) => {
    const t = new Date(start + (i + 0.5) * bucketMs);
    if (days <= 7)  return DNAMES[t.getDay()];
    if (days <= 90) return MONTHS[t.getMonth()] + ' ' + t.getDate();
    return MONTHS[t.getMonth()];
  });

  const counts = new Array(buckets).fill(0);
  articles.forEach(a => {
    if (!a.date || a.date.length < 8) return;
    const y = parseInt(a.date.substring(0,4));
    const m = parseInt(a.date.substring(4,6)) - 1;
    const d = parseInt(a.date.substring(6,8));
    if (isNaN(y) || isNaN(m) || isNaN(d)) return;
    const idx = Math.floor((new Date(y,m,d).getTime() - start) / bucketMs);
    if (idx >= 0 && idx < buckets) counts[idx]++;
  });
  return { labels, values: counts };
}

function keywordSentiment(text) {
  const t = (text||'').toLowerCase();
  const pos = ['surge','record','boost','milestone','breakthrough','success','achieve','improve',
               'expand','gain','increase','rise','advance','agreement','progress','renewable','clean','invest'];
  const neg = ['fail','crisis','threat','decline','loss','damage','disaster','warn','shortage',
               'conflict','drop','fall','bleach','flood','drought','pollution','collapse','deforest'];
  let s = 0;
  pos.forEach(w => { if (t.includes(w)) s++; });
  neg.forEach(w => { if (t.includes(w)) s--; });
  return s > 0 ? 'Positive' : s < 0 ? 'Negative' : 'Neutral';
}

const CATEGORIES = ['renewable','emissions','biodiversity','water','policy'];

const EVAL_KEYWORDS = {
  renewable:    ['solar','wind','renewable','clean energy','battery','turbine','geothermal','hydrogen','nuclear','photovoltaic','ev','electric vehicle'],
  emissions:    ['carbon','emission','co2','methane','greenhouse','fossil fuel','net zero','decarbonize','carbon capture','ghg','coal'],
  biodiversity: ['species','forest','wildlife','ecosystem','biodiversity','deforestation','extinction','habitat','coral','reef','mammal','bird','fish','plant'],
  water:        ['ocean','water','flood','drought','river','sea level','groundwater','rainfall','aquifer','monsoon','glacier','ice'],
  policy:       ['policy','agreement','cop','law','regulation','government','treaty','pledge','summit','legislation','fund','subsidy','mandate'],
};

function classifyKeyword(title, snippet) {
  const text = (title + ' ' + (snippet||'')).toLowerCase();
  let best = null, bestCount = 0;
  for (const cat of CATEGORIES) {
    const count = EVAL_KEYWORDS[cat].filter(kw => text.includes(kw)).length;
    if (count > bestCount) { bestCount = count; best = cat; }
  }
  return best || 'renewable';
}

function computeMetrics(labels, preds) {
  const total    = labels.length;
  const correct  = labels.filter((l,i) => l === preds[i]).length;
  const accuracy = total > 0 ? correct / total : 0;
  const perClass = {};
  for (const cat of CATEGORIES) {
    let tp=0, fp=0, fn=0;
    for (let i=0; i<total; i++) {
      const isG = labels[i]===cat, isP = preds[i]===cat;
      if (isG && isP) tp++;
      else if (!isG && isP) fp++;
      else if (isG && !isP) fn++;
    }
    const p = tp+fp>0 ? tp/(tp+fp) : 0;
    const r = tp+fn>0 ? tp/(tp+fn) : 0;
    perClass[cat] = { precision:p, recall:r, f1: p+r>0 ? 2*p*r/(p+r) : 0, support: labels.filter(l=>l===cat).length };
  }
  const macroF1 = CATEGORIES.reduce((s,c)=>s+perClass[c].f1, 0) / CATEGORIES.length;
  return { accuracy, macroF1, perClass };
}

function padOrTrim(arr, length) {
  const out = arr.slice(0, length).map(Number);
  while (out.length < length) out.push(out[out.length-1] ?? 3.0);
  return out;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// Saves G-Eval + live keyword precision after each /api/analyze run.
// Preserves any existing full-eval (gpt/keyword macroF1) already on disk.
async function saveEvalMetrics(summary, articles, theme) {
  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const sampleTitles = articles.slice(0,5).map((a,i)=>`${i+1}. ${a.title}`).join('\n');

  const scoreRes = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role:'user', content:
      `Rate this sustainability summary against the source article titles.\n\n` +
      `Titles:\n${sampleTitles}\n\nSummary: ${summary}\n\n` +
      `Return ONLY valid JSON (no markdown): {"relevance":X,"coherence":X,"grounding":X} where each X is 1.0-5.0.`
    }],
    max_tokens: 40,
    temperature: 0,
  });
  const geval = JSON.parse(scoreRes.choices[0].message.content.trim());

  // Keyword precision: how many of the fetched articles the keyword classifier
  // correctly identifies as belonging to this theme (gold = the theme we searched).
  const kwPreds = articles.map(a => classifyKeyword(a.title, a.snippet));
  const kwPrecision = articles.length > 0
    ? Math.round(kwPreds.filter(p => p === theme).length / articles.length * 1000) / 1000
    : 0;

  let existing = {};
  try { existing = JSON.parse(fs.readFileSync(EVAL_RESULTS_PATH,'utf8')); } catch(e) {}

  // Merge: keep existing full-eval metrics, overwrite live geval + precision
  const updated = {
    ...existing,
    geval,
    liveKeywordPrecision: kwPrecision,
    lastUpdated: new Date().toISOString(),
  };
  fs.writeFileSync(EVAL_RESULTS_PATH, JSON.stringify(updated, null, 2));
  console.log(`[EVAL] Saved — G-Eval coherence: ${geval.coherence}, KW precision: ${kwPrecision}`);
}

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`AI4Sustain running → http://localhost:${PORT}`));
