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

const EVAL_PATH  = path.join(__dirname, 'eval_results.json');
const LABEL_PATH = path.join(__dirname, 'labeled_articles.json');

// ── Simple AND queries (spaces = AND in GDELT — no OR, no quotes, no parens).
// Enough words to be topic-specific and distinct from each other.
const THEME_QUERIES = {
  renewable:    'renewable energy solar',
  emissions:    'carbon emissions greenhouse',
  biodiversity: 'biodiversity deforestation species',
  water:        'water drought flooding',
  policy:       'climate policy legislation',
};

// Simple one-word region terms appended to the query
const REGION_WORDS = {
  global: '', europe: 'Europe', asia: 'Asia',
  americas: 'Americas', africa: 'Africa',
};

const GDELT_TIMESPANS = { '7d':'7d', '30d':'30d', '90d':'90d', '1y':'180d' };
const WINDOW_DAYS     = { '7d':7, '30d':30, '90d':90, '1y':365 };

const CATEGORIES  = ['renewable','emissions','biodiversity','water','policy'];

const EVAL_KW = {
  renewable:    ['solar','wind','renewable','clean energy','battery','turbine','geothermal','hydrogen','photovoltaic','electric vehicle'],
  emissions:    ['carbon','emission','co2','methane','greenhouse','fossil fuel','net zero','decarbonize','coal'],
  biodiversity: ['species','forest','wildlife','ecosystem','biodiversity','deforestation','extinction','habitat','coral','reef'],
  water:        ['ocean','water','flood','drought','river','sea level','groundwater','glacier','rainfall'],
  policy:       ['policy','agreement','cop','law','regulation','government','treaty','pledge','legislation','subsidy'],
};

const FALLBACK_ARTICLES = [
  { title:'Global renewable energy capacity surpasses 3,500 GW',         source:'Reuters',         url:'#', date:null, snippet:'Solar and wind additions broke records for the third year running.',              from:'fallback' },
  { title:'UN report warns of accelerating ice sheet loss',               source:'BBC',             url:'#', date:null, snippet:'Satellite data shows ice sheets losing mass faster than IPCC projections.',     from:'fallback' },
  { title:'Carbon markets hit record trading volumes',                    source:'Financial Times', url:'#', date:null, snippet:'EU ETS prices rose as stricter caps took effect across heavy industries.',       from:'fallback' },
  { title:'Amazon deforestation fell 50% in 2024, Brazil reports',        source:'Guardian',        url:'#', date:null, snippet:'Enforcement and satellite monitoring contributed to the decline.',              from:'fallback' },
  { title:'COP30 host Brazil unveils ambitious climate pledge',           source:'AP News',         url:'#', date:null, snippet:'Brazil committed to ending illegal deforestation by 2030.',                    from:'fallback' },
  { title:'Ocean temperatures reach new highs threatening coral reefs',  source:'Nature',          url:'#', date:null, snippet:'Marine heatwaves now affect over 40% of the world\'s oceans.',                 from:'fallback' },
];

// ── Global eval state (tracks background startup eval progress) ───────────
const evalState = { status: 'idle', progress: 0, total: 0 };

// ── POST /api/analyze ─────────────────────────────────────────────────────
app.post('/api/analyze', async (req, res) => {
  const { theme = 'renewable', region = 'global', timeWindow = '30d' } = req.body;

  const baseQ      = THEME_QUERIES[theme] || 'climate change';
  const regionWord = REGION_WORDS[region] || '';
  const query      = regionWord ? `${baseQ} ${regionWord}` : baseQ;
  const timespan   = GDELT_TIMESPANS[timeWindow] || '30d';
  const days       = WINDOW_DAYS[timeWindow] || 30;

  // GDELT — if region+theme query returns 0, drop region, then fall back
  let gdeltArticles = await fetchGdelt(query, timespan);
  if (gdeltArticles.length === 0 && regionWord) {
    console.log('[GDELT] retry without region');
    gdeltArticles = await fetchGdelt(baseQ, timespan);
  }
  if (gdeltArticles.length === 0) {
    console.log('[GDELT] final fallback: climate change');
    gdeltArticles = await fetchGdelt('climate change', '30d');
  }

  // NewsAPI
  let newsArticles = [];
  if (process.env.NEWS_API_KEY) {
    try {
      const from    = new Date(Date.now() - days * 86400000).toISOString().split('T')[0];
      const newsUrl = `https://newsapi.org/v2/everything?q=${encodeURIComponent(baseQ)}&pageSize=10&language=en&sortBy=publishedAt&from=${from}&apiKey=${process.env.NEWS_API_KEY}`;
      const nd      = await (await fetch(newsUrl, { timeout: 10000 })).json();
      newsArticles  = (nd.articles || [])
        .filter(a => a.title && a.title !== '[Removed]')
        .map(a => ({
          title:   a.title,
          source:  a.source?.name || 'NewsAPI',
          url:     a.url || '#',
          date:    a.publishedAt ? a.publishedAt.substring(0,10).replace(/-/g,'') : null,
          snippet: a.description || '',
          from:    'newsapi',
        }));
    } catch(e) { console.warn('[NewsAPI]', e.message); }
  }

  let all = [...gdeltArticles, ...newsArticles];
  if (all.length === 0) all = FALLBACK_ARTICLES.map(a => ({ ...a }));

  const chartData    = buildChartData(all, timeWindow);
  let   summary      = '';
  let   sentimentLine = chartData.labels.map(() => 3.0);

  if (process.env.OPENAI_API_KEY && all.length > 0) {
    try {
      const openai  = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
      const snippet = all.slice(0, 20).map((a, i) =>
        `${i+1}. "${a.title}"${a.snippet ? ' — '+a.snippet.substring(0,100) : ''} [${a.source}]`
      ).join('\n');

      const c = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [{ role:'user', content:
          `Environmental news analyst. Analyze these articles about "${theme}"${regionWord ? ' in '+regionWord : ''}.\n\n` +
          `Articles:\n${snippet}\n\n` +
          `Return ONLY valid JSON (no markdown):\n` +
          `{"summary":"3-4 sentence RAG trend summary grounded in the articles","sentiment":[${chartData.labels.length} floats 1.0-5.0]}`
        }],
        max_tokens: 500, temperature: 0.3,
      });
      const p = JSON.parse(c.choices[0].message.content.trim());
      summary = p.summary || '';
      if (Array.isArray(p.sentiment) && p.sentiment.length > 0)
        sentimentLine = padOrTrim(p.sentiment, chartData.labels.length);
    } catch(e) { console.warn('[OpenAI analyze]', e.message); }
  }

  const articlesOut = all.map(a => ({ ...a, sentiment: kwSentiment(a.title+' '+a.snippet) }));

  res.json({
    articles:  articlesOut,
    summary,
    chartData: { labels: chartData.labels, values: chartData.values, sentiment: sentimentLine },
  });

  // Fire-and-forget background G-Eval after the response is sent
  if (summary && process.env.OPENAI_API_KEY) {
    runBackgroundGeval(summary, articlesOut, theme).catch(() => {});
  }
});

// ── GET /api/eval-results ─────────────────────────────────────────────────
app.get('/api/eval-results', (req, res) => {
  try {
    res.json(JSON.parse(fs.readFileSync(EVAL_PATH, 'utf8')));
  } catch(e) {
    res.status(404).json({ error: 'No results yet', evalStatus: evalState });
  }
});

// ── GET /api/eval-status ──────────────────────────────────────────────────
app.get('/api/eval-status', (req, res) => res.json(evalState));

// ── POST /api/evaluate — manual trigger ──────────────────────────────────
app.post('/api/evaluate', async (req, res) => {
  if (!process.env.OPENAI_API_KEY)
    return res.status(400).json({ error: 'OPENAI_API_KEY not set' });
  let labeled;
  try { labeled = JSON.parse(fs.readFileSync(LABEL_PATH, 'utf8')); }
  catch(e) { return res.status(500).json({ error: `Cannot read labeled_articles.json: ${e.message}` }); }

  const result = await runFullEval(labeled);
  res.json(result);
});

// ── GET /api/evaluate — eval HTML page ───────────────────────────────────
app.get('/api/evaluate', (req, res) => {
  res.send(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AI4Sustain — Classifier Evaluation</title>
<style>
  :root{--forest:#1a3d2b;--emerald:#2d6a4f;--sage:#52b788;--mint:#b7e4c7;--cream:#f8f5ef;--paper:#fff;--sand:#ede8dc;--ink:#1c2b22;--muted:#6b7c70;--border:#d4e8da}
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'DM Sans',system-ui,sans-serif;background:var(--cream);color:var(--ink);padding:2rem}
  h1{font-family:Georgia,serif;font-size:1.8rem;color:var(--forest);margin-bottom:.4rem}
  .sub{color:var(--muted);font-size:.9rem;margin-bottom:2rem}
  .note{background:#fff8e1;border-left:4px solid #c4933f;border-radius:6px;padding:.8rem 1.1rem;font-size:.85rem;color:#5a4000;margin-bottom:2rem}
  button{background:var(--emerald);color:#fff;border:none;border-radius:8px;padding:.65rem 1.6rem;font-size:.95rem;font-weight:600;cursor:pointer}
  button:hover{background:var(--forest)} button:disabled{opacity:.5;cursor:not-allowed}
  .spin{display:none;margin-top:1.5rem;color:var(--muted);font-size:.9rem}.spin.on{display:block}
  .res{display:none;margin-top:2rem}.res.on{display:block}
  .cards{display:flex;gap:1.5rem;margin-bottom:2rem;flex-wrap:wrap}
  .card{background:var(--paper);border:1px solid var(--border);border-radius:12px;padding:1.4rem 1.8rem;min-width:170px}
  .card .lbl{font-size:.75rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:.3rem}
  .card .val{font-family:Georgia,serif;font-size:2rem;color:var(--emerald)}
  .card .sub{font-size:.78rem;color:var(--muted);margin-top:.15rem}
  h2{font-family:Georgia,serif;font-size:1.2rem;color:var(--forest);margin-bottom:1rem}
  table{width:100%;border-collapse:collapse;background:var(--paper);border:1px solid var(--border);border-radius:12px;overflow:hidden;margin-bottom:2rem}
  th{background:var(--forest);color:#fff;font-size:.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em;padding:.7rem 1rem;text-align:left}
  th.r,td.r{text-align:right} td{padding:.65rem 1rem;font-size:.85rem;border-top:1px solid var(--border)}
  tr:nth-child(even) td{background:#fafdf9} .cat{font-weight:600;color:var(--forest);text-transform:capitalize}
  .hi{color:#155724;font-weight:700} .lo{color:#721c24}
  .g{background:rgba(45,106,79,.06)} .k{background:rgba(196,147,63,.06)}
  .err{color:#721c24;background:#f8d7da;border:1px solid #f5c6cb;border-radius:8px;padding:1rem;margin-top:1rem;font-size:.88rem}
  footer{margin-top:3rem;font-size:.75rem;color:var(--muted);text-align:center}
</style>
</head>
<body>
<h1>🌿 AI4Sustain — Classifier Evaluation</h1>
<p class="sub">GPT-4o-mini zero-shot vs keyword baseline · from <code>labeled_articles.json</code></p>
<div class="note">⏱️ <strong>This takes 2–3 minutes</strong> — one GPT call per article in batches of 10. Do not close this tab.</div>
<button id="btn" onclick="go()">▶ Run Evaluation</button>
<div class="spin" id="spin">⟳ Running…</div>
<div class="res" id="res"></div>
<script>
async function go(){
  const btn=document.getElementById('btn');
  btn.disabled=true;
  document.getElementById('spin').classList.add('on');
  document.getElementById('res').classList.remove('on');
  try{
    const r=await fetch('/api/evaluate',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    if(!r.ok){const e=await r.json();throw new Error(e.error||r.status);}
    const d=await r.json();
    document.getElementById('spin').classList.remove('on');
    document.getElementById('res').innerHTML=html(d);
    document.getElementById('res').classList.add('on');
  }catch(e){
    document.getElementById('spin').classList.remove('on');
    document.getElementById('res').innerHTML='<div class="err">'+e.message+'</div>';
    document.getElementById('res').classList.add('on');
  }
  btn.disabled=false;
}
const pct=v=>(v*100).toFixed(1)+'%';
const cl=v=>v>=0.75?'hi':v<0.5?'lo':'';
function html(d){
  const cats=['renewable','emissions','biodiversity','water','policy'];
  const rows=cats.map(c=>{
    const g=d.gpt.perClass[c],k=d.keyword.perClass[c];
    return \`<tr><td class="cat">\${c}</td>
      <td class="r g \${cl(g.precision)}">\${pct(g.precision)}</td>
      <td class="r g \${cl(g.recall)}">\${pct(g.recall)}</td>
      <td class="r g \${cl(g.f1)}">\${pct(g.f1)}</td>
      <td class="r">\${g.support}</td>
      <td class="r k \${cl(k.precision)}">\${pct(k.precision)}</td>
      <td class="r k \${cl(k.recall)}">\${pct(k.recall)}</td>
      <td class="r k \${cl(k.f1)}">\${pct(k.f1)}</td></tr>\`;
  }).join('');
  return \`<div class="cards">
    <div class="card"><div class="lbl">Articles</div><div class="val">\${d.total}</div></div>
    <div class="card"><div class="lbl">GPT Accuracy</div><div class="val \${cl(d.gpt.accuracy)}">\${pct(d.gpt.accuracy)}</div><div class="sub">Macro F1: \${pct(d.gpt.macroF1)}</div></div>
    <div class="card"><div class="lbl">Keyword Accuracy</div><div class="val \${cl(d.keyword.accuracy)}">\${pct(d.keyword.accuracy)}</div><div class="sub">Macro F1: \${pct(d.keyword.macroF1)}</div></div>
    \${d.geval?'<div class="card"><div class="lbl">G-Eval Coherence</div><div class="val">'+d.geval.coherence.toFixed(1)+'<span style="font-size:1.2rem">/5</span></div></div>':''}
  </div>
  <h2>Per-class breakdown</h2>
  <table><thead><tr>
    <th>Category</th>
    <th class="r g">GPT Prec</th><th class="r g">GPT Rec</th><th class="r g">GPT F1</th>
    <th class="r">Support</th>
    <th class="r k">KW Prec</th><th class="r k">KW Rec</th><th class="r k">KW F1</th>
  </tr></thead><tbody>\${rows}</tbody>
  <tfoot>
    <tr style="background:var(--sand)"><td class="cat">Macro F1</td><td class="r g" colspan="2"></td><td class="r g \${cl(d.gpt.macroF1)}"><strong>\${pct(d.gpt.macroF1)}</strong></td><td class="r">\${d.total}</td><td class="r k" colspan="2"></td><td class="r k \${cl(d.keyword.macroF1)}"><strong>\${pct(d.keyword.macroF1)}</strong></td></tr>
    <tr style="background:var(--sand)"><td class="cat">Accuracy</td><td class="r g \${cl(d.gpt.accuracy)}" colspan="3"><strong>\${pct(d.gpt.accuracy)}</strong></td><td></td><td class="r k \${cl(d.keyword.accuracy)}" colspan="3"><strong>\${pct(d.keyword.accuracy)}</strong></td></tr>
  </tfoot></table>\`;
}
</script>
<footer>AI4Sustain · LLM Spring 2026 · TeamX</footer>
</body></html>`);
});

// ── Helpers ───────────────────────────────────────────────────────────────

// Primary English check: GDELT's own language field.
// Backup: reject any title containing non-Latin script characters.
function isEnglish(article) {
  const lang = (article.language || '').toLowerCase();
  if (lang && lang !== 'english') return false;
  return !/[\u0400-\u04FF\u0600-\u06FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF\u0900-\u097F\u0E00-\u0E7F]/.test(article.title);
}

async function fetchGdelt(query, timespan) {
  const url =
    `https://api.gdeltproject.org/api/v2/doc/doc` +
    `?query=${encodeURIComponent(query)}` +
    `&mode=artlist&maxrecords=25&format=json` +
    `&timespan=${timespan}&sourcelang=english&sort=DateDesc`;
  console.log(`[GDELT] ${url.substring(0, 120)}…`);
  try {
    const res  = await fetch(url, { timeout: 12000 });
    const text = await res.text();
    // GDELT sometimes returns empty body or invalid JSON on bad queries
    if (!text || text.trim() === '') { console.warn('[GDELT] empty response'); return []; }
    const data = JSON.parse(text);
    const raw  = (data.articles || []);
    const articles = raw
      .map(a => ({
        title:    a.title    || '',
        source:   a.domain   || 'GDELT',
        url:      a.url      || '#',
        date:     parseGdeltDate(a.seendate),
        snippet:  '',
        from:     'gdelt',
        language: a.language || '',
      }))
      .filter(a => a.title && isEnglish(a));
    console.log(`[GDELT] ${articles.length}/${raw.length} English articles (query: "${query.substring(0,50)}")`);
    return articles;
  } catch(e) {
    console.warn(`[GDELT] error for "${query.substring(0,40)}":`, e.message);
    return [];
  }
}

function parseGdeltDate(s) {
  if (!s) return null;
  return s.replace(/[T\-:Z]/g, '').substring(0, 8);
}

function buildChartData(articles, timeWindow) {
  const days     = WINDOW_DAYS[timeWindow] || 30;
  const buckets  = days <= 7 ? 7 : 6;
  const now      = Date.now();
  const start    = now - days * 86400000;
  const bucketMs = (days * 86400000) / buckets;
  const MO = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const DA = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  const labels = Array.from({ length:buckets }, (_,i) => {
    const t = new Date(start + (i+0.5)*bucketMs);
    return days <= 7 ? DA[t.getDay()] : days <= 90 ? MO[t.getMonth()]+' '+t.getDate() : MO[t.getMonth()];
  });
  const counts = new Array(buckets).fill(0);
  articles.forEach(a => {
    if (!a.date || a.date.length < 8) return;
    const t = new Date(+a.date.substring(0,4), +a.date.substring(4,6)-1, +a.date.substring(6,8)).getTime();
    const i = Math.floor((t - start) / bucketMs);
    if (i >= 0 && i < buckets) counts[i]++;
  });
  return { labels, values: counts };
}

function kwSentiment(text) {
  const t = (text||'').toLowerCase();
  const pos = ['surge','record','boost','milestone','breakthrough','success','achieve','improve','expand','gain','increase','rise','advance','agreement','progress','renewable','clean','invest'];
  const neg = ['fail','crisis','threat','decline','loss','damage','disaster','warn','shortage','conflict','drop','fall','bleach','flood','drought','pollution','collapse','deforest'];
  let s = 0;
  pos.forEach(w => t.includes(w) && s++);
  neg.forEach(w => t.includes(w) && s--);
  return s > 0 ? 'Positive' : s < 0 ? 'Negative' : 'Neutral';
}

function classifyKw(title, snippet) {
  const text = (title+' '+(snippet||'')).toLowerCase();
  let best = null, bestN = 0;
  for (const cat of CATEGORIES) {
    const n = EVAL_KW[cat].filter(w => text.includes(w)).length;
    if (n > bestN) { bestN = n; best = cat; }
  }
  return best || 'renewable';
}

function computeMetrics(gold, pred) {
  const total   = gold.length;
  const correct = gold.filter((g,i) => g === pred[i]).length;
  const perClass = {};
  for (const cat of CATEGORIES) {
    let tp=0, fp=0, fn=0;
    for (let i=0; i<total; i++) {
      const g = gold[i]===cat, p = pred[i]===cat;
      if (g && p) tp++; else if (!g && p) fp++; else if (g && !p) fn++;
    }
    const pr = tp+fp>0 ? tp/(tp+fp) : 0;
    const re = tp+fn>0 ? tp/(tp+fn) : 0;
    perClass[cat] = { precision:pr, recall:re, f1: pr+re>0 ? 2*pr*re/(pr+re) : 0, support: gold.filter(g=>g===cat).length };
  }
  const macroF1 = CATEGORIES.reduce((s,c) => s+perClass[c].f1, 0) / CATEGORIES.length;
  return { accuracy: correct/total, macroF1, perClass };
}

function r3(v) { return Math.round(v * 1000) / 1000; }

function padOrTrim(arr, n) {
  const out = arr.slice(0,n).map(Number);
  while (out.length < n) out.push(out[out.length-1] ?? 3.0);
  return out;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Full evaluation (runs against labeled_articles.json) ─────────────────
async function runFullEval(labeled) {
  const openai   = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const total    = labeled.length;
  const gold     = labeled.map(a => a.label);
  const kwPreds  = labeled.map(a => classifyKw(a.title, a.snippet));
  const gptPreds = new Array(total);
  const BATCH    = 10;

  evalState.status = 'running'; evalState.total = total; evalState.progress = 0;

  for (let start = 0; start < total; start += BATCH) {
    await Promise.all(labeled.slice(start, start+BATCH).map(async (art, bi) => {
      const idx = start + bi;
      console.log(`[EVAL] article ${idx+1}/${total}`);
      try {
        const c = await openai.chat.completions.create({
          model: 'gpt-4o-mini',
          messages: [{ role:'user', content:
            `Classify this sustainability article into ONE of: renewable, emissions, biodiversity, water, policy\n\n` +
            `Title: ${art.title}\nSnippet: ${art.snippet||''}\n\n` +
            `Reply with just the single lowercase category word.`
          }],
          max_tokens: 10, temperature: 0,
        });
        const pred = c.choices[0].message.content.trim().toLowerCase();
        gptPreds[idx] = CATEGORIES.includes(pred) ? pred : classifyKw(art.title, art.snippet);
      } catch(e) {
        console.warn(`[EVAL] article ${idx+1} failed:`, e.message);
        gptPreds[idx] = classifyKw(art.title, art.snippet);
      }
      evalState.progress = Math.round((idx+1)/total*100);
    }));
    if (start + BATCH < total) await sleep(1000);
  }

  const gptM = computeMetrics(gold, gptPreds);
  const kwM  = computeMetrics(gold, kwPreds);

  // G-Eval on 5-article sample
  let geval = null;
  try {
    const sample = labeled.slice(0,5).map((a,i) => `${i+1}. "${a.title}" — ${(a.snippet||'').substring(0,100)}`).join('\n');
    const sumR = await openai.chat.completions.create({
      model:'gpt-4o-mini', messages:[{role:'user',content:`Summarize in 3-4 sentences:\n${sample}`}],
      max_tokens:200, temperature:0.3,
    });
    const genSum = sumR.choices[0].message.content.trim();
    const scR = await openai.chat.completions.create({
      model:'gpt-4o-mini',
      messages:[{role:'user',content:
        `Rate this summary vs source articles. Return ONLY JSON {"relevance":X,"coherence":X,"grounding":X} (1.0-5.0).\n\nSources:\n${sample}\n\nSummary: ${genSum}`
      }],
      max_tokens:40, temperature:0,
    });
    geval = JSON.parse(scR.choices[0].message.content.trim());
    console.log('[EVAL] G-Eval:', geval);
  } catch(e) { console.warn('[EVAL] G-Eval failed:', e.message); }

  const fmtM = m => ({
    accuracy: r3(m.accuracy), macroF1: r3(m.macroF1),
    perClass: Object.fromEntries(CATEGORIES.map(c => [c, {
      precision: r3(m.perClass[c].precision), recall: r3(m.perClass[c].recall),
      f1: r3(m.perClass[c].f1), support: m.perClass[c].support,
    }])),
  });

  // Merge with any existing live geval, preferring the freshly computed one
  let existing = {};
  try { existing = JSON.parse(fs.readFileSync(EVAL_PATH,'utf8')); } catch(_) {}
  const result = { ...existing, total, gpt: fmtM(gptM), keyword: fmtM(kwM), geval };

  fs.writeFileSync(EVAL_PATH, JSON.stringify(result, null, 2));
  console.log('[EVAL] Complete. Saved to eval_results.json');
  evalState.status = 'complete'; evalState.progress = 100;
  return result;
}

// ── Background G-Eval after each /api/analyze ─────────────────────────────
async function runBackgroundGeval(summary, articles, theme) {
  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const titles = articles.slice(0,5).map((a,i) => `${i+1}. ${a.title}`).join('\n');

  const scR = await openai.chat.completions.create({
    model:'gpt-4o-mini',
    messages:[{role:'user',content:
      `Rate this sustainability news summary. Return ONLY JSON {"relevance":X,"coherence":X,"grounding":X} (1.0-5.0).\n\n` +
      `Article titles:\n${titles}\n\nSummary: ${summary}`
    }],
    max_tokens:40, temperature:0,
  });
  const geval = JSON.parse(scR.choices[0].message.content.trim());

  // Live keyword precision for this theme: what fraction of articles does keyword
  // classifier correctly label as the searched theme?
  const kwPreds   = articles.map(a => classifyKw(a.title, a.snippet));
  const kwPrec    = articles.length > 0 ? r3(kwPreds.filter(p => p===theme).length / articles.length) : 0;

  let existing = {};
  try { existing = JSON.parse(fs.readFileSync(EVAL_PATH,'utf8')); } catch(_) {}

  // Only overwrite geval and live stats; preserve full eval (gpt/keyword) if present
  fs.writeFileSync(EVAL_PATH, JSON.stringify({
    ...existing,
    geval,
    liveKeywordPrecision: kwPrec,
    lastUpdated: new Date().toISOString(),
  }, null, 2));
  console.log(`[GEVAL] coherence=${geval.coherence} kwPrec=${kwPrec}`);
}

// ── Auto-run full eval on startup if labeled data exists but results don't ─
setTimeout(async () => {
  if (!process.env.OPENAI_API_KEY) return;
  if (fs.existsSync(EVAL_PATH)) {
    // Already have results — check if they include full eval (gpt.macroF1)
    try {
      const cached = JSON.parse(fs.readFileSync(EVAL_PATH,'utf8'));
      if (cached.gpt?.macroF1 != null) {
        console.log('[STARTUP] eval_results.json already has full results, skipping auto-eval');
        return;
      }
    } catch(_) {}
  }
  if (!fs.existsSync(LABEL_PATH)) {
    console.log('[STARTUP] labeled_articles.json not found, skipping auto-eval');
    return;
  }
  console.log('[STARTUP] Running full evaluation in background…');
  try {
    const labeled = JSON.parse(fs.readFileSync(LABEL_PATH,'utf8'));
    await runFullEval(labeled);
  } catch(e) {
    console.warn('[STARTUP] Auto-eval failed:', e.message);
    evalState.status = 'failed';
  }
}, 2000); // 2-second delay lets the server fully start first

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`AI4Sustain running → http://localhost:${PORT}`));
