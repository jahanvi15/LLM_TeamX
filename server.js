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

const THEME_KEYWORDS = {
  renewable:    'renewable energy solar wind power',
  emissions:    'carbon emissions CO2 greenhouse gas climate',
  biodiversity: 'biodiversity wildlife deforestation habitat nature',
  water:        'water scarcity drought oceans flooding rivers',
  policy:       'climate policy COP Paris agreement legislation carbon tax'
};

const REGION_TERMS = {
  global:   '',
  europe:   ' Europe',
  asia:     ' Asia Pacific',
  americas: ' Americas',
  africa:   ' Africa'
};

const GDELT_TIMESPAN = { '7d':'7d', '30d':'30d', '90d':'90d', '1y':'365d' };
const WINDOW_DAYS    = { '7d':7, '30d':30, '90d':90, '1y':365 };

app.post('/api/analyze', async (req, res) => {
  const { theme = 'renewable', region = 'global', timeWindow = '30d' } = req.body;

  const keyword   = THEME_KEYWORDS[theme] || 'climate environment';
  const regionStr = REGION_TERMS[region] || '';
  const query     = (keyword + regionStr).trim();
  const timespan  = GDELT_TIMESPAN[timeWindow] || '30d';
  const days      = WINDOW_DAYS[timeWindow] || 30;

  let gdeltArticles = [];
  let newsArticles  = [];

  // ── GDELT (free, no key) ──────────────────────────────────────────────
  try {
    const gdeltUrl =
      `https://api.gdeltproject.org/api/v2/doc/doc` +
      `?query=${encodeURIComponent(query)}` +
      `&mode=artlist&maxrecords=20&format=json` +
      `&timespan=${timespan}&sourcelang=english`;
    const gdeltRes  = await fetch(gdeltUrl, { timeout: 10000 });
    const gdeltData = await gdeltRes.json();
    gdeltArticles = (gdeltData.articles || []).map(a => ({
      title:   a.title   || '',
      source:  a.domain  || 'GDELT',
      url:     a.url     || '#',
      date:    parseGdeltDate(a.seendate),
      snippet: '',
      from:    'gdelt'
    })).filter(a => a.title);
  } catch (e) {
    console.warn('GDELT fetch failed:', e.message);
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

  const allArticles = [...gdeltArticles, ...newsArticles];

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

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`AI4Sustain running → http://localhost:${PORT}`));
