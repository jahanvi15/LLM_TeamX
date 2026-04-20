const API = "http://localhost:5000/api";

// ── helpers ──────────────────────────────────────────────────────────────────
function sentClass(label){ return label==="Positive"?"green":label==="Negative"?"red":"amber"; }
function sentEmoji(label){ return label==="Positive"?"😊":label==="Negative"?"😟":"😐"; }
const themeLabels = {
  renewable:    "Renewable Energy",
  emissions:    "Emissions & Carbon",
  biodiversity: "Biodiversity",
  water:        "Water & Oceans",
  policy:       "Climate Policy"
};

// ── F1 score: read from the eval card in the DOM, never hardcoded ─────────────
function getF1Score() {
  // Reads the DeBERTa prog-val from the Classification F1 eval card
  const evalCards = document.querySelectorAll(".eval-card");
  for (const card of evalCards) {
    const heading = card.querySelector("h3");
    if (heading && heading.textContent.includes("Classification F1")) {
      const rows = card.querySelectorAll(".progress-row");
      for (const row of rows) {
        const label = row.querySelector(".prog-label");
        if (label && label.textContent.trim() === "DeBERTa") {
          const val = row.querySelector(".prog-val");
          if (val) return val.textContent.trim();
        }
      }
    }
  }
  return "—";
}

// ── status check ─────────────────────────────────────────────────────────────
async function checkHealth(){
  const dot = document.getElementById("dotApi");
  const msg = document.getElementById("statusApi");
  try {
    const r = await fetch(`${API}/health`, {signal: AbortSignal.timeout(5000)});
    const d = await r.json();
    dot.className = "status-dot ok";
    msg.textContent = `API connected`;
    document.getElementById("statusDb").style.display = "";
    document.getElementById("dotDb").className = "status-dot ok";
    document.getElementById("statusDbText").textContent = `${d.articles} articles · ${d.embedded} embedded`;
    loadStats(d);
  } catch(e) {
    dot.className = "status-dot err";
    msg.textContent = "API offline — run: python backend/api.py";
  }
}

async function loadStats(healthData){
  try {
    const r = await fetch(`${API}/stats`);
    const d = await r.json();
    document.getElementById("heroTotal").textContent    = d.total;
    document.getElementById("heroEmbedded").textContent = d.embedded;
    const sent = d.avg_sentiment > 0.1 ? "↑ Positive" : d.avg_sentiment < -0.1 ? "↓ Negative" : "≈ Neutral";
    document.getElementById("heroSent").textContent     = sent;

    // F1 score: pulled from the eval card in the DOM — not hardcoded
    document.getElementById("heroF1Score").textContent = getF1Score();

    // snippet
    const byTheme = d.by_theme || {};
    const lines = Object.entries(byTheme).map(([k,v])=>`${themeLabels[k]||k}: ${v}`).join(" · ");
    document.getElementById("heroSummarySnippet").textContent = lines || "Run fetch_articles.py to populate the database.";
    // hero sparkline from trends
    const tr = await fetch(`${API}/trends?theme=renewable`);
    const td = await tr.json();
    if(td.counts){ buildSparkline(td.counts); }
    drawChart(td.labels||[], td.counts||[], td.sentiment||[]);
  } catch(e){}
}

// ── sparkline ─────────────────────────────────────────────────────────────────
function buildSparkline(counts){
  const el = document.getElementById("sparkline");
  el.innerHTML = "";
  const max = Math.max(...counts, 1);
  counts.forEach((v,i)=>{
    const b = document.createElement("div");
    b.className = "bar" + (i===counts.length-1?" active":"");
    b.style.height = Math.round((v/max)*100)+"%";
    el.appendChild(b);
  });
}

// ── canvas chart ──────────────────────────────────────────────────────────────
function drawChart(labels, values, sentiment){
  const canvas = document.getElementById("trendChart");
  const ctx    = canvas.getContext("2d");
  const dpr    = window.devicePixelRatio||1;
  canvas.width  = canvas.offsetWidth * dpr;
  canvas.height = 160 * dpr;
  ctx.scale(dpr, dpr);
  const W = canvas.offsetWidth, H = 160;
  const pad = {t:10,r:10,b:30,l:32};
  const cw  = W - pad.l - pad.r;
  const ch  = H - pad.t - pad.b;
  ctx.clearRect(0,0,W,H);

  const maxV = Math.max(...values, 1)*1.2;
  const maxS = Math.max(...sentiment.map(Math.abs), .1)*1.5;

  // grid
  ctx.strokeStyle = "#d4e8da"; ctx.lineWidth = 1;
  for(let i=0;i<=4;i++){
    const y = pad.t + ch - (i/4)*ch;
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(pad.l+cw,y); ctx.stroke();
  }

  // bars
  const bw = cw/Math.max(labels.length,1)*.6;
  values.forEach((v,i)=>{
    const x  = pad.l + (i+.5)*(cw/Math.max(labels.length,1)) - bw/2;
    const bh = (v/maxV)*ch;
    ctx.fillStyle = "#2d6a4f99";
    ctx.beginPath();
    ctx.roundRect(x, pad.t+ch-bh, bw, bh, 3);
    ctx.fill();
  });

  // sentiment line
  if(sentiment.length){
    ctx.strokeStyle="#c4933f"; ctx.lineWidth=2;
    ctx.beginPath();
    sentiment.forEach((s,i)=>{
      const x = pad.l+(i+.5)*(cw/Math.max(labels.length,1));
      const y = pad.t+ch - ((s+maxS)/(2*maxS))*ch;
      i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    });
    ctx.stroke();
    ctx.fillStyle="#c4933f";
    sentiment.forEach((s,i)=>{
      const x = pad.l+(i+.5)*(cw/Math.max(labels.length,1));
      const y = pad.t+ch - ((s+maxS)/(2*maxS))*ch;
      ctx.beginPath(); ctx.arc(x,y,3,0,Math.PI*2); ctx.fill();
    });
  }

  // labels
  ctx.fillStyle="#6b7c70"; ctx.font="11px 'DM Sans',sans-serif"; ctx.textAlign="center";
  labels.forEach((l,i)=>{
    ctx.fillText(l, pad.l+(i+.5)*(cw/Math.max(labels.length,1)), H-6);
  });
}

// ── run analysis ──────────────────────────────────────────────────────────────
async function runAnalysis(){
  const btn    = document.getElementById("runBtn");
  const panel  = document.getElementById("resultsPanel");
  const theme  = document.getElementById("themeSelect").value;
  const region = document.getElementById("regionSelect").value;
  const tw     = document.getElementById("timeSelect").value;
  const nlq    = document.getElementById("nlQuery").value;

  btn.disabled = true;
  btn.innerHTML = '<span style="display:inline-block;animation:spin .8s linear infinite">⟳</span> Fetching from GDELT…';

  panel.innerHTML = `
    <div class="result-card" style="opacity:1;transform:none">
      <div class="skeleton" style="width:40%;height:12px"></div>
      <div class="skeleton" style="width:90%;height:16px;margin-top:.6rem"></div>
      <div class="skeleton" style="width:75%"></div>
      <div class="skeleton" style="width:80%"></div>
    </div>
    <div class="result-card" style="opacity:1;transform:none">
      <div class="skeleton" style="width:35%;height:12px"></div>
      <div class="skeleton" style="width:85%;height:16px;margin-top:.6rem"></div>
      <div class="skeleton" style="width:70%"></div>
    </div>`;

  // fetch trends for chart
  try {
    const tr = await fetch(`${API}/trends?theme=${theme}&region=${region==="global"?"":region}`);
    const td = await tr.json();
    document.getElementById("chartTitle").textContent = `${themeLabels[theme]||theme} — Weekly Volume`;
    drawChart(td.labels||[], td.counts||[], td.sentiment||[]);
  } catch(e){}

  // RAG query
  try {
    btn.innerHTML = '<span style="display:inline-block;animation:spin .8s linear infinite">⟳</span> Generating RAG summary…';
    const resp = await fetch(`${API}/query`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({theme, region, time_window: tw, nl_query: nlq})
    });
    const data = await resp.json();

    let html = "";

    // summary card
    html += `<div class="result-card" style="animation-delay:0s;border-left:4px solid var(--sage)">
      <div class="result-meta">
        <span class="meta-chip green">RAG Summary</span>
        <span class="meta-chip">${themeLabels[theme]||theme}</span>
        <span class="meta-chip blue">GPT-3.5-turbo</span>
      </div>
      <div class="result-title">AI-Generated Trend Summary</div>
      <div class="result-summary" id="typedSummary"></div>
    </div>`;

    // article cards
    (data.articles||[]).forEach((a,i)=>{
      const sl = a.sentiment_label||"Neutral";
      html += `<div class="result-card" style="animation-delay:${(i+1)*.08}s">
        <div class="result-meta">
          <span class="meta-chip ${sentClass(sl)}">${sentEmoji(sl)} ${sl}</span>
          <span class="meta-chip">${themeLabels[a.theme]||a.theme||"—"}</span>
          <span class="meta-chip">${a.region||"Global"}</span>
        </div>
        <div class="result-title">${a.title}</div>
        <div class="result-summary">${(a.abstract||"").slice(0,280)}${a.abstract&&a.abstract.length>280?"…":""}</div>
        <div class="result-footer">
          <a class="source-link" href="${a.url||"#"}" target="_blank">📰 ${a.source||"Source"}</a>
          <div class="confidence">
            <span>${new Date(a.published||"").toLocaleDateString("en-GB",{month:"short",day:"numeric"})}</span>
          </div>
        </div>
      </div>`;
    });

    panel.innerHTML = html;

    // typewriter
    const el   = document.getElementById("typedSummary");
    const text = data.summary || "No summary generated.";
    let idx = 0;
    el.classList.add("typing-cursor");
    const iv = setInterval(()=>{
      el.textContent += text[idx++];
      if(idx>=text.length){ clearInterval(iv); el.classList.remove("typing-cursor"); }
    }, 16);

  } catch(e) {
    panel.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠️</div><p>API error: ${e.message}.<br/>Make sure <code style="background:var(--sand);padding:.1rem .4rem;border-radius:4px">python backend/api.py</code> is running.</p></div>`;
  }

  btn.disabled = false;
  btn.innerHTML = '<span>▶</span> Run Analysis';
}

// ── G-Eval / F1 progress bars on scroll ──────────────────────────────────────
// Fixed: threshold lowered to 0.15 so bars animate on all viewport sizes.
// unobserve() added so each card only triggers once (no re-fire jitter).
// On-load pass handles any eval cards already visible without scrolling.
function triggerProgBars(card) {
  card.querySelectorAll(".prog-fill").forEach(f => { f.style.width = f.dataset.val + "%"; });
}

const obs = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      triggerProgBars(e.target);
      obs.unobserve(e.target);
    }
  });
}, { threshold: 0.15 });

document.querySelectorAll(".eval-card").forEach(c => obs.observe(c));

// ── on load ───────────────────────────────────────────────────────────────────
window.addEventListener("load", () => {
  checkHealth();
  // placeholder sparkline
  buildSparkline([3,5,4,8,6,10,9,12,8,14,11,15]);
  setTimeout(() => drawChart(
    ["Nov 4","Nov 11","Nov 18","Nov 25","Dec 2","Dec 9"],
    [0,0,0,0,0,0],[0,0,0,0,0,0]
  ), 200);

  // Animate bars for eval cards already in viewport on load
  document.querySelectorAll(".eval-card").forEach(c => {
    const rect = c.getBoundingClientRect();
    if (rect.top < window.innerHeight && rect.bottom > 0) {
      triggerProgBars(c);
      obs.unobserve(c);
    }
  });
});

window.addEventListener("resize", () => {
  const theme = document.getElementById("themeSelect").value;
  fetch(`${API}/trends?theme=${theme}`).then(r => r.json())
    .then(d => drawChart(d.labels||[], d.counts||[], d.sentiment||[]))
    .catch(() => {});
});
