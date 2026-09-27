const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let mode = "clone";
let voices = [];
const selected = new Set(JSON.parse(localStorage.getItem("ov.selected") || "[]"));
const gpuHist = [];
const jobEls = new Map();

function fmtMB(mb) { return mb == null ? "–" : mb >= 1024 ? (mb / 1024).toFixed(2) + " GB" : Math.round(mb) + " MB"; }

function renderGpu(g) {
  if (!g || g.util == null) return;
  $("#gUtil").textContent = g.util + " %";
  $("#gMem").textContent = `${fmtMB(g.mem_used)} / ${fmtMB(g.mem_total)}`;
  $("#gTorch").textContent = g.torch_alloc != null ? `${fmtMB(g.torch_alloc)} (res ${fmtMB(g.torch_reserved)})` : "–";
  $("#gTemp").textContent = g.temp != null ? g.temp + " °C" : "–";
  $("#gPower").textContent = g.power != null ? `${g.power.toFixed(0)} / ${g.power_limit?.toFixed(0) ?? "?"} W` : "–";
  $("#gClock").textContent = g.clock_sm != null ? g.clock_sm + " MHz" : "–";
  const last = gpuHist[gpuHist.length - 1];
  if (!last || last.t !== g.t) { gpuHist.push(g); if (gpuHist.length > 240) gpuHist.shift(); drawChart(); }
}

function drawChart() {
  const c = $("#gpuChart"), dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = 90;
  c.width = w * dpr; c.height = h * dpr;
  const ctx = c.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.strokeStyle = "#2a303b"; ctx.lineWidth = 1;
  for (const y of [0.25, 0.5, 0.75]) { ctx.beginPath(); ctx.moveTo(0, h * y); ctx.lineTo(w, h * y); ctx.stroke(); }
  const n = 240, step = w / (n - 1), off = n - gpuHist.length;
  const line = (fn, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.beginPath();
    gpuHist.forEach((g, i) => { const x = (off + i) * step, y = h - 2 - (fn(g) / 100) * (h - 4); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  };
  const css = getComputedStyle(document.documentElement);
  line((g) => (g.mem_used / g.mem_total) * 100, css.getPropertyValue("--accent2"));
  line((g) => g.util, css.getPropertyValue("--accent"));
}

function renderModel(m) {
  const el = $("#modelStatus");
  el.className = "pill " + ({ pronto: "ok", carregando: "warn", erro: "err" }[m.status] || "");
  let t = `${m.id} · ${m.status}`;
  if (m.status === "pronto") t += ` · ${m.device} ${m.dtype} · torch ${m.torch} · ASR ${m.asr}`;
  if (m.status === "carregando") t += " (primeira vez baixa ~GBs do HuggingFace)";
  if (m.error) t += " · " + m.error;
  el.textContent = t;
  $("#gpuName").textContent = m.gpu_name || "sem GPU NVIDIA detectada";
  $("#queueInfo").textContent = m.queue ? `${m.queue} na fila` : "";
}

const FMT_LABEL = { wav: "WAV", mp3: "MP3", opus: "OGG", vorbis: "OGG", flac: "FLAC" };
let lastJobs = [];
try { const f = localStorage.getItem("ov.fmt"); if (f) $("#exportFmt").value = f; } catch {}
$("#exportFmt").onchange = (e) => { try { localStorage.setItem("ov.fmt", e.target.value); } catch {} renderJobs(lastJobs); };

const statusCls = { done: "ok", running: "run", queued: "", error: "err", cancelled: "warn" };

function renderJobs(jobs) {
  lastJobs = jobs;
  const box = $("#jobs");
  const ids = new Set(jobs.map((j) => j.id));
  for (const [id, el] of jobEls) if (!ids.has(id)) { el.remove(); jobEls.delete(id); }
  if (!jobs.length) { box.innerHTML = '<div class="empty">Nenhum job ainda.</div>'; return; }
  box.querySelector(".empty")?.remove();

  jobs.forEach((j, idx) => {
    let el = jobEls.get(j.id);
    if (!el) {
      el = $("#jobTpl").content.firstElementChild.cloneNode(true);
      el.querySelector(".cancel").onclick = () => fetch(`/api/jobs/${j.id}/cancel`, { method: "POST" });
      el.querySelector(".del").onclick = () => fetch(`/api/jobs/${j.id}`, { method: "DELETE" });
      jobEls.set(j.id, el);
    }
    if (box.children[idx] !== el) box.insertBefore(el, box.children[idx] || null);

    const p = j.params || {};
    const who = p.mode === "clone" ? `🎙 ${p.voice_name || p.voice_id}` : p.mode === "design" ? `🎨 ${p.instruct || "design"}` : "🤖 auto";
    el.className = "job " + j.status;
    el.querySelector(".job-title").textContent = who;
    const st = el.querySelector(".job-status");
    st.className = "job-status pill " + (statusCls[j.status] || "");
    st.textContent = j.phase;
    el.querySelector(".job-text").textContent = p.text;
    el.querySelector(".job-text").title = p.text;
    el.querySelector(".fill").style.width = j.progress + "%";
    el.querySelector(".pct").textContent = j.status === "running"
      ? `${j.progress}% · passo ${j.step}/${j.num_step}` + (j.n_chunks > 1 ? ` · chunk ${Math.min(j.chunk + 1, j.n_chunks)}/${j.n_chunks}` : "")
      : j.status === "done" ? "100%" : "";

    const m = [];
    if (j.status === "running" || j.status === "done") {
      m.push(`<span>⏱ <b>${j.elapsed.toFixed(1)}s</b></span>`);
      if (j.it_s) m.push(`<span><b>${j.it_s}</b> it/s</span>`);
      if (j.tok_s) m.push(`<span title="tokens de áudio decididos por segundo (frames × 8 codebooks)"><b>${Math.round(j.tok_s).toLocaleString()}</b> audio tok/s</span>`);
      if (j.ctx_tok_s) m.push(`<span title="posições processadas pelo transformer por segundo (cond + uncond, CFG)"><b>${Math.round(j.ctx_tok_s).toLocaleString()}</b> ctx tok/s</span>`);
    }
    if (j.status === "done") {
      m.push(`<span>🔊 <b>${j.audio_seconds}s</b> de áudio</span>`);
      if (j.rtf != null) m.push(`<span>RTF <b>${j.rtf}</b> (${(1 / j.rtf).toFixed(1)}× tempo real)</span>`);
      if (j.peak_vram) m.push(`<span>pico VRAM <b>${fmtMB(j.peak_vram)}</b></span>`);
    }
    m.push(`<span>steps ${p.num_step} · cfg ${p.guidance_scale}${p.language ? " · " + esc(p.language) : ""}</span>`);
    el.querySelector(".job-metrics").innerHTML = m.join("");

    const audio = el.querySelector("audio"), dl = el.querySelector(".dl");
    if (j.output) {
      const src = `/api/outputs/${j.output}`;
      if (audio.getAttribute("src") !== src) audio.src = src;
      const fmt = $("#exportFmt").value;
      dl.href = `${src}?fmt=${fmt}&download=1`;
      dl.textContent = `baixar ${FMT_LABEL[fmt]}`;
      audio.hidden = dl.hidden = false;
    } else { audio.hidden = dl.hidden = true; }
    el.querySelector(".cancel").hidden = !(j.status === "running" || j.status === "queued");
    el.querySelector(".del").hidden = j.status === "running" || j.status === "queued";
    el.querySelector(".job-err").textContent = j.error || "";
    el.querySelector(".job-err").title = j.ref_text_used ? "ref_text: " + j.ref_text_used : "";
  });
}

async function loadVoices() {
  voices = await (await fetch("/api/voices")).json();
  for (const id of [...selected]) if (!voices.find((v) => v.id === id)) selected.delete(id);
  const ul = $("#voiceList");
  if (!voices.length) { ul.innerHTML = '<li class="empty" style="display:block">Nenhuma voz ainda — envie um áudio.</li>'; return; }
  ul.innerHTML = voices.map((v) => `
    <li data-id="${v.id}">
      <input type="checkbox" ${selected.has(v.id) ? "checked" : ""}>
      <div>
        <div class="voice-name">${esc(v.name)}</div>
        <div class="voice-meta">${v.duration}s${v.duration > 20 ? " ⚠ longo (ideal 3–10s)" : ""} · ${v.cached ? "prompt em cache" : "ainda não codificada"}${v.ref_text_auto ? " · transcrição automática" : ""}</div>
        <div class="voice-text" title="clique para editar a transcrição">${v.ref_text ? "“" + esc(v.ref_text) + "”" : "(sem transcrição — o Parakeet v3 transcreve na 1ª geração; clique para escrever)"}</div>
        <audio controls preload="none" src="/api/voices/${v.id}/audio"></audio>
      </div>
      <button class="btn ghost small">✕</button>
    </li>`).join("");
  ul.querySelectorAll("li[data-id]").forEach((li) => {
    const id = li.dataset.id;
    li.querySelector("input").onchange = (e) => { e.target.checked ? selected.add(id) : selected.delete(id); saveSel(); };
    li.querySelector("button").onclick = async () => {
      if (!confirm("Remover esta voz?")) return;
      await fetch(`/api/voices/${id}`, { method: "DELETE" }); selected.delete(id); saveSel(); loadVoices();
    };
    li.querySelector(".voice-text").onclick = async () => {
      const v = voices.find((x) => x.id === id);
      const t = prompt("Transcrição exata do áudio de referência (vazio = transcrição automática):", v.ref_text || "");
      if (t === null) return;
      await fetch(`/api/voices/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ref_text: t }) });
      loadVoices();
    };
  });
}
function saveSel() { localStorage.setItem("ov.selected", JSON.stringify([...selected])); }

let pendingFiles = [];
function addPending(files) {
  for (const f of files) pendingFiles.push({ file: f, name: f.name.replace(/\.[^.]+$/, ""), text: "", url: URL.createObjectURL(f) });
  renderPending();
}
function renderPending() {
  const box = $("#pending");
  if (!pendingFiles.length) { box.innerHTML = ""; return; }
  box.innerHTML = pendingFiles.map((p, i) => `
    <div class="pending-item" data-i="${i}">
      <div class="row">
        <div><label>Nome</label><input class="pn" value="${esc(p.name)}"></div>
        <div><label>Transcrição (opcional — melhora a clonagem)</label><input class="pt" value="${esc(p.text)}" placeholder="vazio = transcrição automática (Parakeet v3)"></div>
      </div>
      <audio controls src="${p.url}"></audio>
    </div>`).join("") +
    `<div class="actions"><button class="btn primary" id="uploadBtn">Enviar ${pendingFiles.length} voz(es)</button><button class="btn ghost" id="clearPending">descartar</button><span class="hint" id="upMsg"></span></div>`;
  box.querySelectorAll(".pending-item").forEach((el) => {
    const p = pendingFiles[+el.dataset.i];
    el.querySelector(".pn").oninput = (e) => (p.name = e.target.value);
    el.querySelector(".pt").oninput = (e) => (p.text = e.target.value);
  });
  $("#clearPending").onclick = () => { pendingFiles = []; renderPending(); };
  $("#uploadBtn").onclick = uploadPending;
}
async function uploadPending() {
  const fd = new FormData();
  pendingFiles.forEach((p) => fd.append("files", p.file, p.file.name));
  fd.append("names", JSON.stringify(pendingFiles.map((p) => p.name)));
  fd.append("ref_texts", JSON.stringify(pendingFiles.map((p) => p.text)));
  $("#uploadBtn").disabled = true; $("#upMsg").textContent = "enviando…";
  const r = await fetch("/api/voices", { method: "POST", body: fd });
  if (!r.ok) { $("#upMsg").textContent = "erro: " + (await r.text()); $("#uploadBtn").disabled = false; return; }
  (await r.json()).forEach((v) => selected.add(v.id)); saveSel();
  pendingFiles = []; renderPending(); loadVoices();
}

const drop = $("#drop");
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); addPending(e.dataTransfer.files); };
$("#fileInput").onchange = (e) => { addPending(e.target.files); e.target.value = ""; };

let rec = null;
$("#recBtn").onclick = async () => {
  const btn = $("#recBtn");
  if (rec) { rec.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const chunks = [];
    rec = new MediaRecorder(stream);
    rec.ondataavailable = (e) => chunks.push(e.data);
    rec.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunks, { type: rec.mimeType });
      addPending([new File([blob], `gravacao-${new Date().toLocaleTimeString().replace(/:/g, "")}.webm`, { type: rec.mimeType })]);
      rec = null; btn.textContent = "● Gravar microfone"; btn.classList.remove("rec");
    };
    rec.start(); btn.textContent = "■ Parar gravação"; btn.classList.add("rec");
  } catch (e) { alert("Microfone indisponível: " + e.message); }
};

document.querySelectorAll("#modeTabs button").forEach((b) => (b.onclick = () => {
  mode = b.dataset.mode;
  document.querySelectorAll("#modeTabs button").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelector(".mode-clone").hidden = mode !== "clone";
  document.querySelector(".mode-design").hidden = mode !== "design";
}));

const val = (id) => $("#" + id).value;
$("#genBtn").onclick = async () => {
  const body = {
    mode, text: val("text"), language: val("language"), instruct: val("instruct"),
    voice_ids: [...selected],
    num_step: +val("num_step"), guidance_scale: +val("guidance_scale"),
    speed: val("speed") ? +val("speed") : null, duration: val("duration") ? +val("duration") : null,
    seed: val("seed") === "" ? null : +val("seed"),
    t_shift: +val("t_shift"), class_temperature: +val("class_temperature"), position_temperature: +val("position_temperature"),
    denoise: $("#denoise").checked, postprocess: $("#postprocess").checked, normalize_text: $("#normalize_text").checked,
    pause_mode: val("pause_mode"), pause_sentence_ms: +val("pause_sentence_ms"), pause_comma_ms: +val("pause_comma_ms"),
  };
  const r = await fetch("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  $("#genMsg").textContent = r.ok ? `${(await r.json()).length} job(s) na fila` : "erro: " + (await r.json()).detail;
  setTimeout(() => ($("#genMsg").textContent = ""), 4000);
};

const formIds = ["text", "language", "instruct", "num_step", "guidance_scale", "speed", "duration", "seed", "t_shift", "class_temperature", "position_temperature", "pause_mode", "pause_sentence_ms", "pause_comma_ms"];

function updatePauseUI() {
  const split = val("pause_mode") === "split";
  document.querySelectorAll(".pause-opt").forEach((el) => (el.hidden = !split));
  $("#pauseHint").textContent = split
    ? "Cada trecho entre pontuações é gerado à parte e unido com silêncio exato. ; e : usam a média entre vírgula e ponto; linha em branco = 1,75× a pausa de frase. Referência (fala lida): pausas curtas ~100–150 ms, de frase ~500 ms; fala espontânea ~80 / ~430 ms. \"Duração fixa\" é ignorada neste modo."
    : "O modelo decide as pausas dentro de uma duração estimada pelo ritmo da referência. Se ele atropelar a pontuação, tente velocidade 0.85–0.9 ou o modo \"Garantidas\".";
}
$("#pause_mode").addEventListener("change", updatePauseUI);
function saveForm() { localStorage.setItem("ov.form", JSON.stringify(Object.fromEntries(formIds.map((i) => [i, val(i)])))); }
formIds.forEach((i) => $("#" + i).addEventListener("change", saveForm));

(async () => {
  const langs = await (await fetch("/api/languages")).json();
  $("#language").innerHTML += langs.map((l) => `<option>${esc(l)}</option>`).join("");
  try {
    const f = JSON.parse(localStorage.getItem("ov.form") || "{}");
    if (f.pause_sentence_ms === "400" && f.pause_comma_ms === "150") { delete f.pause_sentence_ms; delete f.pause_comma_ms; }
    for (const k in f) if ($("#" + k)) $("#" + k).value = f[k];
  } catch {}
  updatePauseUI();

  const s = await (await fetch("/api/state")).json();
  (s.gpu_history || []).forEach((g) => gpuHist.push(g));
  renderModel(s.model); renderGpu(s.gpu); renderJobs(s.jobs); drawChart();
  loadVoices();

  let lastDone = new Set(s.jobs.filter((j) => j.status === "done").map((j) => j.id));
  const es = new EventSource("/api/stream");
  es.onmessage = (e) => {
    const d = JSON.parse(e.data);
    renderModel(d.model); renderGpu(d.gpu); renderJobs(d.jobs);
    const done = d.jobs.filter((j) => j.status === "done").map((j) => j.id);
    if (done.some((id) => !lastDone.has(id))) loadVoices();
    lastDone = new Set(done);
  };
  es.onerror = () => { $("#modelStatus").className = "pill err"; $("#modelStatus").textContent = "servidor desconectado"; };
  window.onresize = drawChart;
})();
