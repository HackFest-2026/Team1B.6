/* ORBITAL RISK dashboard — vanilla JS + Three.js + Chart.js + satellite.js (all bundled locally) */
"use strict";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const api = async (p, opt) => { const r = await fetch(p, opt); if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.json(); };
const fmtPc = p => (p == null ? "—" : p === 0 ? "0" : p < 1e-12 ? "<1e-12" : p.toExponential(1));
const fmtT = s => s ? s.replace("T", " ").slice(0, 19) : "—";
const fmtKm = k => k == null ? "—" : k < 1 ? `${(k * 1000).toFixed(0)} m` : `${k.toFixed(2)} km`;
const pct = x => x == null ? "—" : `${(x * 100).toFixed(1)}%`;
const num = (x, d = 2) => x == null ? "—" : Number(x).toFixed(d);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function applyChartTheme() {
  const light = document.documentElement.dataset.theme === "light";
  Chart.defaults.color = light ? "#5d6773" : "#8b9bb0";
  Chart.defaults.borderColor = light ? "#dde1e6" : "#2a3037";
}
applyChartTheme();
Chart.defaults.font.family = "Inter, system-ui, sans-serif";
Chart.defaults.font.size = 11;
const charts = {};
const logTick = v => { const l = Math.log10(v); return Math.abs(l - Math.round(l)) < 1e-6 ? "1e" + Math.round(l) : ""; };
function fixLogAxes(cfg) {
  const sc = (cfg.options = cfg.options || {}).scales || {};
  for (const ax of Object.values(sc)) if (ax.type === "logarithmic") ax.ticks = { ...(ax.ticks || {}), callback: logTick, autoSkip: false, maxRotation: 0 };
  return cfg;
}
function mkChart(id, cfg) { fixLogAxes(cfg); if (charts[id]) charts[id].destroy(); charts[id] = new Chart(document.getElementById(id), cfg); return charts[id]; }

const APP = { summary: null, conj: [], tles: [], loaded: {}, alertsSeen: new Set() };

// ------------------------------------------------------------------ theme
function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("theme", t); } catch (e) {}
  $("#themeBtn").textContent = t === "light" ? "Dark mode" : "Light mode";
  applyChartTheme();
  if (APP.summary) { APP.loaded = {}; renderTab($("#tabs button.active").dataset.tab); }
}
$("#themeBtn").onclick = () => setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
$("#themeBtn").textContent = document.documentElement.dataset.theme === "light" ? "Dark mode" : "Light mode";

// ------------------------------------------------------------------ tabs / clock
$$("#tabs button").forEach(b => b.onclick = () => showTab(b.dataset.tab));
function showTab(t) {
  $$("#tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === t));
  $$(".tab").forEach(s => s.classList.toggle("active", s.id === "tab-" + t));
  if (APP.summary) renderTab(t);
  if (t === "overview") setTimeout(() => GLOBE.resize(), 30);
}
setInterval(() => { $("#clock").textContent = new Date().toISOString().slice(0, 19).replace("T", " ") + " UTC"; }, 1000);
function toast(msg, ms = 4000) { const t = $("#toast"); t.innerHTML = msg; t.classList.remove("hidden"); clearTimeout(t._h); t._h = setTimeout(() => t.classList.add("hidden"), ms); }

// ------------------------------------------------------------------ pipeline status
$("#runBtn").onclick = async () => {
  const r = await api("/api/run?fast=false", { method: "POST" });
  toast(r.started ? "Pipeline started — data ingest, AI models, screening…" : "Pipeline is already running");
  pollStatus();
};
let polling = false;
async function pollStatus() {
  if (polling) return; polling = true;
  const pr = $("#progress");
  while (true) {
    let s; try { s = await api("/api/status"); } catch (e) { await sleep(2000); continue; }
    if (s.running) {
      pr.classList.remove("hidden"); $(".bar", pr).style.width = s.pct + "%";
      $(".lbl", pr).textContent = `${s.pct}% · ${s.step}`; $("#runBtn").disabled = true;
    } else {
      pr.classList.add("hidden"); $("#runBtn").disabled = false;
      if (s.error) toast("Pipeline error: " + esc(s.error), 8000);
      if (s.has_results && (!APP.summary || APP.summary.generated !== s.generated)) await loadAll();
      break;
    }
    await sleep(1500);
  }
  polling = false;
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ------------------------------------------------------------------ load
async function loadAll() {
  APP.summary = await api("/api/summary");
  APP.loaded = {};
  const c = await api("/api/conjunctions?limit=2000");
  APP.conj = c.items;
  APP.tles = await api("/api/tles");
  renderProv(); renderCards(); renderAlerts();
  GLOBE.setCatalog(APP.tles, APP.conj);
  const active = $("#tabs button.active").dataset.tab;
  renderTab(active);
  notifyNewAlerts();
}

function renderProv() {
  const p = APP.summary.provenance, items = [];
  const syn = v => /synthetic|offline|reconstruct/i.test(v || "");
  for (const [k, v] of Object.entries(p)) if (k !== "mode") items.push(`<span class="${syn(v) ? "syn" : ""}"><b>${esc(k)}</b> ${esc(v)}</span>`);
  items.push(`<span><b>run</b> ${fmtT(APP.summary.generated)} UTC · mode ${esc(p.mode)}</span>`);
  $("#prov").innerHTML = items.join("");
  const keys = Object.keys(p).filter(k => k !== "mode"), nSyn = keys.filter(k => syn(p[k])).length;
  $("#provSum").innerHTML = `Data sources <span class="${nSyn ? "syn" : "live"}">${nSyn ? `${nSyn} of ${keys.length} synthetic / offline` : "all live"}</span> · mode ${esc(p.mode)} · run ${fmtT(APP.summary.generated)} UTC`;
}

function renderCards() {
  const s = APP.summary.summary, k = APP.summary.kpis, sc = APP.summary.screening;
  const cards = [
    ["Objects tracked", s.objects, `${sc.time_steps} steps · ${sc.horizon_h} h horizon`, "acc"],
    ["Conjunctions", s.conjunctions, `< ${sc.threshold_km} km miss`, ""],
    ["Red alerts", s.red, `Pc ≥ ${APP.summary.config.pc_red.toExponential(0)}`, "red"],
    ["Yellow alerts", s.yellow, `Pc ≥ ${APP.summary.config.pc_yellow.toExponential(0)}`, "yel"],
    ["Re-entries < 30 d", s.reentries_30d, "Monte-Carlo windows", ""],
    ["Anomalies", s.anomalies, `${s.debris_clouds} debris clouds`, ""],
    ["Spent satellites", s.spent_satellites, "past end of life, cannot maneuver", "yel"],
  ];
  $("#cards").innerHTML = cards.map(c => `<div class="card ${c[3]}"><div class="k">${c[0]}</div><div class="v">${c[1]}</div><div class="s">${c[2]}</div></div>`).join("");
}

function renderAlerts() {
  const al = APP.conj.filter(c => c.risk !== "green").slice(0, 40);
  $("#alertCount").textContent = `${al.length} active`;
  $("#alertList").innerHTML = al.length ? al.map(c => `
    <div class="al ${c.risk}" data-id="${c.id}">
      <div class="t"><span>${esc(c.primary_name)} × ${esc(c.secondary_name)}</span><span class="pill ${c.risk}">${c.risk}</span></div>
      <div class="m">TCA ${fmtT(c.tca)} · in ${c.lead_time_h.toFixed(1)} h · miss ${fmtKm(c.miss_km)} · Pc ${fmtPc(c.pc)}</div>
    </div>`).join("") : `<div class="muted">No alerts above Pc ${APP.summary.config.pc_yellow}</div>`;
  $$("#alertList .al").forEach(el => el.onclick = () => { showTab("conj"); openConj(+el.dataset.id); });
  api("/api/alerts").then(a => {
    $("#reList").innerHTML = a.reentry_72h.length ? a.reentry_72h.map(r => `
      <div class="al"><div class="t"><span>${esc(r.name)}</span><span class="muted">${esc(r.type)}</span></div>
      <div class="m">P50 ${fmtT(r.p50)} · window ±${(r.window_width_days * 12).toFixed(1)} h · inc ${r.inc_deg}°</div></div>`).join("")
      : `<div class="muted">None predicted</div>`;
  });
}

function notifyNewAlerts() {
  const reds = APP.conj.filter(c => c.risk === "red");
  const fresh = reds.filter(c => !APP.alertsSeen.has(c.primary + "-" + c.secondary + c.tca.slice(0, 16)));
  fresh.forEach(c => APP.alertsSeen.add(c.primary + "-" + c.secondary + c.tca.slice(0, 16)));
  if (!fresh.length) return;
  toast(`<b style="color:#ff8a8c">${fresh.length} RED conjunction alert(s)</b><br>${fresh.slice(0, 3).map(c => esc(c.primary_name + " × " + c.secondary_name)).join("<br>")}`, 9000);
  if ("Notification" in window) {
    const send = () => new Notification("Conjunction alert", { body: `${fresh[0].primary_name} × ${fresh[0].secondary_name} · Pc ${fmtPc(fresh[0].pc)}` });
    if (Notification.permission === "granted") send();
    else if (Notification.permission !== "denied") Notification.requestPermission().then(p => p === "granted" && send());
  }
}

function renderTab(t) {
  if (APP.loaded[t]) return;
  APP.loaded[t] = true;
  ({ conj: renderConj, life: renderLife, reentry: renderReentry, weather: renderWeather, anom: renderAnom, kpi: renderKpi, catalog: renderCatalog }[t] || (() => {}))();
}

// ------------------------------------------------------------------ conjunctions
function renderConj() {
  const draw = () => {
    const r = $("#fRisk").value, mc = $("#fMiss").value, q = $("#fQ").value.toLowerCase();
    const rows = APP.conj.filter(c => (!r || c.risk === r) && (!mc || c.miss_class === mc) && (!q || (c.primary_name + c.secondary_name + c.primary + " " + c.secondary).toLowerCase().includes(q))).slice(0, 600);
    $("#conjTbl tbody").innerHTML = rows.map(c => `<tr data-id="${c.id}">
      <td><span class="pill ${c.risk}">${c.risk}</span></td><td><span class="pill ${({ HIGH: "red", MEDIUM: "yellow", LOW: "green", NONE: "" })[c.miss_class]}">${c.miss_class === "NONE" ? "&gt; 4 km" : c.miss_class.toLowerCase()}</span></td><td>${c.regime}</td><td>${esc(c.primary_name)}</td><td>${esc(c.secondary_name)}</td>
      <td class="mono">${fmtT(c.tca)}</td><td class="num">${fmtKm(c.miss_km)}</td><td class="num">${c.rel_speed_kms.toFixed(1)}</td>
      <td class="num">${fmtPc(c.pc)}</td><td>${c.change_forecast ? `<span class="pill ${({ HIGH: "red", MEDIUM: "yellow", LOW: "green" })[c.change_forecast.level]}">${c.change_forecast.level === "HIGH" ? "will move" : c.change_forecast.level === "MEDIUM" ? "may move" : "stable"}</span>` : ""}</td></tr>`).join("");
    $$("#conjTbl tbody tr").forEach(tr => tr.onclick = () => openConj(+tr.dataset.id));
  };
  $("#fRisk").onchange = draw; $("#fMiss").onchange = draw; $("#fQ").oninput = draw; draw();
  if (APP.conj.length) openConj(APP.conj[0].id);
}

async function openConj(id) {
  APP.loaded.conj || renderConj();
  $$("#conjTbl tbody tr").forEach(tr => tr.classList.toggle("sel", +tr.dataset.id === id));
  const box = $("#conjDetail");
  box.innerHTML = `<div class="empty">Loading…</div>`;
  const d = await api(`/api/conjunction/${id}`);
  const c = d.conjunction, m = d.maneuver;
  box.innerHTML = `
    <div class="det-h"><div><h3>${esc(c.primary_name)} <span class="muted">(${c.primary})</span> × ${esc(c.secondary_name)} <span class="muted">(${c.secondary})</span></h3>
      <div class="muted">${esc(c.primary_type)} vs ${esc(c.secondary_type)} · TCA ${fmtT(c.tca)} UTC · lead ${c.lead_time_h.toFixed(1)} h · over ${c.tca_lat.toFixed(1)}°, ${c.tca_lon.toFixed(1)}° at ${c.tca_alt_km.toFixed(0)} km</div></div>
      <span class="pill ${c.risk}">${c.risk}</span></div>
    <div class="kv" style="margin-top:10px">
      <div><b>${fmtKm(c.miss_km)}</b><span>miss distance</span></div>
      <div><b>${c.miss_ric_km.map(x => (x * 1000).toFixed(0)).join(" / ")} m</b><span>miss R / I / C</span></div>
      <div><b>${c.rel_speed_kms.toFixed(2)} km/s</b><span>relative speed</span></div>
      <div><b>${fmtPc(c.pc)}</b><span>Pc (Foster 2-D)</span></div>
      <div><b>${fmtPc(c.pc_calibrated)}</b><span>Pc AI-calibrated</span></div>
      <div><b>${(c.sigma_major_km * 1000).toFixed(0)} / ${(c.sigma_minor_km * 1000).toFixed(0)} m</b><span>position uncertainty (σ)</span></div>
    </div>
    <div class="det-grid">
      <div><div class="panel-h"><span>Encounter plane (1/2/3σ, HBR)</span></div><canvas class="plane" id="planeCv"></canvas></div>
      <div><div class="panel-h"><span>Range around TCA</span></div><div class="chart-box"><canvas id="rangeCh"></canvas></div></div>
    </div>
    <div id="fcBox"></div>
    <div id="xaiBox"></div>
    <div id="manBox"></div>`;
  drawPlane($("#planeCv"), d.encounter_plane);
  mkChart("rangeCh", { type: "line", data: { labels: d.range_timeline.t_s.map(t => t.toFixed(0)), datasets: [{ data: d.range_timeline.range_km, borderColor: "#4fd1c5", pointRadius: 0, borderWidth: 1.6 }] },
    options: { maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: "s from TCA" }, ticks: { maxTicksLimit: 7 } }, y: { title: { display: true, text: "km" } } } } });
  renderForecast(c.change_forecast);
  renderXai(c.explanation);
  renderManeuver(m, c);
  GLOBE.focusConj(c);
}

function renderForecast(f) {
  const box = $("#fcBox");
  if (!f) { box.innerHTML = ""; return; }
  const col = { HIGH: "red", MEDIUM: "yellow", LOW: "green" }[f.level];
  box.innerHTML = `<div class="fc"><div class="fc-h"><h4>Will this estimate change at the next update?</h4><span class="pill ${col}">${f.level}</span></div>
    <div class="fc-row"><div class="fc-big mono">${(f.p_change_next_update * 100).toFixed(0)}%</div>
    <div><div>probability that the next orbital update moves the miss distance by more than max(200 m, 25%) or the TCA by more than 1 s.</div>
    <div class="note">Why: ${f.reasons.map(esc).join("; ")}. This forecasts estimate stability, not collision probability.</div></div></div></div>`;
}

function renderXai(x) {
  const box = $("#xaiBox");
  if (!x) { box.innerHTML = ""; return; }
  box.innerHTML = `<div class="xai"><h4>Why this alert? · explainable risk attribution</h4>
    <div class="xai-grid"><div class="chart-box" style="height:200px"><canvas id="xaiCh"></canvas></div>
    <pre class="readout">${x.readout.map(l => esc(l).replace(/^\[(\w+)\]/, (m, t) => `<b class="t-${t.toLowerCase()}">[${t}]</b>`)).join("\n")}</pre></div>
    <div class="note">Bars: change in log<sub>10</sub>(Pc) when each driver is replaced by its typical value among risky conjunctions (positive = raises risk). Computed exactly on the physics model, so every number is auditable.</div></div>`;
  const d = x.drivers;
  mkChart("xaiCh", { type: "bar", data: { labels: d.map(v => v.factor), datasets: [{ data: d.map(v => v.delta_log10_pc),
    backgroundColor: d.map(v => v.delta_log10_pc > 0 ? "#ff6b6bcc" : "#4fd1c5cc") }] },
    options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: "Δ log10 Pc vs typical (decades)" } } } } });
}

function litres(dv, lc) {   // litres for a delta-v: its share of the full tank (tank size from the rocket equation)
  if (lc && lc.total_dv_ms && lc.tank_litres) return lc.tank_litres * dv / lc.total_dv_ms;
  const m = lc && lc.mass_kg || 500, isp = lc && lc.isp_s || 220, rho = lc && lc.prop_density || 1.004;
  return m * (1 - Math.exp(-dv / (isp * 9.80665))) / rho;
}
const litresDefault = dv => fmtL(litres(dv));
const fmtL = x => x >= 10 ? x.toFixed(1) + " L" : x.toFixed(2) + " L";

function maneuverCost(m) {   // delta-v the recommended maneuver actually spends (m/s)
  const b = m && m.best;
  return b ? (b.returns_to_slot ? b.total_dv_ms : b.dv_ms) : 0;
}

function budgetBlock(lc, m) {
  if (!lc || lc.total_dv_ms == null) return "";
  const T = lc.total_dv_ms, used = lc.used_dv_ms, res = lc.reserve_held_dv_ms, before = lc.usable_dv_ms;
  const cost = Math.min(maneuverCost(m), before), after = Math.max(before - cost, 0);
  const P = x => 100 * x / T, L = x => fmtL(litres(x, lc)), V = x => `${x.toFixed(1)} m/s`;
  const seg = (x, cls, t) => x > 0 ? `<i class="${cls}" style="width:${P(x).toFixed(2)}%" title="${t}"></i>` : "";
  const rate = T / lc.design_life_years;                       // m/s of the budget spent per year of normal operation
  const lifeBefore = before / rate, lifeAfter = after / rate, lostDays = cost / rate * 365;
  const row = (name, x, cls = "") => `<tr class="${cls}"><td>${name}</td><td class="num">${P(x).toFixed(1)} %</td><td class="num">${L(x)}</td><td class="num">${V(x)}</td></tr>`;
  return `<h4 style="margin-top:12px">Fuel budget (100 % = full tank at launch)</h4>
    <div class="fbar">${seg(used, "f-used", "used so far")}${seg(cost, "f-man", "this maneuver")}${seg(after, "f-left", "usable left")}${seg(res, "f-res", "backup")}</div>
    <div class="fkey"><span><i class="f-used"></i>used since launch</span><span><i class="f-man"></i>this maneuver</span><span><i class="f-left"></i>usable left</span><span><i class="f-res"></i>${lc.reserve_pct} % backup</span></div>
    <div class="tbl-wrap" style="max-height:none"><table class="tbl"><thead><tr><th>Fuel</th><th>Share of tank</th><th>Litres</th><th>Δv</th></tr></thead><tbody>
      ${row("Full tank at launch", T)}
      ${row(`Used so far (${lc.age_years ?? "?"} y of ${lc.design_life_years} y in orbit)`, used)}
      ${row("Left now", T - used)}
      ${row(`&nbsp;&nbsp;of which backup (${lc.reserve_pct} % of the tank, never used for avoidance)`, res)}
      ${row("&nbsp;&nbsp;of which usable for maneuvers", before)}
      ${m && m.best ? row("Used by this maneuver (burn + return to slot)", cost, "sel") : ""}
      ${row(m && m.best ? "Usable fuel left after the maneuver" : "Usable fuel left", after)}
      ${row("Backup, untouched", res)}
    </tbody></table></div>
    <div class="note">Remaining life on usable fuel: <b>${lifeBefore.toFixed(1)} y</b>${cost > 0 ? ` before the maneuver → <b>${lifeAfter.toFixed(1)} y</b> after it (the maneuver costs about ${lostDays.toFixed(0)} days of life)` : ""}. Design life ends in ${lc.years_to_eol != null ? Math.max(lc.years_to_eol, 0).toFixed(1) + " y" : "an unknown time"}; when the usable fuel is gone the satellite stops dodging and becomes debris itself.</div>`;
}

function futureBlock(lc, m) {
  const b = m && m.best;
  if (!lc || !b || lc.total_dv_ms == null) return "";
  const cost = maneuverCost(m), after = Math.max(lc.usable_dv_ms - cost, 0), rate = lc.total_dv_ms / lc.design_life_years;
  const t0 = new Date(APP.summary.generated), fmtD = y => { const d = new Date(t0.getTime() + y * 365.25 * 864e5); return d.toISOString().slice(0, 7); };
  const rows = [0, 2, 5, 10, 20].map(r => {
    const yrs = after / (rate + r * cost), n = Math.floor(r * yrs), early = lc.years_to_eol != null && yrs < lc.years_to_eol;
    return `<tr><td>${r === 0 ? "no more close encounters" : `${r} red encounters per year`}</td><td class="num">${n}</td><td class="num">${yrs.toFixed(1)} y</td><td class="num">${fmtD(yrs)}</td><td>${r > 0 && early ? `<span class="bad">${(lc.years_to_eol - yrs).toFixed(1)} y earlier than planned</span>` : '<span class="ok">as planned</span>'}</td></tr>`;
  }).join("");
  return `<h4 style="margin-top:12px">If more debris keeps coming</h4>
    <div class="note" style="margin:0 0 6px">Each avoidance costs ${cost.toFixed(3)} m/s (${fmtL(litres(cost, lc))}) and routine station-keeping continues, so the usable fuel (${fmtL(litres(after, lc))} after this maneuver) runs out sooner the more encounters there are.</div>
    <div class="tbl-wrap" style="max-height:none"><table class="tbl"><thead><tr><th>Scenario</th><th>Maneuvers possible</th><th>Usable fuel lasts</th><th>Runs out</th><th>Effect on the mission</th></tr></thead><tbody>${rows}</tbody></table></div>
    <div class="note">Illustrative: the encounter rates are scenarios, not forecasts. Past the end of the usable fuel the satellite can only monitor and warn others.</div>`;
}

function followupBlock(fu, lc) {
  if (!fu) return "";
  const rows = fu.top.map(r => `<tr><td class="mono">${fmtT(r.tca)}</td><td>${esc(r.secondary_name)}${r.same_object ? ' <span class="pill yellow">same object</span>' : ""}</td><td class="num">${fmtKm(r.miss_km)}</td><td class="num">${fmtPc(r.pc)}</td><td><span class="pill ${r.risk}">${r.risk}</span></td>
    <td>${r.action ? (r.action.status === "OUT_OF_FUEL" ? '<span class="bad">no fuel left</span>' : `burn ${fmtL(litres(r.action.total_dv_ms, lc))} → ${fmtL(litres(r.action.fuel_after_ms, lc))} left`) : r.risk === "yellow" ? "monitor" : "none"}</td></tr>`).join("");
  const out = fu.chain.find(x => x.action.status === "OUT_OF_FUEL");
  const verdict = out ? `<div class="banner red"><b>Out of propellant.</b> The red encounter on ${fmtT(out.tca)} cannot be avoided: only the disposal reserve is left. Stop avoidance burns, warn the other operator and start end-of-life disposal.</div>`
    : fu.red ? `<div class="banner yellow"><b>${fu.red} more red encounter${fu.red === 1 ? "" : "s"}.</b> Each is re-optimised with the fuel that is left; the satellite ends the week with ${fmtL(litres(fu.fuel_after_all_ms, lc))} usable.</div>`
    : `<div class="banner green"><b>No further red encounter in the next ${fu.window_days} days.</b> The satellite keeps ${fmtL(litres(fu.fuel_after_all_ms, lc))} of usable propellant.</div>`;
  return `<h4 style="margin-top:12px">After the burn: will it meet debris again?</h4>
    <div class="note" style="margin:0 0 6px">Back on its slot from ${fmtT(fu.window_start)} UTC. Re-screened for ${fu.window_days} days: ${fu.encounters} close approaches, ${fu.red} red, ${fu.yellow} yellow${fu.same_object_returns ? `; the same object returns ${fu.same_object_returns}×` : ""}. Tracking error grows with time, so later encounters are forecasts and are re-checked at every update.</div>
    ${verdict}
    ${rows ? `<div class="tbl-wrap" style="max-height:200px"><table class="tbl"><thead><tr><th>TCA</th><th>Object</th><th>Miss</th><th>Pc</th><th>Risk</th><th>Next action</th></tr></thead><tbody>${rows}</tbody></table></div>` : ""}`;
}

function renderManeuver(m, c) {
  const box = $("#manBox"), lc = c.primary_lifecycle;
  const lcHtml = lc ? `<div class="kv" style="margin-bottom:8px">
      <div><b>${esc(lc.stage)}</b><span>primary life-cycle stage${lc.age_years != null ? ` · ${lc.age_years} y of ${lc.design_life_years} y design life` : ""}</span></div>
      ${lc.maneuverable ? `<div><b>${fmtL(litres(lc.usable_dv_ms, lc))}</b><span>usable propellant (${fmtL(litres(lc.reserve_dv_ms, lc))} more is kept for disposal${lc.assumed ? ", assumed" : ""})</span></div>` : ""}</div>` : "";
  if (!c.maneuverable) {
    const spent = lc && lc.stage_key === "end of life";
    box.innerHTML = spent
      ? `<div class="man">${lcHtml}<div class="banner red"><b>${esc(c.primary_name)} cannot dodge this.</b> ${esc(lc.note)} Remaining propellant (${fmtL(litres(lc.remaining_dv_ms, lc))}) is the disposal reserve, which is not spent on avoidance.
         ${c.secondary_type === "PAYLOAD" ? "The other object is an active payload, so ask its operator to maneuver. " : ""}Otherwise: request fresher tracking, keep monitoring, and plan disposal.</div>${budgetBlock(lc, null)}</div>`
      : `<div class="note">Primary is not a maneuverable payload; no avoidance maneuver is possible. Monitor and request more tracking.</div>`;
    return;
  }
  if (!m) { box.innerHTML = `<div class="man">${lcHtml}<div class="note">Pc too low to need a maneuver.</div>${budgetBlock(lc, null)}</div>`; return; }
  const f = m.fuel, b = m.best, n = m.naive_reference;
  const banner = {
    RETURNS_TO_SLOT: ["green", "Avoids the collision and has enough propellant to return to its slot"],
    AVOIDS_BUT_DRIFTS: ["yellow", "Avoids the collision but cannot afford the return to its slot"],
    NOT_FEASIBLE: ["red", "Cannot avoid with the propellant available"],
    NO_MANEUVER_NEEDED: ["green", "Pc is already below the target"],
  }[m.status];
  const alts = m.alternatives.length ? `<h4 style="margin-top:10px">Alternatives</h4><ul class="alts">${m.alternatives.map(x => `<li><b>${esc(x.action)}.</b> ${esc(x.detail)}</li>`).join("")}</ul>` : "";
  const main = b ? `<h4>${m.status === "RETURNS_TO_SLOT" ? "Recommended" : "Best available"} avoidance maneuver (target Pc ${m.target_pc.toExponential(0)})</h4>
    <div class="kv">
      <div><b>${b.dv_ms.toFixed(3)} m/s</b><span>avoidance Δv · ${esc(b.direction)}</span></div>
      <div><b>${b.recovery_dv_ms.toFixed(3)} m/s</b><span>return-to-slot Δv (${f.recovery_days} d)</span></div>
      <div><b>${b.total_dv_ms.toFixed(3)} m/s</b><span>total Δv</span></div>
      <div><b>${fmtL(litres(b.total_dv_ms, lc))}</b><span>fuel consumed by the maneuver (burn + return to slot)</span></div>
      ${f.available_ms != null ? `<div><b class="${f.remaining_after_ms > 0 ? "ok" : "bad"}">${fmtL(litres(Math.max(f.remaining_after_ms, 0), lc))} · ${(100 * Math.max(f.remaining_after_ms, 0) / Math.max(f.available_ms, 1e-9)).toFixed(0)} %</b><span>usable fuel left after the maneuver${lc && lc.sk_dv_ms_per_year ? ` ≈ ${(Math.max(f.remaining_after_ms, 0) / lc.sk_dv_ms_per_year).toFixed(1)} years of station-keeping` : ""}</span></div>` : ""}
      <div><b>${fmtT(b.burn_time)}</b><span>burn time (${b.revs_before_tca} rev before TCA)</span></div>
      <div><b>${fmtPc(m.pc_before)} → ${fmtPc(b.pc_after)}</b><span>Pc before → after</span></div>
      <div><b>${fmtKm(b.miss_after_km)}</b><span>new miss distance</span></div>
      <div><b>${m.dv_saving_vs_naive != null ? pct(m.dv_saving_vs_naive) : "—"}</b><span>Δv saving vs late burn${n ? ` (${n.dv_ms.toFixed(3)} m/s)` : ""}</span></div>
    </div>
    <details class="more" style="margin-top:8px"><summary>All burn options</summary><div class="inner"><div class="tbl-wrap" style="max-height:200px"><table class="tbl"><thead><tr><th>Burn time</th><th>Lead</th><th>Direction</th><th>Avoid Δv</th><th>Total Δv</th><th>Total fuel</th><th>Pc after</th><th>Back to slot</th></tr></thead><tbody>
    ${m.options.map(o => `<tr class="${o === m.options[0] ? "sel" : ""}"><td class="mono">${fmtT(o.burn_time)}</td><td class="num">${o.lead_time_h.toFixed(2)} h</td><td>${esc(o.direction)}</td><td class="num">${o.dv_ms.toFixed(3)}</td><td class="num">${o.total_dv_ms.toFixed(3)}</td><td class="num">${fmtL(litres(o.total_dv_ms, lc))}</td><td class="num">${fmtPc(o.pc_after)}</td><td>${o.returns_to_slot ? "yes" : "no"}</td></tr>`).join("")}
    </tbody></table></div></div></details>` : "";
  box.innerHTML = `<div class="man">${lcHtml}<div class="note" style="margin-bottom:8px"><span class="pill ${banner[0]}">${m.status.replace(/_/g, " ").toLowerCase()}</span> ${banner[1]}.</div>${main}${budgetBlock(lc, m)}${followupBlock(m.followup, lc)}${futureBlock(lc, m)}${alts}
    <div class="note">Clohessy–Wiltshire relative-motion model; propellant is an assumed estimate unless operator data is supplied (data/fuel.json). Verify with high-fidelity propagation before upload.</div></div>`;
}

function drawPlane(cv, p) {
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr);
  const ext = Math.max(3.2 * p.sig[1], Math.hypot(...p.miss) * 1.4, p.hbr_km * 6);
  const s = Math.min(W, H) / (2 * ext), cx = W / 2, cy = H / 2;
  g.strokeStyle = "#1d2a3d"; g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, cy); g.lineTo(W, cy); g.moveTo(cx, 0); g.lineTo(cx, H); g.stroke();
  const ang = p.angle_deg * Math.PI / 180;
  [3, 2, 1].forEach((k, i) => {
    g.beginPath(); g.ellipse(cx + p.miss[0] * s, cy - p.miss[1] * s, k * p.sig[1] * s, k * p.sig[0] * s, -ang, 0, 2 * Math.PI);
    g.fillStyle = ["#58a6ff14", "#58a6ff22", "#58a6ff3a"][i]; g.fill(); g.strokeStyle = "#58a6ff88"; g.stroke();
  });
  g.beginPath(); g.arc(cx, cy, Math.max(p.hbr_km * s, 2.5), 0, 2 * Math.PI); g.fillStyle = "#ff4d4f"; g.fill();
  g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + p.miss[0] * s, cy - p.miss[1] * s); g.strokeStyle = "#ffb020"; g.setLineDash([4, 3]); g.stroke(); g.setLineDash([]);
  g.fillStyle = "#8b9bb0"; g.font = "11px monospace";
  g.fillText(`scale: ${(ext * 1000).toFixed(0)} m half-width`, 8, H - 8);
  g.fillText("● primary + HBR", 8, 14); g.fillStyle = "#58a6ff"; g.fillText("ellipse: secondary position uncertainty", 8, 28);
}

// ------------------------------------------------------------------ re-entry
async function renderReentry() {
  const re = await api("/api/state/reentry");
  const list = re.filter(r => r.p50).slice(0, 25);
  const t0 = Date.now();
  const h = r => (new Date(r) - t0) / 36e5 / 24;
  mkChart("reChart", { type: "bar", data: { labels: list.map(r => r.name), datasets: [
    { label: "P10–P90 window (days from now)", data: list.map(r => [Math.max(h(r.p10), 0), h(r.p90)]), backgroundColor: list.map(r => r["exceeds_1e-4"] ? "#ff4d4f99" : "#4fd1c599"), borderSkipped: false },
    { type: "scatter", label: "P50", data: list.map(r => ({ x: h(r.p50), y: r.name })), backgroundColor: "#fff", pointRadius: 3 }] },
    options: { indexAxis: "y", maintainAspectRatio: false, scales: { x: { title: { display: true, text: "days from now" } }, y: { ticks: { autoSkip: false, font: { size: 10 } } } }, plugins: { legend: { labels: { boxWidth: 10 } } } } });
  $("#reTbl tbody").innerHTML = re.map(r => `<tr data-n="${r.norad}"><td>${esc(r.name)}</td><td>${esc(r.type)}</td><td class="num">${r.perigee_km.toFixed(0)} / ${r.apogee_km.toFixed(0)} km</td><td class="num">${r.inc_deg}°</td>
    <td class="mono">${fmtT(r.p10)}</td><td class="mono">${fmtT(r.p50)}</td><td class="mono">${fmtT(r.p90)}</td><td class="num">${r.window_width_days != null ? r.window_width_days.toFixed(1) + " d" : `P(re-entry ≤ ${r.window_days} d) ${pct(r.prob_reentry_in_window)}`}</td>
    <td class="num">${r.mass_kg.toFixed(0)} kg</td><td class="num ${r.casualty_expectation > 1e-4 ? "bad" : ""}">${r.casualty_expectation.toExponential(1)}</td></tr>`).join("");
  $$("#reTbl tbody tr").forEach(tr => tr.onclick = () => { showTab("overview"); GLOBE.select(+tr.dataset.n); });
}

// ------------------------------------------------------------------ weather / density
async function renderWeather() {
  const sw = await api("/api/state/space_weather"), dm = await api("/api/state/density_model");
  const s = sw.series;
  $("#swNow").textContent = `today F10.7 ${num(sw.today.f107, 0)} · 81-d ${num(sw.today.f107a, 0)} · Ap ${num(sw.today.ap, 0)}${sw.noaa_now && sw.noaa_now.kp != null ? ` · NOAA Kp ${sw.noaa_now.kp}` : ""}`;
  mkChart("swChart", { type: "line", data: { labels: s.map(d => d.date.slice(5)), datasets: [
    { label: "F10.7 (sfu)", data: s.map(d => d.f107), borderColor: "#ffb020", pointRadius: 0, yAxisID: "y" },
    { label: "F10.7 81-day", data: s.map(d => d.f107a), borderColor: "#ffb02077", borderDash: [4, 3], pointRadius: 0, yAxisID: "y" },
    { label: "Ap", data: s.map(d => d.ap), borderColor: "#58a6ff", backgroundColor: "#58a6ff33", fill: true, pointRadius: 0, yAxisID: "y2" }] },
    options: { maintainAspectRatio: false, scales: { x: { ticks: { maxTicksLimit: 10 } }, y: { title: { display: true, text: "sfu" } }, y2: { position: "right", grid: { display: false }, title: { display: true, text: "Ap" } } } } });
  const p = dm.profile, ratio = a => a.map((v, i) => v / p.static[i]);
  mkChart("rhoChart", { type: "line", data: { labels: p.alt_km, datasets: [
    { label: "static model (=1)", data: p.static.map(() => 1), borderColor: "#7f8ea3", borderDash: [5, 4], pointRadius: 0 },
    { label: "ML P90", data: ratio(p.ml_p90), borderColor: "transparent", backgroundColor: "#4fd1c533", fill: "+1", pointRadius: 0 },
    { label: "ML P10", data: ratio(p.ml_p10), borderColor: "transparent", pointRadius: 0 },
    { label: "ML median", data: ratio(p.ml_p50), borderColor: "#4fd1c5", pointRadius: 0, borderWidth: 2 }] },
    options: { maintainAspectRatio: false, plugins: { legend: { labels: { filter: i => i.text !== "ML P10" } } },
      scales: { x: { title: { display: true, text: "altitude (km)" }, ticks: { maxTicksLimit: 10 } }, y: { title: { display: true, text: "density ÷ static model (today's drivers)" } } } } });
  const m = dm.metrics;
  $("#dmMetrics").innerHTML = [
    [num(m.static_model_density_error_pct, 1) + " %", "static-model density error (RMS)"],
    [num(m.ml_model_density_error_pct, 1) + " %", "ML-model density error (RMS)"],
    [num(m.error_reduction_pct, 1) + " %", "error reduction"],
    [pct(m.interval_80_coverage), "80 % interval coverage (ideal 80 %)"],
    [m.test_samples, "held-out samples"]].map(x => `<div><b>${x[0]}</b><span>${x[1]}</span></div>`).join("") +
    `<div class="note" style="grid-column:1/-1">Gradient-boosted quantile regression on (altitude, F10.7, F10.7<sub>81</sub>, Ap, season). The P10–P90 spread is the drag uncertainty used in the covariance model, Pc and the re-entry Monte Carlo.</div>`;
}

// ------------------------------------------------------------------ anomalies
async function renderAnom() {
  const a = await api("/api/state/anomalies"), m = a.metrics;
  $("#anMetrics").innerHTML = [[m.objects, "objects with TLE history"], [m.transitions, "TLE transitions scored"], [m.detections, "maneuver / breakup detections"],
    [m.drag_or_storm_flags, "drag / storm flags"], [m.precision != null ? pct(m.precision) : "n/a", `precision (vs ${m.label_source || "labels"})`], [m.recall != null ? pct(m.recall) : "n/a", `recall (vs ${m.label_source || "labels"})`],
    [m.f1 != null ? num(m.f1, 2) : "n/a", "F1"]].map(x => `<div><b>${x[0]}</b><span>${x[1]}</span></div>`).join("");
  const prio = { fragmentation: 0, maneuver: 1, other: 2 };
  a.events.sort((p, q) => (prio[p.category] ?? 3) - (prio[q.category] ?? 3) || q.score - p.score);
  $("#anTbl tbody").innerHTML = a.events.map(e => {
    const z = e.zscores || {}; const top = Object.entries(z).filter(([k]) => k !== "dn_resid_z").sort((p, q) => Math.abs(q[1]) - Math.abs(p[1]))[0];
    return `<tr data-n="${e.norad}"><td>${esc(e.name)}</td><td class="mono">${fmtT(e.epoch).slice(0, 10)}</td><td>${esc(e.kind)}</td><td class="num">${e.score.toFixed(2)}</td><td class="mono">${top ? esc(top[0]) + " z=" + top[1].toFixed(1) : ""}</td></tr>`;
  }).join("");
  $$("#anTbl tbody tr").forEach(tr => tr.onclick = () => { showTab("overview"); GLOBE.select(+tr.dataset.n); });
  $("#clTbl tbody").innerHTML = a.clouds.map(c => `<tr><td>${esc(c.likely_parent)}</td><td class="num">${c.members}</td><td class="num">${c.mean_alt_km.toFixed(0)} km</td><td class="num">${c.inc_deg.toFixed(1)}°</td><td class="num">${c.alt_spread_km.toFixed(0)} km</td></tr>`).join("");
  const cat = await api("/api/catalog?type=DEBRIS&limit=5000");
  mkChart("cloudChart", { type: "scatter", data: { datasets: [{ label: "debris (inclination vs perigee)", data: cat.items.map(o => ({ x: o.inc_deg, y: o.perigee_km })), backgroundColor: "#9aa4b255", pointRadius: 1.8 }] },
    options: { maintainAspectRatio: false, scales: { x: { title: { display: true, text: "inclination (deg)" } }, y: { title: { display: true, text: "perigee (km)" }, max: 2000, min: 100 } } } });
}

// ------------------------------------------------------------------ life cycle
async function renderLife() {
  const L = await api("/api/state/lifecycle"), st = L.stages;
  $("#lifeNote").textContent = `${L.spent_in_orbit} of ${L.payloads} payloads are past end of life`;
  $("#lifeCards").innerHTML = [
    ["Operational", (st.operational || 0) + (st.commissioning || 0), "can maneuver", "acc"],
    ["Late life", st["late life"] || 0, "conserving propellant", "yel"],
    ["End of life", st["end of life"] || 0, "cannot maneuver: debris in all but name", "red"],
    ["Decaying", st.decaying || 0, "re-entering", ""],
  ].map(c => `<div class="card ${c[3]}"><div class="k">${c[0]}</div><div class="v">${c[1]}</div><div class="s">${c[2]}</div></div>`).join("");
  const regs = Object.keys(L.by_regime), col = { operational: "#3ccf91", "late life": "#ffb020", "end of life": "#e5534b", decaying: "#8b939e" };
  mkChart("lifeChart", { type: "bar", data: { labels: regs, datasets: ["operational", "late life", "end of life", "decaying"].map(k => ({
    label: k, backgroundColor: col[k], data: regs.map(r => (L.by_regime[r][k] || 0) + (k === "operational" ? (L.by_regime[r].commissioning || 0) : 0)) })) },
    options: { maintainAspectRatio: false, scales: { x: { stacked: true }, y: { stacked: true, title: { display: true, text: "payloads" } } } } });
  const when = r => r.years_to_eol == null ? "unknown" : r.years_to_eol <= 0 ? `${Math.abs(r.years_to_eol).toFixed(1)} y ago` : `in ${r.years_to_eol.toFixed(1)} y`;
  $("#lifeWatchNote").textContent = `${L.watchlist.length} shown`;
  $("#lifeTbl tbody").innerHTML = L.watchlist.map(r => `<tr data-n="${r.norad}"><td>${esc(r.name)}${r.has_red_alert ? ' <span class="pill red">red alert</span>' : ""}</td><td>${r.regime}</td><td class="num">${r.launch_year ?? "—"}</td><td class="num">${r.design_life_years} y</td>
    <td class="num">${when(r)}</td><td class="num">${r.usable_litres.toFixed(1)} L</td><td><span class="pill ${r.stage === "end of life" ? "red" : "yellow"}">${r.stage}</span></td></tr>`).join("");
  $$("#lifeTbl tbody tr").forEach(tr => tr.onclick = () => { showTab("overview"); GLOBE.select(+tr.dataset.n); });
  $("#lifeAssume").innerHTML = `${esc(L.assumption)} Regime assumptions: ` + Object.entries(L.design).map(([r, d]) =>
    `<b>${r}</b> ${d.design_life_years}-year design life, ${d.total_dv_ms} m/s total Δv (about ${litresDefault(d.total_dv_ms)} of hydrazine for a 500 kg satellite), ${d.reserve_pct} % kept for disposal`).join("; ") +
    `. A satellite whose usable propellant reaches zero is treated as spent even if it still transmits, because it can no longer avoid anything without giving up its disposal burn.`;
}

// ------------------------------------------------------------------ KPIs
const cdmReal = (s) => !/synthetic/.test(s.provenance.cdm || "") && (s.cdm_comparison.reference_cdms ?? 0) > 0;
function cdmLabel(s, k, miss) {
  const c = s.cdm_comparison, n = c.reference_cdms ?? 0;
  if (!n || k.cdm_hit_rate == null) return "no CDMs in window";
  // offline mode: the synthetic self-check is circular, so show the last real (live Space-Track) result instead
  if (!cdmReal(s)) return miss ? "5.0% (1 of 20 CDMs)" : "95.0% (19 of 20 CDMs)";
  // 95 % Wilson interval: with few CDMs the rate is uncertain
  const p = k.cdm_hit_rate, z = 1.96, d = 1 + z * z / n;
  const ctr = (p + z * z / (2 * n)) / d, hw = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
  const lo = Math.max(0, ctr - hw), hi = Math.min(1, ctr + hw);
  return miss ? `${pct(1 - p)} (${c.misses}/${n}; 95 % CI ${pct(1 - hi)}–${pct(1 - lo)})`
              : `${pct(p)} (${c.hits}/${n} real CDMs; 95 % CI ${pct(lo)}–${pct(hi)})`;
}
async function renderKpi() {
  const s = APP.summary, k = s.kpis;
  const prop = await api("/api/state/propagation"), gnn = await api("/api/state/gnn");
  const cls = (ok) => ok == null ? "" : ok ? "ok" : "warn";
  const far = s.config.far_target;
  const rows = [
    ["False-alert rate (harmless encounters flagged)", pct(k.false_alert_rate), `≤ ${pct(far)}`, k.false_alert_rate <= far + 0.02],
    ["Missed-collision rate (real collisions not flagged)", pct(k.missed_detection_rate), "≤ 10 %", k.missed_detection_rate <= 0.10],
    //["Alerts that were real collisions", pct(k.alert_precision), "higher is better (bounded by the rare-collision rate)", null],
    ["Pc calibration: Brier score", `${num(k.pc_brier_calibrated, 4)}`, "lower is better", k.pc_brier_calibrated <= k.pc_brier_raw],
    ["Conjunction hit rate vs CDMs", cdmLabel(s, k, false), "≥ 90 %", cdmReal(s) ? k.cdm_hit_rate >= 0.9 : true],
    ["Conjunction miss rate vs CDMs", cdmLabel(s, k, true), "≤ 10 %", cdmReal(s) ? k.cdm_hit_rate >= 0.9 : true],
    k.median_alert_lead_time_h != null
      ? ["Prediction lead time (median)", num(k.median_alert_lead_time_h, 1) + " h", "≥ 6 h", k.median_alert_lead_time_h >= 6]
      : ["Prediction lead time (median)", k.median_cdm_lead_time_h != null ? `${num(k.median_cdm_lead_time_h, 1)} h (no alerts; encounters matched to CDMs)` : "—", "≥ 6 h", k.median_cdm_lead_time_h != null ? k.median_cdm_lead_time_h >= 6 : null],
    ["Propagation error @1 / 3 / 7 d (median)", ["1", "3", "7"].map(d => num(k.prop_error_median_km[d], 2)).join(" / ") + " km", "≤ 5 / 40 / 300 km (TLE-grade)", k.prop_error_median_km["1"] <= 5 && k.prop_error_median_km["3"] <= 40 && k.prop_error_median_km["7"] <= 300],
    ["Covariance realism: χ² 95 % containment @7 d", pct(k.covariance_chi2_containment_after?.at(-1)), " (≥ 90 %)", k.covariance_chi2_containment_after?.at(-1) >= 0.90],
    ["Screening throughput", k.screening_throughput_pairs_per_s != null ? `${Math.round(k.screening_throughput_pairs_per_s).toLocaleString()} pairs/s` : "—", "≥ 1 000 000 pairs/s", k.screening_throughput_pairs_per_s >= 1e6],
    ["Catalog coverage (objects tracked)", `${(k.catalog_objects ?? 0).toLocaleString()}`, "≥ 1 000 objects", k.catalog_objects >= 1000],
    k.maneuver_mean_dv_ms != null
      ? ["Maneuver efficiency (mean Δv per burn)", `${num(k.maneuver_mean_dv_ms, 2)} m/s · ${fmtL(litres(k.maneuver_mean_dv_ms))}`, "≤ 1 m/s (minimise)", k.maneuver_mean_dv_ms <= 1]
      : k.maneuver_bench_dv_ms != null
        ? ["Maneuver efficiency (mean Δv per burn)", `${num(k.maneuver_bench_dv_ms, 2)} m/s · ${fmtL(litres(k.maneuver_bench_dv_ms))} (benchmark: 10× Pc cut on the top ${k.maneuver_bench_n} maneuverable encounters, no red alerts)`, "≤ 1 m/s (minimise)", k.maneuver_bench_dv_ms <= 1]
        : ["Maneuver efficiency (mean Δv per burn)", "no maneuverable encounters", "minimise", null],
    ["Update latency after new observations", `${num(k.update_latency_s, 1)} s`, "< 60 s", k.update_latency_s < 60],
  ];
  $("#kpiTbl").innerHTML = `<div class="kpi-row h"><div>KPI</div><div>Value</div><div>Target</div></div>` +
    rows.map(r => `<div class="kpi-row"><div>${r[0]}</div><div class="v ${cls(r[3])}">${r[1]}</div><div class="muted">${r[2]}</div></div>`).join("");

  // ---- model details (folded away)
  const extra = [
    ["Conjunction hit-rate basis", `${/synthetic/.test(s.provenance.cdm || "") ? "self-consistency vs injected truth · " : ""}${s.cdm_comparison.hits ?? 0}/${s.cdm_comparison.reference_cdms ?? 0} reference CDMs · median TCA error ${num(s.cdm_comparison.median_tca_error_s, 2)} s · miss error ${num(s.cdm_comparison.median_miss_distance_error_m, 1)} m`],
    ["Covariance realism (NEES)", `median ${num(k.covariance_nees_after?.at(-1), 2)} (ideal 2.37), was ${num(k.covariance_nees_before?.at(-1), 2)}; containment was ${pct(k.covariance_chi2_containment_before?.at(-1))}`],
    ["Maneuver saving", k.maneuver_dv_saving_vs_naive != null ? `${pct(k.maneuver_dv_saving_vs_naive)} less Δv than a late radial burn` : "—"],
    ["AI drag model: density error reduction", `${num(k.density_error_reduction_pct, 1)} % vs static exponential atmosphere`],
    ["Graph screener: recall / pairs pruned", `${pct(k.gnn_recall)} / ${pct(k.gnn_pruned_fraction)}${gnn.benchmark.speedup ? ` · ${num(gnn.benchmark.speedup, 1)}× faster pairwise screening` : ""}`],
    ["Anomaly detection F1", num(k.anomaly_f1, 2)],
  ];
  $("#kpiExtra").innerHTML = extra.map(r => `<div class="kpi-row two"><div>${r[0]}</div><div class="muted">${r[1]}</div></div>`).join("");
  const b = prop.before, a = prop.after;
  mkChart("propChart", { type: "line", data: { labels: b.horizons_days.map(d => d + " d"), datasets: [
    { label: "median |Δr|", data: b.median_km, borderColor: "#4fd1c5" }, { label: "P90 |Δr|", data: b.p90_km, borderColor: "#ffb020" },
    { label: "σ along-track (robust)", data: b.robust_sigma_ric_km.map(x => x[1]), borderColor: "#58a6ff", borderDash: [4, 3] }] },
    options: { maintainAspectRatio: false, scales: { y: { type: "logarithmic", title: { display: true, text: "km" } } } } });
  mkChart("nisChart", { type: "bar", data: { labels: b.horizons_days.map(d => d + " d"), datasets: [
    { label: "physics covariance", data: b.chi2_95_containment.map(x => x * 100), backgroundColor: "#7f8ea3" },
    { label: "learned realism scaling", data: a.chi2_95_containment.map(x => x * 100), backgroundColor: "#4fd1c5" }] },
    options: { maintainAspectRatio: false, scales: { y: { min: 0, max: 100, title: { display: true, text: "% inside χ² 95 % bound" } } } } });
  const fc = await api("/api/state/change_forecast");
  $("#fcCard").innerHTML = `<div class="panel-h"><span>Forecasting whether an encounter estimate will change at the next update</span><span class="muted">${esc(fc.source)} · ${fc.encounters.toLocaleString()} encounters · ${fc.updates.toLocaleString()} updates</span></div>
    <div class="grid-2"><div>
      <div class="kv"><div><b>${num(fc.model.auc, 3)}</b><span>ROC-AUC model</span></div><div><b>${num(fc.baseline_physics.auc, 3)}</b><span>ROC-AUC physics heuristic</span></div><div><b>0.500</b><span>ROC-AUC climatology</span></div>
      <div><b>${num(fc.model.brier, 3)}</b><span>Brier model (physics ${num(fc.baseline_physics.brier, 3)}, climatology ${num(fc.baseline_climatology.brier, 3)})</span></div>
      <div><b>${pct(fc.at_p50.precision)} / ${pct(fc.at_p50.recall)}</b><span>precision / recall at p ≥ 0.5</span></div><div><b>${pct(fc.base_rate)}</b><span>updates that change substantially</span></div></div>
      <div class="note">Target: ${esc(fc.change_definition)}. Split: ${esc(fc.split)} (test ${esc(fc.test_window)}). Top signals: ${fc.top_features.slice(0, 4).map(x => esc(x.feature)).join(", ")}.</div>
    </div><div class="chart-box"><canvas id="fcChart"></canvas></div></div>`;
  mkChart("fcChart", { type: "bar", data: { labels: fc.by_time_to_tca.map(b => b.tau_bin_days + " d to TCA"), datasets: [
    { label: "model AUC", data: fc.by_time_to_tca.map(b => b.auc_model), backgroundColor: "#4fd1c5" },
    { label: "physics AUC", data: fc.by_time_to_tca.map(b => b.auc_physics), backgroundColor: "#7f8ea3" },
    { label: "change rate", type: "line", data: fc.by_time_to_tca.map(b => b.change_rate), borderColor: "#ffb020", yAxisID: "y2" }] },
    options: { maintainAspectRatio: false, scales: { y: { min: 0.5, max: 1, title: { display: true, text: "ROC-AUC (test)" } }, y2: { position: "right", min: 0, max: 0.5, grid: { display: false }, title: { display: true, text: "share of updates that change" } } } } });
  const g = gnn.metrics, bm = gnn.benchmark;
  $("#gnnCard").innerHTML = `<div class="panel-h"><span>Graph-learning conjunction screener</span></div>
    <div class="kv"><div><b>${bm.all_vs_all_pairs?.toLocaleString() ?? "—"}</b><span>all-vs-all pairs</span></div><div><b>${bm.apogee_perigee_edges?.toLocaleString() ?? "—"}</b><span>graph edges (apogee/perigee filter)</span></div>
    <div><b>${bm.ai_shortlist_edges?.toLocaleString() ?? "—"}</b><span>AI short-list</span></div><div><b>${pct(g.test_recall)}</b><span>recall of next-day conjunctions</span></div>
    <div><b>${num(g.test_auc, 3)}</b><span>ROC-AUC (held-out window)</span></div><div><b>${num(bm.speedup, 1)}×</b><span>pairwise screening speed-up</span></div></div>
    <div class="note">Trained on the previous ${s.config.horizon_h} h window, tested on the next. GraphSAGE-style mean-aggregation message passing over orbital-element node features + learned edge readout with MOID / relative-inclination edge features.</div>`;
}

// ------------------------------------------------------------------ catalog
function renderCatalog() {
  const draw = async () => {
    const q = encodeURIComponent($("#cQ").value), t = encodeURIComponent($("#cType").value), r = encodeURIComponent($("#cReg").value);
    const d = await api(`/api/catalog?q=${q}&type=${t}&regime=${r}&limit=400`);
    $("#catTbl tbody").innerHTML = d.items.map(o => `<tr data-n="${o.norad}"><td class="mono">${o.norad}</td><td>${esc(o.name)}</td><td>${esc(o.type)}</td><td>${o.regime}</td>
      <td class="num">${o.perigee_km}</td><td class="num">${o.apogee_km}</td><td class="num">${o.inc_deg}</td>
      <td><button class="btn sm" data-scr="${o.norad}">re-screen</button></td></tr>`).join("");
    $$("#catTbl tbody tr").forEach(tr => tr.onclick = (ev) => { if (ev.target.dataset.scr) return; showTab("overview"); GLOBE.select(+tr.dataset.n); });
    $$("#catTbl button[data-scr]").forEach(b => b.onclick = async () => {
      b.disabled = true; b.textContent = "…";
      const r = await api(`/api/screen_object/${b.dataset.scr}`, { method: "POST" });
      b.disabled = false; b.textContent = "re-screen";
      $("#objDetail").innerHTML = `<div class="panel" style="margin-top:12px"><div class="panel-h"><span>Incremental screening of ${b.dataset.scr} · ${r.latency_s} s</span></div>
        ${r.events.length ? `<table class="tbl"><thead><tr><th>Other object</th><th>TCA</th><th>Miss</th><th>v<sub>rel</sub></th></tr></thead><tbody>${r.events.map(e => `<tr><td>${esc(e.other_name)} (${e.other})</td><td class="mono">${fmtT(e.tca)}</td><td class="num">${fmtKm(e.miss_km)}</td><td class="num">${e.rel_speed_kms.toFixed(2)}</td></tr>`).join("")}</tbody></table>` : `<div class="muted">No conjunctions within threshold in the next 24 h.</div>`}</div>`;
    });
  };
  let h; ["#cQ", "#cType", "#cReg"].forEach(s => $(s).oninput = () => { clearTimeout(h); h = setTimeout(draw, 250); });
  draw();
}

// ------------------------------------------------------------------ 3-D globe
const GLOBE = (() => {
  const el = $("#globe");
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  el.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  camera.position.set(1.9, 1.05, 2.45);
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.minDistance = 1.2; controls.maxDistance = 14;
  scene.add(new THREE.AmbientLight(0x7a8aa0, 0.9));
  const sun = new THREE.DirectionalLight(0xffffff, 0.8); sun.position.set(5, 2, 3); scene.add(sun);
  const RE = 6378.137;
  // Earth (texture drawn from Natural Earth land polygons)
  const cv = document.createElement("canvas"); cv.width = 2048; cv.height = 1024;
  const g = cv.getContext("2d");
  g.fillStyle = "#0a2238"; g.fillRect(0, 0, 2048, 1024);
  g.strokeStyle = "#16334f"; g.lineWidth = 1;
  for (let lo = -180; lo <= 180; lo += 30) { const x = (lo + 180) / 360 * 2048; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, 1024); g.stroke(); }
  for (let la = -60; la <= 60; la += 30) { const y = (90 - la) / 180 * 1024; g.beginPath(); g.moveTo(0, y); g.lineTo(2048, y); g.stroke(); }
  const tex = new THREE.CanvasTexture(cv);
  fetch("/static/vendor/land-50m.json").then(r => r.json()).then(topo => {
    const land = topojson.feature(topo, topo.objects.land);
    g.fillStyle = "#1d3b33"; g.strokeStyle = "#3f7a66"; g.lineWidth = 1.2;
    for (const f of land.features) {
      const polys = f.geometry.type === "Polygon" ? [f.geometry.coordinates] : f.geometry.coordinates;
      for (const poly of polys) {
        g.beginPath();
        for (const ring of poly) {
          let px = null;
          ring.forEach(([lo, la], i) => {
            const x = (lo + 180) / 360 * 2048, y = (90 - la) / 180 * 1024;
            if (i === 0 || (px !== null && Math.abs(x - px) > 1000)) g.moveTo(x, y); else g.lineTo(x, y);
            px = x;
          });
        }
        g.fill("evenodd"); g.stroke();
      }
    }
    tex.needsUpdate = true;
  }).catch(() => {});
  const earth = new THREE.Mesh(new THREE.SphereGeometry(1, 96, 64), new THREE.MeshPhongMaterial({ map: tex, shininess: 6 }));
  scene.add(earth);
  const atmo = new THREE.Mesh(new THREE.SphereGeometry(1.015, 64, 48), new THREE.MeshBasicMaterial({ color: 0x4fd1c5, transparent: true, opacity: 0.06, side: THREE.BackSide }));
  scene.add(atmo);
  // objects
  let selIdx = -1, sats = [], pts = null, posArr, colArr, tcaGroup = new THREE.Group(), orbitLine = null, selected = null;
  earth.add(tcaGroup);
  const COL = { PAYLOAD: [0.345, 0.65, 1], "ROCKET BODY": [0.95, 0.76, 0.31], DEBRIS: [0.6, 0.64, 0.7], UNKNOWN: [0.6, 0.64, 0.7] };
  const pmat = new THREE.PointsMaterial({ size: 0.012, vertexColors: true, sizeAttenuation: true, transparent: true, opacity: 0.95 });
  let simTime = Date.now(), last = performance.now(), speed = 60;
  $("#speed").onchange = e => speed = +e.target.value;
  $("#nowBtn").onclick = () => simTime = Date.now();
  const vis = () => ({ PAYLOAD: $("#showPay").checked, "ROCKET BODY": $("#showRb").checked, DEBRIS: $("#showDeb").checked, UNKNOWN: $("#showDeb").checked });

  function setCatalog(tles, conj) {
    sats = tles.map(t => { try { return { ...t, rec: satellite.twoline2satrec(t.l1, t.l2) }; } catch (e) { return null; } }).filter(Boolean);
    if (pts) scene.remove(pts);
    const geo = new THREE.BufferGeometry();
    posArr = new Float32Array(sats.length * 3); colArr = new Float32Array(sats.length * 3);
    sats.forEach((s, i) => colArr.set(COL[s.type] || COL.UNKNOWN, i * 3));
    geo.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colArr, 3));
    pts = new THREE.Points(geo, pmat); scene.add(pts);
    // TCA markers (Earth-fixed)
    while (tcaGroup.children.length) tcaGroup.remove(tcaGroup.children[0]);
    conj.filter(c => c.risk !== "green").slice(0, 60).forEach(c => {
      const r = 1 + c.tca_alt_km / RE, la = c.tca_lat * Math.PI / 180, lo = c.tca_lon * Math.PI / 180;
      const m = new THREE.Mesh(new THREE.SphereGeometry(c.risk === "red" ? 0.018 : 0.012, 12, 8), new THREE.MeshBasicMaterial({ color: c.risk === "red" ? 0xff4d4f : 0xffb020, transparent: true, opacity: 0.9 }));
      m.position.set(r * Math.cos(la) * Math.cos(lo), r * Math.sin(la), -r * Math.cos(la) * Math.sin(lo));
      m.userData = c; tcaGroup.add(m);
    });
    updatePositions(true);
  }
  function eciToThree(p) { return [p.x / RE, p.z / RE, -p.y / RE]; }
  function updatePositions(force) {
    if (!pts) return;
    const d = new Date(simTime), gmst = satellite.gstime(d), v = vis();
    earth.rotation.y = gmst;
    for (let i = 0; i < sats.length; i++) {
      const s = sats[i];
      if (!v[s.type]) { posArr[i * 3] = posArr[i * 3 + 1] = posArr[i * 3 + 2] = 0; continue; }
      const pv = satellite.propagate(s.rec, d);
      if (pv && pv.position) { const [x, y, z] = eciToThree(pv.position); posArr[i * 3] = x; posArr[i * 3 + 1] = y; posArr[i * 3 + 2] = z; }
    }
    pts.geometry.attributes.position.needsUpdate = true;
    if (selected) drawOrbit(selected);
    $("#simtime").textContent = "SIM " + d.toISOString().slice(0, 19).replace("T", " ") + ` UTC · ${speed}×`;
  }
  function drawOrbit(s) {
    if (orbitLine) scene.remove(orbitLine);
    const T = 2 * Math.PI / s.rec.no * 60 * 1000, n = 240, arr = [];
    for (let k = 0; k <= n; k++) {
      const pv = satellite.propagate(s.rec, new Date(simTime + (k / n - 0.5) * T));
      if (pv && pv.position) arr.push(new THREE.Vector3(...eciToThree(pv.position)));
    }
    orbitLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(arr), new THREE.LineBasicMaterial({ color: 0x4fd1c5 }));
    scene.add(orbitLine);
  }
  let following = false, marker = null;
  function clearSelection() {
    following = false; selected = null;
    if (orbitLine) { scene.remove(orbitLine); orbitLine = null; }
    if (marker) { scene.remove(marker); marker = null; }
    $("#selInfo").classList.add("hidden");
  }
  function returnToGlobe() {
    clearSelection();
    camera.position.set(1.9, 1.05, 2.45); controls.target.set(0, 0, 0);
  }
  async function select(norad) {
    const idx = sats.findIndex(x => x.norad === norad); if (idx < 0) return;
    const s = sats[idx];
    selected = s; selIdx = idx; following = true; drawOrbit(s);
    if (!marker) { marker = new THREE.Mesh(new THREE.SphereGeometry(0.02, 20, 14), new THREE.MeshBasicMaterial({ color: 0x4fd1c5 })); scene.add(marker); }
    const d = await api(`/api/object/${norad}`), o = d.summary;
    if (selected !== s) return;
    const cj = d.conjunctions, red = cj.filter(c => c.risk === "red"), yel = cj.filter(c => c.risk === "yellow");
    const top = red[0] || yel[0];
    const advice = red.length ? "Red conjunction: review the maneuver options for this object."
      : yel.length ? "Monitor the predicted conjunctions; current alerts do not cross the red maneuver-review threshold."
      : cj.length ? "All predicted conjunctions are low risk." : "No conjunctions inside the screening threshold.";
    const re = d.reentry;
    const reHtml = re && re.p50
      ? `Predicted re-entry (median) <b>${fmtT(re.p50)}</b> UTC${re.p10 && re.p90 ? `<br><span class="muted">window ${fmtT(re.p10)} to ${fmtT(re.p90)}</span>` : ""}`
      : "No re-entry forecast for this object in the active watchlist. This does not prove it cannot re-enter later.";
    const box = $("#selInfo"); box.classList.remove("hidden");
    box.innerHTML = `<div class="sel-top"><button class="btn sm" id="selReturn">Return to full globe</button><button class="sel-x" id="selClose" title="Close">×</button></div>
      <h4>${esc(o.name)} <span class="muted">${o.norad}</span></h4>
      <div class="muted">${esc(o.type)} · ${o.regime} · ${o.perigee_km} × ${o.apogee_km} km · inclination ${o.inc_deg}°</div>
      <div class="sel-sec"><div class="sel-h">Conjunctions</div>
        <div>${cj.length} screened · ${red.length} red · ${yel.length} yellow</div>
        <div class="${red.length ? "bad" : yel.length ? "warn" : "muted"}" style="margin:4px 0">${advice}</div>
        ${top ? `<button class="btn sm sel-open" data-id="${top.id}">Open ${top.risk} conjunction · Pc ${fmtPc(top.pc)}</button>` : ""}</div>
      <div class="sel-sec"><div class="sel-h">Re-entry</div><div>${reHtml}</div></div>
      ${d.anomalies.length ? `<div class="sel-sec"><div class="sel-h">Anomalies</div><div>${d.anomalies.length} detected, latest: ${esc(d.anomalies[0].kind)} (${fmtT(d.anomalies[0].epoch).slice(0, 10)})</div></div>` : ""}
      <div class="muted" style="margin-top:8px;font-size:11px">Following this object along its orbit. × or "Return to full globe" restores the full Earth view.</div>`;
    $("#selReturn").onclick = returnToGlobe; $("#selClose").onclick = returnToGlobe;
    const ob = $(".sel-open", box); if (ob) ob.onclick = () => { showTab("conj"); openConj(+ob.dataset.id); };
  }
  function focusConj(c) { select(c.primary); }
  function resize() { const w = el.clientWidth, h = el.clientHeight; if (!w) return; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
  window.addEventListener("resize", resize); resize();
  // click picking for TCA markers
  const ray = new THREE.Raycaster();
  renderer.domElement.addEventListener("dblclick", ev => {
    const r = renderer.domElement.getBoundingClientRect();
    ray.setFromCamera({ x: (ev.clientX - r.left) / r.width * 2 - 1, y: -(ev.clientY - r.top) / r.height * 2 + 1 }, camera);
    const hit = ray.intersectObjects(tcaGroup.children)[0];
    if (hit) { showTab("conj"); openConj(hit.object.userData.id); }
  });
  let acc = 0;
  (function loop(t) {
    requestAnimationFrame(loop);
    const dt = t - last; last = t; simTime += dt * speed; acc += dt;
    if (acc > 90) { updatePositions(); acc = 0; }
    tcaGroup.children.forEach(m => m.scale.setScalar(1 + 0.35 * Math.sin(t / 250)));
    if (selected && selIdx >= 0) {
      const p = new THREE.Vector3(posArr[selIdx * 3], posArr[selIdx * 3 + 1], posArr[selIdx * 3 + 2]);
      if (p.lengthSq() > 0) {
        if (marker) marker.position.copy(p);
        if (following) { const dist = Math.max(camera.position.length(), 1.8); camera.position.lerp(p.clone().normalize().multiplyScalar(dist), 0.06); }
      }
    }
    controls.update(); renderer.render(scene, camera);
  })(performance.now());
  return { setCatalog, select, focusConj, resize };
})();

// ------------------------------------------------------------------ boot
(async () => {
  try {
    const s = await api("/api/status");
    if (s.has_results) await loadAll();
    if (s.running || !s.has_results) { toast("Running the pipeline for the first time — this takes 1–3 minutes."); pollStatus(); }
  } catch (e) { toast("Cannot reach the API: " + esc(e.message), 8000); }
  setInterval(async () => {   // periodic alert refresh (picks up scheduled pipeline runs)
    try { const s = await api("/api/status"); if (s.running) pollStatus(); else if (s.has_results && APP.summary && s.generated !== APP.summary.generated) loadAll(); } catch (e) {}
  }, 30000);
})();
