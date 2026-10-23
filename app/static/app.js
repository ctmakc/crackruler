(() => {
  "use strict";
  document.body.classList.remove("no-js");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reduced) document.body.classList.add("js-ready");

  const EXPOSURES = [
    ["dry_air", "Dry air or protective membrane", 0.41],
    ["humid", "Humidity, moist air, soil", 0.30],
    ["deicing", "Deicing chemicals", 0.18],
    ["seawater", "Seawater spray, wetting and drying", 0.15],
    ["water_retaining", "Water-retaining structures", 0.10],
  ];
  const ACTIONS = {
    no_crack: ["No crack found", "ok"], log: ["Log it", "ok"], monitor: ["Monitor", "watch"],
    engineer_review: ["Engineer review", "over"], retake: ["Retake the photo", "watch"],
    human_review: ["Person must look", "watch"],
  };
  const $ = (s) => document.querySelector(s);
  const state = { image: null, sample: null, report: null };
  const status = (t, err) => { const s = $("#status"); s.textContent = t; s.classList.toggle("err", !!err); };
  const fmt = (v, d = 2) => (v === null || v === undefined ? "–" : Number(v).toFixed(d));

  // exposure picker: each option shows its guideline as a printed line (8 px per mm, min 1 px)
  const exp = $("#exp");
  EXPOSURES.forEach(([k, label, mm], i) => {
    const l = document.createElement("label");
    l.innerHTML = `<input type="radio" name="exp" value="${k}" ${k === "humid" ? "checked" : ""}>
      <span>${label}</span><span class="line" aria-hidden="true"><i style="height:${Math.max(1, mm * 12).toFixed(1)}px"></i></span>
      <span class="mm">${mm.toFixed(2)} mm</span>`;
    exp.appendChild(l);
  });

  // empty state: the comparator card itself, with the guideline of the chosen exposure
  function preview() {
    if (state.report) return;
    const k = document.querySelector('input[name="exp"]:checked').value;
    const e = EXPOSURES.find((x) => x[0] === k);
    gauge({ unit: "mm", calibrated: true, exposure: { guideline_mm: e[2] }, cracks: [], decision: {} }, "#gauge-preview");
  }
  exp.addEventListener("change", preview);

  // samples
  fetch("/api/samples").then((r) => r.json()).then((items) => {
    const ul = $("#samples");
    items.sort((a, b) => (a.order ?? 99) - (b.order ?? 99));
    items.forEach((it) => {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button";
      b.setAttribute("aria-pressed", "false");
      b.title = `${it.label}. ${it.note || ""}`;
      b.innerHTML = `<img src="/samples/${it.name}" alt="" loading="lazy" width="120" height="80">
        <b>${it.short || it.label}</b><small>${it.synthetic ? "synthetic" : "real, no marker"}</small>`;
      b.addEventListener("click", () => {
        document.querySelectorAll(".samples button").forEach((x) => x.setAttribute("aria-pressed", "false"));
        b.setAttribute("aria-pressed", "true");
        state.sample = it.name; state.image = null;
        $("#thumb").hidden = true;
        $("#run").disabled = false;
        status(`Sample selected: ${it.label || it.name}`);
      });
      li.appendChild(b); ul.appendChild(li);
    });
  }).catch(() => status("Could not load the sample list.", true));

  // upload: downscale on the device to keep the request under the 6 MB Lambda limit
  async function loadFile(f) {
    if (!f) return;
    if (!/^image\/(jpeg|png)$/.test(f.type)) { status("Use a JPEG or PNG photo.", true); return; }
    const bmp = await createImageBitmap(f);
    const k = Math.min(1, 3200 / Math.max(bmp.width, bmp.height));
    const c = document.createElement("canvas");
    c.width = Math.round(bmp.width * k); c.height = Math.round(bmp.height * k);
    c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
    state.image = c.toDataURL("image/jpeg", 0.92);
    state.sample = null;
    document.querySelectorAll(".samples button").forEach((x) => x.setAttribute("aria-pressed", "false"));
    const t = $("#thumb"); t.src = state.image; t.hidden = false;
    $("#run").disabled = false;
    status(`Photo ready: ${c.width} × ${c.height} px${k < 1 ? " (reduced on this device)" : ""}.`);
  }
  $("#file").addEventListener("change", (e) => loadFile(e.target.files[0]));
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => loadFile(e.dataTransfer.files[0]));

  $("#run").addEventListener("click", async () => {
    const btn = $("#run");
    const exposure = document.querySelector('input[name="exp"]:checked').value;
    const marker_mm = parseFloat($("#mm").value);
    if (!(marker_mm >= 5 && marker_mm <= 500)) { status("Marker size must be between 5 and 500 mm.", true); return; }
    const body = { exposure, marker_mm };
    if (state.sample) body.sample = state.sample; else if (state.image) body.image_b64 = state.image; else return;
    btn.disabled = true; btn.classList.add("busy"); btn.textContent = "Measuring…";
    status("Running the inspection agent on AWS Lambda, about 10-25 seconds…");
    const t0 = performance.now();
    try {
      const r = await fetch("/api/inspect", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
      state.report = j;
      render(j, Math.round(performance.now() - t0));
      status(`Done in ${Math.round(performance.now() - t0)} ms (server ${Math.round(j.server_ms)} ms${j.cold_start ? ", cold start" : ""}).`);
    } catch (e) {
      status(`Inspection failed: ${e.message}`, true);
    } finally {
      btn.disabled = false; btn.classList.remove("busy"); btn.textContent = "Measure cracks";
    }
  });

  function render(rep, rtt) {
    $("#empty").hidden = true;
    const res = $("#result"); res.hidden = false;
    const d = rep.decision || {};
    const [word, tone] = ACTIONS[d.action] || [d.action, "watch"];
    const v = $("#verdict"); v.classList.remove("in"); void v.offsetWidth; v.classList.add("in");
    const w = $(".v-word"); w.dataset.tone = tone;
    $("#v-word").textContent = word;
    $("#v-reason").textContent = d.reason || "";
    $("#v-next").textContent = d.next_step || "";
    $("#v-human").hidden = !d.needs_human;

    const img = $("#ann");
    const fig = $(".photo"); fig.classList.remove("in");
    if (rep.annotated_jpeg_b64) { img.src = "data:image/jpeg;base64," + rep.annotated_jpeg_b64; void fig.offsetWidth; fig.classList.add("in"); }
    const q = rep.quality || {};
    const parts = [];
    parts.push(rep.calibrated ? `scale ${fmt(rep.px_per_mm, 2)} px/mm from the marker` : "no marker: pixel units");
    const rect = (rep.trace || []).find((t) => t.tool === "rectify");
    if (rect && rect.observation.psf_sigma_px) parts.push(`blur σ ${fmt(rect.observation.psf_sigma_px, 2)} px`);
    if (rep.resolution) parts.push(`smallest measurable width ${fmt(rep.resolution.min_resolvable_mm, 2)} mm`);
    parts.push(`OpenCV ${rep.engine.opencv} · ${rep.engine.primary} + ${rep.engine.secondary || "none"} · ${rep.arch || ""}`);
    parts.push(`input ${rep.input_size ? rep.input_size.join(" × ") : "?"} px · sharpness ${q.sharpness ?? "–"}`);
    $("#ann-cap").textContent = parts.join(" · ");
    $("#cvv").textContent = rep.engine.opencv;

    gauge(rep);
    table(rep);
    trace(rep, true);
    const blob = new Blob([JSON.stringify(Object.assign({}, rep, { annotated_jpeg_b64: undefined }), null, 2)], { type: "application/json" });
    const a = $("#json"); if (a.dataset.url) URL.revokeObjectURL(a.dataset.url);
    a.href = a.dataset.url = URL.createObjectURL(blob);
    a.download = `crackruler-${rep.id}.json`;
    $("#evidence").focus({ preventScroll: true });
    if (innerWidth < 900) $("#evidence").scrollIntoView({ behavior: reduced ? "auto" : "smooth" });
  }

  // comparator card: printed reference lines with true relative thickness, sqrt scale
  function gauge(rep, target = "#gauge") {
    const unit = rep.unit, cal = rep.calibrated;
    const limit = rep.exposure.guideline_mm;
    const cracks = (rep.cracks || []).filter((c) => c.width_p95 !== null);
    const W = 900, H = cracks.length ? 116 + 12 * Math.min(3, cracks.length) : 110, x0 = 24, x1 = W - 24;
    let max = cal ? Math.max(2, ...cracks.map((c) => c.width_p95 * 1.15)) : Math.max(10, ...cracks.map((c) => c.width_p95 * 1.15));
    const refs = cal ? [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3, 5, 8, 12].filter((r) => r <= max)
                     : [1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 40, 60].filter((r) => r <= max);
    const X = (v) => x0 + (x1 - x0) * Math.sqrt(Math.max(0, v) / max);
    const thick = (v) => Math.max(0.6, Math.min(26, v * (cal ? 9 : 0.9)));
    let s = `<svg viewBox="0 0 ${W} ${H}" aria-hidden="true"><g fill="currentColor">`;
    refs.forEach((r) => {
      s += `<rect x="${(X(r) - thick(r) / 2).toFixed(1)}" y="16" width="${thick(r).toFixed(2)}" height="54"/>`;
      s += `<text x="${X(r).toFixed(1)}" y="88" font-family="Martian Mono, monospace" font-size="10" text-anchor="middle">${r}</text>`;
    });
    s += `<rect x="${x0}" y="16" width="1" height="54" opacity=".35"/><text x="${x0}" y="88" font-family="Martian Mono, monospace" font-size="10" text-anchor="middle">0</text>`;
    s += `</g><text x="${x1}" y="104" font-family="Martian Mono, monospace" font-size="10" text-anchor="end" style="fill:var(--ink-dim)">${cal ? "mm" : "px (no scale reference)"}</text>`;
    if (cal) {
      const lx = X(limit);
      s += `<path d="M${lx - 7} 4 L${lx + 7} 4 L${lx} 14 Z" style="fill:var(--over)"/><line x1="${lx}" x2="${lx}" y1="14" y2="74" style="stroke:var(--over)" stroke-width="1.5" stroke-dasharray="3 3"/>`;
      s += `<text x="${lx + 10}" y="12" font-family="Public Sans, sans-serif" font-size="11" font-weight="700" style="fill:var(--over)">guideline ${limit.toFixed(2)} mm</text>`;
    }
    cracks.forEach((c, i) => {
      const y = 112 + (i % 3) * 12;
      const unc = cal && rep.decision && rep.decision.uncertainty_mm ? rep.decision.uncertainty_mm : 0;
      const a = X(Math.max(0, c.width_p95 - unc)), b = X(c.width_p95 + unc), m = X(c.width_p95);
      const weak = c.confirmed === false;
      s += `<g opacity="${weak ? 0.45 : 1}"><rect class="tickline" x="${x0}" y="${y - 3}" width="${(m - x0).toFixed(1)}" height="6" style="fill:var(--crayon)" stroke="currentColor" stroke-width=".6"/>`;
      if (unc) s += `<line x1="${a}" x2="${b}" y1="${y}" y2="${y}" stroke="currentColor" stroke-width="2"/>`;
      s += `<text x="${(Math.max(m, b) + 8).toFixed(1)}" y="${y + 4}" font-family="Martian Mono, monospace" font-size="10" style="fill:var(--ink)">#${c.id} ${fmt(c.width_p95, cal ? 2 : 1)}${weak ? " unconfirmed" : ""}</text></g>`;
    });
    s += `</svg>`;
    const g = $(target);
    g.innerHTML = s;
    g.setAttribute("aria-label", cracks.length
      ? `Comparator: ${cracks.map((c) => `crack ${c.id} p95 ${fmt(c.width_p95)} ${unit}`).join(", ")}${cal ? `; guideline ${limit} mm` : ""}`
      : "Comparator: no measured crack");
    if (!reduced) g.querySelectorAll(".tickline").forEach((el, i) =>
      el.animate([{ transform: "scaleX(0)" }, { transform: "scaleX(1)" }], { duration: 700, delay: 250 + i * 90, easing: "cubic-bezier(.16,1,.3,1)", fill: "backwards" }));
  }

  function table(rep) {
    const tb = $("#cracks tbody"); tb.innerHTML = "";
    const u = rep.unit, lim = rep.exposure.guideline_mm, cal = rep.calibrated;
    if (!rep.cracks || !rep.cracks.length) { tb.innerHTML = `<tr><td colspan="6">No crack measured.</td></tr>`; return; }
    rep.cracks.forEach((c) => {
      const cls = cal && c.width_p95 !== null ? (c.width_p95 > lim ? "w-over" : c.width_p95 > 0.8 * lim ? "w-watch" : "") : "";
      const tr = document.createElement("tr");
      if (c.confirmed === false) tr.className = "weak";
      const sup = c.support === null ? "n/a" : c.confirmed ? `agrees (${Math.round(c.support * 100)} %)` : `does not see it`;
      tr.innerHTML = `<td>${c.id}</td><td>${fmt(c.length, 0)} ${u}</td><td>${fmt(c.width_median)} ${u}</td>
        <td class="${cls}">${fmt(c.width_p95)} ${u}</td><td>${fmt(c.width_max)} ${u}</td><td>${sup}</td>`;
      tb.appendChild(tr);
    });
  }

  // agent trace, replayed at the real step timings (compressed so the slowest step takes ~600 ms)
  function trace(rep, animate) {
    const ol = $("#trace"); ol.innerHTML = "";
    const steps = rep.trace || [];
    const maxMs = Math.max(1, ...steps.map((t) => t.ms || 0));
    let at = 0;
    steps.forEach((t) => {
      const li = document.createElement("li");
      if (t.tool === "request_retake" || t.tool === "flag_disagreement" || t.tool === "decide") li.classList.add("stop");
      li.innerHTML = `<span class="k">${t.step}</span><span class="tool">${t.tool}${t.args && t.args.method ? ` · ${t.args.method}${t.args.sensitive ? " (sensitive)" : ""}` : ""}${t.args && t.args.enhanced ? " · enhanced" : ""}</span><span class="note">${t.note || ""}</span><span class="ms">${t.ms !== undefined ? Math.round(t.ms) + " ms" : ""}</span>`;
      ol.appendChild(li);
      if (animate && !reduced) {
        li.classList.add("pending");
        at += 90 + 600 * ((t.ms || 0) / maxMs);
        setTimeout(() => li.classList.remove("pending"), at);
      }
    });
  }
  preview();
  $("#replay").addEventListener("click", () => state.report && trace(state.report, true));

  // facts from the committed evaluation results
  fetch("/static/eval-summary.json").then((r) => r.ok ? r.json() : null).then((f) => {
    if (!f) return;
    $("#facts").innerHTML = f.map((x) => `<div class="fact"><b>${x.value}</b><span>${x.label}</span></div>`).join("");
  }).catch(() => {});
})();
