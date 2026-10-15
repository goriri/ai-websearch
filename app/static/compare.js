/* Prism — Compare Architectures logic */
(() => {
  // SerpApi serves identical queries from a free 1-hour cache in ~0.3 s; such
  // results are labeled and excluded from the "fastest" badge.
  const isCached = (p) => p.arch !== "gemini" && p.total_ms != null && p.total_ms < 1500;

  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  const form = $("#compareForm");
  const input = $("#compareInput");
  const chips = $("#compareChips");
  const columns = $$(".compare-col");

  let availableArchs = ["gemini"];
  const panelState = {};

  function initPanels() {
    for (const col of columns) {
      const arch = col.dataset.arch;
      panelState[arch] = {
        arch,
        col,
        iframe: $("iframe", col),
        disabledCard: $(".disabled-card", col),
        running: false,
        done: false,
        error: false,
        ttft_ms: null,
        total_ms: null,
        images: 0,
        sources: 0,
      };
    }
  }

  function setMetric(col, metric, val, isBest = false) {
    const el = col.querySelector(`[data-metric="${metric}"]`);
    if (!el) return;
    el.textContent = val;
    if (isBest) {
      const badge = document.createElement("span");
      badge.className = "badge-best";
      badge.textContent = "best";
      el.append(" ", badge);
    }
  }

  function evaluateBest() {
    const active = availableArchs
      .map((a) => panelState[a])
      .filter((p) => p && p.done && !p.error && p.total_ms != null && !isCached(p));

    if (active.length === 0) return;

    // Fastest Full answer
    let minTime = Infinity;
    for (const p of active) {
      if (p.total_ms < minTime) minTime = p.total_ms;
    }
    if (minTime < Infinity) {
      for (const p of active) {
        if (p.total_ms === minTime) {
          const valStr = (p.total_ms / 1000).toFixed(1) + "s";
          setMetric(p.col, "total", valStr, true);
        }
      }
    }

    // Most Images
    let maxImages = 0;
    for (const p of active) {
      if (p.images > maxImages) maxImages = p.images;
    }
    if (maxImages > 0) {
      for (const p of active) {
        if (p.images === maxImages) {
          setMetric(p.col, "images", String(p.images), true);
        }
      }
    }
  }

  function runCompare(query) {
    const q = query.trim();
    if (!q) return;

    for (const arch of Object.keys(panelState)) {
      const p = panelState[arch];
      if (!availableArchs.includes(arch)) continue;

      p.running = true;
      p.done = false;
      p.error = false;
      p.ttft_ms = null;
      p.total_ms = null;
      p.images = 0;
      p.sources = 0;

      setMetric(p.col, "ttft", "…");
      setMetric(p.col, "total", "…");
      setMetric(p.col, "images", "…");
      setMetric(p.col, "sources", "…");

      p.iframe.src = `/?embed=1&arch=${encodeURIComponent(arch)}&q=${encodeURIComponent(q)}`;
    }
  }

  window.addEventListener("message", (e) => {
    const d = e.data;
    if (!d || d.type !== "prism-metrics") return;

    const p = panelState[d.arch];
    if (!p) return;

    if (d.phase === "ttft") {
      p.ttft_ms = d.ttft_ms;
      if (d.ttft_ms != null) {
        setMetric(p.col, "ttft", (d.ttft_ms / 1000).toFixed(1) + "s");
      }
    } else if (d.phase === "done") {
      p.running = false;
      p.done = true;
      p.error = false;
      p.ttft_ms = d.ttft_ms;
      p.total_ms = d.total_ms;
      p.images = d.images != null ? d.images : 0;
      p.sources = d.sources != null ? d.sources : 0;

      if (d.ttft_ms != null) {
        setMetric(p.col, "ttft", (d.ttft_ms / 1000).toFixed(1) + "s");
      }
      if (d.total_ms != null) {
        setMetric(p.col, "total", (d.total_ms / 1000).toFixed(1) + "s" + (isCached({ arch: d.arch, total_ms: d.total_ms }) ? " (cached)" : ""));
      }
      setMetric(p.col, "images", String(p.images));
      setMetric(p.col, "sources", String(p.sources));
    } else if (d.phase === "error") {
      p.running = false;
      p.done = true;
      p.error = true;
      p.total_ms = d.total_ms;
      p.images = d.images != null ? d.images : 0;
      p.sources = d.sources != null ? d.sources : 0;

      setMetric(p.col, "total", "Error");
      if (p.images) setMetric(p.col, "images", String(p.images));
      if (p.sources) setMetric(p.sources, "sources", String(p.sources));
    }

    const allFinished = availableArchs.every((a) => {
      const s = panelState[a];
      return !s || s.done;
    });

    if (allFinished) {
      evaluateBest();
    }
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    history.pushState(null, "", `?q=${encodeURIComponent(q)}`);
    runCompare(q);
  });

  chips.addEventListener("click", (e) => {
    const btn = e.target.closest("button.example");
    if (!btn) return;
    const q = btn.dataset.q;
    if (!q) return;
    input.value = q;
    history.pushState(null, "", `?q=${encodeURIComponent(q)}`);
    runCompare(q);
  });

  (async () => {
    initPanels();

    try {
      const cfg = await (await fetch("/api/config")).json();
      if (Array.isArray(cfg.archs)) availableArchs = cfg.archs;
    } catch (_) {
      availableArchs = ["gemini"];
    }

    for (const arch of Object.keys(panelState)) {
      const p = panelState[arch];
      if (!availableArchs.includes(arch)) {
        p.iframe.hidden = true;
        p.disabledCard.hidden = false;
        p.col.classList.add("disabled");
      }
    }

    const params = new URLSearchParams(location.search);
    const q = params.get("q");
    if (q) {
      input.value = q;
      runCompare(q);
    } else {
      input.focus();
    }
  })();
})();
