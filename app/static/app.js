/* Prism — multimodal AI search front-end (no build step). */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const el = (tag, attrs = {}, ...children) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) if (c != null) node.append(c);
    return node;
  };
  const favicon = (domain) =>
    `https://www.google.com/s2/favicons?domain=${encodeURIComponent(domain || "")}&sz=32`;

  const EXAMPLES = [
    ["🏔️", "Best beginner hikes in Zion National Park"],
    ["🍜", "How to make authentic tonkotsu ramen at home"],
    ["🏛️", "Most iconic buildings designed by Zaha Hadid"],
    ["🐦", "Birds you can spot in Central Park in spring"],
    ["🌌", "What has the James Webb telescope discovered recently?"],
    ["🛋️", "Japandi living room design ideas"],
  ];

  const STAGES = {
    search: "Searching the web",
    images: "Finding & verifying images",
    retry: "Retrying with a backup model",
    finishing: "Adding sources",
  };

  const MODE_CHIPS = { search: "⌕ Search", visual: "✦ Visual", aimode: "◎ AI Mode" };
  const ARCH_CHIPS = { gemini: "A · Gemini-native", thirdparty: "B · SerpApi + Gemini", aimode: "C · SerpApi AI Mode", vertexsearch: "D · Vertex AI Search images" };
  const FIGURE_WAIT = {
    search: "Finding the best image…",
    visual: "Generating illustration…",
    aimode: "Loading image…",
  };

  const state = {
    mode: "search",
    arch: null,
    embed: false,
    model: "fast",
    images: "web",
    imageGeneration: false, // from /api/config
    aiMode: false, // from /api/config (server has a SerpApi key)
    attachment: null, // data URL
    history: [], // [{role, text}]
    busy: false,
    controller: null,
    turns: [],
  };

  // ---------------------------------------------------------------- Composer
  const composer = $("#composerTpl").content.firstElementChild.cloneNode(true);
  const q = $("#q", composer);
  const sendBtn = $("#sendBtn", composer);
  const fileInput = $("#fileInput", composer);
  const attachPreview = $("#attachPreview", composer);
  const attachImg = $("#attachImg", composer);
  const SEND_ICON = sendBtn.innerHTML;
  const STOP_ICON =
    '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2.5" fill="currentColor"/></svg>';

  function mountComposer() {
    const landing = document.body.classList.contains("landing");
    const slot = landing ? $("#heroComposerSlot") : $("#dockComposerSlot");
    if (composer.parentElement !== slot) slot.append(composer);
    q.placeholder = landing ? "Ask anything…" : "Ask a follow-up…";
  }

  function autosize() {
    q.style.height = "auto";
    q.style.height = Math.min(q.scrollHeight, 200) + "px";
    updateSendState();
  }
  function updateSendState() {
    sendBtn.disabled = !state.busy && !q.value.trim() && !state.attachment;
  }

  q.addEventListener("input", autosize);
  q.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      composer.requestSubmit();
    }
  });
  composer.addEventListener("submit", (e) => {
    e.preventDefault();
    if (state.busy) return stopSearch();
    const text = q.value.trim();
    if (!text && !state.attachment) return;
    runSearch(text || "What is in this image? Tell me about it.");
  });

  composer.querySelectorAll(".seg button").forEach((b) =>
    b.addEventListener("click", () => setMode(b.dataset.mode))
  );
  function setMode(mode) {
    state.mode = mode;
    composer.querySelectorAll(".seg button").forEach((x) =>
      x.setAttribute("aria-checked", String(x.dataset.mode === mode))
    );
    // Image preference only applies to Search (and only when generation is on);
    // AI Mode answers come from a single SerpApi call, so the model is fixed.
    $("#imgPrefWrap", composer).hidden = mode !== "search" || !state.imageGeneration;
    $("#modelSel", composer).closest("label").hidden = mode === "aimode";
  }
  $("#modelSel", composer).addEventListener("change", (e) => (state.model = e.target.value));
  $("#imgSel", composer).addEventListener("change", (e) => (state.images = e.target.value));

  // Image attachment (button, paste, drag & drop). Downscaled client-side.
  async function attachFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    const url = await new Promise((res, rej) => {
      const r = new FileReader();
      r.onload = () => res(r.result);
      r.onerror = rej;
      r.readAsDataURL(file);
    });
    state.attachment = await downscale(url, 1536);
    attachImg.src = state.attachment;
    attachPreview.hidden = false;
    updateSendState();
    q.focus();
  }
  function downscale(dataUrl, max) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const s = Math.min(1, max / Math.max(img.width, img.height));
        const c = document.createElement("canvas");
        c.width = Math.round(img.width * s);
        c.height = Math.round(img.height * s);
        c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
        resolve(c.toDataURL("image/jpeg", 0.88));
      };
      img.onerror = () => resolve(dataUrl);
      img.src = dataUrl;
    });
  }
  function clearAttachment() {
    state.attachment = null;
    attachPreview.hidden = true;
    attachImg.removeAttribute("src");
    fileInput.value = "";
    updateSendState();
  }
  fileInput.addEventListener("change", () => attachFile(fileInput.files[0]));
  $("#attachRemove", composer).addEventListener("click", clearAttachment);
  q.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) {
      e.preventDefault();
      attachFile(item.getAsFile());
    }
  });
  composer.addEventListener("dragover", (e) => e.preventDefault());
  composer.addEventListener("drop", (e) => {
    e.preventDefault();
    attachFile(e.dataTransfer.files[0]);
  });

  // ------------------------------------------------------------- Landing UI
  const examples = $("#examples");
  for (const [emoji, text] of EXAMPLES) {
    examples.append(
      el("button", { class: "example", type: "button", onclick: () => runSearch(text) },
        el("span", { class: "em", text: emoji }), text)
    );
  }
  function goHome() {
    stopSearch();
    state.history = [];
    state.turns = [];
    $("#thread").replaceChildren();
    document.body.classList.add("landing");
    history.replaceState(null, "", location.pathname);
    q.value = "";
    clearAttachment();
    mountComposer();
    autosize();
    updateCompareHref();
    q.focus();
  }
  $("#homeBtn").addEventListener("click", goHome);
  $("#newBtn").addEventListener("click", goHome);

  function setupCompareBtn() {
    const newBtn = $("#newBtn");
    if (!newBtn || $("#compareBtn")) return;
    const compareBtn = el("a", {
      class: "ghost-btn",
      id: "compareBtn",
      href: "/compare",
      title: "Compare architectures",
    },
    el("svg", { viewBox: "0 0 24 24", width: "16", height: "16", "aria-hidden": "true" },
      el("path", { d: "M9 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h4m6-18h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4M12 3v18", fill: "none", stroke: "currentColor", "stroke-width": "2", "stroke-linecap": "round" })),
    el("span", { text: "Compare" }));
    newBtn.before(compareBtn);
  }

  function updateCompareHref() {
    const btn = $("#compareBtn");
    if (!btn) return;
    const currentQ = q.value.trim() || state.turns[state.turns.length - 1]?.query || "";
    btn.href = currentQ ? `/compare?q=${encodeURIComponent(currentQ)}` : "/compare";
  }
  q.addEventListener("input", updateCompareHref);
  document.addEventListener("keydown", (e) => {
    const typing = ["TEXTAREA", "INPUT", "SELECT"].includes(document.activeElement?.tagName);
    if (e.key === "/" && !typing && $("#lightbox").hidden) {
      e.preventDefault();
      q.focus();
    }
    if (e.key === "Escape") closeLightbox();
  });

  // --------------------------------------------------------------- Lightbox
  function openLightbox(src, caption, href, domain) {
    $("#lbImg").src = src;
    const cap = $("#lbCap");
    cap.replaceChildren(caption || "");
    if (href) {
      cap.append(caption ? " · " : "", el("a", { href, target: "_blank", rel: "noopener", text: domain || "source" }));
    }
    $("#lightbox").hidden = false;
  }
  function closeLightbox() {
    $("#lightbox").hidden = true;
  }
  $("#lightbox").addEventListener("click", (e) => {
    if (e.target.id === "lightbox" || e.target.id === "lbClose") closeLightbox();
  });

  // ------------------------------------------------------------------ Turn
  class Turn {
    constructor(query, opts) {
      this.query = query;
      this.opts = opts;
      this.md = "";
      this.slots = new Map();
      this.sources = [];
      this.renderQueued = false;
      this.done = false;
      this.t0 = performance.now();

      this.statusEl = el("span", { class: "status" },
        el("span", { class: "dots" }, el("i"), el("i"), el("i")),
        el("span", { class: "status-text", text: "Thinking" }));
      this.metaEl = el("div", { class: "turn-meta" }, this.statusEl);
      this.sourcesEl = el("div", { class: "sources", hidden: true });
      this.answerEl = el("div", { class: "answer streaming" });
      this.gsearchEl = el("div", { class: "gsearch", hidden: true });
      this.relatedEl = el("div", { class: "related", hidden: true });
      this.footerEl = el("div", { class: "footer-meta" });
      this.el = el("article", { class: "turn" },
        el("h2", { class: "turn-q", text: query }),
        opts.image ? el("img", { class: "turn-qimg", src: opts.image, alt: "Query image" }) : null,
        this.metaEl, this.sourcesEl, this.answerEl, this.gsearchEl, this.relatedEl, this.footerEl);
    }

    setStatus(text) {
      $(".status-text", this.statusEl).textContent = text;
    }

    // --- event handlers
    cost(d) { this.costData = d; }
    status(d) { this.setStatus(STAGES[d.stage] || d.message); }
    text(d) { this.md += d.delta; this.queueRender(); }
    slot(d) {
      this.md += `\n\n@@SLOT:${d.id}@@\n\n`;
      this.slots.set(d.id, this.makeFigure(d));
      this.queueRender();
    }
    image(d) {
      const fig = this.slots.get(d.id);
      if (fig) this.fillFigure(fig, d);
    }
    image_failed(d) {
      const fig = this.slots.get(d.id);
      if (fig) fig.classList.add("failed");
    }
    final(d) {
      if (d.markdown) this.md = d.markdown;
      this.sources = d.sources || [];
      this.model = d.model;
      this.renderSources();
      this.render();
      this.answerEl.classList.remove("streaming");
      this.renderSearchEntryPoint(d.search_entry_point);
      const pending = [...this.slots.values()].some((f) => !f.classList.contains("ready") && !f.classList.contains("failed"));
      this.setStatus(pending ? STAGES.images : "Wrapping up");
    }
    related(d) {
      const qs = d.questions || [];
      if (!qs.length) return;
      this.relatedEl.replaceChildren(
        el("h4", {}, svgIcon("related"), "Related"),
        ...qs.map((x) => el("button", { type: "button", onclick: () => runSearch(x) },
          el("span", { text: x }), svgIcon("plus")))
      );
      this.relatedEl.hidden = false;
    }
    error(d) {
      this.answerEl.after(el("div", { class: "error-box", text: d.message || "Something went wrong." }));
    }
    finish(d) {
      if (this.done) return;
      this.done = true;
      this.answerEl.classList.remove("streaming");
      this.statusEl.remove();
      const secs = ((d && d.elapsed_ms) || performance.now() - this.t0) / 1000;
      const nImg = [...this.slots.values()].filter((f) => f.classList.contains("ready") || f._filled).length;
      const label = ARCH_CHIPS[state.arch] || MODE_CHIPS[this.opts.mode] || MODE_CHIPS.search;
      let costChip = null;
      if (this.costData && typeof this.costData.total_usd === "number") {
        const total = this.costData.total_usd;
        const text = (total > 0 && total < 0.0001) ? "< $0.0001" : `≈ $${total.toFixed(4)}`;
        const lines = (this.costData.items || []).map((it) => {
          const usdStr = typeof it.usd === "number" ? `$${it.usd.toFixed(4)}` : "";
          return `${it.label} · ${it.detail} — ${usdStr}`;
        });
        if (this.costData.note) lines.push(this.costData.note);
        costChip = el("span", { class: "chip", title: lines.join("\n"), text });
      }
      this.metaEl.append(...[  // DOM append() would render null as text: filter first.
        el("span", { class: "chip", text: label }),
        this.sources.length ? el("span", { class: "chip", text: `${this.sources.length} sources` }) : null,
        nImg ? el("span", { class: "chip", text: `${nImg} images` }) : null,
        el("span", { class: "chip", text: `${secs.toFixed(1)}s` }),
        costChip,
        secs < 1.5 && ["thirdparty", "aimode"].includes(state.arch)
          ? el("span", { class: "chip", title: "Served from SerpApi's 1-hour cache", text: "cached" }) : null,
      ].filter(Boolean));
      // Remove figures that never resolved.
      for (const f of this.slots.values()) if (!f.classList.contains("ready") && !f._filled) f.classList.add("failed");
      if (this.model) {
        this.footerEl.textContent = this.model.startsWith("serpapi:")
          ? "Answer by Google AI Mode (via SerpApi)"
          : `Answer by ${this.model} on Gemini Enterprise Agent Platform (Vertex AI)`;
      }
    }

    plainText() {
      return this.md.replace(/@@(SLOT|CITE):[^@]*@@/g, "").replace(/\n{3,}/g, "\n\n").trim();
    }

    // --- rendering
    queueRender() {
      if (this.renderQueued) return;
      this.renderQueued = true;
      requestAnimationFrame(() => {
        this.renderQueued = false;
        this.render();
      });
    }

    render() {
      const src = this.md
        .replace(/@@SLOT:([\w-]+)@@/g, (_, id) => `<div data-slot="${id}"></div>`)
        .replace(/@@CITE:([\d,]+)@@/g, (_, n) => `<span data-cite="${n}"></span>`);
      const html = DOMPurify.sanitize(marked.parse(src, { gfm: true, breaks: false }));
      this.answerEl.innerHTML = html;
      this.answerEl.querySelectorAll("div[data-slot]").forEach((a) => {
        const fig = this.slots.get(a.dataset.slot);
        if (fig) a.replaceWith(fig);
        else a.remove();
      });
      this.answerEl.querySelectorAll("span[data-cite]").forEach((s) => {
        const nums = s.dataset.cite.split(",").map(Number);
        const links = nums
          .map((n) => this.sources.find((x) => x.n === n))
          .filter(Boolean)
          .map((src) => el("a", { href: src.url, target: "_blank", rel: "noopener", title: `${src.title} — ${src.domain}`, text: String(src.n) }));
        if (links.length) s.replaceWith(el("span", { class: "cite" }, links));
        else s.remove();
      });
      this.answerEl.querySelectorAll("table").forEach((t) => {
        if (!t.parentElement.classList.contains("table-wrap")) {
          const w = el("div", { class: "table-wrap" });
          t.replaceWith(w);
          w.append(t);
        }
      });
      this.answerEl.querySelectorAll("a[href]").forEach((a) => {
        if (!a.closest(".slot")) {
          a.target = "_blank";
          a.rel = "noopener";
        }
      });
    }

    makeFigure(d) {
      const img = el("img", { alt: d.caption || "", decoding: "async" });
      const bg = el("div", { class: "bgblur" });
      const finding = el("div", { class: "finding" },
        el("div", {}, FIGURE_WAIT[this.opts.mode] || FIGURE_WAIT.search,
          d.query ? el("b", { text: `“${d.query}”` }) : null));
      const frame = el("div", { class: "frame" }, bg, img, el("div", { class: "shimmer" }), finding);
      const cap = el("figcaption", {}, d.caption ? el("span", { class: "cap", text: d.caption }) : null);
      const fig = el("figure", { class: "slot" }, frame, cap);
      fig._img = img; fig._bg = bg; fig._cap = cap; fig._frame = frame;
      fig._filled = false;
      setTimeout(() => (fig.style.animation = "none"), 400); // Avoid replaying on re-render.
      return fig;
    }

    fillFigure(fig, d) {
      fig._filled = true;
      fig._img.onload = () => fig.classList.add("ready");
      fig._img.src = d.src;
      if (fig._img.complete && fig._img.naturalWidth > 0) fig.classList.add("ready");
      fig._img.alt = d.caption || d.title || "";
      fig._bg.style.backgroundImage = `url("${d.src}")`;
      if (d.width && d.height) fig._img.style.aspectRatio = `${d.width} / ${d.height}`;
      const meta = [];
      if (d.kind === "generated") {
        meta.push(el("span", { class: "badge ai", title: "This image was generated by Gemini and may not be accurate." }, "✦ AI-generated"));
      } else if (d.source_url) {
        meta.push(el("a", { class: "srcl", href: d.source_url, target: "_blank", rel: "noopener", title: d.title || d.domain },
          el("img", { src: favicon(d.domain), alt: "", loading: "lazy" }),
          el("span", { text: d.domain || "source" })));
        if (d.license) meta.push(el("span", { class: "srcl", text: d.license }));
      }
      if (!d.caption && d.kind === "generated") fig.classList.add("compact");
      fig._cap.append(...meta);
      fig._frame.onclick = () => openLightbox(d.src, d.caption || d.title, d.source_url, d.domain);
    }

    renderSources() {
      if (!this.sources.length) return;
      const row = el("div", { class: "sources-row" },
        this.sources.map((s) =>
          el("a", { class: "src-card", href: s.url, target: "_blank", rel: "noopener", title: s.title },
            el("span", { class: "t", text: sourceTitle(s) }),
            el("span", { class: "d" }, el("img", { src: favicon(s.domain), alt: "", loading: "lazy" }),
              el("span", { text: s.domain }), el("span", { class: "n", text: String(s.n) })))
        ));
      this.sourcesEl.replaceChildren(
        el("div", { class: "sources-head" }, svgIcon("sources"), "Sources"), row);
      this.sourcesEl.hidden = false;
    }

    renderSearchEntryPoint(html) {
      // Google Search grounding requires showing the provided Search Suggestions.
      if (!html) return;
      const host = el("div", { class: "gsearch-host" });
      host.attachShadow({ mode: "open" }).innerHTML = html;
      this.gsearchEl.replaceChildren(host);
      this.gsearchEl.hidden = false;
    }
  }

  // When the page title is unknown (it equals the domain), derive a readable
  // title from the URL slug, e.g. ".../best-easy-hikes-zion/" -> "Best easy hikes zion".
  function sourceTitle(s) {
    if (s.title && s.title !== s.domain) return s.title;
    try {
      const u = new URL(s.url);
      // Unresolved grounding redirects carry an opaque token, not a page slug.
      if (u.hostname.endsWith("vertexaisearch.cloud.google.com")) return s.title || s.domain;
      const segs = u.pathname.split("/").filter(Boolean);
      const slug = decodeURIComponent(segs[segs.length - 1] || "")
        .replace(/\.(html?|php|aspx?)$/i, "")
        .replace(/[-_+]+/g, " ")
        .trim();
      if (slug.length > 3 && /[a-z]/i.test(slug) && !/^\d+$/.test(slug)) {
        return slug.charAt(0).toUpperCase() + slug.slice(1);
      }
    } catch (_) { /* ignore */ }
    return s.title || s.domain;
  }

  function svgIcon(name) {
    const paths = {
      related: '<path d="M4 6h16M4 12h10M4 18h7" stroke="currentColor" stroke-width="2" stroke-linecap="round" fill="none"/>',
      plus: '<path d="M12 5v14M5 12h14" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
      sources: '<path d="M4 5h11a3 3 0 0 1 3 3v11H7a3 3 0 0 1-3-3z M8 9h6M8 13h6" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linejoin="round"/>',
    };
    const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    s.setAttribute("viewBox", "0 0 24 24");
    s.setAttribute("width", "16");
    s.setAttribute("height", "16");
    s.setAttribute("aria-hidden", "true");
    s.innerHTML = paths[name] || "";
    return s;
  }

  // ------------------------------------------------------------ Networking
  async function streamSSE(body, onEvent, signal) {
    const resp = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    if (!resp.ok) {
      let msg = `${resp.status} ${resp.statusText}`;
      try {
        const j = await resp.json();
        msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
      } catch (_) { /* ignore */ }
      throw new Error(msg);
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        let event = "message";
        const data = [];
        for (const line of raw.split("\n")) {
          if (line.startsWith(":")) continue;
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
        }
        if (data.length) onEvent(event, JSON.parse(data.join("\n")));
      }
    }
  }

  function setBusy(b) {
    state.busy = b;
    sendBtn.classList.toggle("busy", b);
    sendBtn.innerHTML = b ? STOP_ICON : SEND_ICON;
    sendBtn.title = b ? "Stop" : "Search (Enter)";
    updateSendState();
  }

  function stopSearch() {
    if (state.controller) state.controller.abort();
  }

  async function runSearch(query) {
    if (state.busy) stopSearch();
    const opts = { mode: state.mode, model: state.model, images: state.images, image: state.attachment, arch: state.arch };
    const first = document.body.classList.contains("landing");
    document.body.classList.remove("landing");
    mountComposer();
    if (first && !state.embed) history.replaceState(null, "", `?q=${encodeURIComponent(query)}${state.mode !== "search" ? `&mode=${state.mode}` : ""}`);

    const turn = new Turn(query, opts);
    state.turns.push(turn);
    $("#thread").append(turn.el);
    if (!state.embed) {
      turn.el.scrollIntoView({ behavior: first ? "auto" : "smooth", block: "start" });
    }

    q.value = "";
    clearAttachment();
    autosize();

    const controller = new AbortController();
    state.controller = controller;
    setBusy(true);
    const historySnapshot = state.history.slice(-8);

    const t0 = performance.now();
    let ttft_ms = null;
    let total_ms = null;
    let hadError = false;

    function sendMetrics(phase) {
      if (!state.embed || !window.parent) return;
      const nImg = [...turn.slots.values()].filter((f) => f.classList.contains("ready") || f._filled).length;
      const payload = {
        type: "prism-metrics",
        arch: state.arch,
        phase,
        ttft_ms,
        total_ms,
        images: nImg,
        sources: turn.sources ? turn.sources.length : 0,
        chars: turn.plainText().length,
      };
      if (phase === "done" && turn.costData) {
        payload.cost_usd = turn.costData.total_usd;
        payload.cost_items = turn.costData.items;
      }
      window.parent.postMessage(payload, "*");
    }

    const body = {
      query,
      mode: opts.mode,
      model: opts.model,
      images: opts.images,
      image: opts.image,
      history: historySnapshot,
    };
    if (state.arch) body.arch = state.arch;

    try {
      await streamSSE(
        body,
        (event, data) => {
          if (event === "text" && ttft_ms === null) {
            ttft_ms = Math.round(performance.now() - t0);
            sendMetrics("ttft");
          }
          if (event === "error") {
            hadError = true;
            total_ms = Math.round(performance.now() - t0);
            turn.error(data);
            sendMetrics("error");
            return;
          }
          if (event === "done") {
            total_ms = Math.round(performance.now() - t0);
            turn.finish(data);
            if (!hadError) sendMetrics("done");
            return;
          }
          const fn = turn[event];
          if (typeof fn === "function") fn.call(turn, data);
        },
        controller.signal
      );
    } catch (err) {
      hadError = true;
      total_ms = Math.round(performance.now() - t0);
      if (err.name !== "AbortError") turn.error({ message: err.message });
      else turn.error({ message: "Stopped." });
      sendMetrics("error");
    } finally {
      turn.finish();
      if (state.controller === controller) {
        state.controller = null;
        setBusy(false);
      }
      state.history.push({ role: "user", text: query }, { role: "model", text: turn.plainText().slice(0, 6000) });
      updateCompareHref();
    }
  }

  // When the server disables image generation (cost control), hide Visual
  // mode and the image-source choices that would generate images. AI Mode is
  // offered only when the server has a SerpApi key. The mode switch stays
  // visible whenever there is more than one mode to choose from.
  function applyServerConfig(cfg) {
    state.imageGeneration = !!cfg.image_generation;
    state.aiMode = !!cfg.ai_mode;
    const seg = composer.querySelector(".seg");
    if (!state.aiMode) seg.querySelector('button[data-mode="aimode"]')?.remove();
    if (!state.imageGeneration) {
      seg.querySelector('button[data-mode="visual"]')?.remove();
      state.images = "web";
    }
    seg.hidden = seg.querySelectorAll("button").length < 2;
    setMode("search");
  }

  // ------------------------------------------------------------------ Boot
  (async () => {
    const params = new URLSearchParams(location.search);
    state.embed = params.get("embed") === "1";
    state.arch = params.get("arch") || null;
    if (state.embed) {
      document.body.classList.add("embed");
    }
    mountComposer();
    autosize();
    setupCompareBtn();
    try {
      applyServerConfig(await (await fetch("/api/config")).json());
    } catch (_) {
      applyServerConfig({ image_generation: false, ai_mode: false });
    }
    if (params.get("mode") === "visual" && state.imageGeneration) setMode("visual");
    if (params.get("mode") === "aimode" && state.aiMode) setMode("aimode");
    if (params.get("q")) runSearch(params.get("q"));
    else if (!state.embed) q.focus();
  })();
})();
