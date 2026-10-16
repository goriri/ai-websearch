# Design decisions

Short, ADR-style notes on the main choices behind Prism and the alternatives
that were considered.

## 1. Gemini on Vertex AI as the only AI and search provider

**Decision.** Use Gemini on Vertex AI (Gemini Enterprise Agent Platform) for
answering, web research (Grounding with Google Search), image retrieval
(`image_search` grounding), visual verification and image generation.

**Why.** One provider, one auth model (ADC or service account, no API keys),
and grounding that comes with citations and search suggestions. Gemini's image
models can both *find* real images on the web and *generate* new ones, which
covers both meanings of "image results".

**Alternatives.** The Google Programmable Search JSON API (image search),
Unsplash or Pexels (stock only, keys required), and Bing image search. None of
these is needed. The `Candidate` provider interface in `app/images.py` makes it
easy to add one.

## 2. Image slots instead of asking one model to place retrieved images

**Decision.** The answer model writes `[[IMAGE: query | caption]]` markers. A
separate resolver fills each marker.

**Why.**

*   Text models (the best writers and the fastest to first token) don't support
    `image_search`.
*   When the image model does a single combined search-and-write call, its
    internal references (`image_0…N`) don't map reliably onto the returned
    grounding chunks. In testing, images ended up under the wrong headings.
*   Slots let text stream immediately and let images resolve in parallel.
*   A caption is a precise, checkable spec for what the image must show. That
    is what makes visual verification possible.

## 3. Visual re-ranking with a small multimodal model

**Decision.** Download up to 4 candidates and let `gemini-3.1-flash-lite` look
at them and pick one, or none.

**Why.** Search result order and page titles are weak signals. Candidates
often include logos, maps, text-heavy banners, collages or an image of a
nearby place. A visual judge fixes most of these for about 1–2 s per slot, and
it runs off the critical path for text. With `RERANK_IMAGES=false`, Prism
takes the first valid candidate.

## 4. Fallback chain: web → Wikimedia → generated

**Decision.** In the default `auto` mode, an unfilled slot falls back to
Wikimedia Commons (free, licensed, keyless), then to Gemini image generation
with an explicit "AI-generated" badge. `web` disables generation. `ai`
generates every image.

**Why.** Every slot the model asked for should end up with an image if at all
possible, while staying honest about where each image came from.

## 5. Server-Sent Events over `fetch`, not WebSockets

**Decision.** `POST /api/search` returns `text/event-stream`. The browser reads
it with `fetch` + `ReadableStream`.

**Why.** It's one-directional, needs no connection state, works through Cloud
Run and IAP, and the request needs a JSON body (which `EventSource` can't send
as a GET). Cancelling is just `AbortController.abort()`, which cancels the
server task and its image subtasks.

## 6. Images delivered as normalized data URLs

**Decision.** The server re-encodes every image to JPEG (≤1024 px) and sends it
inline in the `image` event.

**Why.** Hotlinking third-party images often fails (hotlink protection, mixed
content, slow hosts). Re-encoding also sanitizes the bytes (no SVG, no animated
or huge files), and keeps the service stateless: no image cache to share
between Cloud Run instances. The cost is about 100–300 KB per image on the
stream, which is acceptable.

## 7. Citations inserted by text matching, not offsets

**Decision.** Use `grounding_supports[].segment.text` and search for it in the
assembled Markdown, instead of using byte offsets.

**Why.** The assembled Markdown differs from the raw model output, because
markers are replaced by slot tokens. Searching in order, starting after the
previous match, keeps repeated phrases attached to the right occurrence.
Matches inside headings are skipped.

## 8. Plain front end with no build step

**Decision.** Plain HTML, CSS and JS, with `marked` and `DOMPurify` vendored.

**Why.** Zero tooling, instant load, easy to read, and the container stays
Python-only. A throttled re-render with persistent figure nodes gives smooth
streaming without a framework.

## 9. Two answer tiers and a fallback model

**Decision.** "Fast" uses `gemini-3.5-flash-lite` with low thinking (it was
`gemini-3.8-flash` until 2026-10-16; in a side-by-side run Flash-Lite reached
first text in 3.0 s vs 5.1 s and finished in 6.2 s vs 16.9 s, at a lower token
price). "Pro" uses
`gemini-3.1-pro-preview` with high thinking. `gemini-2.5-flash` is the fallback
when a model fails before any text is emitted. All model IDs are environment
variables.

**Why.** Users can trade latency for depth, and model availability changes
over time and across projects.

## 10. Locked-down deployment

**Decision.** Cloud Run behind IAP (or private plus `gcloud run services
proxy`), with a dedicated service account that has only `roles/aiplatform.user`
and a per-IP rate limit.

**Why.** Every search spends model and grounding quota, so a public,
unauthenticated endpoint would be an open cost risk.

## 11. Pluggable search providers

**Decision.** Abstract web grounding and image retrieval behind pluggable provider interfaces (`WEB_PROVIDER` and `IMAGE_PROVIDER` in `app/providers.py`), defaulting to Gemini grounding (`gemini`) while supporting SerpApi (Google Web & Images), Brave, and Openverse.

**Why.** While Gemini-native search is the best default, decoupling search retrieval allows benchmarking alternative backends, integrating third-party search APIs when specific non-Gemini web sources are preferred, and testing keyless open providers (Openverse) without altering core streaming or slot-rendering logic.

## 12. SerpApi AI Mode evaluated and not adopted

**Decision.** Evaluate SerpApi Google AI Mode (`engine=google_ai_mode`) as an alternative architecture, support it as an experimental option (`arch="aimode"`), but retain Architecture A (Gemini-native) as the default.

**Why.** Live benchmark measurements demonstrate that SerpApi AI Mode does not satisfy Prism's core requirements:
* **Image scarcity:** Across 8 visual test queries, AI Mode returned images for only one query (a single thumbnail that subsequently failed to download), yielding only 0.1 images/query compared to 2.9 for Gemini-native.
* **Latency and streaming:** While the full answer arrives in 8.5 s, it arrives all at once without streaming, making time-to-first-token worse than Gemini-native (4.8 s) or SerpApi + Gemini (4.3 s).
* **Content depth:** Answers are roughly 3× shorter (1,272 characters vs 3,813 characters) and provide no mechanism for inserting contextual image slots into relevant sections.
* **Vendor complexity:** Introduces third-party scraping dependencies and per-query API costs without quality benefits.

## 13. Compare portal via iframes and postMessage

**Decision.** Build the multi-architecture comparison portal (`/compare`) by embedding three sandboxed `<iframe>` panels running embed mode (`/?embed=1&arch=...`), communicating client-side performance metrics back to the parent window using `postMessage`.

**Why.**
* **Code reuse:** Directly reuses the production search interface, Markdown renderer, and SSE handling without duplicating front-end logic.
* **Isolation:** Iframes isolate DOM trees, styles, and simultaneous SSE streams, preventing race conditions or stream multiplexing bugs.
* **Accurate client metrics:** Measures true end-to-end client-observed user experience (TTFT upon first text event, total duration upon `done`, image count) and relays them via structured `prism-metrics` messages to drive the live scoreboard.
