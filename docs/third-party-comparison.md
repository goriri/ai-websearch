# Gemini-native vs third-party (SerpApi) architectures

All numbers are from live runs on 2026-10-15, using the SerpApi free plan and Gemini on Vertex AI. The SerpApi cache was turned off (`no_cache`) so latencies are fair, and runs went one after another so they didn't load each other. Images were judged by `gemini-3.8-flash`. Total SerpApi usage was 127 of the 250 free searches.

## Architectures compared

| | A. Gemini-native (default) | B. Third-party search | C. SerpApi Google AI Mode |
|---|---|---|---|
| Answer text | `gemini-3.8-flash` + Grounding with Google Search, streamed | SerpApi `engine=google` top results → `gemini-3.8-flash` writes with `[n]` citations, streamed | Google AI Mode answer (`engine=google_ai_mode`), arrives all at once |
| Images | per section: `gemini-3.1-flash-image` image search → check by `flash-lite` | per section: SerpApi `engine=google_images` → same check | `inline_images` / list thumbnails from the AI Mode response |
| Calls per query | ~10 Gemini (1 answer + ~4 image searches + ~4 checks + 1 related) | ~5 SerpApi + ~6 Gemini | **1 SerpApi**, 0 Gemini |

## 1. Image search alone: same 24 image slots, same download, check and dedupe path

| Image provider | Slots filled | **Correct images (of all slots)** | Correct when filled | Search p50 / p90 | Per-slot p50 / p90 | Empty or timed out |
|---|---|---|---|---|---|---|
| **Gemini image search** | 79% (92% earlier run) | **62.5%** (75% earlier run) | 79% (82%) | **4.2 s / 5.2 s** | **7.1 s / 9.2 s** | 3 |
| **SerpApi Google Images** | 79% | **62.5%** | 79% | 10.2 s / **30 s** (timeout) | 13.3 s / 30 s | 4 (timeouts) |
| Openverse (keyless) | 62% | 38% | 60% | 5.9 s / 9.6 s | 6.2 s / 12.0 s | 4 |

- **Quality is identical** between Gemini image search and SerpApi. Both draw on the Google image index.
- **SerpApi is about 2.5× slower, with a long tail.** Uncached Google Images searches took 1–17 s when probed directly. With a 12 s timeout, 8 of 24 searches failed; with 30 s, 4 still did.

## 2. End to end: 8 queries, production code paths

| Architecture | First text (p50) | Full answer incl. images (p50) | Sources / query | **Images / query** | Images relevant to their text | Answer length |
|---|---|---|---|---|---|---|
| **A. Gemini-native** | 4.8 s | **20.5 s** | 4.9 | **2.9** | **19/23 (83%)** | 3,813 chars |
| **B. Third-party search** | **4.3 s** | 33.2 s | **7.5** | 2.4 | 16/19 (84%) | 3,067 chars |
| **C. SerpApi AI Mode** | 8.5 s (no streaming) | **8.5 s** | 3.6 | **0.1** | 1/1 | 1,272 chars |

### What the AI Mode responses actually contained

- **Images:** across 8 diverse, visual queries (hikes, buildings, birds, street food, interiors, phones, jet engines, Kyoto), AI Mode returned images for **one** query. That was a single thumbnail for Kyoto, which then failed to download. `inline_images`, list thumbnails and `related_questions` were missing almost everywhere. Example response keys: `text_blocks`, `references`, `reconstructed_markdown`.
- **Answer length:** answers are about 3× shorter (a paragraph plus a table, often ending with "let me know if…"). Several had 0–1 sources.
- **Latency:** the full answer takes 7–8.5 s with no streaming. That is faster to complete than A or B, but first text is later than either.

## Conclusions

1. **AI Mode via SerpApi is not a better architecture for this product.** It is the fastest and simplest path to *a* cited text answer, but it **doesn't deliver images** in practice, which is the core requirement. It also gives no control over format or length, and is a third-party scraping dependency. A hybrid (AI Mode text plus Prism image slots) would also lose here: AI Mode writes no image slots, so the images would have to be guessed, and its short answers leave few places for them.
2. **B (SerpApi search plus Gemini writing) works.** Quality matches A (84% vs 83% relevant images), with slightly faster first text and more sources. But it is about 60% slower end to end (33 s vs 20.5 s), because SerpApi image search is slow and unpredictable. It also needs a second vendor plus legal review, and costs about 5 SerpApi searches per query ($0.05–0.125 per query on paid plans).
3. **A (Gemini-native) remains the best default:** fastest full result, most images per query, equal quality, and one first-party bill and set of terms.

**Where SerpApi still makes sense:** as a *web context* provider (`WEB_PROVIDER=serpapi`) when you want more or specific sources or a non-Gemini search backend. Keep `IMAGE_PROVIDER=gemini` for speed. Every combination is now a config switch.

## How to try it

You can compare all three architectures side-by-side in real time using the **Compare portal**:

1. Start the server locally with `SERPAPI_API_KEY` set (required for Architectures B and C).
2. Open `/compare` in your browser (or click the **Compare** button in the top navigation bar).
3. Type any search query to observe live client-side latency (time to first token, full answer time), source counts, and filled image figures across all three architectures simultaneously.

## Reproduce from CLI

```bash
export GOOGLE_CLOUD_PROJECT=<project> SERPAPI_API_KEY=<key> SERPAPI_NO_CACHE=1
WEB_PROVIDER=serpapi IMAGE_PROVIDER=serpapi python -m scripts.cli_search "best beginner hikes in Zion"
```

Select architecture C in the UI with the **AI Mode** toggle (shown when `SERPAPI_API_KEY` is set).
