# Prism Benchmark Suite

This directory contains the benchmarking and empirical validation tools used to evaluate Prism's search architectures, grounding pipelines, and provider integrations. The numbers, latency measurements, and support matrices in the project documentation were generated using these scripts.

## Benchmark Tools

| Script | What It Measures | Documented Result / Backed Doc | Rough Cost |
|---|---|---|---|
| [`bench_arch.py`](bench_arch.py) | End-to-end comparison of **Architecture A** (Gemini-native), **Architecture B** (SerpApi + Gemini), and **Architecture C** (SerpApi AI Mode): time to first token (TTFT), total latency, source counts, image count, and judged relevance across 8 diverse visual queries. | [`docs/third-party-comparison.md`](../../docs/third-party-comparison.md) (Section 2: End to end), [`README.md`](../../README.md) (Three architectures table) | ~130 Gemini calls + ~50 SerpApi searches (~$0.08–0.22/query) |
| [`bench_providers.py`](bench_providers.py) | Head-to-head comparison of image search providers (Gemini image search, SerpApi Google Images, Openverse, Brave) on 24 identical image slots: slot fill rate, judged relevance, search p50/p90, and slot resolution p50/p90. | [`docs/third-party-comparison.md`](../../docs/third-party-comparison.md) (Section 1: Image search alone: same 24 image slots) | ~60–70 model calls / provider (slots cached in `bench_slots.json`) |
| [`bench_calls.py`](bench_calls.py) | Evaluates fewer-Gemini-calls pipeline architectures: current (~2N+1 calls), without visual re-ranking, batched search (~3 calls), and unified single-call (2–3 calls) on latency, yield, and precision. | [`docs/third-party-comparison.md`](../../docs/third-party-comparison.md) (Cost per query / calls per query trade-offs), [`docs/image-search-grounding.md`](../../docs/image-search-grounding.md) | ~30–180 Gemini calls depending on variant |
| [`imgsearch_matrix.py`](imgsearch_matrix.py) | Grounding with Google Search support matrix across 16 text models and 5 image models on Vertex AI, documenting `400 INVALID_ARGUMENT` errors vs supported configurations for `web_search` and `image_search`. | [`docs/image-search-grounding.md`](../../docs/image-search-grounding.md) (Section 1: Model Support Matrix) | ~50–80 Gemini calls (text models fail fast on T1) |
| [`image_models_deep.py`](image_models_deep.py) | Deep-dive comparison of `gemini-nano-banana-2.1` vs `gemini-3.1-flash-image` on single-call web+image search: search trigger reliability, unwanted generated images, grounding support links, attribution relevance, and streaming behavior. | [`docs/image-search-grounding.md`](../../docs/image-search-grounding.md) (Section 6.2: Streaming Incompatibility, Section 6.3: Alternative Image Models) | ~50 Gemini calls (24 generation + ~25 judge calls) |
| [`image_search_no_generation.py`](image_search_no_generation.py) | Verifies that capping `max_output_tokens` (e.g. 128 or 64 tokens) on `gemini-3.1-flash-image` prevents expensive image generation (>1,000 output tokens) while preserving grounding metadata chunks. | [`docs/image-search-grounding.md`](../../docs/image-search-grounding.md) (Section 2: The Exact Image Search Request, Section 5: Zero Image Output Tokens) | 30 Gemini calls to `gemini-3.1-flash-image` |
| [`live_no_generation_check.py`](live_no_generation_check.py) | End-to-end verification of the production pipeline across diverse preferences (`auto`, `ai`, `web`), confirming 0 billable image output tokens and 100% authentic web/Wikimedia photos. | [`docs/image-search-grounding.md`](../../docs/image-search-grounding.md) (Section 5: Slot Fill Accuracy and Zero Image Output Tokens) | ~40 Gemini calls across 4 production queries |
| [`combined_search_placement.py`](combined_search_placement.py) | Evaluates experimental unified web+image search prompting with `{{image_N}}` markers, demonstrating the disconnect between grounding chunk indices and text section placement. | [`docs/image-search-grounding.md`](../../docs/image-search-grounding.md) (Section 6.1: Chunk Mapping Disconnect) | 8 Gemini calls to `gemini-3.1-flash-image` |

## General Notes

1. **Judge Model:** All automated image and caption quality evaluations use `gemini-3.8-flash` with low thinking level (`thinking_level="LOW"`). The judge answers binary or ternary questions (`yes`, `partial`, or `no`) to determine whether the displayed image accurately illustrates the accompanying section text.
2. **Directional Measurements:** Results are empirical and directional. Real latencies, search yields, and token counts depend on live search index updates and network conditions.
3. **Sequential Execution for Latency:** To avoid client-side concurrency bottlenecks or API rate limiting during latency benchmarks, runs should execute sequentially or with bounded concurrency semaphores.
4. **Environment Configuration:**
   - `GOOGLE_CLOUD_PROJECT`: Required for all benchmarks. Must have Vertex AI API enabled.
   - `GOOGLE_CLOUD_LOCATION`: Vertex AI region (default: `global`).
   - `SERPAPI_API_KEY`: Required for SerpApi benchmarks (`bench_arch.py` architectures B/C and `bench_providers.py` SerpApi provider).
   - `SERPAPI_NO_CACHE=1`: Strongly recommended for latency runs to bypass SerpApi's 1-hour cache and measure true search time.
   - `BENCH_OUT_DIR` / `--out-dir`: Specifies output directory for result JSON files (default: `/tmp`).
