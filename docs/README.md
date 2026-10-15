# Prism Documentation

Technical documentation, architecture guides, and empirical benchmarks for Prism.

## Documents

*   [Architecture](architecture.md): System overview, request lifecycle, streaming Server-Sent Events protocol, slot resolution, three-architecture routing, and the `/compare` portal.
*   [Design Decisions](design-decisions.md): Architecture Decision Records (ADRs) explaining key design trade-offs, including image slots, client-side streaming, provider abstractions, and why AI Mode was not chosen as the default.
*   [Image Search Grounding](image-search-grounding.md): How Prism uses Gemini's image models to retrieve real web photos via Google image search grounding without generating images or spending output tokens.
*   [Third-Party Search Comparison](third-party-comparison.md): Empirical benchmark comparing Gemini-native search against SerpApi (Google Web + Images) and SerpApi Google AI Mode across latency, image yield, source count, and relevance.
