# Demo recordings

These are real-time screen recordings of Prism, captured with headless Chrome by
[`scripts/record_demo.py`](../scripts/record_demo.py) against a live
deployment (Gemini on Vertex AI, Google Search grounding, SerpApi). Answers,
sources, images and costs come straight from the APIs; nothing is mocked.

| Recording | What it shows |
|---|---|
| [compare.mp4](compare.mp4) ([GIF, 2×](compare.gif)) | The **compare portal** (`/compare`): one query, "Most beautiful libraries in the world", answered side by side by A Gemini-native, B SerpApi + Gemini and C SerpApi AI Mode. Live metrics per panel: first text, full answer, images, sources and **measured cost**, with "best" badges. Result: A 20.8 s / 3 images / $0.121; B 39.6 s / 4 images / $0.084; C 8.7 s / 0 images / $0.015. |
| [search-mode.mp4](search-mode.mp4) ([GIF, 2×](search-mode.gif)) | Search mode: the grounded answer streams with citations, and image slots fill with verified **real web photos** (generation is off). It ends with search suggestions, related questions and a cost chip (≈ $0.19). |
| [image-query.mp4](image-query.mp4) | Multimodal input: a photo is attached with "What is this building and can I go to the top?" Prism identifies the Burj Khalifa and illustrates the answer with a real photo. |
| [mobile.mp4](mobile.mp4) | The responsive phone layout (390×844), started from an example chip. |

[`older/`](older/) holds recordings from the first version, kept for reference:
Visual mode (Gemini-generated illustrations, now off by default via
`ALLOW_IMAGE_GENERATION=false`) and a conversational follow-up.

`assets/query-photo.jpg`, the photo used in the image-query demo, was
generated with Gemini.

Re-record (needs Chrome, ffmpeg and a running server; `compare` needs
`SERPAPI_API_KEY` and uses about 6 SerpApi searches):

```bash
python -m scripts.record_demo --url http://localhost:8080/ --out demo \
  --only compare search-mode image-query mobile --gif
```
