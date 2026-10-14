# Demo recordings

Real-time screen recordings of Prism. They were captured with headless Chrome
by [`scripts/record_demo.py`](../scripts/record_demo.py) against a live
deployment (Gemini on Vertex AI). Answers, sources and images come straight
from the models and Google Search. Nothing is mocked.

| Recording | What it shows |
|---|---|
| [search-mode.mp4](search-mode.mp4) ([GIF, 2x](search-mode.gif)) | Typing a query. The grounded answer streams in, image slots appear as shimmer cards and are filled with verified real web photos. Then citations, source cards, Google Search suggestions and related searches. |
| [visual-mode.mp4](visual-mode.mp4) ([GIF, 2x](visual-mode.gif)) | Visual mode: Gemini's native interleaved output, with generated illustrations placed inline (each labeled "AI-generated"). |
| [follow-up.mp4](follow-up.mp4) | A conversational follow-up ("Which of these can I visit in Europe…") that uses the previous answer as context. |
| [image-query.mp4](image-query.mp4) | Multimodal input: attach a photo and ask "What is this building and can I go to the top?" |
| [mobile.mp4](mobile.mp4) | Responsive layout on a phone-sized viewport, started from an example chip. |

`assets/query-photo.jpg`, the photo used in the image-query demo, was
generated with Gemini.

Re-record (needs Chrome, ffmpeg and a running server):

```bash
python -m scripts.record_demo --url http://localhost:8080/ --out demo --gif
```
