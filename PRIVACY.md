# Privacy Policy: Prism Multimodal Search

_Last updated: 2026-10-15_

Prism is a demo search website that answers natural-language questions with
text and images. This policy describes what the hosted demo does with your data.

## What we collect

- **Sign-in identity.** Access is protected by Google Cloud Identity-Aware
  Proxy (IAP). IAP uses your Google account email only to decide whether you may
  open the site. Prism does not store your email or profile.
- **Search requests.** Your query text, optional uploaded image, and the recent
  conversation are sent to the server to produce an answer. They are not stored
  in a database.
- **Logs.** The server writes operational logs to Google Cloud Logging. These
  include the request time and the first 200 characters of each query. Logs are
  kept for the default Cloud Logging retention period (30 days) and are used only
  for debugging and abuse prevention.

## How your request is processed

To answer a query, Prism sends it (and any uploaded image) to:

- **Google Gemini on Google Cloud (Vertex AI / Gemini Enterprise Agent
  Platform)**, including Grounding with Google Search, under Google Cloud's terms.
- **SerpApi** (only for the "Third-party search" and "AI Mode" architectures in
  the compare portal), which runs the query on Google Search.

Images shown in answers are downloaded from public web pages on your behalf and
linked to their sources.

## What we don't do

- We don't sell or share your data for advertising.
- We don't use your queries to train models.
- We don't use cookies for tracking. The only cookies are the ones IAP sets to
  keep you signed in.

## Your choices

Don't submit personal or confidential information in queries or images. To ask
a question or request deletion of log entries, contact the support email shown
on the sign-in consent screen, or open an issue in this repository.
