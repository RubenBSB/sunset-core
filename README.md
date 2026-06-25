# 🌅 sunset-core

Reusable **async** services for AI applications — auth, LLM routing, RAG, storage, pub/sub, and more — in one pip-installable package.

Every service is async-first (`asyncpg`, `aio` methods), lazy-loaded (you only import the heavy dependency you actually use), and provider-agnostic where it makes sense (LLM routing, multi-provider email, etc.).

## Install

```bash
pip install sunset-core              # base: auth-less, light footprint
pip install "sunset-core[llm]"       # add the LLM providers
pip install "sunset-core[all]"       # everything
```

Extras map to service groups: `auth`, `llm`, `retrieval`, `gcp`, `analytics`, `email`, `crawl`, `redis`, `asr`, `youtube`, `observability`.

## Usage

Services are imported from the `sunset.services` namespace:

```python
from sunset.services import LLMService, RetrievalService, AuthService

llm = LLMService(...)
response = await llm.chat(model="gemini-2.5-flash", messages=[...])
```

### RetrievalService (RAG)

```python
from sunset.services.retrieval import RetrievalService

svc = RetrievalService(dsn=DATABASE_URL, project=GCP_PROJECT)
await svc.connect()
chunks = await svc.ingest_document("report.pdf", describe_images=True)
results = await svc.query("What is the refund policy?", top_k=5)
await svc.close()
```

## Available services

| Service                 | What it does                                                        |
| ----------------------- | ------------------------------------------------------------------ |
| `AuthService`           | JWT with refresh rotation, OAuth (Google, Apple, Microsoft)        |
| `LLMService`            | Multi-provider LLM (OpenAI / Gemini), tool use, structured output  |
| `RetrievalService`      | RAG: Docling parsing, chunking, Vertex AI embeddings, pgvector     |
| `StorageService`        | GCS upload, signed URLs, deletion                                  |
| `PubSubService`         | Google Cloud Pub/Sub messaging                                     |
| `SecretsService`        | Google Secret Manager                                              |
| `EmailSendService`      | Transactional email (Resend / SendGrid)                            |
| `WhatsappService`       | WhatsApp Cloud API messaging + webhooks                            |
| `SlackService`          | Slack OAuth v2 + Web API                                           |
| `AnalyticsService`      | PostHog event tracking                                             |
| `MonitoringService`     | Google Cloud Monitoring                                            |
| `CrawlService`          | Web crawling (Playwright / Firecrawl)                             |
| `RedisService`          | Async Redis client                                                 |
| `ASRService`            | Speech-to-text (Deepgram)                                          |
| `init_observability`    | OpenTelemetry bootstrap: traces + metrics, LLM cost/token metrics  |

…plus `InstagramService`, `YouTubeService`, `ShopifyService`, `GoogleDriveService`, `HubspotService`, `SEOService`, `DuffelService`, `BookingService`, `ChatService`, `MultimodalEmbeddingService`.

Each service ships its own integration guide at `sunset/services/<name>/README.md`.

## License

[Apache-2.0](LICENSE)
