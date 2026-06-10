"""Multimodal embedding via Vertex AI ``multimodalembedding@001``.

Sibling of :class:`~sunset.services.retrieval.RetrievalService`. That service is
text-only (``gemini-embedding-001``, 768-d) and cannot embed images or video.
This one maps text, images and video into the *same* 1408-d space, so a text
query embedding can be compared directly (cosine) against image/video
embeddings — true cross-modal search.

It calls the REST ``:predict`` endpoint directly rather than the deprecated
``vertexai.vision_models`` SDK (removed 2026-06-24). Embedding only — vector
storage / search stays in the consuming project (the domain data lives there),
mirroring how RetrievalService leaves your ``Document`` model to you.
"""

import base64
import logging
import os
import random
import time
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import google.auth
from google.auth.transport.requests import AuthorizedSession

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "multimodalembedding@001"
# 1408 is the only dimension the video path supports; used everywhere so
# image/text/video vectors live in one comparable space.
EMBEDDING_DIMENSIONS = 1408
# One embedding per N seconds of video (a row per segment → search lands on a moment).
VIDEO_INTERVAL_SEC = 15
# multimodalembedding@001 has a low per-minute quota; retry transient throttling
# (429) and unavailability (503) with exponential backoff + jitter so a burst of
# files drains at the sustainable rate instead of failing.
_MAX_RETRIES = 6
_RETRY_STATUS = (429, 503)


@lru_cache(maxsize=1)
def _default_creds():
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    return creds


class MultimodalEmbeddingService:
    """Cross-modal (text / image / video) embeddings from a single Vertex model.

    All three methods return vectors in the same 1408-d space. Image/video bytes
    are sent base64-inline; for very large videos (beyond the request size limit)
    pass them via a GCS URI instead — see ``embed_video``.
    """

    def __init__(
        self,
        project: str | None = None,
        region: str | None = None,
        dimensions: int = EMBEDDING_DIMENSIONS,
    ):
        self.project = project or os.environ["GCP_PROJECT"]
        self.region = region or os.environ.get("GCP_REGION", "europe-west1")
        self.dimensions = dimensions

    # ─── Internals ────────────────────────────────────────────────────────

    @property
    def _endpoint(self) -> str:
        return (
            f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project}"
            f"/locations/{self.region}/publishers/google/models/{EMBEDDING_MODEL}:predict"
        )

    def _predict(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        body = {"instances": [instance], "parameters": {"dimension": self.dimensions}}
        for attempt in range(_MAX_RETRIES + 1):
            # Session per call (cheap; creds cached) keeps it safe across worker threads.
            resp = AuthorizedSession(_default_creds()).post(self._endpoint, json=body)
            if resp.status_code in _RETRY_STATUS and attempt < _MAX_RETRIES:
                retry_after = resp.headers.get("Retry-After", "")
                delay = (
                    float(retry_after)
                    if retry_after.isdigit()
                    else min(2**attempt, 32) + random.uniform(0, 1)
                )
                logger.warning(
                    "multimodalembedding %s — backing off %.1fs (attempt %d/%d)",
                    resp.status_code,
                    delay,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                time.sleep(delay)
                continue
            resp.raise_for_status()
            return resp.json()["predictions"][0]

    # ─── Public API ───────────────────────────────────────────────────────

    def embed_text(self, text: str) -> List[float]:
        return self._predict({"text": text})["textEmbedding"]

    def embed_image(
        self, image_bytes: bytes, mime_type: str = "image/jpeg"
    ) -> List[float]:
        b64 = base64.b64encode(image_bytes).decode()
        return self._predict({"image": {"bytesBase64Encoded": b64}})["imageEmbedding"]

    def embed_video(
        self, path: str, interval_sec: int = VIDEO_INTERVAL_SEC
    ) -> List[Tuple[int, List[float]]]:
        """Return ``[(start_offset_sec, embedding), ...]`` — one entry per segment."""
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        pred = self._predict(
            {
                "video": {
                    "bytesBase64Encoded": b64,
                    "videoSegmentConfig": {"intervalSec": interval_sec},
                }
            }
        )
        return [
            (int(seg.get("startOffsetSec", 0)), seg["embedding"])
            for seg in pred["videoEmbeddings"]
        ]
