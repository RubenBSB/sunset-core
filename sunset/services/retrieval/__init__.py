"""pgvector-based retrieval service for RAG pipelines.

Provides text embedding (Vertex AI gemini-embedding-001), document ingestion
with Docling-powered parsing and chunking, and cosine-similarity search over
a Cloud SQL PostgreSQL database with the pgvector extension.
"""

import io
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import asyncpg
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from google import genai
from google.genai import types
from pgvector.asyncpg import register_vector
from pydantic import ConfigDict

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768

IMAGE_DESCRIBE_PROMPT = (
    "Describe this image from a document in detail, including any text, "
    "data, charts, diagrams, or visual elements present."
)
IMAGE_DESCRIBE_MODEL = "gemini-2.5-flash"


class VertexAITokenizer(BaseTokenizer):
    """Tokenizer aligned with Gemini embedding models.

    Uses Google's local SentencePiece tokenizer (same vocabulary as
    Gemini/Gemma models) for fast, offline token counting.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_tokens: int = 2000
    _local_tokenizer: Any = None

    def model_post_init(self, __context: Any) -> None:
        from google.genai.local_tokenizer import LocalTokenizer

        self._local_tokenizer = LocalTokenizer(model_name="gemini-2.5-flash")

    def count_tokens(self, text: str) -> int:
        return self._local_tokenizer.count_tokens(text).total_tokens

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self) -> Any:
        return self._local_tokenizer


_FILTER_OPS = {
    "$eq": "=",
    "$ne": "!=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}

_COLUMN_KEYS = {"source_file", "content_type"}


def _build_where(
    filter_dict: Dict[str, Any], param_offset: int
) -> Tuple[str, List[Any]]:
    """Compile a metadata filter dict into a parameterised SQL WHERE clause.

    Supports:
      - Equality shorthand: ``{"key": "value"}``
      - Operators: ``{"key": {"$gt": 10, "$lte": 100}}``
      - ``$in``: ``{"key": {"$in": [1, 2, 3]}}``
      - Top-level columns: ``source_file`` and ``content_type`` target
        their columns directly instead of ``metadata->>``.

    ``param_offset`` is the next available ``$N`` placeholder index (1-based).
    Returns ``(sql_fragment, params)`` where *sql_fragment* starts with
    ``" WHERE ..."`` and *params* is the list of bind values.
    """
    clauses: List[str] = []
    params: List[Any] = []
    idx = param_offset

    for key, value in filter_dict.items():
        col = key if key in _COLUMN_KEYS else f"metadata->>'{key}'"

        if isinstance(value, dict):
            for op, operand in value.items():
                if op == "$in":
                    if not isinstance(operand, (list, tuple)) or len(operand) == 0:
                        raise ValueError(f"$in requires a non-empty list for '{key}'")
                    placeholders = ", ".join(f"${idx + i}" for i in range(len(operand)))
                    clauses.append(f"{col} IN ({placeholders})")
                    params.extend(str(v) for v in operand)
                    idx += len(operand)
                elif op in _FILTER_OPS:
                    sql_op = _FILTER_OPS[op]
                    if isinstance(operand, (int, float)) and key not in _COLUMN_KEYS:
                        clauses.append(f"({col})::float {sql_op} ${idx}")
                        params.append(float(operand))
                    else:
                        clauses.append(f"{col} {sql_op} ${idx}")
                        params.append(str(operand))
                    idx += 1
                else:
                    raise ValueError(f"Unknown operator '{op}' for key '{key}'")
        else:
            clauses.append(f"{col} = ${idx}")
            params.append(str(value))
            idx += 1

    sql = " WHERE " + " AND ".join(clauses)
    return sql, params


class RetrievalService:
    """Async pgvector retrieval service with Vertex AI embeddings and Docling parsing.

    Args:
        dsn: PostgreSQL connection string
            (e.g. ``postgresql://user:pass@localhost:5432/dbname``).
        project: GCP project ID for Vertex AI embedding calls.
        location: GCP region for the Vertex AI endpoint.
    """

    def __init__(
        self,
        dsn: str,
        project: str,
        location: str = "europe-west1",
    ):
        self.dsn = dsn
        self.project = project
        self.location = location
        self._pool: Optional[asyncpg.Pool] = None
        self._genai = genai.Client(vertexai=True, project=project, location=location)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Create the asyncpg connection pool and register the vector type."""
        self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=10)
        async with self._pool.acquire() as conn:
            await register_vector(conn)
        logger.info("RetrievalService connected to database")

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("RetrievalService disconnected")

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> List[float]:
        """Embed a single text string using Vertex AI gemini-embedding-001."""
        response = await self._genai.aio.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config={"output_dimensionality": EMBEDDING_DIMENSIONS},
        )
        return list(response.embeddings[0].values)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in a single API call."""
        response = await self._genai.aio.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config={"output_dimensionality": EMBEDDING_DIMENSIONS},
        )
        return [list(e.values) for e in response.embeddings]

    # ------------------------------------------------------------------
    # Ingestion — raw text (backwards-compatible)
    # ------------------------------------------------------------------

    async def ingest(
        self,
        text: str,
        source_file: str,
        metadata: Optional[Dict[str, Any]] = None,
        max_tokens: int = 2000,
    ) -> int:
        """Chunk raw text with Docling's HybridChunker, embed, and insert.

        Returns the number of chunks inserted.
        """
        from docling.chunking import HybridChunker
        from docling_core.types.doc import DoclingDocument

        doc = DoclingDocument(name=source_file)
        doc.add_text(text=text)

        tokenizer = VertexAITokenizer(max_tokens=max_tokens)
        chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
        chunks = list(chunker.chunk(dl_doc=doc))

        if not chunks:
            return 0

        texts_to_embed = [chunker.contextualize(chunk) for chunk in chunks]
        embeddings = await self.embed_batch(texts_to_embed)

        meta_str = json.dumps(metadata) if metadata else None
        now = datetime.now(timezone.utc)

        rows = [
            (
                uuid.uuid4(),
                chunk.text,
                embedding,
                source_file,
                meta_str,
                "text_chunk",
                list(chunk.meta.headings) if chunk.meta.headings else None,
                now,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        async with self._pool.acquire() as conn:
            await register_vector(conn)
            await conn.executemany(
                """
                INSERT INTO knowledge_chunks
                    (id, content, embedding, source_file, metadata, content_type, headings, created_at)
                VALUES ($1, $2, $3::vector, $4, $5::jsonb, $6, $7, $8)
                """,
                rows,
            )

        logger.info(f"Ingested {len(rows)} chunk(s) from '{source_file}'")
        return len(rows)

    # ------------------------------------------------------------------
    # Ingestion — document files (Docling-powered)
    # ------------------------------------------------------------------

    async def ingest_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        describe_images: bool = False,
        max_tokens: int = 2000,
    ) -> int:
        """Parse a document with Docling, chunk, embed, and insert into pgvector.

        Supports PDF, DOCX, PPTX, XLSX, HTML, Markdown, and more.
        When ``describe_images`` is True, images/figures are sent to Gemini
        for description, and the descriptions are embedded alongside text chunks.

        Returns the total number of chunks inserted (text + image descriptions).
        """
        from docling.chunking import HybridChunker
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.types.doc import PictureItem

        # Configure converter with image extraction if needed
        pipeline_options = PdfPipelineOptions()
        if describe_images:
            pipeline_options.generate_picture_images = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )

        logger.info(f"Converting document: {file_path}")
        result = converter.convert(file_path)
        doc = result.document

        # Chunk the document
        tokenizer = VertexAITokenizer(max_tokens=max_tokens)
        chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
        chunks = list(chunker.chunk(dl_doc=doc))

        import os

        source_file = os.path.basename(file_path)
        base_metadata = metadata or {}
        base_metadata["path"] = file_path
        now = datetime.now(timezone.utc)
        total = 0

        # --- Text chunks ---
        if chunks:
            texts_to_embed = [chunker.contextualize(chunk) for chunk in chunks]
            embeddings = await self.embed_batch(texts_to_embed)

            rows = [
                (
                    uuid.uuid4(),
                    chunk.text,
                    embedding,
                    source_file,
                    json.dumps(base_metadata),
                    "text_chunk",
                    list(chunk.meta.headings) if chunk.meta.headings else None,
                    now,
                )
                for chunk, embedding in zip(chunks, embeddings)
            ]

            async with self._pool.acquire() as conn:
                await register_vector(conn)
                await conn.executemany(
                    """
                    INSERT INTO knowledge_chunks
                        (id, content, embedding, source_file, metadata, content_type, headings, created_at)
                    VALUES ($1, $2, $3::vector, $4, $5::jsonb, $6, $7, $8)
                    """,
                    rows,
                )
            total += len(rows)
            logger.info(f"Ingested {len(rows)} text chunk(s) from '{source_file}'")

        # --- Image descriptions ---
        if describe_images:
            image_count = 0
            for element, _level in doc.iterate_items():
                if not isinstance(element, PictureItem):
                    continue

                pil_image = element.get_image(doc)
                if pil_image is None:
                    continue

                # Convert PIL image to bytes for Gemini
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                image_bytes = buf.getvalue()

                # Generate description with Gemini
                description = await self._describe_image(image_bytes)
                if not description:
                    continue

                # Embed the description
                embedding = await self.embed(description)
                image_count += 1

                image_meta = {
                    **base_metadata,
                    "image_index": image_count,
                    "content_type_detail": "image_description",
                }

                async with self._pool.acquire() as conn:
                    await register_vector(conn)
                    await conn.execute(
                        """
                        INSERT INTO knowledge_chunks
                            (id, content, embedding, source_file, metadata, content_type, headings, created_at)
                        VALUES ($1, $2, $3::vector, $4, $5::jsonb, $6, $7, $8)
                        """,
                        uuid.uuid4(),
                        description,
                        embedding,
                        source_file,
                        json.dumps(image_meta),
                        "image_description",
                        None,
                        now,
                    )

            total += image_count
            if image_count:
                logger.info(
                    f"Ingested {image_count} image description(s) from '{source_file}'"
                )

        return total

    async def _describe_image(self, image_bytes: bytes) -> Optional[str]:
        """Send an image to Gemini and get a text description."""
        try:
            response = await self._genai.aio.models.generate_content(
                model=IMAGE_DESCRIBE_MODEL,
                contents=[
                    types.Content(
                        parts=[
                            types.Part(text=IMAGE_DESCRIBE_PROMPT),
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/png", data=image_bytes
                                )
                            ),
                        ]
                    )
                ],
            )
            return response.text
        except Exception:
            logger.warning("Failed to describe image", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete(self, where: Union[Dict[str, Any], str]) -> int:
        """Delete chunks matching a metadata filter.

        Args:
            where: Metadata filter (same format as ``query(where=...)``).
                Required — call with an explicit filter to avoid accidental
                full-table deletes.

        Returns the number of rows deleted.
        """
        if isinstance(where, dict) and where:
            where_sql, params = _build_where(where, param_offset=1)
        elif isinstance(where, str) and where:
            where_sql = f" WHERE {where}"
            params = []
        else:
            raise ValueError("where filter is required for delete()")

        sql = f"DELETE FROM knowledge_chunks{where_sql}"

        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, *params)

        count = int(result.split()[-1])
        logger.info(f"Deleted {count} chunk(s)")
        return count

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def query(
        self,
        query_text: str,
        top_k: int = 5,
        where: Optional[Union[Dict[str, Any], str]] = None,
    ) -> List[Dict[str, Any]]:
        """Embed the query and return the top-k most similar chunks.

        Args:
            query_text: The text to search for.
            top_k: Number of results to return.
            where: Optional metadata filter. Accepts either:

                - A **dict** compiled to a parameterised WHERE clause::

                    {"doctor_id": "dr_123"}                        # equality
                    {"specialty": {"$in": ["cardio", "neuro"]}}    # in
                    {"confidence": {"$gte": 0.8}}                  # range
                    {"status": {"$ne": "archived"}}                # not equal

                  Supported operators: ``$eq``, ``$ne``, ``$gt``,
                  ``$gte``, ``$lt``, ``$lte``, ``$in``.

                - A raw **SQL string** injected as-is (caller is
                  responsible for safety)::

                    "metadata->>'doctor_id' = 'dr_123'"

        Returns a list of dicts with keys:
        ``id``, ``content``, ``source_file``, ``metadata``, ``score``, ``created_at``.
        Score is 1 - cosine_distance (higher = more similar).
        """
        query_embedding = await self.embed(query_text)

        # Base params: $1 = embedding, $2 = top_k
        base_params: List[Any] = [query_embedding, top_k]

        if isinstance(where, dict) and where:
            where_sql, extra_params = _build_where(where, param_offset=3)
            base_params.extend(extra_params)
        elif isinstance(where, str) and where:
            where_sql = f" WHERE {where}"
        else:
            where_sql = ""

        sql = f"""
            SELECT
                id,
                content,
                source_file,
                metadata,
                content_type,
                headings,
                created_at,
                1 - (embedding <=> $1::vector) AS score
            FROM knowledge_chunks{where_sql}
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """

        async with self._pool.acquire() as conn:
            await register_vector(conn)
            rows = await conn.fetch(sql, *base_params)

        return [
            {
                "id": str(row["id"]),
                "content": row["content"],
                "source_file": row["source_file"],
                "metadata": row["metadata"],
                "content_type": row["content_type"],
                "headings": row["headings"],
                "score": float(row["score"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
