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

LLM_CHUNK_MODEL = "gemini-2.5-flash"
LLM_CHUNK_PROMPT = """\
You are a document chunking assistant. Given the attached document, extract its \
full text content and split it into semantic chunks suitable for a RAG retrieval \
system.

Rules:
- Each chunk should be a coherent, self-contained passage (200-1500 words).
- Preserve the original text exactly — do not summarise, paraphrase, or omit content.
- Split at natural boundaries: section headings, topic changes, paragraph breaks.
- Include all content — headers, footers, tables (as text), captions, footnotes.
- For tables, convert them into readable text rows.
- For images, replace them with an [image] tag with detailed description (it might contain text, you should report it exactly too)
- Return ONLY a JSON array of strings, one per chunk. No markdown fences, no \
explanation.

Example output:
["First chunk text...", "Second chunk text...", "Third chunk text..."]
"""


def _make_vertex_ai_tokenizer(max_tokens: int = 2000):
    """Create a VertexAITokenizer instance (imports docling lazily)."""
    from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer

    class VertexAITokenizer(BaseTokenizer):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        max_tokens: int = max_tokens
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

    return VertexAITokenizer(max_tokens=max_tokens)


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


def _merge_small_chunks(
    texts: List[str], min_chars: int = 400, max_chars: int = 3000
) -> List[str]:
    """Merge consecutive small chunks to reduce embedding calls and improve context."""
    merged: List[str] = []
    buffer = ""
    for text in texts:
        if buffer and len(buffer) + len(text) > max_chars:
            merged.append(buffer)
            buffer = text
        elif len(text) < min_chars or (buffer and len(buffer) < min_chars):
            buffer = buffer + "\n\n" + text if buffer else text
        else:
            if buffer:
                merged.append(buffer)
            buffer = text
    if buffer:
        merged.append(buffer)
    return merged


class RetrievalService:
    """Async pgvector retrieval service with Vertex AI embeddings and Docling parsing.

    Args:
        dsn: PostgreSQL connection string
            (e.g. ``postgresql://user:pass@localhost:5432/dbname``).
        project: GCP project ID for Vertex AI embedding calls.
        location: GCP region for the Vertex AI endpoint.
        llm_service: Optional LLM service instance (any ``LLMService`` subclass).
            Required when using ``engine="llm"`` in ``ingest_document``.
    """

    def __init__(
        self,
        dsn: str,
        project: str,
        location: str = "europe-west1",
        llm_service: Optional[Any] = None,
    ):
        self.dsn = dsn
        self.project = project
        self.location = location
        self.llm_service = llm_service
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

        tokenizer = _make_vertex_ai_tokenizer(max_tokens=max_tokens)
        chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
        chunks = list(chunker.chunk(dl_doc=doc))

        if not chunks:
            return 0

        texts_to_embed = [chunker.contextualize(chunk) for chunk in chunks]
        texts_to_embed = _merge_small_chunks(texts_to_embed)
        embeddings = await self.embed_batch(texts_to_embed)

        meta_str = json.dumps(metadata) if metadata else None
        now = datetime.now(timezone.utc)

        rows = [
            (
                uuid.uuid4(),
                text,
                embedding,
                source_file,
                meta_str,
                "text_chunk",
                None,
                now,
            )
            for text, embedding in zip(texts_to_embed, embeddings)
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
        do_ocr: bool = True,
        do_table_structure: bool = True,
        num_threads: int = 4,
        engine: str = "docling",
        llm_model: str = LLM_CHUNK_MODEL,
    ) -> int:
        """Parse a document, chunk, embed, and insert into pgvector.

        Supports PDF, DOCX, PPTX, XLSX, HTML, Markdown, and more.
        When ``describe_images`` is True, images/figures are sent to Gemini
        for description, and the descriptions are embedded alongside text chunks.

        Args:
            engine: ``"docling"`` (default) uses Docling for local parsing
                and chunking. ``"reducto"`` uses the Reducto API for
                cloud-based parsing with OCR, table extraction, and
                chunking — requires ``REDUCTO_API_KEY`` env var and
                ``reductoai`` package. ``"llm"`` sends the file to a
                Gemini model which extracts and chunks the text — faster,
                no heavy dependencies, but costs per-token.
            llm_model: Gemini model to use when ``engine="llm"``.
            do_ocr: Run OCR on pages (docling engine only).
            do_table_structure: Recognise table structure (docling engine only).
            num_threads: CPU threads for the Docling pipeline (docling engine only).

        Returns the total number of chunks inserted (text + image descriptions).
        """
        if engine == "reducto":
            return await self._ingest_document_reducto(
                file_path,
                metadata=metadata,
                describe_images=describe_images,
                max_tokens=max_tokens,
            )

        if engine == "llm":
            return await self._ingest_document_llm(
                file_path, metadata=metadata, llm_model=llm_model
            )

        # Plain text files are not supported by Docling's DocumentConverter.
        # Read them directly and delegate to the raw-text ingest method.
        import os

        if os.path.splitext(file_path)[1].lower() in (".txt", ".text"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            source_file = os.path.basename(file_path)
            return await self.ingest(
                text, source_file, metadata=metadata, max_tokens=max_tokens
            )

        from docling.chunking import HybridChunker
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            AcceleratorDevice,
            AcceleratorOptions,
            PdfPipelineOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.types.doc import PictureItem

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = do_ocr
        pipeline_options.do_table_structure = do_table_structure
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=num_threads,
            device=AcceleratorDevice.AUTO,
        )
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
        tokenizer = _make_vertex_ai_tokenizer(max_tokens=max_tokens)
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
            texts_to_embed = _merge_small_chunks(texts_to_embed)
            embeddings = await self.embed_batch(texts_to_embed)

            rows = [
                (
                    uuid.uuid4(),
                    text,
                    embedding,
                    source_file,
                    json.dumps(base_metadata),
                    "text_chunk",
                    None,
                    now,
                )
                for text, embedding in zip(texts_to_embed, embeddings)
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
    # Ingestion — Reducto-powered parsing
    # ------------------------------------------------------------------

    async def _ingest_document_reducto(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        describe_images: bool = False,
        max_tokens: int = 2000,
    ) -> int:
        """Parse a document with the Reducto API, then embed and insert chunks."""
        import asyncio
        import os
        from pathlib import Path

        from reducto import Reducto

        client = Reducto()
        source_file = os.path.basename(file_path)

        # Reducto SDK is synchronous — run in a thread to avoid blocking.
        upload = await asyncio.to_thread(client.upload, file=Path(file_path))

        parse_kwargs: Dict[str, Any] = {
            "input": upload,
            "formatting": {"table_output_format": "md"},
            "retrieval": {
                "chunking": {
                    "chunk_mode": "variable",
                    "chunk_size": max_tokens,
                },
                "embedding_optimized": True,
            },
        }
        if describe_images:
            parse_kwargs["enhance"] = {"summarize_figures": True}

        result = await asyncio.to_thread(client.parse.run, **parse_kwargs)

        chunks = result.result.chunks
        if not chunks:
            logger.warning(f"Reducto returned no chunks for '{source_file}'")
            return 0

        # Prefer the embedding-optimized `embed` field, fall back to `content`.
        texts = []
        for chunk in chunks:
            text = getattr(chunk, "embed", None) or chunk.content
            if text:
                texts.append(text)

        if not texts:
            return 0

        texts = _merge_small_chunks(texts)
        embeddings = await self.embed_batch(texts)

        base_metadata = metadata or {}
        base_metadata["path"] = file_path
        base_metadata["engine"] = "reducto"
        meta_str = json.dumps(base_metadata)
        now = datetime.now(timezone.utc)

        rows = [
            (
                uuid.uuid4(),
                text,
                embedding,
                source_file,
                meta_str,
                "text_chunk",
                None,
                now,
            )
            for text, embedding in zip(texts, embeddings)
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

        logger.info(
            f"Ingested {len(rows)} Reducto chunk(s) from '{source_file}' "
            f"(job_id={result.job_id}, pages={result.usage.num_pages})"
        )
        return len(rows)

    # ------------------------------------------------------------------
    # Ingestion — LLM-powered chunking
    # ------------------------------------------------------------------

    async def _ingest_document_llm(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        llm_model: str = LLM_CHUNK_MODEL,
    ) -> int:
        """Send a document to an LLM for text extraction and chunking."""
        import mimetypes
        import os

        if not self.llm_service:
            raise ValueError(
                "llm_service is required for engine='llm'. "
                "Pass it to the RetrievalService constructor."
            )

        source_file = os.path.basename(file_path)
        mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        logger.info(f"LLM chunking document: {file_path} ({mime_type})")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": LLM_CHUNK_PROMPT},
                    {
                        "type": "inline_data",
                        "data": file_bytes,
                        "mime_type": mime_type,
                    },
                ],
            }
        ]

        response = await self.llm_service.generate_response(
            input=messages,
            model=llm_model,
        )

        raw = response["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        chunk_texts = json.loads(raw)
        if not isinstance(chunk_texts, list) or not chunk_texts:
            logger.warning(f"LLM returned no chunks for '{source_file}'")
            return 0

        chunk_texts = _merge_small_chunks(chunk_texts)
        embeddings = await self.embed_batch(chunk_texts)

        base_metadata = metadata or {}
        base_metadata["path"] = file_path
        base_metadata["engine"] = "llm"
        meta_str = json.dumps(base_metadata)
        now = datetime.now(timezone.utc)

        rows = [
            (
                uuid.uuid4(),
                text,
                embedding,
                source_file,
                meta_str,
                "text_chunk",
                None,
                now,
            )
            for text, embedding in zip(chunk_texts, embeddings)
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

        logger.info(f"Ingested {len(rows)} LLM chunk(s) from '{source_file}'")
        return len(rows)

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
    # List sources
    # ------------------------------------------------------------------

    async def list_sources(
        self, where: Union[Dict[str, Any], str, None] = None
    ) -> List[Dict[str, Any]]:
        """Return distinct source files with chunk counts and earliest creation date.

        Args:
            where: Optional metadata filter (same format as ``query()`` and
                ``delete()``). Pass ``None`` to list all sources.

        Returns a list of dicts with keys:
        ``source_file``, ``chunks_count``, ``created_at``.
        """
        params: List[Any] = []

        if isinstance(where, dict) and where:
            where_sql, params = _build_where(where, param_offset=1)
        elif isinstance(where, str) and where:
            where_sql = f" WHERE {where}"
        else:
            where_sql = ""

        sql = f"""
            SELECT
                source_file,
                COUNT(*) AS chunks_count,
                MIN(created_at) AS created_at
            FROM knowledge_chunks{where_sql}
            GROUP BY source_file
            ORDER BY created_at
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [
            {
                "source_file": row["source_file"],
                "chunks_count": row["chunks_count"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

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
