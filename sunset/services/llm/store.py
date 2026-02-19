import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)


class FileInfo(TypedDict):
    id: str
    name: str
    size_bytes: Optional[int]
    created_at: Optional[str]
    state: str
    metadata: Optional[Dict[str, Any]]


class StoreInfo(TypedDict):
    id: str
    name: str
    provider: str


class FileStore(ABC):
    """Async interface for managing documents in an LLM provider's knowledge store."""

    @abstractmethod
    async def upload(
        self,
        *,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        name: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> FileInfo:
        """Upload a file or inline content to the store.

        Provide exactly one of file_path or content.
        """
        ...

    @abstractmethod
    async def list(self) -> List[FileInfo]:
        """List all documents in the store."""
        ...

    @abstractmethod
    async def delete(self, file_id: str) -> bool:
        """Delete a document by ID. Returns True if deleted."""
        ...

    @abstractmethod
    async def get(self) -> StoreInfo:
        """Get store metadata."""
        ...


class OpenAIFileStore(FileStore):
    """OpenAI vector store file management."""

    def __init__(self, client, store_id: str):
        self._client = client
        self._store_id = store_id

    @classmethod
    async def create(cls, client, name: str = "knowledge-base") -> "OpenAIFileStore":
        """Create a new OpenAI vector store and return a ready FileStore."""
        store = await client.vector_stores.create(name=name)
        logger.info(
            f"Created OpenAI vector store: {store.id} (name: {name}). "
            "Save this ID to OPENAI_FILE_STORE_ID to reuse."
        )
        return cls(client, store.id)

    @classmethod
    async def from_id(cls, client, store_id: str) -> "OpenAIFileStore":
        """Retrieve an existing OpenAI vector store by ID."""
        await client.vector_stores.retrieve(vector_store_id=store_id)
        return cls(client, store_id)

    async def upload(
        self,
        *,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        name: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> FileInfo:
        if not file_path and not content:
            raise ValueError("Provide file_path or content")

        if content:
            # Write content to a temp file for upload
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write(content)
                file_path = f.name

        try:
            with open(file_path, "rb") as fileb:
                file = await self._client.files.create(
                    file=(name, fileb), purpose="user_data"
                )
            await self._client.vector_stores.files.create(
                vector_store_id=self._store_id, file_id=file.id
            )
            size = os.path.getsize(file_path) if os.path.exists(file_path) else None
            return FileInfo(
                id=file.id,
                name=name,
                size_bytes=size,
                created_at=datetime.now(timezone.utc).isoformat(),
                state="active",
                metadata=metadata,
            )
        finally:
            if content and file_path:
                os.unlink(file_path)

    async def list(self) -> List[FileInfo]:
        files = await self._client.vector_stores.files.list(
            vector_store_id=self._store_id
        )
        return [
            FileInfo(
                id=f.id,
                name=f.id,
                size_bytes=None,
                created_at=str(getattr(f, "created_at", None)),
                state=getattr(f, "status", "active"),
                metadata=None,
            )
            for f in files.data
        ]

    async def delete(self, file_id: str) -> bool:
        try:
            await self._client.vector_stores.files.delete(
                vector_store_id=self._store_id, file_id=file_id
            )
            await self._client.files.delete(file_id=file_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete OpenAI file {file_id}: {e}")
            return False

    async def get(self) -> StoreInfo:
        store = await self._client.vector_stores.retrieve(self._store_id)
        return StoreInfo(id=store.id, name=store.name, provider="openai")


class GeminiFileStore(FileStore):
    """Gemini file search store management."""

    def __init__(self, client, store_name: str):
        self._client = client
        self._store_name = store_name

    @classmethod
    async def create(cls, client, name: str = "knowledge-base") -> "GeminiFileStore":
        """Create a new Gemini file search store and return a ready FileStore."""
        store = client.file_search_stores.create(config={"display_name": name})
        logger.info(
            f"Created Gemini file search store: {store.name} (display_name: {name}). "
            "Save this ID to GEMINI_FILE_STORE_ID to reuse."
        )
        return cls(client, store.name)

    @classmethod
    async def from_id(cls, client, store_id: str) -> "GeminiFileStore":
        """Retrieve an existing Gemini file search store by ID."""
        await client.aio.file_search_stores.get(name=store_id)
        return cls(client, store_id)

    async def upload(
        self,
        *,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        name: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> FileInfo:
        if not file_path and not content:
            raise ValueError("Provide file_path or content")

        if content:
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write(content)
                file_path = f.name

        try:
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

            operation = self._client.file_search_stores.upload_to_file_search_store(
                file=file_path,
                file_search_store_name=self._store_name,
                config={"display_name": name},
            )

            while not operation.done:
                await asyncio.sleep(2)
                operation = self._client.operations.get(operation)

            doc_name = ""
            doc_state = "active"
            if hasattr(operation, "response") and operation.response:
                doc_name = getattr(operation.response, "name", "")
                doc_state = str(getattr(operation.response, "state", "active"))
            elif hasattr(operation, "name"):
                doc_name = operation.name

            return FileInfo(
                id=doc_name,
                name=name,
                size_bytes=file_size,
                created_at=datetime.now(timezone.utc).isoformat(),
                state=doc_state,
                metadata=metadata,
            )
        finally:
            if content and file_path:
                os.unlink(file_path)

    async def list(self) -> List[FileInfo]:
        try:
            documents = self._client.file_search_stores.documents.list(
                parent=self._store_name
            )
            return [
                FileInfo(
                    id=getattr(doc, "name", ""),
                    name=getattr(doc, "display_name", ""),
                    size_bytes=getattr(doc, "size_bytes", 0),
                    created_at=str(getattr(doc, "create_time", None)),
                    state=str(getattr(doc, "state", "active")),
                    metadata=None,
                )
                for doc in documents
            ]
        except Exception as e:
            logger.exception(f"Failed to list files from Gemini: {e}")
            return []

    async def delete(self, file_id: str) -> bool:
        try:
            self._client.file_search_stores.documents.delete(
                name=file_id, config={"force": True}
            )
            logger.info(f"Deleted file: {file_id}")
            return True
        except Exception as e:
            logger.exception(f"Failed to delete file {file_id}: {e}")
            return False

    async def get(self) -> StoreInfo:
        store = await self._client.aio.file_search_stores.get(name=self._store_name)
        return StoreInfo(
            id=store.name,
            name=getattr(store, "display_name", store.name),
            provider="gemini",
        )


class VertexFileStore(FileStore):
    """Vertex AI Search (Discovery Engine) document management."""

    def __init__(self, project: str, data_store_id: str):
        self._project = project
        self._data_store_id = data_store_id
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google.cloud import discoveryengine_v1 as discoveryengine

            self._client = discoveryengine.DocumentServiceClient()
        return self._client

    def _branch_path(self):
        return self._get_client().branch_path(
            project=self._project,
            location="global",
            data_store=self._data_store_id,
            branch="default_branch",
        )

    async def upload(
        self,
        *,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        name: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> FileInfo:
        if content is None and file_path:
            with open(file_path) as f:
                content = f.read()
        if content is None:
            raise ValueError("Provide file_path or content")

        def _create():
            from google.cloud import discoveryengine_v1 as discoveryengine

            doc_data = {"content": content}
            if metadata:
                doc_data.update(metadata)

            document = discoveryengine.Document(json_data=json.dumps(doc_data))
            request = discoveryengine.CreateDocumentRequest(
                parent=self._branch_path(),
                document_id=name,
                document=document,
            )
            return self._get_client().create_document(request=request)

        response = await asyncio.get_event_loop().run_in_executor(None, _create)

        return FileInfo(
            id=response.name,
            name=name,
            size_bytes=len(content.encode()) if content else None,
            created_at=datetime.now(timezone.utc).isoformat(),
            state="active",
            metadata=metadata,
        )

    async def list(self) -> List[FileInfo]:
        def _list():
            from google.cloud import discoveryengine_v1 as discoveryengine

            request = discoveryengine.ListDocumentsRequest(parent=self._branch_path())
            return [
                FileInfo(
                    id=doc.name,
                    name=doc.id,
                    size_bytes=None,
                    created_at=None,
                    state="active",
                    metadata={"json_data": doc.json_data} if doc.json_data else None,
                )
                for doc in self._get_client().list_documents(request=request)
            ]

        return await asyncio.get_event_loop().run_in_executor(None, _list)

    async def delete(self, file_id: str) -> bool:
        def _delete():
            from google.cloud import discoveryengine_v1 as discoveryengine

            request = discoveryengine.DeleteDocumentRequest(name=file_id)
            self._get_client().delete_document(request=request)

        try:
            await asyncio.get_event_loop().run_in_executor(None, _delete)
            logger.info(f"Deleted document: {file_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {file_id}: {e}")
            return False

    async def get(self) -> StoreInfo:
        return StoreInfo(
            id=self._data_store_id,
            name=self._data_store_id,
            provider="vertex",
        )

    # --- Vertex-specific ---

    async def import_from_gcs(self, gcs_uri: str) -> Dict[str, Any]:
        """Bulk import documents from GCS. Vertex AI only."""

        def _import():
            from google.cloud import discoveryengine_v1 as discoveryengine

            request = discoveryengine.ImportDocumentsRequest(
                parent=self._branch_path(),
                gcs_source=discoveryengine.GcsSource(
                    input_uris=[gcs_uri],
                    data_schema="content",
                ),
                reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
            )
            operation = self._get_client().import_documents(request=request)
            response = operation.result()
            return {
                "data_store_id": self._data_store_id,
                "error_samples": [str(e) for e in (response.error_samples or [])],
            }

        return await asyncio.get_event_loop().run_in_executor(None, _import)
