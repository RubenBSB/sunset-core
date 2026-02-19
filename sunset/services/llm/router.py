import logging
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from . import _FILE_SEARCH, LLMResponse, ToolExecutor, _FileSearchSentinel
from .gemini import GeminiService
from .openai import OpenAIService
from .store import FileStore
from .vertex import VertexAIGeminiService

logger = logging.getLogger(__name__)


class LLMServiceRouter:
    """
    Unified LLM service that routes requests to OpenAI or Gemini based on model name.

    Model routing:
    - Models starting with 'gemini' -> GeminiService or VertexAIGeminiService
    - All other models -> OpenAIService (default)

    Gemini Provider Options:
    - use_vertex_ai=False (default): Uses Gemini Developer API with API key.
      Supports File Search Stores for RAG.
    - use_vertex_ai=True: Uses Vertex AI with GCP authentication.
      Uses Vertex AI Search (Discovery Engine) for RAG via grounded generation.
      Set search_engine_id and search_data_store_ids for file search.

    File stores are created lazily if file_store_id is not provided (Gemini API only).
    """

    def __init__(
        self,
        openai_api_key: str,
        gemini_api_key: Optional[str] = None,
        openai_file_store_id: Optional[str] = None,
        gemini_file_store_id: Optional[str] = None,
        file_store_name: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
        # Vertex AI options
        use_vertex_ai: bool = False,
        vertex_project: Optional[str] = None,
        vertex_project_number: Optional[str] = None,
        vertex_location: str = "europe-west1",
        search_engine_id: Optional[str] = None,
        search_data_store_ids: Optional[List[str]] = None,
        # Monitoring
        monitoring_project: Optional[str] = None,
        # Retrieval (pgvector RAG)
        retrieval: Optional[Any] = None,
    ):
        self._openai = OpenAIService(
            api_key=openai_api_key,
            file_store_id=openai_file_store_id,
            file_store_name=file_store_name,
        )

        self.use_vertex_ai = use_vertex_ai

        # Create monitoring service if a GCP project is available
        monitoring = None
        effective_project = monitoring_project or vertex_project
        if effective_project:
            from sunset.services.monitoring import MonitoringService

            monitoring = MonitoringService(project=effective_project)

        if use_vertex_ai:
            if not vertex_project:
                raise ValueError("vertex_project is required when use_vertex_ai=True")
            if not vertex_project_number:
                raise ValueError(
                    "vertex_project_number is required when use_vertex_ai=True"
                )
            self._gemini = VertexAIGeminiService(
                project=vertex_project,
                project_number=vertex_project_number,
                location=vertex_location,
                search_engine_id=search_engine_id,
                search_data_store_ids=search_data_store_ids,
                monitoring=monitoring,
                retrieval=retrieval,
            )
            logger.info(
                f"Using Vertex AI Gemini (project={vertex_project}, "
                f"location={vertex_location}, search_engine={search_engine_id})"
            )
        else:
            if not gemini_api_key:
                raise ValueError("gemini_api_key is required when use_vertex_ai=False")
            self._gemini = GeminiService(
                api_key=gemini_api_key,
                file_store_id=gemini_file_store_id,
                file_store_name=file_store_name,
                monitoring=monitoring,
                retrieval=retrieval,
            )

        self.default_model = default_model

    @property
    def openai_file_store_id(self) -> Optional[str]:
        """Get OpenAI file store ID (may be set after lazy creation)."""
        return self._openai.store._store_id if self._openai.store else None

    @property
    def gemini_file_store_id(self) -> Optional[str]:
        """Get Gemini file store ID (may be set after lazy creation). None if using Vertex AI."""
        store = self._gemini.store
        return getattr(store, "_store_name", None) if store else None

    @property
    def search_engine_id(self) -> Optional[str]:
        """Get search engine ID (only available when using Vertex AI)."""
        if self.use_vertex_ai and hasattr(self._gemini, "search_engine_id"):
            return self._gemini.search_engine_id
        return None

    @property
    def search_data_store_ids(self) -> List[str]:
        """Get search data store IDs (only available when using Vertex AI)."""
        if self.use_vertex_ai and hasattr(self._gemini, "search_data_store_ids"):
            return self._gemini.search_data_store_ids
        return []

    def _get_service(self, model: str):
        """Route to the correct service based on model name."""
        if model.startswith("gemini"):
            return self._gemini
        return self._openai

    async def generate_response(
        self,
        input: str | List[Dict[str, Any]],
        model: Optional[str] = None,
        function_tools: Optional[List[Any]] = None,
        tool_executor: Optional[ToolExecutor] = None,
        text_format: Optional[Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        metric_tag: str = "",
        # Deprecated — use function_tools=[file_search] instead
        file_search: bool = False,
    ) -> LLMResponse:
        model = model or self.default_model

        # Backward compat: convert file_search=True to sentinel
        if file_search:
            function_tools = list(function_tools or [])
            if not any(isinstance(t, _FileSearchSentinel) for t in function_tools):
                function_tools.insert(0, _FILE_SEARCH)

        service = self._get_service(model)
        return await service.generate_response(
            input,
            model,
            function_tools=function_tools,
            tool_executor=tool_executor,
            text_format=text_format,
            temperature=temperature,
            metric_tag=metric_tag,
        )

    @property
    def store(self) -> Optional[FileStore]:
        """Primary file store (Gemini/Vertex provider)."""
        return self._gemini.store

    async def generate_json(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        text_format: Optional[Type[BaseModel]] = None,
        metric_tag: str = "",
    ) -> Dict[str, Any]:
        """Generate structured JSON using the appropriate service based on model."""
        model = model or self.default_model
        service = self._get_service(model)
        return await service.generate_json(
            messages, model, temperature, text_format, metric_tag=metric_tag
        )
