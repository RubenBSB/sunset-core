from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OutputFormat(str, Enum):
    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"


@dataclass
class CrawlPage:
    url: str
    title: str
    content: str
    depth: int
    json_data: dict[str, Any] | None = None
    links: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass
class CrawlFile:
    url: str
    filename: str
    content: str
    mime_type: str
    source_page: str
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass
class CrawlResult:
    pages: list[CrawlPage]
    files: list[CrawlFile]
    output: str


class CrawlService(ABC):
    @abstractmethod
    async def crawl(
        self,
        url: str,
        *,
        exclude_paths: list[str] | None = None,
        max_depth: int = 2,
        max_pages: int = 50,
        output_format: OutputFormat = OutputFormat.MARKDOWN,
        json_schema: dict[str, Any] | None = None,
        prompt: str | None = None,
    ) -> CrawlResult:
        raise NotImplementedError

    async def close(self) -> None:
        pass
