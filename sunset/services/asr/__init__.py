"""Automatic Speech Recognition service using Deepgram."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, List, Optional, Union

from deepgram import AsyncDeepgramClient

logger = logging.getLogger(__name__)


@dataclass
class Word:
    word: str
    start: float
    end: float
    confidence: float
    speaker: Optional[int] = None


@dataclass
class Transcript:
    text: str
    words: List[Word] = field(default_factory=list)
    language: Optional[str] = None
    duration: Optional[float] = None


class ASRService:
    """Async speech-to-text service powered by Deepgram."""

    def __init__(self, api_key: Optional[str] = None, model: str = "nova-3"):
        self._api_key = api_key or os.getenv("DEEPGRAM_API_KEY", "")
        self._model = model
        self._client = AsyncDeepgramClient(api_key=self._api_key)

    async def transcribe(
        self,
        source: Union[bytes, str, Path],
        *,
        streaming: bool = False,
        language: Optional[str] = None,
        punctuate: bool = True,
        smart_format: bool = True,
        diarize: bool = False,
        **kwargs,
    ) -> Union[Transcript, AsyncIterator[str]]:
        """Transcribe audio from bytes, file path, or URL.

        Args:
            source: Audio data as bytes, a local file path, or a public URL.
            streaming: If True, return an async iterator of partial transcripts
                       instead of waiting for the full result.
            language: Explicit language code (e.g. "en", "fr"). If None,
                      auto-detection is used.
            punctuate: Add punctuation to transcript.
            smart_format: Apply smart formatting (dates, numbers, etc.).
            diarize: Enable speaker diarization.
            **kwargs: Additional Deepgram options passed through.
        """
        if streaming:
            return self._stream(
                source,
                language=language,
                punctuate=punctuate,
                smart_format=smart_format,
                **kwargs,
            )

        return await self._transcribe_prerecorded(
            source,
            language=language,
            punctuate=punctuate,
            smart_format=smart_format,
            diarize=diarize,
            **kwargs,
        )

    async def _transcribe_prerecorded(
        self,
        source: Union[bytes, str, Path],
        *,
        language: Optional[str] = None,
        punctuate: bool = True,
        smart_format: bool = True,
        diarize: bool = False,
        **kwargs,
    ) -> Transcript:
        options = {
            "model": self._model,
            "punctuate": punctuate,
            "smart_format": smart_format,
            "diarize": diarize,
            **kwargs,
        }

        if language:
            options["language"] = language
        else:
            options["detect_language"] = True

        source = (
            Path(source)
            if isinstance(source, str)
            and not source.startswith(("http://", "https://"))
            else source
        )

        if isinstance(source, Path):
            with open(source, "rb") as f:
                audio_data = f.read()
            response = await self._client.listen.v1.media.transcribe_file(
                request=audio_data,
                **options,
            )
        elif isinstance(source, bytes):
            response = await self._client.listen.v1.media.transcribe_file(
                request=source,
                **options,
            )
        elif isinstance(source, str):
            response = await self._client.listen.v1.media.transcribe_url(
                url=source,
                **options,
            )
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        return self._parse_response(response)

    async def _stream(
        self,
        source: Union[bytes, str, Path],
        *,
        language: Optional[str] = None,
        punctuate: bool = True,
        smart_format: bool = True,
        **kwargs,
    ) -> AsyncIterator[str]:
        options = {
            "model": self._model,
            "punctuate": punctuate,
            "smart_format": smart_format,
            **kwargs,
        }

        if language:
            options["language"] = language
        else:
            options["detect_language"] = True

        source = (
            Path(source)
            if isinstance(source, str)
            and not source.startswith(("http://", "https://"))
            else source
        )

        if isinstance(source, Path):
            with open(source, "rb") as f:
                audio_data = f.read()
        elif isinstance(source, bytes):
            audio_data = source
        elif isinstance(source, str):
            import httpx

            async with httpx.AsyncClient() as http:
                resp = await http.get(source)
                resp.raise_for_status()
                audio_data = resp.content
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        return self._stream_audio(audio_data, options)

    async def _stream_audio(
        self, audio_data: bytes, options: dict
    ) -> AsyncIterator[str]:
        import asyncio

        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        async def on_transcript(_, result, **kw):
            transcript = (
                result.channel.alternatives[0].transcript
                if result.channel and result.channel.alternatives
                else ""
            )
            if transcript:
                await queue.put(transcript)

        async def on_error(_, error, **kw):
            logger.error(f"Deepgram streaming error: {error}")

        async def on_close(_, *args, **kw):
            await queue.put(None)

        async with self._client.listen.v1.connect(**options) as connection:
            connection.on("Results", on_transcript)
            connection.on("Error", on_error)
            connection.on("Close", on_close)

            chunk_size = 4096
            for i in range(0, len(audio_data), chunk_size):
                await connection.send(audio_data[i : i + chunk_size])

            await connection.finish()

            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item

    def _parse_response(self, response) -> Transcript:
        results = response.results
        channel = results.channels[0]
        alt = channel.alternatives[0]

        words = [
            Word(
                word=w.word,
                start=w.start,
                end=w.end,
                confidence=w.confidence,
                speaker=getattr(w, "speaker", None),
            )
            for w in (alt.words or [])
        ]

        language = getattr(channel, "detected_language", None) or getattr(
            results, "detected_language", None
        )

        duration = getattr(results, "duration", None)

        return Transcript(
            text=alt.transcript,
            words=words,
            language=language,
            duration=duration,
        )
