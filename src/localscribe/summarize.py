"""Суммаризация транскрипта через локальную LLM (Ollama).

Для длинных транскриптов, которые не влезают в контекстное окно модели целиком,
используется map-reduce: текст режется на части, каждая часть суммаризируется
отдельно ("map"), затем промежуточные выжимки сводятся в одну финальную ("reduce").
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from localscribe.config import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_SUMMARY_STYLE,
    OLLAMA_TIMEOUT_SECONDS,
    SUMMARY_CHUNK_CHAR_LIMIT,
)
from localscribe.errors import OllamaConnectionError, OllamaRequestError

logger = logging.getLogger(__name__)

STYLE_INSTRUCTIONS = {
    "краткая": (
        "Сделай очень краткую выжимку текста (2-4 предложения), отражающую "
        "только главную суть, без второстепенных деталей."
    ),
    "подробная": (
        "Сделай подробную выжимку текста связным текстом из нескольких абзацев, "
        "сохранив ключевые детали, аргументы и выводы."
    ),
    "тезисы": (
        "Выдели ключевые тезисы текста в виде маркированного списка "
        "(5-10 пунктов), каждый пункт - отдельная законченная мысль."
    ),
}

_DIRECT_PROMPT = (
    "Вот транскрипт аудио- или видеозаписи на русском языке.\n"
    "{style_instruction}\n\n"
    "Транскрипт:\n{text}"
)

_MAP_PROMPT = (
    "Ниже - фрагмент транскрипта длинной записи (это только часть текста, "
    "не весь текст целиком). Выпиши кратко основные мысли и факты из этого "
    "фрагмента, по существу, без вступлений и выводов, 3-6 предложений.\n\n"
    "Фрагмент:\n{chunk}"
)

_REDUCE_PROMPT = (
    "Ниже приведены краткие пересказы последовательных частей одного длинного "
    "транскрипта, в порядке по тексту.\n"
    "{style_instruction}\n\n"
    "Части:\n{combined}"
)

_MAP_NUM_PREDICT = 200
_REDUCE_NUM_PREDICT = 500
_DIRECT_NUM_PREDICT = 500

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class SummaryResult:
    summary: str
    style: str
    model_name: str
    chunks_used: int  # 1, если текст суммаризирован за один проход (без map-reduce)


def _call_ollama(
    prompt: str,
    *,
    model: str,
    host: str,
    num_predict: int | None = None,
    timeout: float = OLLAMA_TIMEOUT_SECONDS,
) -> str:
    payload: dict = {"model": model, "prompt": prompt, "stream": False}
    if num_predict is not None:
        payload["options"] = {"num_predict": num_predict}

    try:
        response = httpx.post(f"{host}/api/generate", json=payload, timeout=timeout)
    except httpx.ConnectError as exc:
        raise OllamaConnectionError(host, exc) from exc

    if response.status_code != 200:
        raise OllamaRequestError(
            f"Ollama вернул ошибку {response.status_code} для модели '{model}': "
            f"{response.text.strip()}"
        )

    data = response.json()
    return data["response"].strip()


def _split_into_chunks(text: str, max_chars: int = SUMMARY_CHUNK_CHAR_LIMIT) -> list[str]:
    """Режет текст на части по границам предложений, не превышая max_chars на часть."""
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current and current_len + len(sentence) > max_chars:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks or [text]


def summarize(
    text: str,
    *,
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    style: str = DEFAULT_SUMMARY_STYLE,
    chunk_char_limit: int = SUMMARY_CHUNK_CHAR_LIMIT,
    timeout: float = OLLAMA_TIMEOUT_SECONDS,
) -> SummaryResult:
    """Делает выжимку текста транскрипта через локальную LLM в Ollama.

    Args:
        text: текст транскрипта.
        model: имя модели в Ollama (должна быть предварительно скачана через
            `ollama pull <model>`).
        host: адрес сервера Ollama.
        style: "краткая" | "подробная" | "тезисы".
        chunk_char_limit: порог длины (в символах), после которого включается
            map-reduce вместо суммаризации за один проход.
        timeout: таймаут одного запроса к Ollama, в секундах.

    Raises:
        ValueError: пустой текст или неизвестный style.
        OllamaConnectionError: сервер Ollama недоступен.
        OllamaRequestError: Ollama вернул ошибку (например, модель не найдена).
    """
    if style not in STYLE_INSTRUCTIONS:
        raise ValueError(
            f"Неизвестный стиль выжимки: {style!r}. "
            f"Доступные: {', '.join(STYLE_INSTRUCTIONS)}"
        )

    text = text.strip()
    if not text:
        raise ValueError("Пустой транскрипт - нечего суммаризировать.")

    style_instruction = STYLE_INSTRUCTIONS[style]
    chunks = _split_into_chunks(text, chunk_char_limit)

    if len(chunks) == 1:
        prompt = _DIRECT_PROMPT.format(style_instruction=style_instruction, text=chunks[0])
        summary = _call_ollama(
            prompt, model=model, host=host, num_predict=_DIRECT_NUM_PREDICT, timeout=timeout
        )
        return SummaryResult(summary=summary, style=style, model_name=model, chunks_used=1)

    logger.info(
        "Транскрипт длинный (%d символов), разбиваю на %d частей для map-reduce.",
        len(text), len(chunks),
    )
    partial_summaries = []
    for i, chunk in enumerate(chunks, start=1):
        logger.info("Суммаризирую часть %d/%d...", i, len(chunks))
        partial = _call_ollama(
            _MAP_PROMPT.format(chunk=chunk),
            model=model, host=host, num_predict=_MAP_NUM_PREDICT, timeout=timeout,
        )
        partial_summaries.append(partial)

    combined = "\n\n".join(f"Часть {i}: {s}" for i, s in enumerate(partial_summaries, start=1))
    reduce_prompt = _REDUCE_PROMPT.format(style_instruction=style_instruction, combined=combined)
    summary = _call_ollama(
        reduce_prompt, model=model, host=host, num_predict=_REDUCE_NUM_PREDICT, timeout=timeout
    )

    return SummaryResult(summary=summary, style=style, model_name=model, chunks_used=len(chunks))


def save_summary(result: SummaryResult, output_path: str | Path) -> Path:
    """Сохраняет текст выжимки в файл (UTF-8)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.summary + "\n", encoding="utf-8")
    return output_path


def check_ollama_available(host: str = DEFAULT_OLLAMA_HOST, timeout: float = 5.0) -> bool:
    """Быстрая проверка, что сервер Ollama отвечает."""
    try:
        response = httpx.get(f"{host}/api/version", timeout=timeout)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def list_ollama_models(host: str = DEFAULT_OLLAMA_HOST, timeout: float = 5.0) -> list[str]:
    """Возвращает список имён моделей, уже скачанных в Ollama."""
    try:
        response = httpx.get(f"{host}/api/tags", timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaConnectionError(host, exc) from exc

    data = response.json()
    return [m["name"] for m in data.get("models", [])]
