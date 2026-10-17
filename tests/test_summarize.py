"""Тесты для localscribe.summarize: суммаризация через Ollama.

Часть тестов (краткая/длинная суммаризация) обращаются к реально запущенному
серверу Ollama с моделью qwen2.5:7b-instruct - как и остальной проект, мы
проверяем настоящий локальный пайплайн, а не только мокаем HTTP. Ошибочные
пути (сервер недоступен, HTTP-ошибка) тестируются через monkeypatch, без
реальной сети.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from localscribe.config import DEFAULT_OLLAMA_MODEL
from localscribe.errors import OllamaConnectionError, OllamaRequestError
from localscribe.summarize import (
    SummaryResult,
    _split_into_chunks,
    check_ollama_available,
    save_summary,
    summarize,
)

SAMPLE_TEXT = (
    "Сегодня мы обсудили план разработки локального приложения для "
    "транскрибации видео и аудио с последующей суммаризацией текста. "
    "Решили использовать faster-whisper для распознавания речи на GPU и "
    "Ollama с моделью Qwen для генерации выжимки. Всё будет работать "
    "полностью локально, без отправки данных в облако."
)


def test_summarize_short_text_direct() -> None:
    result = summarize(SAMPLE_TEXT, style="краткая")

    assert isinstance(result, SummaryResult)
    assert result.chunks_used == 1
    assert result.model_name == DEFAULT_OLLAMA_MODEL
    assert len(result.summary) > 0
    assert len(result.summary) < len(SAMPLE_TEXT) * 2  # действительно выжимка, а не перепев


def test_summarize_bullet_style_contains_list_markers() -> None:
    result = summarize(SAMPLE_TEXT, style="тезисы")

    assert result.style == "тезисы"
    # Маркированный список обычно содержит дефисы/звёздочки/цифры построчно.
    assert any(marker in result.summary for marker in ("-", "*", "1", "•"))


def test_summarize_long_text_triggers_map_reduce() -> None:
    long_text = " ".join([SAMPLE_TEXT] * 8)  # заведомо длиннее лимита в один чанк

    result = summarize(long_text, style="краткая", chunk_char_limit=400)

    assert result.chunks_used > 1
    assert len(result.summary) > 0


def test_summarize_unknown_style_raises() -> None:
    with pytest.raises(ValueError):
        summarize(SAMPLE_TEXT, style="несуществующий-стиль")


def test_summarize_empty_text_raises() -> None:
    with pytest.raises(ValueError):
        summarize("   ", style="краткая")


def test_summarize_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(OllamaConnectionError):
        summarize(SAMPLE_TEXT, style="краткая")


def test_summarize_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status_code = 404
        text = "model not found"

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(OllamaRequestError):
        summarize(SAMPLE_TEXT, style="краткая", model="no-such-model")


def test_check_ollama_available_true_when_server_responds() -> None:
    assert check_ollama_available() is True


def test_check_ollama_available_false_on_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", fake_get)

    assert check_ollama_available() is False


def test_split_into_chunks_respects_limit() -> None:
    text = "Одно предложение. " * 50
    chunks = _split_into_chunks(text, max_chars=100)

    assert len(chunks) > 1
    assert all(len(c) <= 150 for c in chunks)  # небольшой запас на границы предложений


def test_split_into_chunks_single_chunk_for_short_text() -> None:
    chunks = _split_into_chunks(SAMPLE_TEXT, max_chars=10_000)
    assert chunks == [SAMPLE_TEXT]


def test_save_summary_writes_utf8_text(tmp_path: Path) -> None:
    result = SummaryResult(summary="Короткая выжимка.", style="краткая", model_name="x", chunks_used=1)
    out = tmp_path / "out.txt"

    saved = save_summary(result, out)

    assert saved == out
    assert out.read_text(encoding="utf-8") == "Короткая выжимка.\n"
