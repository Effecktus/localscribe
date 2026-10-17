"""Тесты для localscribe.transcribe: распознавание речи через faster-whisper.

Везде используется модель 'tiny' - она маленькая и достаточно точная для
короткой чистой синтезированной речи, при этом тесты остаются быстрыми.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localscribe import transcribe as transcribe_module
from localscribe.transcribe import TranscriptionResult, save_transcript, transcribe

FIXTURE = Path(__file__).parent / "fixtures" / "sample_ru.wav"


def test_transcribe_recognizes_russian_speech() -> None:
    result = transcribe(FIXTURE, model_name="tiny", device="cuda")

    assert result.language == "ru"
    assert result.device_used in {"cuda", "cpu"}
    assert result.model_name == "tiny"

    lowered = result.text.lower()
    assert "тестовая запись" in lowered
    assert "погода" in lowered
    assert len(result.segments) >= 1


def test_transcribe_on_cpu_explicitly() -> None:
    result = transcribe(FIXTURE, model_name="tiny", device="cpu")

    assert result.device_used == "cpu"
    assert "погода" in result.text.lower()


def test_transcribe_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        transcribe(tmp_path / "missing.wav", model_name="tiny")


def test_transcribe_falls_back_to_cpu_on_gpu_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Если инференс на GPU падает (например, отсутствует CUDA-библиотека),
    transcribe() должен прозрачно повторить попытку на CPU, а не падать."""
    original_run = transcribe_module._run
    calls = {"count": 0}

    def flaky_run(model, audio_path, language):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("симулированная ошибка CUDA-инференса")
        return original_run(model, audio_path, language)

    monkeypatch.setattr(transcribe_module, "_cuda_available", lambda: True)
    monkeypatch.setattr(transcribe_module, "_run", flaky_run)

    result = transcribe(FIXTURE, model_name="tiny", device="cuda")

    assert calls["count"] == 2
    assert result.device_used == "cpu"
    assert "погода" in result.text.lower()


def test_save_transcript_writes_utf8_text(tmp_path: Path) -> None:
    result = TranscriptionResult(
        text="Привет, мир.",
        segments=[],
        language="ru",
        language_probability=1.0,
        model_name="tiny",
        device_used="cpu",
    )
    out = tmp_path / "out.txt"

    saved = save_transcript(result, out)

    assert saved == out
    assert out.read_text(encoding="utf-8") == "Привет, мир.\n"
