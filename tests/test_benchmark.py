"""Метрики, изоляция прогонов и защита от неполных ответов."""
import json
import weakref
from pathlib import Path

import httpx
import pytest

from localscribe.benchmark import run_benchmark, word_errors
from localscribe.errors import OllamaRequestError
from localscribe.summarize import _call_ollama, _split_into_chunks


def test_wer_normalization_and_edits():
    assert word_errors("Ёж, пришёл!", "еж пришел")["wer"] == 0
    assert word_errors("один два три", "один четыре")["word_errors"] == 2
    assert word_errors("один", "")["wer"] == 1
    with pytest.raises(ValueError):
        word_errors("", "текст")


def test_unpunctuated_text_and_invalid_limit():
    text = "длинная речь без пунктуации " * 100
    parts = _split_into_chunks(text, 100)
    assert all(len(part) <= 100 for part in parts)
    assert "".join(parts) == text.strip()
    with pytest.raises(ValueError):
        _split_into_chunks(text, 0)


@pytest.mark.parametrize("payload", [
    {"response": "Оборванная фраза", "done_reason": "length"},
    {"response": "<think>только мысли</think>", "done_reason": "stop"},
])
def test_incomplete_or_empty_summary_is_rejected(monkeypatch, payload):
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: httpx.Response(200, json=payload))
    with pytest.raises(OllamaRequestError):
        _call_ollama("текст", model="test", host="http://localhost", num_predict=10)


def test_generation_metrics_and_seed(monkeypatch):
    sent = []
    def post(*args, **kwargs):
        sent.append(kwargs["json"])
        return httpx.Response(200, json={"response": "Выжимка", "done_reason": "stop", "eval_count": 4})
    monkeypatch.setattr(httpx, "post", post)
    metrics = []
    assert _call_ollama("текст", model="test", host="http://localhost", seed=42, metrics=metrics) == "Выжимка"
    assert sent[0]["options"]["seed"] == 42
    assert metrics[0]["eval_count"] == 4


def test_release_whisper_drops_cached_models(monkeypatch):
    from localscribe import transcribe
    class Model:
        pass
    monkeypatch.setattr(transcribe, "WhisperModel", lambda *a, **kw: Model())
    transcribe.release_models()
    model = transcribe._load_model("fake", "cpu", "int8")
    reference = weakref.ref(model)
    del model
    assert reference() is not None
    transcribe.release_models()
    assert reference() is None


def test_benchmark_separate_process_and_persistent_errors(tmp_path, monkeypatch):
    # Реальный дочерний процесс получает неизвестную модель без сервера;
    # такой прогон должен остаться в отчёте как ошибка, а не исчезнуть.
    source = tmp_path / "source.txt"
    source.write_text("Проверочный текст.", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [{"id": "sample", "task": "summary", "text": "source.txt"}]}))
    report = run_benchmark(manifest, tmp_path / "result", asr_models=[],
                           llm_models=["missing"], host="http://127.0.0.1:1")
    assert report["runs"][0]["status"] == "error"
    assert "OllamaConnectionError" in report["runs"][0]["error"]
    assert (tmp_path / "result" / "report.json").exists()
    with pytest.raises(FileExistsError):
        run_benchmark(manifest, tmp_path / "result", asr_models=[], llm_models=["missing"])
