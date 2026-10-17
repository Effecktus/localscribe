"""Тесты CLI (Typer) localscribe.

Как и остальной проект, гоняем настоящий пайплайн (faster-whisper на GPU/CPU,
реальный Ollama) на маленьком видео-фикстуре - модель 'tiny' держит тесты
быстрыми.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from localscribe.cli import app

runner = CliRunner()

FIXTURE_CLIP = Path(__file__).parent / "fixtures" / "sample_clip.mp4"


def test_process_full_pipeline(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "process", str(FIXTURE_CLIP),
            "--out-dir", str(tmp_path),
            "--whisper-model", "tiny",
        ],
    )

    assert result.exit_code == 0, result.output
    transcript_path = tmp_path / "sample_clip.transcript.txt"
    summary_path = tmp_path / "sample_clip.summary.txt"
    assert transcript_path.exists()
    assert summary_path.exists()

    transcript_text = transcript_path.read_text(encoding="utf-8").lower()
    assert "погода" in transcript_text
    assert len(summary_path.read_text(encoding="utf-8").strip()) > 0
    # По умолчанию промежуточный WAV не должен остаться.
    assert not (tmp_path / "sample_clip.audio.wav").exists()


def test_process_keep_audio_flag(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "process", str(FIXTURE_CLIP),
            "--out-dir", str(tmp_path),
            "--whisper-model", "tiny",
            "--keep-audio",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "sample_clip.audio.wav").exists()


def test_process_missing_input_file_fails() -> None:
    result = runner.invoke(app, ["process", "no_such_file.mp4"])
    assert result.exit_code != 0


def test_transcribe_then_summarize_standalone(tmp_path: Path) -> None:
    transcript_path = tmp_path / "t.txt"
    result = runner.invoke(
        app,
        [
            "transcribe", str(FIXTURE_CLIP),
            "--out", str(transcript_path),
            "--whisper-model", "tiny",
        ],
    )
    assert result.exit_code == 0, result.output
    assert transcript_path.exists()
    assert "погода" in transcript_path.read_text(encoding="utf-8").lower()

    summary_path = tmp_path / "s.txt"
    result = runner.invoke(
        app,
        [
            "summarize", str(transcript_path),
            "--out", str(summary_path),
            "--summary-style", "тезисы",
        ],
    )
    assert result.exit_code == 0, result.output
    assert summary_path.exists()
    assert len(summary_path.read_text(encoding="utf-8").strip()) > 0


def test_summarize_unknown_style_fails_gracefully(tmp_path: Path) -> None:
    transcript_path = tmp_path / "t.txt"
    transcript_path.write_text("Какой-то текст транскрипта.", encoding="utf-8")

    result = runner.invoke(
        app,
        ["summarize", str(transcript_path), "--summary-style", "несуществующий"],
    )

    assert result.exit_code == 1
    assert "Ошибка" in result.output


def test_doctor_runs() -> None:
    result = runner.invoke(app, ["doctor"])
    # На этой машине всё должно быть настроено (см. M0/M3) - код 0.
    assert result.exit_code == 0, result.output
    assert "ffmpeg" in result.output
