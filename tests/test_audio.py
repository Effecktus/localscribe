"""Тесты для localscribe.audio: извлечение/нормализация аудио через ffmpeg."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import pytest

from localscribe.audio import extract_audio, temporary_audio
from localscribe.errors import AudioExtractionError


@pytest.fixture(scope="module")
def sample_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Короткий синтетический WAV (sine-тон) для тестов, без реальной речи."""
    path = tmp_path_factory.mktemp("audio") / "tone.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-ar", "44100", "-ac", "2",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return path


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory: pytest.TempPathFactory, sample_audio: Path) -> Path:
    """Синтетическое видео (цветная заливка + тот же звук) - проверяем, что видеопоток
    отбрасывается и остаётся только нормализованное аудио."""
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=blue:size=320x240:duration=2",
            "-i", str(sample_audio),
            "-c:v", "libx264", "-c:a", "aac",
            "-shortest",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return path


def _wav_info(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as wf:
        return wf.getframerate(), wf.getnchannels()


def test_extract_audio_from_audio_file(sample_audio: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.wav"
    result = extract_audio(sample_audio, out, sample_rate=16000, channels=1)

    assert result == out
    assert out.exists()
    assert _wav_info(out) == (16000, 1)


def test_extract_audio_from_video_file(sample_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.wav"
    result = extract_audio(sample_video, out, sample_rate=16000, channels=1)

    assert result.exists()
    assert _wav_info(result) == (16000, 1)


def test_extract_audio_custom_sample_rate_and_channels(sample_audio: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.wav"
    extract_audio(sample_audio, out, sample_rate=22050, channels=2)

    assert _wav_info(out) == (22050, 2)


def test_extract_audio_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_audio(tmp_path / "does_not_exist.mp4", tmp_path / "out.wav")


def test_extract_audio_corrupt_input_raises(tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a real media file")

    with pytest.raises(AudioExtractionError):
        extract_audio(broken, tmp_path / "out.wav")


def test_extract_audio_default_output_is_temp_file(sample_audio: Path) -> None:
    result = extract_audio(sample_audio)
    try:
        assert result.exists()
        assert result.suffix == ".wav"
    finally:
        result.unlink(missing_ok=True)


def test_temporary_audio_cleans_up(sample_audio: Path) -> None:
    with temporary_audio(sample_audio) as wav_path:
        assert wav_path.exists()
        captured_path = wav_path
    assert not captured_path.exists()


def test_temporary_audio_cleans_up_on_exception(sample_audio: Path) -> None:
    captured_path = None
    with pytest.raises(RuntimeError):
        with temporary_audio(sample_audio) as wav_path:
            captured_path = wav_path
            raise RuntimeError("boom")
    assert captured_path is not None
    assert not captured_path.exists()
