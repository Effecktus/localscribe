"""Извлечение/нормализация аудио из любого видео- или аудиофайла через ffmpeg.

Вход: произвольный видео- или аудиофайл.
Выход: WAV, моно, 16kHz, 16-бит PCM — формат, который ожидает faster-whisper.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from localscribe.config import AUDIO_CHANNELS, AUDIO_SAMPLE_RATE
from localscribe.errors import AudioExtractionError, FFmpegNotFoundError


def _ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise FFmpegNotFoundError()
    return path


def extract_audio(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    channels: int = AUDIO_CHANNELS,
    overwrite: bool = True,
) -> Path:
    """Извлекает аудиодорожку из `input_path` и сохраняет её как нормализованный WAV.

    Работает одинаково для видео (просто берёт звук, видеопоток отбрасывается)
    и для аудио (переконвертирует в нужный формат/частоту/число каналов).

    Args:
        input_path: путь к исходному видео/аудио файлу.
        output_path: куда сохранить результат. Если None — создаётся временный
            файл (см. `temporary_audio`, если нужна автоматическая очистка).
        sample_rate: частота дискретизации на выходе (Гц).
        channels: число каналов на выходе (1 = моно).
        overwrite: перезаписывать output_path, если он уже существует.

    Returns:
        Путь к созданному WAV-файлу.

    Raises:
        FileNotFoundError: input_path не существует.
        FFmpegNotFoundError: ffmpeg не найден в PATH.
        AudioExtractionError: ffmpeg завершился с ошибкой (битый файл, нет
            аудиопотока и т.п.).
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    ffmpeg = _ffmpeg_path()

    if output_path is None:
        fd, tmp_name = tempfile.mkstemp(prefix="localscribe_", suffix=".wav")
        os.close(fd)
        output_path = Path(tmp_name)
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-i", str(input_path),
        "-vn",                      # без видео - нужен только звук
        "-acodec", "pcm_s16le",     # 16-бит PCM
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-loglevel", "error",
        str(output_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0 or not output_path.exists():
        raise AudioExtractionError(str(input_path), result.stderr)

    return output_path


@contextmanager
def temporary_audio(
    input_path: str | Path,
    *,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    channels: int = AUDIO_CHANNELS,
) -> Iterator[Path]:
    """Как `extract_audio`, но во временный файл, который удаляется при выходе из `with`.

    Пример:
        with temporary_audio("lecture.mp4") as wav_path:
            transcribe(wav_path)
        # wav_path уже удалён
    """
    wav_path = extract_audio(input_path, sample_rate=sample_rate, channels=channels)
    try:
        yield wav_path
    finally:
        wav_path.unlink(missing_ok=True)
