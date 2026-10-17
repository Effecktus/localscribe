"""Транскрибация аудио в текст через faster-whisper (GPU/CUDA с fallback на CPU)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import ctranslate2
from faster_whisper import WhisperModel

from localscribe.config import (
    DEFAULT_LANGUAGE,
    DEFAULT_WHISPER_COMPUTE_TYPE,
    DEFAULT_WHISPER_DEVICE,
    DEFAULT_WHISPER_MODEL,
)

logger = logging.getLogger(__name__)

# Если GPU недоступна или инференс на ней не удался, откатываемся на CPU с этим compute_type -
# float16 на CPU не поддерживается/неэффективен, int8 - стандартный выбор для CPU-инференса.
CPU_FALLBACK_COMPUTE_TYPE = "int8"


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    segments: list[TranscriptSegment]
    language: str
    language_probability: float
    model_name: str
    device_used: str  # реально использованное устройство (после возможного fallback)


def _cuda_available() -> bool:
    """Проверяет доступность CUDA через сам ctranslate2, без torch.

    Важно: torch и ctranslate2 в одном процессе конфликтуют по именам CUDA DLL
    (например, оба пытаются загрузить свою версию cudnn_cnn64_9.dll - кто
    первый импортирован, тот и "занимает" имя в процессе, второй ломается
    с WinError 127). faster-whisper не нуждается в torch вообще - у
    ctranslate2 есть собственный независимый способ узнать число GPU.
    """
    try:
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


@lru_cache(maxsize=4)
def _load_model(model_name: str, device: str, compute_type: str) -> WhisperModel:
    logger.info("Загружаю модель Whisper '%s' (device=%s, compute_type=%s)...",
                model_name, device, compute_type)
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def _run(
    model: WhisperModel, audio_path: Path, language: str | None
) -> tuple[list[TranscriptSegment], object]:
    """Запускает инференс и сразу разворачивает ленивый генератор сегментов -
    любая ошибка CUDA-инференса (например, отсутствующая DLL) происходит именно
    здесь, а не при создании модели."""
    segments_iter, info = model.transcribe(str(audio_path), language=language)
    segments = [
        TranscriptSegment(start=s.start, end=s.end, text=s.text.strip())
        for s in segments_iter
    ]
    return segments, info


def transcribe(
    audio_path: str | Path,
    *,
    model_name: str = DEFAULT_WHISPER_MODEL,
    device: str = DEFAULT_WHISPER_DEVICE,
    compute_type: str = DEFAULT_WHISPER_COMPUTE_TYPE,
    language: str | None = DEFAULT_LANGUAGE,
) -> TranscriptionResult:
    """Транскрибирует аудиофайл (WAV 16kHz mono, см. `localscribe.audio`) в текст.

    Если `device="cuda"` недоступна или инференс на GPU завершается с ошибкой
    (например, не хватило VRAM или не найдены CUDA-библиотеки), автоматически
    и прозрачно откатывается на CPU - вызывающий код получает результат в любом
    случае, а в `TranscriptionResult.device_used` видно, что было использовано
    по факту.

    Args:
        audio_path: путь к аудиофайлу.
        model_name: размер/имя модели Whisper (например "large-v3-turbo").
        device: "cuda" или "cpu".
        compute_type: точность вычислений ("float16" для GPU, "int8" для CPU).
        language: код языка ("ru"), либо None для автоопределения.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Файл не найден: {audio_path}")

    if device == "cuda" and not _cuda_available():
        logger.warning("CUDA недоступна в этой системе, использую CPU (будет медленнее).")
        device = "cpu"

    if device == "cpu" and compute_type == DEFAULT_WHISPER_COMPUTE_TYPE:
        # float16 (дефолт для GPU) неэффективен/не поддерживается на CPU -
        # если пользователь не указал свой compute_type явно, подставляем int8.
        compute_type = CPU_FALLBACK_COMPUTE_TYPE

    try:
        model = _load_model(model_name, device, compute_type)
        segments, info = _run(model, audio_path, language)
    except Exception as exc:
        if device != "cpu":
            logger.warning(
                "Инференс на GPU не удался (%s). Повторяю на CPU (int8).", exc
            )
            device, compute_type = "cpu", CPU_FALLBACK_COMPUTE_TYPE
            model = _load_model(model_name, device, compute_type)
            segments, info = _run(model, audio_path, language)
        else:
            raise

    full_text = " ".join(s.text for s in segments if s.text).strip()

    return TranscriptionResult(
        text=full_text,
        segments=segments,
        language=info.language,
        language_probability=info.language_probability,
        model_name=model_name,
        device_used=device,
    )


def save_transcript(result: TranscriptionResult, output_path: str | Path) -> Path:
    """Сохраняет чистый текст транскрипта в файл (UTF-8)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.text + "\n", encoding="utf-8")
    return output_path
