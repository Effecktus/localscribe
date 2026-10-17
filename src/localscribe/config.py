"""Значения по умолчанию для всего пайплайна."""

from __future__ import annotations

# --- Аудио ---
# Whisper ожидает моно WAV 16kHz - это стандарт де-факто для ASR.
AUDIO_SAMPLE_RATE = 16_000
AUDIO_CHANNELS = 1

# --- Whisper / faster-whisper ---
DEFAULT_WHISPER_MODEL = "large-v3-turbo"
DEFAULT_WHISPER_DEVICE = "cuda"  # fallback на "cpu" делает transcribe.py при недоступности GPU
DEFAULT_WHISPER_COMPUTE_TYPE = "float16"
DEFAULT_LANGUAGE = "ru"

# --- Ollama / суммаризация ---
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"
DEFAULT_SUMMARY_STYLE = "подробная"  # краткая | подробная | тезисы
OLLAMA_TIMEOUT_SECONDS = 180.0
# Размер кусков транскрипта для map-reduce суммаризации (в символах). Подобран
# с запасом под контекстное окно 7B-моделей (обычно 4096 токенов у Ollama по
# умолчанию) - оставляет место под промпт и ответ.
SUMMARY_CHUNK_CHAR_LIMIT = 3000

# Расширения файлов, которые мы считаем "медиа" (видео или аудио) на входе.
SUPPORTED_EXTENSIONS = {
    # видео
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v",
    # аудио
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
}
