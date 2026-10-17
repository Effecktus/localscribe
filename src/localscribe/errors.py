"""Общие исключения проекта."""


class LocalscribeError(Exception):
    """Базовое исключение для всех ошибок localscribe."""


class FFmpegNotFoundError(LocalscribeError):
    """ffmpeg не найден в PATH."""

    def __init__(self) -> None:
        super().__init__(
            "ffmpeg не найден в PATH. Установите его, например:\n"
            "  winget install -e --id Gyan.FFmpeg"
        )


class AudioExtractionError(LocalscribeError):
    """Ошибка при извлечении/конвертации аудио через ffmpeg."""

    def __init__(self, input_path: str, stderr: str) -> None:
        self.input_path = input_path
        self.stderr = stderr
        super().__init__(
            f"Не удалось извлечь аудио из '{input_path}'.\n"
            f"Вывод ffmpeg:\n{stderr.strip()}"
        )


class UnsupportedInputError(LocalscribeError):
    """Входной файл не существует или не является медиафайлом."""


class OllamaConnectionError(LocalscribeError):
    """Не удалось подключиться к серверу Ollama."""

    def __init__(self, host: str, cause: Exception) -> None:
        super().__init__(
            f"Не удалось подключиться к Ollama по адресу {host}.\n"
            f"Запущен ли сервер? Проверьте командой: ollama serve\n"
            f"Исходная ошибка: {cause}"
        )


class OllamaRequestError(LocalscribeError):
    """Ollama вернул ошибку на запрос (например, модель не найдена)."""
