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
