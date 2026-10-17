"""CLI-интерфейс localscribe: видео/аудио -> транскрипт -> выжимка."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from localscribe import audio, summarize as summarize_mod, transcribe as transcribe_mod
from localscribe.config import (
    DEFAULT_LANGUAGE,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_SUMMARY_STYLE,
    DEFAULT_WHISPER_DEVICE,
    DEFAULT_WHISPER_MODEL,
)
from localscribe.errors import LocalscribeError

# Разрешаем "-h" как короткий алиас для "--help" (по умолчанию Typer/click
# понимают только "--help").
CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

app = typer.Typer(
    help="Локальная (офлайн) транскрибация видео/аудио и суммаризация через LLM.",
    no_args_is_help=True,
    context_settings=CONTEXT_SETTINGS,
)
console = Console()
logger = logging.getLogger("localscribe")


@app.callback(context_settings=CONTEXT_SETTINGS)
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Подробный (debug) вывод: показывать debug-логи (загрузка моделей, HTTP-запросы к Ollama и т.п.)."),
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_time=False, show_path=False, markup=True)],
    )


def _resolve_language(lang: str) -> Optional[str]:
    return None if lang.lower() == "auto" else lang


def _fail(message: str) -> "typer.Exit":
    console.print(f"[bold red]Ошибка:[/bold red] {message}")
    return typer.Exit(code=1)


# Ожидаемые/обрабатываемые ошибки пайплайна - показываем пользователю понятное
# сообщение вместо traceback. LocalscribeError - сбои внешних систем (ffmpeg,
# Ollama); ValueError/FileNotFoundError - невалидный вход (например, пустой
# транскрипт или неизвестный --summary-style).
EXPECTED_ERRORS = (LocalscribeError, ValueError, FileNotFoundError)


@app.command(context_settings=CONTEXT_SETTINGS)
def process(
    input_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True,
        help="Путь к видео- или аудиофайлу (любой формат, который понимает ffmpeg).",
    ),
    out_dir: Optional[Path] = typer.Option(
        None, "--out-dir",
        help="Куда сохранить <имя>.transcript.txt и <имя>.summary.txt. "
             "По умолчанию - рядом с входным файлом.",
    ),
    whisper_model: str = typer.Option(
        DEFAULT_WHISPER_MODEL, "--whisper-model",
        help="Размер/имя модели faster-whisper для распознавания речи "
             "(tiny, base, small, medium, large-v3, large-v3-turbo, ...).",
    ),
    llm_model: str = typer.Option(
        DEFAULT_OLLAMA_MODEL, "--llm-model",
        help="Имя модели в Ollama для суммаризации (должна быть скачана: ollama pull <модель>).",
    ),
    lang: str = typer.Option(
        DEFAULT_LANGUAGE, "--lang",
        help="Код языка речи (например 'ru', 'en') или 'auto' для автоопределения Whisper'ом.",
    ),
    device: str = typer.Option(
        DEFAULT_WHISPER_DEVICE, "--device",
        help="Устройство для транскрибации: 'cuda' (GPU) или 'cpu'. "
             "При сбое/отсутствии CUDA автоматически откатывается на CPU.",
    ),
    summary_style: str = typer.Option(
        DEFAULT_SUMMARY_STYLE, "--summary-style",
        help="Стиль выжимки: 'краткая' (4-6 предложений), "
             "'подробная' (несколько абзацев) или 'тезисы' (маркированный список).",
    ),
    keep_audio: bool = typer.Option(
        False, "--keep-audio",
        help="Не удалять промежуточный WAV-файл после обработки "
             "(сохранится как <имя>.audio.wav в --out-dir).",
    ),
    ollama_host: str = typer.Option(
        DEFAULT_OLLAMA_HOST, "--ollama-host",
        help="Адрес сервера Ollama (если запущен не локально или на другом порту).",
    ),
) -> None:
    """Полный пайплайн: входной файл -> транскрипт -> выжимка."""
    out_dir = out_dir or input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    transcript_path = out_dir / f"{stem}.transcript.txt"
    summary_path = out_dir / f"{stem}.summary.txt"
    language = _resolve_language(lang)

    try:
        console.print(f"[bold]\\[1/3][/bold] Извлекаю аудио из [cyan]{input_path.name}[/cyan]...")

        if keep_audio:
            wav_path = audio.extract_audio(input_path, out_dir / f"{stem}.audio.wav")
            transcript_text = _transcribe_and_save(
                wav_path, transcript_path, whisper_model, device, language
            )
        else:
            with audio.temporary_audio(input_path) as wav_path:
                transcript_text = _transcribe_and_save(
                    wav_path, transcript_path, whisper_model, device, language
                )

        console.print("[bold]\\[3/3][/bold] Делаю выжимку...")
        summary = summarize_mod.summarize(
            transcript_text, model=llm_model, host=ollama_host, style=summary_style
        )
        summarize_mod.save_summary(summary, summary_path)
        console.print(f"   -> выжимка сохранена: [green]{summary_path}[/green]")

        console.print()
        console.print("[bold green]Готово![/bold green]")
        console.print(summary.summary)
    except EXPECTED_ERRORS as exc:
        raise _fail(str(exc))


def _transcribe_and_save(
    wav_path: Path,
    transcript_path: Path,
    whisper_model: str,
    device: str,
    language: Optional[str],
) -> str:
    console.print("[bold]\\[2/3][/bold] Распознаю речь...")
    result = transcribe_mod.transcribe(
        wav_path, model_name=whisper_model, device=device, language=language
    )
    transcribe_mod.save_transcript(result, transcript_path)
    console.print(
        f"   -> устройство: [cyan]{result.device_used}[/cyan], "
        f"язык: [cyan]{result.language}[/cyan] (p={result.language_probability:.2f})"
    )
    console.print(f"   -> транскрипт сохранён: [green]{transcript_path}[/green]")
    return result.text


@app.command(name="transcribe", context_settings=CONTEXT_SETTINGS)
def transcribe_cmd(
    input_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True,
        help="Путь к видео- или аудиофайлу (любой формат, который понимает ffmpeg).",
    ),
    out_path: Optional[Path] = typer.Option(
        None, "--out",
        help="Куда сохранить транскрипт. По умолчанию - <имя>.transcript.txt рядом с входным файлом.",
    ),
    whisper_model: str = typer.Option(
        DEFAULT_WHISPER_MODEL, "--whisper-model",
        help="Размер/имя модели faster-whisper для распознавания речи "
             "(tiny, base, small, medium, large-v3, large-v3-turbo, ...).",
    ),
    lang: str = typer.Option(
        DEFAULT_LANGUAGE, "--lang",
        help="Код языка речи (например 'ru', 'en') или 'auto' для автоопределения Whisper'ом.",
    ),
    device: str = typer.Option(
        DEFAULT_WHISPER_DEVICE, "--device",
        help="Устройство для транскрибации: 'cuda' (GPU) или 'cpu'. "
             "При сбое/отсутствии CUDA автоматически откатывается на CPU.",
    ),
    keep_audio: bool = typer.Option(
        False, "--keep-audio",
        help="Не удалять промежуточный WAV после обработки "
             "(сохранится как <имя>.audio.wav рядом с входным файлом).",
    ),
) -> None:
    """Только транскрибация (без суммаризации)."""
    out_path = out_path or input_path.with_name(f"{input_path.stem}.transcript.txt")
    language = _resolve_language(lang)

    try:
        if keep_audio:
            wav_path = audio.extract_audio(input_path, input_path.with_name(f"{input_path.stem}.audio.wav"))
            _transcribe_and_save(wav_path, out_path, whisper_model, device, language)
        else:
            with audio.temporary_audio(input_path) as wav_path:
                _transcribe_and_save(wav_path, out_path, whisper_model, device, language)
    except EXPECTED_ERRORS as exc:
        raise _fail(str(exc))


@app.command(name="summarize", context_settings=CONTEXT_SETTINGS)
def summarize_cmd(
    transcript_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True,
        help="Текстовый файл (UTF-8) с готовым транскриптом.",
    ),
    out_path: Optional[Path] = typer.Option(
        None, "--out",
        help="Куда сохранить выжимку. По умолчанию - <имя>.summary.txt рядом с транскриптом.",
    ),
    llm_model: str = typer.Option(
        DEFAULT_OLLAMA_MODEL, "--llm-model",
        help="Имя модели в Ollama для суммаризации (должна быть скачана: ollama pull <модель>).",
    ),
    summary_style: str = typer.Option(
        DEFAULT_SUMMARY_STYLE, "--summary-style",
        help="Стиль выжимки: 'краткая' (4-6 предложений), "
             "'подробная' (несколько абзацев) или 'тезисы' (маркированный список).",
    ),
    ollama_host: str = typer.Option(
        DEFAULT_OLLAMA_HOST, "--ollama-host",
        help="Адрес сервера Ollama (если запущен не локально или на другом порту).",
    ),
) -> None:
    """Только суммаризация готового текстового транскрипта (без распознавания речи)."""
    out_path = out_path or transcript_path.with_name(
        transcript_path.stem.removesuffix(".transcript") + ".summary.txt"
    )
    text = transcript_path.read_text(encoding="utf-8")

    try:
        summary = summarize_mod.summarize(text, model=llm_model, host=ollama_host, style=summary_style)
        summarize_mod.save_summary(summary, out_path)
        console.print(f"Выжимка сохранена: [green]{out_path}[/green]")
        console.print()
        console.print(summary.summary)
    except EXPECTED_ERRORS as exc:
        raise _fail(str(exc))


@app.command(context_settings=CONTEXT_SETTINGS)
def doctor(
    llm_model: str = typer.Option(
        DEFAULT_OLLAMA_MODEL, "--llm-model",
        help="Какую модель в Ollama проверять на наличие.",
    ),
    ollama_host: str = typer.Option(
        DEFAULT_OLLAMA_HOST, "--ollama-host",
        help="Адрес сервера Ollama (если запущен не локально или на другом порту).",
    ),
) -> None:
    """Проверяет окружение: ffmpeg, GPU/CUDA, доступность Ollama и нужной модели."""
    ok = True

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        console.print(f"[green]OK[/green]  ffmpeg найден: {ffmpeg_path}")
    else:
        ok = False
        console.print("[red]FAIL[/red]  ffmpeg не найден в PATH (winget install -e --id Gyan.FFmpeg).")

    if ffmpeg_path is None:
        console.print("[yellow]SKIP[/yellow]  Проверка GPU/транскрибации пропущена - нет ffmpeg.")
    else:
        console.print("Проверяю GPU/CUDA (сквозной тест на крошечном аудио)...")
        tmp_tone: Optional[Path] = None
        try:
            fd, tmp_name = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            tmp_tone = Path(tmp_name)
            subprocess.run(
                [ffmpeg_path, "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                 "-ar", "16000", "-ac", "1", "-loglevel", "error", str(tmp_tone)],
                check=True, capture_output=True,
            )
            result = transcribe_mod.transcribe(tmp_tone, model_name="tiny", device="cuda", language="ru")
            if result.device_used == "cuda":
                console.print("[green]OK[/green]  Транскрибация реально работает на GPU (CUDA).")
            else:
                console.print(
                    "[yellow]WARN[/yellow]  Транскрибация откатилась на CPU. Если на машине есть "
                    "NVIDIA GPU, запустите: python scripts/fix_ctranslate2_cuda_dlls.py"
                )
        except Exception as exc:
            ok = False
            console.print(f"[red]FAIL[/red]  Не удалось протестировать транскрибацию: {exc}")
        finally:
            if tmp_tone is not None:
                tmp_tone.unlink(missing_ok=True)

    if summarize_mod.check_ollama_available(ollama_host):
        console.print(f"[green]OK[/green]  Ollama отвечает на {ollama_host}.")
        try:
            models = summarize_mod.list_ollama_models(ollama_host)
        except LocalscribeError:
            models = []
        if llm_model in models:
            console.print(f"[green]OK[/green]  Модель '{llm_model}' скачана в Ollama.")
        else:
            ok = False
            console.print(
                f"[red]FAIL[/red]  Модель '{llm_model}' не найдена в Ollama. "
                f"Скачайте: ollama pull {llm_model}"
            )
    else:
        ok = False
        console.print(
            f"[red]FAIL[/red]  Ollama не отвечает на {ollama_host}. Запущен ли сервис? (ollama serve)"
        )

    console.print()
    if ok:
        console.print("[bold green]Всё готово к работе.[/bold green]")
    else:
        console.print("[bold red]Есть проблемы, см. сообщения выше.[/bold red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
