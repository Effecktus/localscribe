"""Суммаризация транскрипта через локальную LLM (Ollama).

По умолчанию текст суммаризируется за один проход целиком (с явно увеличенным
контекстным окном модели, см. OLLAMA_NUM_CTX в config.py) - это даёт связную
выжимку без потери смысловых связей между частями текста. Map-reduce (текст
режется на части, каждая суммаризируется отдельно, затем сводится в одну)
включается только для действительно длинных транскриптов, превышающих
SUMMARY_CHUNK_CHAR_LIMIT - это запасной вариант, а не основной путь, так как
фрагментация неизбежно хуже сохраняет связность и суть исходного текста.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from time import perf_counter
from pathlib import Path

import httpx

from localscribe.config import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_SUMMARY_STYLE,
    OLLAMA_NUM_CTX,
    OLLAMA_THINK,
    OLLAMA_TIMEOUT_SECONDS,
    SUMMARY_CHUNK_CHAR_LIMIT,
)
from localscribe.errors import OllamaConnectionError, OllamaRequestError

logger = logging.getLogger(__name__)

STYLE_INSTRUCTIONS = {
    "краткая": (
        "Сделай краткую выжимку текста (4-6 предложений), отражающую основную "
        "суть и самые важные детали."
    ),
    "подробная": (
        "Сделай подробную выжимку текста в виде нескольких абзацев (каждый абзац - "
        "отдельная тема/эпизод из текста). Пройдись по ВСЕМ значимым темам, которые "
        "затрагиваются в тексте, а не только по первой/главной - если в тексте "
        "несколько разных тем или эпизодов, каждая должна получить отражение в "
        "выжимке, пусть и компактное."
    ),
    "тезисы": (
        "Выдели ключевые тезисы текста в виде маркированного списка "
        "(5-10 пунктов), каждый пункт - отдельная законченная мысль."
    ),
}

# Общее правило для всех промптов: это должно быть СОКРАЩЕНИЕ исходного текста,
# а не пересказ/интерпретация своими словами, и форма повествования (лицо)
# должна сохраняться как в оригинале. Без этого модель склонна превращать
# "я сделал..." в "автор/рассказчик/владелец канала сделал..." и добавлять
# собственные выводы, которых в тексте не было.
_VOICE_AND_FIDELITY_RULE = (
    "Важные правила:\n"
    "1. Это должно быть СОКРАЩЕНИЕ исходного текста, а не пересказ своими словами "
    "и не интерпретация. Убирай повторы, воду, оговорки и несущественные детали, "
    "но не добавляй выводов, оценок и фактов, которых не было в тексте.\n"
    "2. Сохраняй форму повествования оригинала. Если текст от первого лица "
    "('я', 'мы') - выжимка тоже должна быть от первого лица. ЗАПРЕЩЕНО описывать "
    "текст со стороны конструкциями вида 'в этом видео автор...', 'в ролике "
    "рассказчик сообщает...', 'герой текста...', 'владелец канала...' - даже "
    "если сам текст называет себя видео/роликом. Пример:\n"
    "   ПЛОХО: \"В этом видео автор делится событиями прошедшей недели.\"\n"
    "   ХОРОШО: \"На этой неделе я...\"\n"
    "   Если исходный текст в третьем лице, сохрани третье лицо. Прямая речь "
    "и цитаты внутри текста не делают весь текст рассказом от первого лица. "
    "Не выдавай себя за участника событий и не превращай диалог в монолог.\n"
    "3. Отвечай только на русском языке от начала до конца. Не переключайся на "
    "другие языки (в том числе китайский или английский) ни в одном предложении.\n"
    "4. Не смягчай и не цензурируй содержание. Сохраняй лексику и тон оригинала, "
    "включая резкие, грубые и нецензурные выражения, если они есть в тексте: не "
    "заменяй их эвфемизмами ('нехорошее слово', '[censored]') и не пропускай "
    "неудобные темы. Не добавляй от себя оговорок, предупреждений и моральных "
    "оценок ('важно помнить, что...', 'следует отметить, что...', 'не стоит...'). "
    "Твоя задача - сократить текст, а не улучшить, смягчить или прокомментировать "
    "его.\n"
    "5. НИЧЕГО НЕ ДОДУМЫВАЙ. Если детали, причины или связи между событиями нет "
    "в тексте - не вставляй её, даже если она кажется очевидной. Не объединяй "
    "разные эпизоды в один, не меняй их порядок и не достраивай причинность "
    "('из-за этого', 'несмотря на'), если её нет в оригинале. Если не уверен "
    "в детали (что именно произошло, с кем, когда) - напиши обобщённо или "
    "опусти, но НЕ подменяй её похожей. Пример:\n"
    "   ПЛОХО: в тексте 'починил кран' -> в выжимке 'сделал ремонт в ванной'.\n"
    "   ХОРОШО: в тексте 'починил кран' -> в выжимке 'починил кран'.\n"
    "6. Сохраняй, кому принадлежит утверждение: мнение, обвинение, прогноз "
    "или предположение не превращай в установленный факт. Сохраняй даты и "
    "периоды статистики, условия и исключения. Не смешивай численность группы "
    "с числом задержанных и не меняй участников событий местами."
)

# Короткое напоминание сразу перед местом, где модель начинает генерировать
# ответ - модели заметно сильнее следуют инструкции, расположенной прямо перед
# точкой генерации, чем той же инструкции в начале длинного промпта.
_VOICE_REMINDER = (
    "Напоминание: сохрани лицо повествования исходника: первое для личного "
    "рассказа, третье для сообщения о других людях. Цитаты не меняют лицо "
    "всего текста. Начни с содержания, без описания текста или видео. "
    "Сохрани авторство мнений, числа, даты и важные условия."
)

_DIRECT_PROMPT = (
    "Вот транскрипт аудио- или видеозаписи на русском языке.\n"
    "{style_instruction}\n"
    f"{_VOICE_AND_FIDELITY_RULE}\n\n"
    "Транскрипт:\n{text}\n\n"
    f"{_VOICE_REMINDER}\n\n"
    "Выжимка:"
)

_MAP_PROMPT = (
    "Ниже - фрагмент транскрипта длинной записи (это только часть текста, "
    "не весь текст целиком). Сократи этот фрагмент, оставив основные мысли и "
    "факты по существу, без добавленных вступлений и выводов. Если в этом "
    "фрагменте затронуто несколько разных тем - кратко отрази каждую из них, "
    "не пропускай ни одну (обычно это 5-8 предложений).\n"
    f"{_VOICE_AND_FIDELITY_RULE}\n\n"
    "Фрагмент:\n{chunk}\n\n"
    f"{_VOICE_REMINDER}\n\n"
    "Сокращённый фрагмент:"
)

_REDUCE_PROMPT = (
    "Ниже приведены сокращённые версии последовательных частей одного длинного "
    "транскрипта, в порядке по тексту.\n"
    "{style_instruction}\n"
    f"{_VOICE_AND_FIDELITY_RULE}\n\n"
    "Части:\n{combined}\n\n"
    f"{_VOICE_REMINDER}\n\n"
    "Итоговая выжимка:"
)

_DIRECT_NUM_PREDICT = 1600

# Низкая temperature: суммаризация - задача фактическая (сократить, а не
# сочинить), а на рыхлых разговорных транскриптах с ошибками ASR высокая
# температура заметно увеличивает конфабуляции.
# Эти значения отобраны замером на реальном транскрипте (8 прогонов на модель,
# см. историю в PLAN.md): qwen3:8b на них не исказил ни одного проверяемого
# факта и покрыл 7.2/10 эпизодов против 5.9 у gemma3:4b.
# Примечание: официальный пресет Qwen3 для режима без размышлений - другой
# (temperature=0.7 / top_p=0.8 / top_k=20). Если захочется тюнинга, начинать
# стоит оттуда, но вслепую не менять: здешние значения проверены на задаче.
_SUMMARY_TEMPERATURE = 0.3
_SUMMARY_TOP_P = 0.95
_SUMMARY_TOP_K = 64

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Блок рассуждений гибридных моделей (Qwen3 и подобные), см. OLLAMA_THINK.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class SummaryResult:
    summary: str
    style: str
    model_name: str
    chunks_used: int  # 1, если текст суммаризирован за один проход (без map-reduce)
    calls: list[dict] = field(default_factory=list)


def _call_ollama(
    prompt: str,
    *,
    model: str,
    host: str,
    num_predict: int | None = None,
    timeout: float = OLLAMA_TIMEOUT_SECONDS,
    metrics: list[dict] | None = None,
    seed: int | None = None,
) -> str:
    # num_ctx явно: по умолчанию Ollama грузит модель с контекстом всего 4096
    # токенов, даже если сама модель поддерживает намного больше - без этого
    # длинные транскрипты обрезались бы/требовали map-reduce без необходимости.
    options: dict = {
        "temperature": _SUMMARY_TEMPERATURE,
        "top_p": _SUMMARY_TOP_P,
        "top_k": _SUMMARY_TOP_K,
        "num_ctx": OLLAMA_NUM_CTX,
    }
    if num_predict is not None:
        options["num_predict"] = num_predict
    if seed is not None:
        options["seed"] = seed
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": OLLAMA_THINK,
        "options": options,
    }

    started = perf_counter()
    try:
        response = httpx.post(f"{host}/api/generate", json=payload, timeout=timeout, trust_env=False)
    except httpx.HTTPError as exc:
        raise OllamaConnectionError(host, exc) from exc

    if response.status_code != 200:
        raise OllamaRequestError(
            f"Ollama вернул ошибку {response.status_code} для модели '{model}': "
            f"{response.text.strip()}"
        )

    data = response.json()
    if metrics is not None:
        metrics.append({
            "seconds": perf_counter() - started,
            "model": data.get("model", model),
            "options": options.copy(),
            **{key: data.get(key) for key in (
                "done_reason", "prompt_eval_count", "eval_count", "total_duration",
                "load_duration", "prompt_eval_duration", "eval_duration",
            )},
        })
    if data.get("done_reason") == "length":
        raise OllamaRequestError(
            f"Ответ модели '{model}' обрезан по лимиту {num_predict} токенов. "
            "Увеличьте num_predict; неполная выжимка не сохранена."
        )
    # think=False Ollama обычно соблюдает, но если модель всё же вернула блок
    # размышлений (другой раннер, другая версия) - он не должен попасть в выжимку.
    result = _THINK_BLOCK_RE.sub("", data["response"]).strip()
    if not result:
        raise OllamaRequestError(f"Модель '{model}' вернула пустую выжимку.")
    return result


def _split_into_chunks(text: str, max_chars: int = SUMMARY_CHUNK_CHAR_LIMIT) -> list[str]:
    """Режет текст на части по границам предложений, не превышая max_chars на часть."""
    if max_chars < 1:
        raise ValueError("Лимит части должен быть положительным.")
    # Без пунктуации ASR может вернуть одно предложение длиннее всего окна.
    sentences = [part for sentence in _SENTENCE_SPLIT_RE.split(text.strip())
                 for part in (sentence[i:i + max_chars]
                              for i in range(0, len(sentence), max_chars))]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current and current_len + len(sentence) > max_chars:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks or [text]


def summarize(
    text: str,
    *,
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    style: str = DEFAULT_SUMMARY_STYLE,
    chunk_char_limit: int = SUMMARY_CHUNK_CHAR_LIMIT,
    timeout: float = OLLAMA_TIMEOUT_SECONDS,
    num_predict: int = _DIRECT_NUM_PREDICT,
    seed: int | None = None,
) -> SummaryResult:
    """Делает выжимку текста транскрипта через локальную LLM в Ollama.

    Args:
        text: текст транскрипта.
        model: имя модели в Ollama (должна быть предварительно скачана через
            `ollama pull <model>`).
        host: адрес сервера Ollama.
        style: "краткая" | "подробная" | "тезисы".
        chunk_char_limit: порог длины (в символах), после которого включается
            map-reduce вместо суммаризации за один проход.
        timeout: таймаут одного запроса к Ollama, в секундах.

    Raises:
        ValueError: пустой текст или неизвестный style.
        OllamaConnectionError: сервер Ollama недоступен.
        OllamaRequestError: Ollama вернул ошибку (например, модель не найдена).
    """
    if style not in STYLE_INSTRUCTIONS:
        raise ValueError(
            f"Неизвестный стиль выжимки: {style!r}. "
            f"Доступные: {', '.join(STYLE_INSTRUCTIONS)}"
        )

    text = text.strip()
    if num_predict < 1:
        raise ValueError("num_predict должен быть положительным.")
    if not text:
        raise ValueError("Пустой транскрипт - нечего суммаризировать.")

    style_instruction = STYLE_INSTRUCTIONS[style]
    chunks = _split_into_chunks(text, chunk_char_limit)
    metrics: list[dict] = []

    if len(chunks) == 1:
        prompt = _DIRECT_PROMPT.format(style_instruction=style_instruction, text=chunks[0])
        summary = _call_ollama(
            prompt, model=model, host=host, num_predict=num_predict, timeout=timeout,
            metrics=metrics, seed=seed,
        )
        return SummaryResult(summary=summary, style=style, model_name=model, chunks_used=1, calls=metrics)

    logger.info(
        "Транскрипт длинный (%d символов), разбиваю на %d частей для map-reduce.",
        len(text), len(chunks),
    )
    partial_summaries = []
    for i, chunk in enumerate(chunks, start=1):
        logger.info("Суммаризирую часть %d/%d...", i, len(chunks))
        partial = _call_ollama(
            _MAP_PROMPT.format(chunk=chunk),
            model=model, host=host, num_predict=num_predict, timeout=timeout,
            metrics=metrics, seed=seed,
        )
        partial_summaries.append(partial)

    combined = "\n\n".join(f"Часть {i}: {s}" for i, s in enumerate(partial_summaries, start=1))
    # Даже промежуточные выжимки очень длинной записи могут не войти в окно.
    for _ in range(8):
        groups = _split_into_chunks(combined, max(chunk_char_limit, SUMMARY_CHUNK_CHAR_LIMIT))
        if len(groups) == 1:
            break
        reduced = [
            _call_ollama(_MAP_PROMPT.format(chunk=group), model=model, host=host,
                         num_predict=num_predict, timeout=timeout, metrics=metrics, seed=seed)
            for group in groups
        ]
        next_combined = "\n\n".join(reduced)
        if len(next_combined) >= len(combined):
            raise OllamaRequestError("Промежуточные выжимки не сокращаются до размера контекста.")
        combined = next_combined
    else:
        raise OllamaRequestError("Слишком много уровней суммаризации; увеличьте размер части.")
    reduce_prompt = _REDUCE_PROMPT.format(style_instruction=style_instruction, combined=combined)
    summary = _call_ollama(
        reduce_prompt, model=model, host=host, num_predict=num_predict, timeout=timeout,
        metrics=metrics, seed=seed,
    )

    return SummaryResult(summary=summary, style=style, model_name=model, chunks_used=len(chunks), calls=metrics)


def save_summary(result: SummaryResult, output_path: str | Path) -> Path:
    """Сохраняет текст выжимки в файл (UTF-8)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.summary + "\n", encoding="utf-8")
    return output_path


def check_ollama_available(host: str = DEFAULT_OLLAMA_HOST, timeout: float = 5.0) -> bool:
    """Быстрая проверка, что сервер Ollama отвечает."""
    try:
        response = httpx.get(f"{host}/api/version", timeout=timeout)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def list_ollama_models(host: str = DEFAULT_OLLAMA_HOST, timeout: float = 5.0) -> list[str]:
    """Возвращает список имён моделей, уже скачанных в Ollama."""
    try:
        response = httpx.get(f"{host}/api/tags", timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaConnectionError(host, exc) from exc

    data = response.json()
    return [m["name"] for m in data.get("models", [])]
