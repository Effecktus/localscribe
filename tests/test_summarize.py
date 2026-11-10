"""Тесты для localscribe.summarize: суммаризация через Ollama.

Часть тестов (краткая/длинная суммаризация) обращаются к реально запущенному
серверу Ollama с моделью qwen3:8b - как и остальной проект, мы
проверяем настоящий локальный пайплайн, а не только мокаем HTTP. Ошибочные
пути (сервер недоступен, HTTP-ошибка) тестируются через monkeypatch, без
реальной сети.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest

from localscribe.config import DEFAULT_OLLAMA_MODEL, DEFAULT_SUMMARY_STYLE
from localscribe.errors import OllamaConnectionError, OllamaRequestError
from localscribe.summarize import (
    SummaryResult,
    _split_into_chunks,
    check_ollama_available,
    save_summary,
    summarize,
)

SAMPLE_TEXT = (
    "Сегодня мы обсудили план разработки локального приложения для "
    "транскрибации видео и аудио с последующей суммаризацией текста. "
    "Решили использовать faster-whisper для распознавания речи на GPU и "
    "Ollama с моделью Qwen для генерации выжимки. Всё будет работать "
    "полностью локально, без отправки данных в облако."
)

FIRST_PERSON_TEXT = (
    "Я сегодня наконец-то купил себе новый компьютер, собрал его сам. "
    "Потратил на это дело почти всю зарплату, но я не жалею, он мне нужен "
    "для работы. Ещё я решил, что в отпуск в этом году поеду к родителям, "
    "нужно будет заранее договориться с ними о датах."
)

# Диапазон символов китайского письма (CJK Unified Ideographs) - используем
# для проверки, что модель не "съехала" на китайский язык посреди ответа.
# Исторически это был глитч Qwen2.5 на длинной генерации. На qwen3:8b и
# gemma3:4b он не наблюдался, но проверка остаётся как дешёвый общий guard
# на смену языка - все эти модели мультиязычные.
_CJK_RE = re.compile(r"[一-鿿]")

# Длинный синтетический текст в стиле неформального видео-дневника от первого
# лица (~9000 символов - больше старого порога в 3000, меньше нового в 24000).
# На реальном транскрипте именно такого стиля модель регрессировала в
# "в этом видео автор делится..." - это распространённый шаблон описания
# видео в обучающих данных, который перетягивает модель в третье лицо.
_VLOG_STYLE_TEXT = (
    "Короче, давно не записывал видео, решил наконец вернуться к этому делу, "
    "расскажу, что у меня произошло за последнюю неделю, по дням более-менее.\n\n"
    "В понедельник я с утра поехал на работу, по пути заехал в кофейню, купил "
    "себе кофе и булочку с корицей, прям вкусно было. На работе было спокойно, "
    "доделал один проект, который тянулся уже две недели, наконец-то скинул его "
    "с плеч. Вечером встретился с другом, посидели в баре, обсудили планы на "
    "выходные, думаем съездить куда-нибудь на природу.\n\n"
    "Во вторник я проспал будильник, опоздал на встречу, было неловко, но все "
    "отнеслись с пониманием. Купил себе новые наушники, давно хотел, старые "
    "совсем расклеились. Вечером читал книгу, которую начал ещё месяц назад, "
    "надо бы её уже добить. Ещё я заказал себе новую клавиатуру, механическую, "
    "давно присматривался к такой.\n\n"
    "В среду была куча встреч на работе, устал прям конкретно. Коллега "
    "рассказал, что увольняется, переходит в другую контору, буду скучать по "
    "нему, хороший человек. Вечером готовил ужин, пытался повторить рецепт из "
    "видео, получилось не очень, но съедобно. Параллельно думал над тем, "
    "сделать ли ремонт в квартире в следующем году, цены на материалы "
    "конкретно выросли.\n\n"
    "Четверг прошёл довольно скучно, если честно, ничего особенного не "
    "произошло. Поработал, посмотрел сериал вечером, лёг пораньше, потому что "
    "в последнее время плохо высыпаюсь. Думаю, надо пересмотреть режим сна, "
    "ложиться раньше и вставать раньше, но пока только думаю об этом, на деле "
    "ничего не меняется.\n\n"
    "В пятницу наконец выбрался с друзьями в караоке, давно хотели собраться. "
    "Попели песни, посмеялись, хорошо провели время. После караоке пошли в "
    "парк, погода была отличная, тёплая, посидели на лавочке, поговорили о "
    "жизни. Домой вернулся поздно, но настроение было отличное весь вечер.\n\n"
    "В субботу занимался домашними делами, убрался в квартире, съездил на "
    "рынок за продуктами. Вечером играл с друзьями в настольную игру, засиделись "
    "допоздна, было весело, особенно когда я неожиданно выиграл последний раунд. "
    "Ещё обсудили с друзьями планы на отпуск осенью, думаем куда-то поехать все "
    "вместе, пока не решили куда конкретно.\n\n"
    "В воскресенье просто отдыхал, никуда не выходил почти. Посмотрел пару "
    "фильмов, поел что-то вкусное, написал несколько сообщений друзьям, с "
    "которыми давно не общался. Вечером сел писать этот текст, чтобы не "
    "забыть, что было за неделю, пока свежо в памяти. На этом, пожалуй, всё, "
    "на следующей неделе постараюсь записать что-нибудь ещё, если будет время."
)


def test_summarize_short_text_direct() -> None:
    result = summarize(SAMPLE_TEXT, style="краткая")

    assert isinstance(result, SummaryResult)
    assert result.chunks_used == 1
    assert result.model_name == DEFAULT_OLLAMA_MODEL
    assert len(result.summary) > 0
    assert len(result.summary) < len(SAMPLE_TEXT) * 2  # действительно выжимка, а не перепев


def test_summarize_default_style_is_detailed() -> None:
    result = summarize(SAMPLE_TEXT)
    assert result.style == DEFAULT_SUMMARY_STYLE == "подробная"


def test_summarize_preserves_first_person_voice() -> None:
    """Регрессия: раньше модель пересказывала 'я' как 'владелец канала'/'автор'
    в третьем лице. Выжимка текста от первого лица должна остаться от первого лица.

    Буквального 'я' НЕ требуем: в русском подлежащее часто опускается, и
    'Сегодня купил компьютер, собрал сам' - такое же первое лицо, как
    'Я сегодня купил'. qwen3 местоимение опускает систематически, а лицо при
    этом держит. По той же причине нельзя опознать лицо и по глаголу: в
    прошедшем времени 'купил' одинаково годится и для 'я', и для 'он'.
    Поэтому настоящий признак регрессии - появление стороннего субъекта
    (автор/рассказчик/он), его и проверяем. Ровно так же устроен
    test_summarize_does_not_describe_text_in_third_person.
    """
    result = summarize(FIRST_PERSON_TEXT, style="краткая")

    lowered = result.summary.lower()
    for forbidden in (
        "автор", "рассказчик", "владелец канала", "герой",
        "он решил", "он купил", "он потратил", "он поехал",
    ):
        assert forbidden not in lowered, f"Нашёл '{forbidden}' - модель ушла в третье лицо"


def test_summarize_stays_in_russian_no_language_drift() -> None:
    """Регрессия: на длинной генерации модель иногда 'съезжает' на китайский
    язык посреди русского ответа (ловилось на Qwen2.5). Проверяем все три стиля."""
    for style in ("краткая", "подробная", "тезисы"):
        result = summarize(SAMPLE_TEXT, style=style)
        assert not _CJK_RE.search(result.summary), (
            f"Обнаружены китайские символы в выжимке (style={style!r}): {result.summary!r}"
        )


def test_summarize_moderate_length_text_stays_single_pass() -> None:
    """Регрессия: раньше порог map-reduce (3000 символов) был искусственно низким
    относительно реального контекста модели (16384 токена) и дробил даже не очень
    длинные тексты без необходимости, теряя связность. Текст заметно длиннее
    старого порога должен укладываться в один проход при новом (24000 символов)."""
    long_text = "\n\n".join([_VLOG_STYLE_TEXT] * 3)
    assert len(long_text) > 6000  # заметно больше старого порога в 3000

    result = summarize(long_text, style="подробная")

    assert result.chunks_used == 1


def test_summarize_does_not_describe_text_in_third_person() -> None:
    """Регрессия: на неформальном видео-дневниковом тексте от первого лица модель
    склонна скатываться в шаблон описания видео со стороны ('в этом видео автор
    делится...') - распространённый паттерн из обучающих данных. Явно проверяем,
    что выжимка остаётся от первого лица."""
    result = summarize(_VLOG_STYLE_TEXT, style="подробная")

    # Примечание: не требуем буквального "я" в тексте - в русском языке
    # подлежащее часто опускается (глагол сам несёт лицо: "завершил проект"
    # и так же легитимно, как "я завершил проект"). Регрессия, которую мы
    # на самом деле проверяем - именно описание текста СО СТОРОНЫ.
    lowered = result.summary.lower()
    for forbidden in (
        "в этом видео", "в этом ролике", "автор делится", "автор рассказывает",
        "рассказчик", "герой видео", "владелец канала",
    ):
        assert forbidden not in lowered, f"Нашёл '{forbidden}' - модель описывает текст со стороны"


def test_news_with_quotes_preserves_third_person() -> None:
    result = summarize(
        "Мария и Виктор объявили о создании партии. Мария сказала: "
        "«Мы хотим объединить сторонников». Совет продолжает работать отдельно "
        "и не является политической партией.", seed=42, style="краткая",
    )
    assert not re.search(r"\bя\b", result.summary.lower())
    assert "мария" in result.summary.lower()


BLUNT_TEXT = (
    "Короче, сходил я вчера в этот их сервисный центр и это полный бардак. "
    "Записывался за неделю, приехал к назначенному времени - а они мне говорят, "
    "что мастера нет и вообще записи моей в системе не существует. Я им говорю: "
    "вы чё, издеваетесь? Сидят трое, пялятся в телефоны, никто ничего не знает. "
    "Короче, два часа потерял впустую на эту хрень. Девка на ресепшене хамила, "
    "как будто я ей лично должен. Деньги за диагностику при этом списали заранее "
    "и возвращать отказались, сволочи. Больше туда ни ногой, и всем знакомым "
    "скажу, чтобы обходили стороной эту контору."
)


def test_summarize_does_not_soften_or_moralize() -> None:
    """Регрессия на правило 4 промпта: модель не должна смягчать резкий текст,
    заменять грубости эвфемизмами, отказываться или добавлять от себя моральные
    оговорки. Модели из коробки склонны именно к этому - сглаживать тон и
    приписывать 'важно отметить, что...' к неудобному содержанию."""
    result = summarize(BLUNT_TEXT, style="краткая")

    lowered = result.summary.lower()

    # 1. Отказа/дисклеймера быть не должно.
    for forbidden in (
        "не могу", "извин", "неуместн", "оскорбительн", "некорректн",
        "important to note", "i cannot",
    ):
        assert forbidden not in lowered, (
            f"Похоже на отказ/дисклеймер ('{forbidden}'): {result.summary!r}"
        )

    # 2. Выжимка должна остаться про конфликт, а не превратиться в нейтральный
    #    пересказ "посетил сервисный центр, остался недоволен обслуживанием".
    assert any(
        marker in lowered
        for marker in ("бардак", "хам", "хрень", "сволоч", "деньги", "не верн", "отказал")
    ), f"Содержание сглажено до неузнаваемости: {result.summary!r}"


def test_summarize_bullet_style_contains_list_markers() -> None:
    result = summarize(SAMPLE_TEXT, style="тезисы")

    assert result.style == "тезисы"
    # Маркированный список обычно содержит дефисы/звёздочки/цифры построчно.
    assert any(marker in result.summary for marker in ("-", "*", "1", "•"))


def test_summarize_long_text_triggers_map_reduce() -> None:
    long_text = " ".join([SAMPLE_TEXT] * 8)  # заведомо длиннее лимита в один чанк

    result = summarize(long_text, style="краткая", chunk_char_limit=400)

    assert result.chunks_used > 1
    assert len(result.summary) > 0


def test_summarize_unknown_style_raises() -> None:
    with pytest.raises(ValueError):
        summarize(SAMPLE_TEXT, style="несуществующий-стиль")


def test_summarize_empty_text_raises() -> None:
    with pytest.raises(ValueError):
        summarize("   ", style="краткая")


def test_summarize_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(OllamaConnectionError):
        summarize(SAMPLE_TEXT, style="краткая")


def test_summarize_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status_code = 404
        text = "model not found"

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(OllamaRequestError):
        summarize(SAMPLE_TEXT, style="краткая", model="no-such-model")


def test_check_ollama_available_true_when_server_responds() -> None:
    assert check_ollama_available() is True


def test_check_ollama_available_false_on_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", fake_get)

    assert check_ollama_available() is False


def test_split_into_chunks_respects_limit() -> None:
    text = "Одно предложение. " * 50
    chunks = _split_into_chunks(text, max_chars=100)

    assert len(chunks) > 1
    assert all(len(c) <= 150 for c in chunks)  # небольшой запас на границы предложений


def test_split_into_chunks_single_chunk_for_short_text() -> None:
    chunks = _split_into_chunks(SAMPLE_TEXT, max_chars=10_000)
    assert chunks == [SAMPLE_TEXT]


def test_save_summary_writes_utf8_text(tmp_path: Path) -> None:
    result = SummaryResult(summary="Короткая выжимка.", style="краткая", model_name="x", chunks_used=1)
    out = tmp_path / "out.txt"

    saved = save_summary(result, out)

    assert saved == out
    assert out.read_text(encoding="utf-8") == "Короткая выжимка.\n"
