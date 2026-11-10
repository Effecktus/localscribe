# localscribe

CLI-утилита для локальной (офлайн) транскрибации видео/аудио и генерации краткой выжимки.

- Транскрибация: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (GPU/CUDA, с fallback на CPU)
- Суммаризация: локальная LLM через [Ollama](https://ollama.com/)
- Работает полностью локально — ничего не уходит в облако

Подробный план/история разработки: [PLAN.md](PLAN.md)

## Установка (Windows)

### 1. Системные зависимости

```powershell
winget install -e --id Python.Python.3.12
winget install -e --id Gyan.FFmpeg
winget install -e --id Ollama.Ollama
```

Откройте новый терминал после установки (чтобы подтянулся обновлённый `PATH`), затем скачайте
модель для суммаризации:

```powershell
ollama pull qwen3:8b
```

### 2. Python-окружение

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[test]"
```

### 3. Windows-специфичный фикс для GPU (NVIDIA)

`ctranslate2` (бэкенд faster-whisper) на Windows не находит `cublas64_12.dll` и часть
`cudnn_*64_9.dll` из pip-пакетов `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` — ни `PATH`, ни
`os.add_dll_directory` тут не помогают. Нужно скопировать эти DLL прямо в папку пакета
`ctranslate2` один раз после установки зависимостей:

```powershell
.venv\Scripts\python.exe scripts\fix_ctranslate2_cuda_dlls.py
```

Подробности и другие найденные особенности окружения (Blackwell/RTX 50xx, конфликт
`torch`↔`ctranslate2` и т.п.) — в [PLAN.md](PLAN.md), раздел M0/M2.

### 4. Проверка окружения

```powershell
.venv\Scripts\python.exe -m localscribe.cli doctor
```

Должно показать `OK` по всем пунктам (ffmpeg, GPU/CUDA, Ollama, модель). Если что-то не так —
команда подскажет, что именно исправить.

## Использование

```powershell
# Полный пайплайн: видео/аудио -> транскрипт + выжимка
.venv\Scripts\python.exe -m localscribe.cli process lecture.mp4

# Только транскрибация
.venv\Scripts\python.exe -m localscribe.cli transcribe lecture.mp4 --out lecture.txt

# Только суммаризация готового текста
.venv\Scripts\python.exe -m localscribe.cli summarize lecture.txt --summary-style тезисы
```

Основные опции `process`:

| Опция | По умолчанию | Описание |
|---|---|---|
| `--out-dir` | рядом с входным файлом | Куда сохранить `*.transcript.txt` и `*.summary.txt` |
| `--whisper-model` | `large-v3-turbo` | Модель faster-whisper |
| `--llm-model` | `qwen3:8b` | Модель в Ollama |
| `--lang` | `ru` | Код языка или `auto` |
| `--device` | `cuda` | `cuda` \| `cpu` (автооткат на CPU при сбое GPU) |
| `--summary-style` | `подробная` | `краткая` \| `подробная` \| `тезисы` |
| `--hint` | нет | Подсказка с именами собственными (см. ниже) |
| `--keep-audio` | выкл. | Не удалять промежуточный WAV |

Полный список опций и подкоманд: `--help` на любом уровне (`localscribe.cli --help`,
`localscribe.cli process --help` и т.д.).

### Подсказка `--hint` — как резко поднять качество распознавания имён

Whisper систематически перевирает имена собственные, которых не ожидает. На
реальной записи «Свияжск» выходил тремя разными неверными способами и ни разу
правильно. Лечится подсказкой — её текст подаётся модели как предшествующий
контекст:

```powershell
localscribe process video.mp4 --hint "Мы с Эмилем съездили из Казани в Свияжск, по дороге заглянули в Иннополис."
```

Два правила, проверенные замерами на одной и той же записи:

| подсказка | «Свияжск» верно | «Иннополис» верно |
|---|---|---|
| нет | 0 из 3 | 3 из 3 |
| список через запятую | 3 из 3 | **0 из 3** |
| **связное предложение** | **3 из 3** | **3 из 3** |

1. Давайте **связное предложение**, а не перечисление: список ломает
   распознавание соседних имён.
2. Называйте **реальные** сущности записи. Нейтральная подсказка без конкретных
   имён не помогает, а вредит.

## Тесты

### Сравнение моделей

Скачать небольшой открытый набор (12 уникальных русских фрагментов FLEURS test
и статьи Gazeta test с эталонными выжимками):

```powershell
.venv\Scripts\python.exe scripts\prepare_public_benchmark.py
ollama pull qwen3.5:4b
.venv\Scripts\python.exe -m localscribe.cli benchmark outputs/public-data/manifest.json --out-dir outputs/comparison-01 --asr-model large-v3-turbo --asr-model large-v3 --llm-model qwen3:8b --llm-model qwen3.5:4b --repeats 2
```

Папка результата должна быть новой. Первый запуск ASR скачивает модели, если их
ещё нет в кэше Hugging Face. Можно передать путь к локальной CTranslate2-модели
вместо её имени. Скачивание данных/моделей требует интернета; само распознавание
и суммаризация остаются локальными.

`report.json` содержит тексты ответов, WER, длительность, параметры, seed,
версии библиотек, SHA-256 входов, сведения о моделях Ollama и причины ошибок.
WER нормализует регистр, пунктуацию и ё/е; числа словами и цифрами остаются
разными. Итоговый WER взвешен по числу слов, не усреднён по фрагментам.
Время ASR включает извлечение аудио и загрузку модели в отдельном процессе;
загрузку модели Ollama можно отличить по `load_duration` в метриках ответа.
Замеры GPU показывают общую занятую память, включая рабочий стол, с опросом
примерно раз в полсекунды; это не точная память конкретного процесса.

`review.md` предназначен для проверки выжимок по исходникам: пропуски фактов,
искажения, выдумки, читаемость. Лексическое совпадение с эталоном не используется
как автоматическая оценка достоверности. `long-stress` — искусственная склейка
статей для проверки обработки длинного входа, а не связная лекция.
FLEURS содержит чтение текста и не заменяет тест на спонтанной речи с шумом.

Для сквозной проверки на связной длинной записи можно добавить
`--include-long` к команде подготовки. Она скачает русскую постановку
[«Предложение» Чехова, LibriVox](https://librivox.org/predlozhenie-by-anton-chekhov/)
длительностью 31 минуту. Запись проверяет весь пайплайн и работу с диалогом;
точного эталона распознавания в этом тесте нет. `scripts/run_long_smoke.py`
запускает эту проверку с бюджетом ответа 3200 токенов и сохраняет лог и время.
Для обычного запуска достаточно команды `process` с
`--whisper-model large-v3-turbo --num-predict 3200`.

Свой манифест — JSON с массивом `cases`; пути задаются относительно манифеста:

```json
{
  "cases": [
    {"id": "speech", "task": "asr", "audio": "speech.wav", "reference": "speech.txt"},
    {"id": "text", "task": "summary", "text": "text.txt", "reference_summary": "reference.txt", "style": "краткая"}
  ]
}
```

Каждый прогон изолирован в отдельном процессе. Перед сравнением команда
выгружает выбранные модели Ollama, чтобы они не занимали память во время ASR.
Запускайте сравнение без одновременной генерации в других приложениях.
Для `process` и `summarize` доступен `--num-predict` (по умолчанию 1600).
Ответы с `done_reason=length` и пустые ответы отклоняются с понятной ошибкой,
чтобы неполная выжимка не сохранилась как готовая.

Источники данных и условия использования:
[FLEURS, CC-BY-4.0](https://huggingface.co/datasets/google/fleurs),
[Gazeta, некоммерческое использование, права у gazeta.ru](https://github.com/IlyaGusev/gazeta).
Скачанные материалы и результаты находятся в игнорируемой папке `outputs/`.

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

Тесты гоняют настоящий пайплайн (faster-whisper + реальный сервер Ollama с моделью
`qwen3:8b`), поэтому перед запуском Ollama должен быть запущен и модель скачана.

## Статус

✅ MVP готов: извлечение аудио, транскрибация (GPU/CPU), суммаризация (map-reduce для длинных
текстов), CLI (`process` / `transcribe` / `summarize` / `doctor` / `benchmark`), 48 тестов.

Первичные замеры и ограничения выбора моделей: [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).
