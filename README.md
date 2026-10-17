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
ollama pull qwen2.5:7b-instruct
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
| `--llm-model` | `qwen2.5:7b-instruct` | Модель в Ollama |
| `--lang` | `ru` | Код языка или `auto` |
| `--device` | `cuda` | `cuda` \| `cpu` (автооткат на CPU при сбое GPU) |
| `--summary-style` | `краткая` | `краткая` \| `подробная` \| `тезисы` |
| `--keep-audio` | выкл. | Не удалять промежуточный WAV |

Полный список опций и подкоманд: `--help` на любом уровне (`localscribe.cli --help`,
`localscribe.cli process --help` и т.д.).

## Тесты

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

Тесты гоняют настоящий пайплайн (faster-whisper + реальный сервер Ollama с моделью
`qwen2.5:7b-instruct`), поэтому перед запуском Ollama должен быть запущен и модель скачана.

## Статус

✅ MVP готов: извлечение аудио, транскрибация (GPU/CPU), суммаризация (map-reduce для длинных
текстов), CLI (`process` / `transcribe` / `summarize` / `doctor`), 31 тест.
