"""
Windows-специфичный фикс: ctranslate2 (бэкенд faster-whisper) не находит
cublas64_12.dll и часть cudnn_*64_9.dll, потому что сам их не поставляет,
а pip-пакеты nvidia-cublas-cu12 / nvidia-cudnn-cu12 кладут DLL в свою
собственную папку site-packages, откуда Windows их не подхватывает
(ни PATH, ни os.add_dll_directory на практике не срабатывают для
загрузки нативного расширения ctranslate2 - см. PLAN.md, раздел M0).

Рабочее решение: скопировать недостающие DLL прямо в папку пакета
ctranslate2, рядом с _ext.*.pyd - Windows ищет зависимости DLL в первую
очередь рядом с загружаемым модулем.

Запускать один раз после `pip install -r requirements.txt` / `pip install -e .`
(или добавить вызов в install-скрипт настройки окружения).
"""

import shutil
import sys
from pathlib import Path


def find_site_packages() -> Path:
    import sysconfig

    return Path(sysconfig.get_path("purelib"))


def main() -> int:
    site_packages = find_site_packages()
    ct2_dir = site_packages / "ctranslate2"
    cublas_bin = site_packages / "nvidia" / "cublas" / "bin"
    cudnn_bin = site_packages / "nvidia" / "cudnn" / "bin"

    if not ct2_dir.is_dir():
        print(f"ctranslate2 package not found at {ct2_dir}. Установлен ли faster-whisper?")
        return 1

    missing_sources = [p for p in (cublas_bin, cudnn_bin) if not p.is_dir()]
    if missing_sources:
        print(
            "Не найдены папки с DLL: "
            + ", ".join(str(p) for p in missing_sources)
            + ". Установлены ли nvidia-cublas-cu12 и nvidia-cudnn-cu12?"
        )
        return 1

    copied = []
    for dll in cublas_bin.glob("*.dll"):
        dest = ct2_dir / dll.name
        shutil.copy2(dll, dest)
        copied.append(dest.name)

    for dll in cudnn_bin.glob("*.dll"):
        if dll.name == "cudnn64_9.dll":
            # ctranslate2 уже поставляет свою версию этого файла - не перезаписываем
            continue
        dest = ct2_dir / dll.name
        shutil.copy2(dll, dest)
        copied.append(dest.name)

    print(f"Скопировано {len(copied)} DLL в {ct2_dir}:")
    for name in sorted(copied):
        print(" -", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
