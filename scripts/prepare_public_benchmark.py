"""Скачать небольшой фиксированный набор FLEURS ru_ru / Gazeta test.

Данные сохраняются в outputs и не включаются в исходный код. Архив FLEURS
читается потоком, на диск попадают только выбранные WAV. Никаких dataset scripts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import httpx


class StreamReader(io.RawIOBase):
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.buffer = bytearray()

    def readable(self):
        return True

    def readinto(self, target):
        while len(self.buffer) < len(target):
            try:
                self.buffer.extend(next(self.chunks))
            except StopIteration:
                break
        size = min(len(target), len(self.buffer))
        target[:size] = self.buffer[:size]
        del self.buffer[:size]
        return size


def prepare(out: Path, count: int) -> None:
    if count < 1:
        raise ValueError("count должен быть положительным")
    out.mkdir(parents=True, exist_ok=True)
    cases, provenance = [], []
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        info = client.get("https://huggingface.co/api/datasets/google/fleurs")
        info.raise_for_status()
        revision = info.json()["sha"]
        base = f"https://huggingface.co/datasets/google/fleurs/resolve/{revision}/data/ru_ru"
        table = client.get(f"{base}/test.tsv")
        table.raise_for_status()
        rows = list(csv.reader(io.StringIO(table.text), delimiter="\t"))
        # Первые уникальные тексты: несколько голосов одного предложения не
        # должны создавать иллюзию более широкого тематического покрытия.
        selected, seen = [], set()
        for row in rows:
            if row[3] not in seen:
                seen.add(row[3])
                selected.append(row)
            if len(selected) >= count:
                break
        wanted = {row[1]: row for row in selected}
        print(f"FLEURS: скачиваю {len(wanted)} выбранных WAV из потокового архива...", flush=True)
        missing = {name for name in wanted if not (out / name).exists()}
        if missing:
            with client.stream("GET", f"{base}/audio/test.tar.gz") as response:
                response.raise_for_status()
                stream = io.BufferedReader(StreamReader(response.iter_bytes()))
                with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                    for member in archive:
                        name = Path(member.name).name
                        if name in missing and member.isfile():
                            source = archive.extractfile(member)
                            if source is None:
                                continue
                            (out / name).write_bytes(source.read())
                            missing.remove(name)
                            print(f"  {name}; осталось {len(missing)}", flush=True)
                            if not missing:
                                break
            if missing:
                raise RuntimeError(f"Не найдены аудиофайлы: {missing}")
        for index, row in enumerate(selected):
            reference = f"fleurs-{index:02d}.txt"
            (out / reference).write_text(row[2], encoding="utf-8")
            cases.append({"id": f"fleurs-{index:02d}", "task": "asr", "audio": row[1],
                          "reference": reference, "duration_seconds": int(row[5]) / 16000,
                          "source": "google/fleurs", "source_id": row[0], "gender": row[6]})
        provenance.append({"dataset": "google/fleurs", "config": "ru_ru", "split": "test",
                           "revision": revision, "license": "CC-BY-4.0",
                           "url": "https://huggingface.co/datasets/google/fleurs",
                           "selection": "first unique normalized transcripts in test.tsv",
                           "tsv_sha256": hashlib.sha256(table.content).hexdigest()})
        print("Gazeta: скачиваю тестовые статьи и эталонные выжимки...", flush=True)
        response = client.get("https://datasets-server.huggingface.co/rows", params={
            "dataset": "IlyaGusev/gazeta", "config": "default", "split": "test",
            "offset": 0, "length": 30})
        response.raise_for_status()
        data = response.json()
        (out / "gazeta-source.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        selected_news = data["rows"][:5]
        longest = max(data["rows"], key=lambda r: len(r["row"]["text"]))
        if longest not in selected_news:
            selected_news.append(longest)
        for entry in selected_news:
            row, index = entry["row"], entry["row_idx"]
            name = f"gazeta-{index:02d}"
            (out / f"{name}.txt").write_text(row["text"], encoding="utf-8")
            (out / f"{name}.reference.txt").write_text(row["summary"], encoding="utf-8")
            cases.append({"id": name, "task": "summary", "text": f"{name}.txt",
                          "reference_summary": f"{name}.reference.txt", "style": "краткая",
                          "source": row["url"], "title": row["title"]})
        # Отдельный тест размера и map-reduce; это НЕ связная длинная лекция.
        combined = "\n\n".join(e["row"]["text"] for e in data["rows"][:10])
        (out / "long-stress.txt").write_text(combined, encoding="utf-8")
        cases.append({"id": "long-stress", "task": "summary", "text": "long-stress.txt",
                      "style": "подробная", "purpose": "synthetic concatenation: robustness only"})
        provenance.append({"dataset": "IlyaGusev/gazeta", "split": "test",
                           "url": "https://huggingface.co/datasets/IlyaGusev/gazeta",
                           "usage": "non-commercial evaluation; rights remain with gazeta.ru",
                           "revision": data.get("dataset_git_revision"),
                           "source_sha256": hashlib.sha256(response.content).hexdigest()})
    (out / "manifest.json").write_text(json.dumps({"sources": provenance, "cases": cases},
                                                  ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово: {out / 'manifest.json'}", flush=True)


def prepare_long() -> None:
    """Общедоступная постановка LibriVox, не эталонный ASR-корпус."""
    output = Path("outputs/long-data")
    output.mkdir(parents=True, exist_ok=True)
    url = "https://archive.org/download/predlozhenie_1404_librivox/predlozhenie_01_chekhov_64kb.mp3"
    audio = output / "predlozhenie.mp3"
    if not audio.exists():
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            audio.write_bytes(response.content)
    metadata = {"title": "Предложение", "author": "Антон Чехов", "audio_url": url,
                "source": "https://librivox.org/predlozhenie-by-anton-chekhov/",
                "rss": "https://librivox.org/rss/8692", "duration_seconds": 1861,
                "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "purpose": "long coherent multi-speaker end-to-end test; no verbatim WER reference"}
    (output / "source.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Длинная запись: {audio}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/public-data"))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--include-long", action="store_true", help="Также скачать 31-минутную постановку LibriVox (~15 МБ).")
    args = parser.parse_args()
    prepare(args.out_dir, args.count)
    if args.include_long:
        prepare_long()
