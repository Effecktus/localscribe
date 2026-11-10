"""Сквозной прогон CLI на уже скачанной длинной записи, с отчётом времени.

Подготовка аудио: outputs/long-data/predlozhenie.mp3 (LibriVox, source.json).
Запускать отдельно от benchmark, чтобы GPU была свободна.
"""
import json
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from localscribe.benchmark import _stop_worker


def main(output: Path):
    source = Path("outputs/long-data/predlozhenie.mp3")
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(exist_ok=False, parents=True)
    model_file = Path("outputs/asr-models.json")
    models = (json.loads(model_file.read_text(encoding="utf-8")) if model_file.exists()
              else {"large-v3-turbo": "large-v3-turbo"})
    for model in ("qwen3:8b", "qwen3.5:4b"):
        response = httpx.post("http://localhost:11434/api/generate",
                              json={"model": model, "keep_alive": 0}, timeout=60, trust_env=False)
        if response.status_code == 404 and model == "qwen3.5:4b":
            continue
        response.raise_for_status()
    command = [sys.executable, "-m", "localscribe.cli", "process", str(source),
               "--out-dir", str(output), "--whisper-model", models["large-v3-turbo"],
               "--llm-model", "qwen3:8b", "--num-predict", "3200"]
    print("Long pipeline started; log:", output / "process.log", flush=True)
    started = time.perf_counter()
    with (output / "process.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log,
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        try:
            process.wait(timeout=1800)
        finally:
            _stop_worker(process)
    transcript = output / "predlozhenie.transcript.txt"
    report = {"command": command, "seconds": time.perf_counter() - started,
              "exit_code": process.returncode, "audio_seconds": 1861,
              "source": json.loads(source.with_name("source.json").read_text(encoding="utf-8")),
              "transcript_chars": len(transcript.read_text(encoding="utf-8")) if transcript.exists() else None,
              "summary_exists": (output / "predlozhenie.summary.txt").exists(),
              "limitation": "No verbatim reference; this run does not measure WER."}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/long-pipeline-01"))
    main(parser.parse_args().out_dir)
