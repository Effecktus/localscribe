"""Воспроизводимое сравнение на локальном JSON-манифесте.

Каждый прогон выполняется в отдельном процессе: CUDA-кэш предыдущей модели
не влияет на следующую. Выжимки оцениваются человеком по сохранённым текстам.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import httpx


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower().replace("ё", "е"))


def word_errors(reference: str, hypothesis: str) -> dict:
    """Расстояние Левенштейна; регистр, пунктуация и ё/е нормализуются."""
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        raise ValueError("Пустой эталон для WER.")
    previous = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        current = [i]
        for j, b in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return {"word_errors": previous[-1], "reference_words": len(ref),
            "wer": previous[-1] / len(ref)}


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8-sig").strip()


def _stop_worker(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    # Windows venv использует launcher: нужно завершить также его дочерний
    # python, иначе после отмены сравнения инференс продолжит занимать GPU.
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, timeout=15)
    else:
        process.kill()
    process.wait(timeout=15)


def _worker(request: dict) -> dict:
    for filename, expected in request.get("implementation_sha256", {}).items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError("Исходный код изменился во время benchmark; повторите сравнение.")
    from localscribe import audio, summarize, transcribe
    started = time.perf_counter()
    if request["task"] == "asr":
        with audio.temporary_audio(Path(request["audio"])) as wav:
            result = transcribe.transcribe(wav, model_name=request["model"],
                                          device=request["device"], language="ru")
        data = asdict(result)
        data.update(word_errors(_read(request["reference"]), result.text))
        duration = max((s.end for s in result.segments), default=0)
        data["last_speech_end_seconds"] = duration
    else:
        result = summarize.summarize(_read(request["text"]), model=request["model"],
                                     host=request["host"], seed=request["seed"],
                                     style=request["style"], num_predict=request["num_predict"],
                                     chunk_char_limit=request["chunk_char_limit"])
        data = asdict(result)
        try:
            response = httpx.get(f"{request['host']}/api/ps", timeout=5, trust_env=False)
            response.raise_for_status()
            data["ollama_runtime"] = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            data["ollama_runtime_error"] = str(exc)
    data["seconds"] = time.perf_counter() - started
    if request.get("duration_seconds"):
        data["real_time_factor"] = data["seconds"] / request["duration_seconds"]
    return data


def run_benchmark(manifest: Path, output: Path, *, asr_models: list[str],
                  llm_models: list[str], repeats: int = 1, host: str = "http://localhost:11434",
                  device: str = "cuda", seed: int = 42, num_predict: int = 1600,
                  chunk_char_limit: int = 24000) -> dict:
    if repeats < 1 or num_predict < 1 or chunk_char_limit < 1:
        raise ValueError("Число повторов и лимиты должны быть положительными.")
    spec = json.loads(_read(str(manifest)))
    cases = spec["cases"]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Нужны непустой набор и уникальные id примеров.")
    root = manifest.resolve().parent
    implementation = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in Path(__file__).resolve().parent.glob("*.py")}
    requests = []
    inputs = {}
    for case in cases:
        resolved = dict(case)
        for key in ("audio", "reference", "text", "reference_summary"):
            if key in case:
                path = (root / case[key]).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                resolved[key] = str(path)
                inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if case["task"] not in {"asr", "summary"}:
            raise ValueError(f"Неизвестная задача: {case['task']}")
        required = ("audio", "reference") if case["task"] == "asr" else ("text",)
        if any(key not in resolved for key in required):
            raise ValueError(f"Нет обязательных полей {required}: {case['id']}")
        models = asr_models if case["task"] == "asr" else llm_models
        for model in models:
            for repeat in range(repeats):
                requests.append({**resolved, "model": model, "repeat": repeat,
                                 "implementation_sha256": implementation,
                                 "seed": seed + repeat, "host": host, "device": device,
                                 "style": case.get("style", "подробная"),
                                 "num_predict": num_predict, "chunk_char_limit": chunk_char_limit})
    if not requests:
        raise ValueError("Не выбрано ни одного прогона.")
    requests.sort(key=lambda request: (request["task"], request["model"], request["repeat"], request["id"]))
    # Не перезаписываем результаты предыдущего сравнения.
    output.mkdir(parents=True, exist_ok=False)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "python": sys.version,
              "versions": {p: version(p) for p in ("faster-whisper", "ctranslate2", "httpx")},
              "manifest": spec, "input_sha256": inputs,
              "implementation_sha256": implementation, "runs": []}
    try:
        for endpoint, key in (("version", "ollama_version"), ("tags", "ollama_models")):
            response = httpx.get(f"{host}/api/{endpoint}", timeout=5, trust_env=False)
            response.raise_for_status()
            report[key] = response.json()
        # Ollama может удерживать LLM после предыдущего пользовательского запуска.
        for model in llm_models:
            response = httpx.post(f"{host}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30, trust_env=False)
            response.raise_for_status()
    except (httpx.HTTPError, ValueError) as exc:
        report["ollama_preflight_error"] = str(exc)
    for index, request in enumerate(requests, 1):
        print(f"[{index}/{len(requests)}] {request['id']} / {request['model']}", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            req, res = Path(tmp) / "request.json", Path(tmp) / "response.json"
            req.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
            with (output / f"run-{index:03d}.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, "-m", __name__, str(req), str(res)],
                                           stdout=log, stderr=log,
                                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                samples = []
                started = time.perf_counter()
                try:
                    while process.poll() is None:
                        try:
                            gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                                  "--format=csv,noheader,nounits"],
                                                 capture_output=True, text=True, timeout=5)
                            if gpu.returncode == 0:
                                samples.append(sum(int(x) for x in gpu.stdout.splitlines()))
                        except (OSError, ValueError, subprocess.TimeoutExpired):
                            pass
                        if time.perf_counter() - started > 1800:
                            break
                        time.sleep(0.5)
                finally:
                    _stop_worker(process)
            row = {"request": request, "exit_code": process.returncode,
                   "wall_seconds": time.perf_counter() - started,
                   "gpu_total_used_peak_mib": max(samples, default=None),
                   "gpu_memory_note": "Суммарная память всех GPU, включая рабочий стол; опрос ~0.5 с."}
            if res.exists():
                row.update(json.loads(_read(str(res))))
            else:
                row.update(status="error", error="Процесс завершился без результата; см. log.")
            report["runs"].append(row)
            totals = {}
            for run in report["runs"]:
                if run["request"]["task"] != "asr" or run["status"] != "ok":
                    continue
                aggregate = totals.setdefault(run["request"]["model"],
                    {"word_errors": 0, "reference_words": 0, "seconds": 0, "successful_runs": 0})
                for key in ("word_errors", "reference_words", "seconds"):
                    aggregate[key] += run[key]
                aggregate["successful_runs"] += 1
                aggregate["wer"] = aggregate["word_errors"] / aggregate["reference_words"]
            report["asr_totals"] = totals
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    review = ["# Проверка выжимок", "", "Сверяйте с исходником; эталон — один из допустимых вариантов.",
              "Заполните для каждого ответа: покрытие фактов, искажения, выдумки, читаемость.", ""]
    for i, row in enumerate(report["runs"], 1):
        if row["request"]["task"] == "summary":
            review.extend([f"## Прогон {i}: {row['request']['id']} / {row['request']['model']}",
                           "", row.get("summary", row.get("error", "")), "",
                           "Оценка: НЕ ПРОВЕРЕНО", ""])
    (output / "review.md").write_text("\n".join(review), encoding="utf-8")
    return report


if __name__ == "__main__":
    try:
        result = {"status": "ok", **_worker(json.loads(_read(sys.argv[1])))}
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
