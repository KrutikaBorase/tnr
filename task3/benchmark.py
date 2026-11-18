"""Compare two Qwen GGUF quantizations using a local llama.cpp server."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import threading
import time
from pathlib import Path

import psutil
import requests


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "task3"
RESULTS = ROOT / "results"
PROMPTS = json.loads((TASK_DIR / "prompts.json").read_text(encoding="utf-8"))
MODELS = {
    "Q4_K_M": TASK_DIR / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "Q8_0": TASK_DIR / "models" / "qwen2.5-1.5b-instruct-q8_0.gguf",
}


def find_binary(name: str) -> Path:
    matches = list((TASK_DIR / "bin").rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} missing; run task3\\download_models.ps1 first")
    return matches[0]


class ProcessMonitor:
    def __init__(self, process: subprocess.Popen, interval: float = 0.5):
        self.process = process
        self.interval = interval
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        try:
            handle = psutil.Process(self.process.pid)
        except psutil.Error:
            return
        while not self._stop.is_set():
            try:
                self.peak_rss = max(self.peak_rss, handle.memory_info().rss)
            except psutil.Error:
                break
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=3)


def get_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_until_ready(base_url: str, process: subprocess.Popen, log_path: Path) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"llama-server exited with code {process.returncode}; see {log_path}"
            )
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"llama-server did not become ready; see {log_path}")


def available_devices(cli_binary: Path) -> list[str]:
    result = subprocess.run(
        [str(cli_binary), "--list-devices"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() not in ("Available devices:", "(none)")
    ]


def host_gpu_name() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def benchmark_model(
    label: str, model_path: Path, server_binary: Path, gpu_layers: int
) -> dict:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model missing: {model_path}; run the download script")
    port = get_free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = TASK_DIR / "output" / f"server-{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(server_binary),
        "--model",
        str(model_path),
        "--ctx-size",
        "4096",
        "--n-gpu-layers",
        str(gpu_layers),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-warmup",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            with ProcessMonitor(process) as monitor:
                wait_until_ready(base_url, process, log_path)
                warmup = requests.post(
                    f"{base_url}/v1/chat/completions",
                    json={
                        "model": "qwen",
                        "messages": [{"role": "user", "content": "Reply with OK."}],
                        "temperature": 0,
                        "max_tokens": 8,
                        "seed": 42,
                    },
                    timeout=120,
                )
                warmup.raise_for_status()
                outputs = []
                elapsed_total = 0.0
                tokens_total = 0
                for prompt in PROMPTS:
                    start = time.perf_counter()
                    response = requests.post(
                        f"{base_url}/v1/chat/completions",
                        json={
                            "model": "qwen",
                            "messages": [{"role": "user", "content": prompt["prompt"]}],
                            "temperature": 0,
                            "max_tokens": 128,
                            "seed": 42,
                        },
                        timeout=180,
                    )
                    response.raise_for_status()
                    elapsed = time.perf_counter() - start
                    body = response.json()
                    text = body["choices"][0]["message"]["content"].strip()
                    tokens = body.get("usage", {}).get("completion_tokens", 0)
                    outputs.append(
                        {
                            "id": prompt["id"],
                            "prompt": prompt["prompt"],
                            "output": text,
                            "completion_tokens": tokens,
                            "elapsed_seconds": elapsed,
                            "wall_tokens_per_second": tokens / elapsed if elapsed else None,
                        }
                    )
                    elapsed_total += elapsed
                    tokens_total += tokens
                time.sleep(0.5)
                peak_rss = monitor.peak_rss
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    return {
        "quantization": label,
        "model_file": model_path.name,
        "model_file_bytes": model_path.stat().st_size,
        "llama_cpp_version": "b12477",
        "gpu_layers": gpu_layers,
        "context_size": 4096,
        "prompt_count": len(PROMPTS),
        "warmup_prompts_excluded": 1,
        "completion_tokens_total": tokens_total,
        "wall_time_seconds_total": elapsed_total,
        "tokens_per_second": tokens_total / elapsed_total if elapsed_total else None,
        "peak_process_rss_mib": peak_rss / (1024**2),
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS / "task3_metrics.json")
    args = parser.parse_args()
    server_binary = find_binary("llama-server.exe")
    cli_binary = find_binary("llama-cli.exe")
    devices = available_devices(cli_binary)
    gpu_layers = 99 if devices else 0
    accelerator = ", ".join(devices) if devices else "CPU (llama.cpp reported no available devices)"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for label, model_path in MODELS.items():
        print(f"Benchmarking {label} ...", flush=True)
        results[label] = benchmark_model(label, model_path, server_binary, gpu_layers)
        print(
            f"{label}: {results[label]['tokens_per_second']:.2f} tokens/s, "
            f"peak process RSS {results[label]['peak_process_rss_mib']:.0f} MiB",
            flush=True,
        )
    output = {
        "model_family": "Qwen2.5-1.5B-Instruct",
        "accelerator": accelerator,
        "llama_cpp_devices": devices,
        "host_cpu": platform.processor(),
        "host_gpu": host_gpu_name(),
        "measurement_method": (
            "Local llama.cpp OpenAI-compatible server; wall generation throughput "
            "is completion token count divided by sequential client request time. "
            "One warm-up is excluded. Peak process RSS is sampled from the server "
            "process; GPU VRAM is not included when llama.cpp uses CPU."
        ),
        "results": results,
    }
    args.output.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value["tokens_per_second"] for key, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
