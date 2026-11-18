# Small-model adaptation, retrieval, and local inference

This repository contains independent, reproducible implementations for the
three assignment tasks. The scripts save raw predictions and measurements
under `results/`; the included report summarizes those measured results.

## Environment used for the measured run

- Windows 11, NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM), CUDA-capable
  NVIDIA driver; 16 GiB system memory.
- Python 3.11 in `.venv`.
- Task 1: Qwen2.5-0.5B-Instruct with PEFT LoRA.
- Task 2: RFC 9110 (HTTP Semantics), `all-MiniLM-L6-v2`, NumPy cosine search.
- Task 3: Qwen2.5-1.5B-Instruct with llama.cpp; Q4_K_M vs Q8_0. The host GPU
  was visible to PyTorch but not enumerated by this llama.cpp build, so the
  measured Task 3 runs explicitly used CPU and report process RSS.

## Setup (PowerShell)

Run from the repository root. The current venv package set supports the
scripts; install a matching CUDA-enabled PyTorch wheel separately if pip's
default Windows wheel does not expose CUDA on your machine.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For machines with a different supported CUDA/PyTorch combination, use the
official PyTorch selector and keep the remaining dependencies in
`requirements.txt`.

## Run the tasks

```powershell
# Task 1: preserve the original 80-example results while repeating training
# and comparing both models on the mixed-intent challenge set.
.\.venv\Scripts\python.exe task1\run.py --results-dir results\task1_recheck --output-dir task1\output\recheck

# Task 2: downloads/uses the RFC and embedding model, evaluates recall@5
# for both chunk sizes, and writes the 15-question evidence table.
.\.venv\Scripts\python.exe task2\rag.py --evaluate

# Task 3: downloads the pinned llama.cpp build and both GGUF files.
.\task3\download_models.ps1
.\.venv\Scripts\python.exe task3\benchmark.py --output results\task3_recheck_metrics.json

# Validate all gold passages and chunk-boundary coverage.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For one-off RAG questions after building the index:

```powershell
.\.venv\Scripts\python.exe task2\rag.py --question "What does Retry-After tell a user agent?"
```

The evaluator uses the same deterministic test questions and retrieval
pipeline as the benchmark. `task2/questions.json` gives the complete questions
and source passages. Original Task 3 outputs and measurements are in
`results/task3_metrics.json`; the report uses the repeat run in
`results/task3_recheck_metrics.json`, which also records model-file sizes.

## Reproducibility and interpretation

- All benchmark scripts record their model identifiers, settings, and
  measurements. Model binaries, adapters, the downloaded RFC text, and
  transient logs are ignored; source code, compact results, and the final
  report are tracked. The RAG script fetches RFC 9110 directly from the RFC
  Editor when it is absent; the document includes its own copyright notice.
- Training and generation are seeded, but GPU kernels and library versions
  can still cause small run-to-run variation.
- Retrieval relevance is defined in advance by the gold RFC passage in each
  question. A hit means at least one of the top-five chunks contains that
  passage, not merely that the same section heading appears.
- Task 3 checks llama.cpp's available devices before choosing GPU offload.
  Its memory figure is peak server-process RSS; GPU VRAM is not reported.
- The 7B training-memory calculation is a back-of-envelope estimate; actual
  activation memory depends strongly on sequence length, batch size, and
  checkpointing.
