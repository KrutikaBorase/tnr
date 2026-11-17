# Task 3 — Local inference and quantization

The benchmark runs the same ten authored prompts against Qwen2.5-1.5B-Instruct
in Q4_K_M and Q8_0 GGUF formats, using the pinned llama.cpp b12477 Windows
build. It checks `llama-cli --list-devices` and uses GPU offload only when the
build enumerates an accelerator. On the machine used for these results,
llama.cpp reported no available device, so the measured run used CPU and
explicitly set zero GPU layers. It warms each model with one excluded prompt,
then records outputs, completion-token wall throughput, server-process RSS,
and whether an accelerator was available.

```powershell
.\task3\download_models.ps1
.\.venv\Scripts\python.exe task3\benchmark.py
```

Models and executables are downloaded locally and excluded from git. The
prompt/output pairs and measurements are saved in
`results/task3_metrics.json`; qualitative observations are summarized in the
final PDF report.

The KV cache stores prior attention keys and values so each next token does
not require recomputing the full prefix. Its size increases with active
sequence count and context length. Quantization reduces parameter storage
from (for example) 16/32-bit values to packed low-bit values plus scale
metadata; this saves memory and often bandwidth at a possible quality cost.

For a 24 GB serving plan, the report assumes a 7B 4-bit GQA model, 20
sequences capped at 2,048 tokens, FP16 KV, and continuous/paged batching.
The calculation is an approximate capacity budget, not a latency/SLA claim.
