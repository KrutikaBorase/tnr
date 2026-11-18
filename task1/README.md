# Task 1 — LoRA fine-tuning

`run.py` fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` to classify a support message
as `billing`, `technical_support`, `account_access`, or `order_status` and emit
only `{"intent":"..."}`. It creates 240 deterministic training examples and
80 separately templated held-out messages. It measures both the base and
fine-tuned model on the identical test set and separately saves deliberately
mixed-intent challenge cases (not included in the headline score).
Both models use the same chat prompt, instruction, messages and greedy decoding.
The held-out wording differs from training templates, but remains in the same
small, synthetic task; it is not evidence of production generalization.

The separate recheck in `results/task1_recheck/` compares base and tuned models
on the 20 mixed-intent challenges after retraining. Some examples can reasonably
have more than one label, so this score measures agreement with the selected
single-label references, not definitive correctness. The original held-out
run and its failure examples remain under `results/`.

Run from the root with:

```powershell
.\.venv\Scripts\python.exe task1\run.py

# Preserve the original results when repeating the harder comparison:
.\.venv\Scripts\python.exe task1\run.py --results-dir results\task1_recheck --output-dir task1\output\recheck
```

The adapter is written to ignored `task1/output/adapter/`; comparable metrics
and raw predictions go to `results/task1_metrics.json`,
`results/task1_predictions.csv`, and `results/task1_challenges.json`.

The run uses LoRA rank 8, alpha 16, q_proj/v_proj targets, dropout 0.05,
learning rate 2e-4, three epochs, batch size 4, and gradient accumulation 2.
The base weights stay frozen. A small rank limits the adapter and optimizer
state; q/v projections provide attention adaptation; three epochs and dropout
regularize a small synthetic dataset. These are pragmatic settings for a
small 0.5B checkpoint, not a universal optimum.

The report's 7B memory arithmetic assumes 16 bytes/parameter for mixed-
precision AdamW full tuning. LoRA retains the 16-bit base but only updates
adapter parameters; QLoRA stores the base in 4-bit and still pays for
activations and trainable adapters. Sequence length, batch size, quantization
metadata, and checkpointing change the actual peak.
