# Task 2 — NumPy dense RAG

`rag.py` fetches RFC 9110, **HTTP Semantics**, from
<https://www.rfc-editor.org/rfc/rfc9110.txt> when needed. The text is
copyrighted RFC material, retains the RFC's notice locally, and is excluded
from git; download it by running the script. The official RFC PDF has 194
pages (verified by counting its page objects), exceeding the 30-page minimum.
Fifteen hand-written questions,
their exact source passages, and section ids are in `questions.json`.

```powershell
.\.venv\Scripts\python.exe task2\rag.py --evaluate
.\.venv\Scripts\python.exe task2\rag.py --question "How does Retry-After work?"
```

The implementation normalizes mean-pooled `sentence-transformers/all-MiniLM-L6-v2`
embeddings and ranks with NumPy dot products. It evaluates 120-word chunks
(40-word overlap) and 240-word chunks (80-word overlap). A top-five hit must
contain the complete gold passage. It uses no RAG framework, external vector
database, or LLM for retrieval. For answer generation, `--answer-model` and
`--llama-server` can use the local llama.cpp model from Task 3; by default the
script returns the five evidence chunks. For example:

```powershell
$server = (Get-ChildItem .\task3\bin -Recurse -Filter llama-server.exe |
  Select-Object -First 1).FullName
.\.venv\Scripts\python.exe task2\rag.py --question "How does Retry-After work?" `
  --answer-model .\task3\models\qwen2.5-1.5b-instruct-q4_k_m.gguf --llama-server $server
```

BM25 is complementary sparse lexical ranking: it favors exact and rare term
matches, including names and identifiers, that dense embeddings may blur.
A cross-encoder re-ranker scores the query together with each retrieved
candidate and reorders a small candidate list; it adds latency and cannot
recover evidence not retrieved in the first stage.

Results and the evidence table are saved to `results/task2_metrics.json` and
`results/task2_questions.csv`.
