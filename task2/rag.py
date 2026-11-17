"""Minimal dense RAG over RFC 9110 using Transformers and NumPy only."""

from __future__ import annotations

import argparse
import csv
import json
import re
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import requests
import torch
import transformers
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "task2" / "data" / "rfc9110.txt"
QUESTIONS = ROOT / "task2" / "questions.json"
RESULTS = ROOT / "results"
CORPUS_URL = "https://www.rfc-editor.org/rfc/rfc9110.txt"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_CONFIGS = ((120, 40), (240, 80))
HEADING = re.compile(r"^\s*((?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\.?)\s{2,}(.+?)\s*$")


def ensure_corpus() -> None:
    if CORPUS.exists():
        return
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        CORPUS_URL, headers={"User-Agent": "small-model-assignment/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        CORPUS.write_bytes(response.read())


def clean_paragraph(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_sections(text: str) -> dict[str, str]:
    starts = list(re.finditer(r"(?m)^\s*1\.\s+Introduction\s*$", text))
    if len(starts) < 2:
        raise ValueError("Could not locate the beginning of RFC 9110 body text")
    body = text[starts[1].start() :]
    sections: dict[str, list[str]] = {}
    current = "1"
    paragraph: list[str] = []

    def flush() -> None:
        nonlocal paragraph
        value = clean_paragraph(" ".join(paragraph))
        if value:
            sections.setdefault(current, []).append(value)
        paragraph = []

    for line in body.splitlines():
        match = HEADING.match(line)
        if match:
            flush()
            current = match.group(1).rstrip(".")
            sections.setdefault(current, [])
        elif not line.strip():
            flush()
        elif re.fullmatch(r"\s*(?:RFC 9110.*|[A-Z][a-z]+, et al\..*Standards Track.*|\d+)\s*", line):
            flush()
        else:
            paragraph.append(line.strip())
    flush()
    return {
        section: " ".join(paragraphs)
        for section, paragraphs in sections.items()
        if paragraphs
    }


def make_chunks(sections: dict[str, str], size: int, overlap: int) -> list[dict]:
    chunks = []
    step = size - overlap
    if step <= 0:
        raise ValueError("Chunk overlap must be smaller than chunk size")
    for section, text in sections.items():
        words = text.split()
        for start in range(0, len(words), step):
            excerpt = " ".join(words[start : start + size])
            if len(excerpt.split()) < 20:
                continue
            chunks.append(
                {
                    "section": section,
                    "start_word": start,
                    "text": excerpt,
                }
            )
            if start + size >= len(words):
                break
    return chunks


def embed_texts(tokenizer, model, texts: list[str], device) -> np.ndarray:
    vectors = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), 64):
            batch = tokenizer(
                texts[start : start + 64],
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            ).to(device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            vectors.append(pooled.cpu().numpy())
    return np.concatenate(vectors, axis=0).astype(np.float32, copy=False)


def load_encoder(device):
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(device)
    return tokenizer, model


def normalise(text: str) -> str:
    return clean_paragraph(text).casefold()


def run_evaluation(tokenizer, model, sections, device) -> dict:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    all_results = {}
    for size, overlap in CHUNK_CONFIGS:
        chunks = make_chunks(sections, size, overlap)
        matrix = embed_texts(tokenizer, model, [item["text"] for item in chunks], device)
        query_matrix = embed_texts(
            tokenizer, model, [item["question"] for item in questions], device
        )
        scores = query_matrix @ matrix.T
        per_question = []
        hits = 0
        for row, question in enumerate(questions):
            top_indices = np.argsort(-scores[row], kind="stable")[:5]
            gold = normalise(question["gold_passage"])
            found = any(gold in normalise(chunks[index]["text"]) for index in top_indices)
            hits += int(found)
            per_question.append(
                {
                    "id": question["id"],
                    "question": question["question"],
                    "gold_section": question["gold_section"],
                    "gold_passage": question["gold_passage"],
                    "hit_at_5": found,
                    "top5": [
                        {
                            "section": chunks[index]["section"],
                            "score": float(scores[row, index]),
                            "text": chunks[index]["text"],
                        }
                        for index in top_indices
                    ],
                }
            )
        all_results[str(size)] = {
            "chunk_words": size,
            "overlap_words": overlap,
            "chunk_count": len(chunks),
            "questions": len(questions),
            "hits_at_5": hits,
            "recall_at_5": hits / len(questions),
            "details": per_question,
        }
    return all_results


def ask_question(tokenizer, model, sections, device, question: str, size: int) -> list[dict]:
    chunks = make_chunks(sections, size, size // 4)
    matrix = embed_texts(tokenizer, model, [item["text"] for item in chunks], device)
    query = embed_texts(tokenizer, model, [question], device)[0]
    indices = np.argsort(-(matrix @ query), kind="stable")[:5]
    return [chunks[index] for index in indices]


def generate_answer(
    llama_server: Path, model_path: Path, question: str, evidence: list[dict]
) -> str:
    if not model_path.is_file():
        raise FileNotFoundError(f"Answer model not found: {model_path}")
    context = "\n\n".join(
        f"[RFC 9110 Section {row['section']}] {row['text']}" for row in evidence[:3]
    )
    port = get_free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [
        str(llama_server),
        "--model",
        str(model_path),
        "--ctx-size",
        "4096",
        "--n-gpu-layers",
        "0",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-warmup",
    ]
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    log.flush()
                    log.seek(0)
                    raise RuntimeError(
                        f"llama-server exited with code {process.returncode}: "
                        f"{log.read()[-2000:]}"
                    )
                try:
                    if requests.get(f"{base_url}/health", timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.5)
            else:
                raise TimeoutError("llama-server did not become ready within 180 seconds.")

            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": "qwen",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Answer using only the supplied RFC 9110 context. "
                                "If it does not contain the answer, say that the "
                                "context is insufficient. Cite section numbers.\n"
                                f"Context:\n{context}"
                            ),
                        },
                        {"role": "user", "content": question},
                    ],
                    "temperature": 0,
                    "max_tokens": 160,
                },
                timeout=180,
            )
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"].strip()
            if not answer:
                raise RuntimeError("llama-server returned an empty answer.")
            return answer
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def get_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--question")
    parser.add_argument("--chunk-words", type=int, default=120)
    parser.add_argument("--answer-model", type=Path)
    parser.add_argument("--llama-server", type=Path)
    args = parser.parse_args()
    if not args.evaluate and not args.question:
        parser.error("Choose --evaluate or provide --question.")
    ensure_corpus()
    sections = extract_sections(CORPUS.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_encoder(device)
    if args.evaluate:
        started = time.perf_counter()
        evaluation = run_evaluation(tokenizer, model, sections, device)
        RESULT_FILE = RESULTS / "task2_metrics.json"
        RESULTS.mkdir(parents=True, exist_ok=True)
        RESULT_FILE.write_text(
            json.dumps(
                {
                    "corpus": "RFC 9110: HTTP Semantics",
                    "corpus_url": CORPUS_URL,
                    "embedding_model": EMBEDDING_MODEL,
                    "numpy_version": np.__version__,
                    "torch_version": torch.__version__,
                    "transformers_version": transformers.__version__,
                    "device": str(device),
                    "question_count": len(json.loads(QUESTIONS.read_text(encoding="utf-8"))),
                    "elapsed_seconds": time.perf_counter() - started,
                    "results": evaluation,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        rows = []
        for size, result in evaluation.items():
            for question in result["details"]:
                rows.append(
                    {
                        "chunk_words": size,
                        "id": question["id"],
                        "question": question["question"],
                        "gold_section": question["gold_section"],
                        "gold_passage": question["gold_passage"],
                        "hit_at_5": question["hit_at_5"],
                        "retrieved_sections": ";".join(
                            str(item["section"]) for item in question["top5"]
                        ),
                        "top1_excerpt": question["top5"][0]["text"],
                    }
                )
        with (RESULTS / "task2_questions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(
            json.dumps(
                {
                    size: {
                        "chunk_count": result["chunk_count"],
                        "hits_at_5": result["hits_at_5"],
                        "recall_at_5": result["recall_at_5"],
                    }
                    for size, result in evaluation.items()
                },
                indent=2,
            )
        )
        return

    evidence = ask_question(tokenizer, model, sections, device, args.question, args.chunk_words)
    if args.answer_model:
        if not args.llama_server:
            parser.error("--answer-model requires --llama-server.")
        answer = generate_answer(
            args.llama_server, args.answer_model, args.question, evidence
        )
        print(answer)
    else:
        for rank, item in enumerate(evidence, start=1):
            print(f"[{rank}] RFC 9110 Section {item['section']}\n{item['text']}\n")


if __name__ == "__main__":
    main()
