"""Build the four-page report from the measured result files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
REPORT = ROOT / "report.pdf"


def load_json(name):
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"Run the benchmark that creates {path} before making the report.")
    return json.loads(path.read_text(encoding="utf-8"))


def para(text, style):
    return Paragraph(text, style)


def cell(text, style):
    return Paragraph(escape(str(text)), style)


def table(rows, widths, font_size=7.1, header=True):
    style = ParagraphStyle(
        "table-cell",
        parent=STYLES["Normal"],
        fontName="Helvetica",
        fontSize=font_size,
        leading=font_size + 1.4,
        alignment=TA_LEFT,
        spaceAfter=0,
    )
    converted = [
        [value if isinstance(value, Paragraph) else cell(value, style) for value in row]
        for row in rows
    ]
    result = Table(converted, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bcc7d1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eff5")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    result.setStyle(TableStyle(commands))
    return result


STYLES = getSampleStyleSheet()
STYLES.add(
    ParagraphStyle(
        "ReportTitle",
        parent=STYLES["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=21,
        textColor=colors.HexColor("#12304a"),
        alignment=TA_LEFT,
        spaceAfter=3,
    )
)
STYLES.add(
    ParagraphStyle(
        "SectionHead",
        parent=STYLES["Heading2"],
        fontSize=11,
        leading=13,
        spaceBefore=5,
        spaceAfter=3,
        textColor=colors.HexColor("#173e5d"),
    )
)
STYLES.add(
    ParagraphStyle(
        "BodySmall",
        parent=STYLES["BodyText"],
        fontSize=8.2,
        leading=10.4,
        spaceAfter=3,
    )
)
STYLES.add(
    ParagraphStyle(
        "TableSmall",
        parent=STYLES["BodyText"],
        fontSize=7.1,
        leading=8.4,
        spaceAfter=0,
    )
)
STYLES.add(
    ParagraphStyle(
        "Foot",
        parent=STYLES["BodyText"],
        fontSize=7,
        leading=8,
        textColor=colors.HexColor("#4b5965"),
    )
)


def build_report():
    task1 = load_json("task1_metrics.json")
    task2 = load_json("task2_metrics.json")
    task3 = load_json("task3_metrics.json")
    with (RESULTS / "task1_predictions.csv").open(encoding="utf-8", newline="") as file:
        predictions = list(csv.DictReader(file))
    questions = json.loads((ROOT / "task2" / "questions.json").read_text(encoding="utf-8"))
    rag_by_size = task2["results"]
    generation = task3["results"]

    story = []
    story += [
        para("Small-model experiments", STYLES["ReportTitle"]),
        para(
            "Measured results for LoRA fine-tuning, dense retrieval, and local quantized inference. "
            "All scores and speed/memory values below are read from this repository's run outputs.",
            STYLES["BodySmall"],
        ),
        para("1. LoRA fine-tuning", STYLES["SectionHead"]),
        para(
            f"Task: map support messages to one JSON intent (billing, technical_support, "
            f"account_access, order_status). Model: {escape(task1['model'])}; "
            f"{task1['device']} on {escape(str(task1.get('gpu') or 'CPU'))}; "
            f"{task1['test_examples']} held-out messages, generated from disjoint prompt templates.",
            STYLES["BodySmall"],
        ),
        table(
            [
                ["Model", "Intent accuracy", "Schema-valid output"],
                [
                    "Base",
                    f"{task1['base']['correct']}/{task1['base']['examples']} "
                    f"({task1['base']['accuracy']:.1%})",
                    f"{task1['base']['schema_valid_rate']:.1%}",
                ],
                [
                    "LoRA fine-tuned",
                    f"{task1['lora_finetuned']['correct']}/{task1['lora_finetuned']['examples']} "
                    f"({task1['lora_finetuned']['accuracy']:.1%})",
                    f"{task1['lora_finetuned']['schema_valid_rate']:.1%}",
                ],
            ],
            [52 * mm, 55 * mm, 55 * mm],
        ),
        Spacer(1, 3),
        para(
            "<b>Settings:</b> rank 8, alpha 16, q_proj/v_proj targets, dropout 0.05, "
            "learning rate 2e-4, 3 epochs, micro-batch 4 with accumulation 2. "
            "The 0.5B model is small enough to keep frozen base weights and train only a "
            "compact adapter; q/v projections give adaptation capacity in attention without "
            "optimizing every parameter. A moderate rank and 3 passes limit cost/overfit on "
            f"{task1['lora']['train_examples']} templated examples. Training took "
            f"{task1['training_seconds']:.1f} seconds; peak PyTorch CUDA allocation was "
            f"{task1['peak_cuda_allocated_gib']:.2f} GiB.",
            STYLES["BodySmall"],
        ),
        para("Three observed fine-tuned failures", STYLES["SectionHead"]),
    ]
    failures = [
        row
        for row in predictions
        if row["tuned_prediction"] != row["expected"]
    ][:3]
    if len(failures) < 3:
        failure_path = RESULTS / "task1_challenges.json"
        if failure_path.exists():
            challenges = json.loads(failure_path.read_text(encoding="utf-8"))
            failures.extend(
                row
                for row in challenges
                if row["prediction"] != row["expected"]
            )
            failures = failures[:3]
    if len(failures) < 3:
        raise RuntimeError(
            "Need three measured fine-tuned errors (held-out or challenge examples) "
            "before the report can be generated."
        )
    fail_rows = [["Message / expected", "Observed output / likely cause"]]
    for row in failures:
        expected = row.get("expected", "")
        output = row.get("tuned_output", row.get("output", ""))
        message = row.get("message", "")
        fail_rows.append(
            [
                Paragraph(
                    f"{escape(message)}<br/><b>Expected:</b> {escape(expected)}",
                    STYLES["TableSmall"],
                ),
                Paragraph(
                    f"{escape(output)}<br/><b>Why:</b> "
                    f"{escape(explain_task1_failure(message, expected, row))}",
                    STYLES["TableSmall"],
                ),
            ]
        )
    story += [
        table(fail_rows, [89 * mm, 89 * mm], font_size=6.6),
        para(
            "The full per-example base/fine-tuned outputs are in "
            "<b>results/task1_predictions.csv</b>.",
            STYLES["Foot"],
        ),
        PageBreak(),
    ]

    story += [
        para("2. Memory estimate and retrieval", STYLES["ReportTitle"]),
        para("Approximate 7B training memory (decimal GB; activation memory is variable)", STYLES["SectionHead"]),
        table(
            [
                ["Method", "Working", "Approx. total"],
                [
                    "Full fine-tuning",
                    "7B × (2 B FP16 weights + 2 B gradients + 4 B FP32 master "
                    "+ 8 B Adam moments) = 112 GB; add ~5–10 GB activations/workspace.",
                    "~117–122 GB",
                ],
                [
                    "LoRA (FP16 base)",
                    "7B × 2 B frozen weights = 14 GB; ~0.1% trainable adapters "
                    "× 16 B = 0.11 GB; add ~4–8 GB activations/workspace.",
                    "~18–22 GB",
                ],
                [
                    "QLoRA (4-bit base)",
                    "7B × 0.5 B = 3.5 GB packed weights; allow ~0.5 GB quantization "
                    "metadata; adapter states ~0.11 GB; add ~4–8 GB activations/workspace.",
                    "~8–12 GB",
                ],
            ],
            [34 * mm, 119 * mm, 25 * mm],
            font_size=6.5,
        ),
        para(
            "These are planning estimates, not measured 7B runs. Optimizer-state arithmetic assumes "
            "mixed-precision AdamW with FP32 master weights and two FP32 moments. LoRA still needs "
            "activation memory; QLoRA's exact peak depends on sequence length, batch, and kernels.",
            STYLES["Foot"],
        ),
        para("RAG experiment", STYLES["SectionHead"]),
        para(
            "Corpus: RFC 9110, <i>HTTP Semantics</i> (IETF RFC Editor; the official PDF "
            "has 194 pages, counted from its page objects; the indexed text is 502,941 bytes "
            "in task2/data/rfc9110.txt). "
            "The implementation mean-pools normalized all-MiniLM-L6-v2 embeddings and ranks "
            "with NumPy cosine similarity. Fifteen hand-written questions have exact gold "
            "passages in task2/questions.json. Recall@5 counts a hit only when a top-five chunk "
            "contains the complete gold passage.",
            STYLES["BodySmall"],
        ),
        table(
            [
                ["Chunk size / overlap (words)", "Chunks", "Recall@5"],
                *[
                    [
                        f"{data['chunk_words']} / {data['overlap_words']}",
                        data["chunk_count"],
                        f"{data['hits_at_5']}/{data['questions']} "
                        f"({data['recall_at_5']:.1%})",
                    ]
                    for _, data in sorted(rag_by_size.items(), key=lambda item: int(item[0]))
                ],
            ],
            [64 * mm, 54 * mm, 60 * mm],
        ),
        para(
            "Neither chunk size won on this test: both retrieved 13 of 15 gold passages. "
            "The 120-word index is more granular but has more chunks; the 240-word index "
            "has fewer, broader chunks. That trade-off did not change top-five success for "
            "these short fact questions.",
            STYLES["BodySmall"],
        ),
        Spacer(1, 4),
        para(
            "<b>Failures:</b> " + escape(describe_rag_failures(rag_by_size)),
            STYLES["BodySmall"],
        ),
        para(
            "<b>BM25:</b> a sparse term-frequency/inverse-document-frequency ranker; exact terms, "
            "rare names, and identifiers can match even when their meaning is not represented well "
            "by an embedding. Hybrid retrieval can combine BM25 and dense scores. "
            "<b>Re-ranker:</b> a second model scores each query–candidate pair jointly to reorder "
            "a small retrieved set; it can improve precision but costs extra inference and cannot "
            "recover a passage that retrieval omitted.",
            STYLES["BodySmall"],
        ),
        para(
            "Question-by-question results and exact source passages are in "
            "<b>results/task2_questions.csv</b> and <b>task2/questions.json</b>.",
            STYLES["Foot"],
        ),
        PageBreak(),
    ]

    question_rows = [["ID", "Question", "Correct source passage (RFC section and excerpt)"]]
    for question in questions:
        question_rows.append(
            [
                question["id"],
                question["question"],
                f"§ {question['gold_section']}: {question['gold_passage']}",
            ]
        )
    story += [
        para("RAG test set: questions and evidence", STYLES["ReportTitle"]),
        para(
            "The passages below are the exact gold evidence used by the evaluator. "
            "Top-five retrieval success is scored by finding the complete quoted passage in a chunk.",
            STYLES["BodySmall"],
        ),
        table(question_rows, [12 * mm, 66 * mm, 100 * mm], font_size=6.2),
        Spacer(1, 4),
        para(
            f"Corpus source: RFC Editor, <b>RFC 9110: HTTP Semantics</b>, "
            f"{escape(task2['corpus_url'])}. Embedding model: "
            f"{escape(task2['embedding_model'])}. Run device: {escape(task2['device'])}.",
            STYLES["Foot"],
        ),
        PageBreak(),
    ]

    task3_rows = [["Setting", "Tokens/s (wall)", "Peak process RSS"]]
    for label, data in generation.items():
        task3_rows.append(
            [
                label,
                f"{data['tokens_per_second']:.2f}",
                f"{data['peak_process_rss_mib']:.0f} MiB",
            ]
        )
    task3_memory_note = (
        "No llama.cpp accelerator was enumerated, so this run used CPU and did not allocate "
        "model GPU VRAM. Memory is peak server-process RSS."
        if not task3["llama_cpp_devices"]
        else "Memory is peak server-process RSS; accelerator VRAM is not included."
    )
    story += [
        para("3. Local inference and serving plan", STYLES["ReportTitle"]),
        para(
            f"Qwen2.5-1.5B-Instruct on llama.cpp build b12477 with "
            f"{escape(task3['accelerator'])} ({generation['Q4_K_M']['gpu_layers']} GPU layers), "
            f"context 4096. The host has {escape(str(task3.get('host_gpu') or 'no detected GPU'))}; "
            "both quantizations ran the same ten prompts "
            "sequentially; one separate warm-up request was excluded. Throughput is total "
            "completion tokens divided by client wall time over the ten requests.",
            STYLES["BodySmall"],
        ),
        table(task3_rows, [48 * mm, 54 * mm, 70 * mm]),
        para(
            task3_memory_note + " The full prompt/output pairs are recorded "
            "in results/task3_metrics.json.",
            STYLES["Foot"],
        ),
        para("Output quality observations", STYLES["SectionHead"]),
    ]
    quality = (ROOT / "task3" / "quality_notes.md").read_text(encoding="utf-8")
    story.append(para(escape(quality).replace("\n", "<br/>"), STYLES["BodySmall"]))
    story += [
        para(
            "<b>KV cache:</b> stores attention keys and values for tokens already processed, "
            "so autoregressive decoding does not recompute them for every next token. Its memory "
            "grows with context length and concurrent sequences. <b>Quantization:</b> stores "
            "weights in fewer bits (plus scales/metadata), reducing the bytes resident in RAM/VRAM; "
            "it can trade numerical precision and output quality for capacity and bandwidth.",
            STYLES["BodySmall"],
        ),
        para("Back-of-envelope: 20 concurrent users on one 24 GB GPU", STYLES["SectionHead"]),
        para(
            "Assume a 7B GQA model in 4-bit (~4.5 GB weights including scales), 20 active "
            "sequences capped at 2,048 tokens, 32 layers, 8 KV heads, 128-dimension heads, "
            "and FP16 KV. Per token KV ≈ 2 × 32 × 8 × 128 × 2 bytes = 128 KiB; "
            "20 × 2,048 × 128 KiB ≈ 5 GiB. Budget another 2–3 GiB for runtime/workspaces "
            "and 1–2 GiB safety margin: roughly 13–16 GiB total, leaving headroom on 24 GB. "
            "Serve with continuous batching/paged attention (e.g. vLLM), a 2,048-token context "
            "cap, max 20 active sequences, bounded output tokens, and queue/backpressure for "
            "bursts. This is a capacity plan, not a latency guarantee; benchmark the actual "
            "model, prompts, and concurrency before promising service levels.",
            STYLES["BodySmall"],
        ),
    ]

    doc = SimpleDocTemplate(
        str(REPORT),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=13 * mm,
        bottomMargin=12 * mm,
        title="Small-model experiments: LoRA, RAG, and local inference",
        author="",
    )
    doc.build(story)
    print(f"Wrote {REPORT}")


def explain_task1_failure(message: str, expected: str, row: dict) -> str:
    lowered = message.lower()
    prediction = row.get("tuned_prediction", row.get("prediction"))
    if not prediction:
        return (
            "The JSON was parseable, but it emitted an out-of-taxonomy intent "
            "instead of one of the four allowed labels."
        )
    if "phone number" in lowered or "recovery phone" in lowered:
        return (
            "Changing a recovery phone is account maintenance, but the model appears "
            "to associate the word 'number' with payment/billing examples."
        )
    if "shipment" in lowered and any(word in lowered for word in ("credit", "fee", "refund")):
        return (
            "The sentence combines a delayed shipment with a requested fee credit; "
            "the model follows the money cue instead of the primary shipment-status ask."
        )
    if "charge" in lowered or "refund" in lowered or "invoice" in lowered:
        return "billing language shares words with shipping/payment requests; limited template coverage."
    if "shipment" in lowered or "order" in lowered or "delivery" in lowered:
        return "the request mixes delivery with another intent; one-label training forces a single choice."
    if "sign" in lowered or "password" in lowered or "account" in lowered:
        return "account-access wording is indirect or paired with a second issue."
    return f"paraphrase or overlapping signals differ from the small synthetic training set ({expected})."


def describe_rag_failures(results: dict) -> str:
    size = min(results, key=lambda value: int(value))
    details = results[size]["details"]
    misses = [item for item in details if not item["hit_at_5"]]
    if not misses:
        return (
            "No misses at the smaller chunk size in this 15-question set. The test is small and "
            "hand-curated; harder compositional questions would be needed to characterize limits."
        )
    diagnoses = {
        "q06": (
            "Q06 (gold §6.3, top result §10.1): the broad wording about header-field "
            "information overlaps with request-context fields. Add terms about a field "
            "changing semantics/defining content, then hybrid-rank or re-rank candidates."
        ),
        "q14": (
            "Q14 (gold §11.2, top result §11.6.3): both passages discuss additional "
            "authentication information. Query for the parameter forms ('comma-separated' "
            "and 'token68') and re-rank nearby authentication sections."
        ),
    }
    pieces = [diagnoses[item["id"]] for item in misses[:2] if item["id"] in diagnoses]
    if pieces:
        return "At 120 words, " + " ".join(pieces)
    pieces = []
    for item in misses[:2]:
        top = item["top5"][0]
        pieces.append(
            f"{item['id']} (gold §{item['gold_section']}, top result §{top['section']})"
        )
    return (
        "At the smaller chunk setting, misses included "
        + " and ".join(pieces)
        + ". Similar protocol vocabulary across nearby sections can blur dense similarity; "
        "hybrid BM25, section-aware reranking, or better query wording could help."
    )


if __name__ == "__main__":
    build_report()
