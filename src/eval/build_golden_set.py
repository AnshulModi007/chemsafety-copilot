"""Draft candidate golden-set QA pairs from ingested CSB report chunks via the local LLM.

Selects a diverse sample of chunks (aiming for one per report) and asks the
model to write a question that chunk alone answers, plus a reference answer.
Domain accuracy matters here, so these drafts are meant to be reviewed/edited
by a human (a chemical engineer) before being treated as ground truth.
"""
import json
import random
import sys
from pathlib import Path

import ollama
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR, GOLDEN_QA_PATH, OLLAMA_MODEL  # noqa: E402

TARGET_PAIRS = 22
SKIP_SECTIONS = {"front matter"}
SKIP_SECTION_SUBSTRINGS = ("reference",)
MIN_WORDS = 80

SYSTEM_PROMPT = """You are helping build an evaluation set for a RAG system over U.S. Chemical Safety \
Board (CSB) investigation reports. Given a single excerpt, write ONE factual question that can be \
answered using ONLY this excerpt, plus a concise reference answer drawn only from the excerpt. \
Prefer questions about root causes, contributing factors, timeline, or findings when the excerpt \
supports it. Do not ask about information not present in the excerpt.

Respond with ONLY a JSON object: {"question": "<question>", "answer": "<answer>"}
"""


class QAPair(BaseModel):
    question: str
    answer: str


def pick_chunks(chunks: list[dict], target: int) -> list[dict]:
    eligible = [
        c for c in chunks
        if c["section"].strip().lower() not in SKIP_SECTIONS
        and not any(s in c["section"].strip().lower() for s in SKIP_SECTION_SUBSTRINGS)
        and len(c["text"].split()) >= MIN_WORDS
    ]
    by_report: dict[str, list[dict]] = {}
    for c in eligible:
        by_report.setdefault(c["report_id"], []).append(c)

    random.seed(42)
    report_ids = list(by_report.keys())
    random.shuffle(report_ids)

    picked, picked_ids = [], set()
    for report_id in report_ids:
        if len(picked) >= target:
            break
        c = random.choice(by_report[report_id])
        picked.append(c)
        picked_ids.add(c["chunk_id"])

    remaining_pool = [c for c in eligible if c["chunk_id"] not in picked_ids]
    random.shuffle(remaining_pool)
    while len(picked) < target and remaining_pool:
        picked.append(remaining_pool.pop())

    return picked[:target]


def draft_qa(chunk: dict) -> QAPair | None:
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Excerpt:\n\n{chunk['text']}"},
        ],
        format=QAPair.model_json_schema(),
        options={"num_ctx": 4096},
    )
    try:
        return QAPair.model_validate_json(response["message"]["content"])
    except Exception as e:
        print(f"  WARNING: failed to parse QA for {chunk['chunk_id']}: {e}")
        return None


def main():
    chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text())
    selected = pick_chunks(chunks, TARGET_PAIRS)
    print(f"Drafting QA pairs from {len(selected)} chunks across "
          f"{len(set(c['report_id'] for c in selected))} reports...")

    golden_set = []
    for i, chunk in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {chunk['report_id']} :: {chunk['section']}")
        qa = draft_qa(chunk)
        if qa is None:
            continue
        golden_set.append({
            "id": f"q{i:03d}",
            "question": qa.question,
            "reference_answer": qa.answer,
            "source_report_id": chunk["report_id"],
            "source_chunk_ids": [chunk["chunk_id"]],
            "section": chunk["section"],
        })

    GOLDEN_QA_PATH.write_text(json.dumps(golden_set, indent=2))
    print(f"\nDrafted {len(golden_set)} QA pairs -> {GOLDEN_QA_PATH}")


if __name__ == "__main__":
    main()
