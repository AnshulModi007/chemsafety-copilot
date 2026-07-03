"""Evaluation harness: retrieval metrics (Precision@k, Recall@k, MRR) computed
directly against the golden set, and generation metrics (Faithfulness, Answer
Relevancy) via RAGAS using the local Ollama model + bge-base embeddings as the
judge -- no paid API involved.

Run repeatedly as the retrieval/generation pipeline changes (hybrid search,
reranking, CRAG, ...) to build a before/after metrics table rather than a
single snapshot -- each run is stored under its own key in
baseline_metrics.json and the README table is regenerated from all of them.
"""
import argparse
import json
import sys
import types
from pathlib import Path

# ragas imports langchain_community.chat_models.vertexai purely for an
# isinstance() check it never exercises with a local Ollama LLM; the real
# module needs the (heavy, unused) langchain-google-vertexai package. Stub
# it out rather than installing Google Cloud SDK deps for nothing.
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _stub

from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402
from ragas import evaluate, EvaluationDataset, SingleTurnSample  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import answer_relevancy, faithfulness  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    GOLDEN_QA_PATH, BASELINE_METRICS_PATH, OLLAMA_MODEL, OLLAMA_HOST,
    EMBEDDING_MODEL, TOP_K, PROJECT_ROOT,
)
from src.retrieval.retriever import dense_retrieve, hybrid_retrieve, reranked_retrieve  # noqa: E402
from src.generation.generate import generate, generate_from_hits, INSUFFICIENT_RETRIEVAL_MESSAGE  # noqa: E402
from src.generation.crag import retrieve_with_crag  # noqa: E402

RUN_LABELS = {
    "dense": "Week 1 - Dense only",
    "hybrid": "Week 2 - Hybrid (dense + BM25, RRF)",
    "reranked": "Week 2 - Hybrid + Reranker",
    "crag": "Week 2 - + CRAG",
}
RETRIEVERS = {"dense": dense_retrieve, "hybrid": hybrid_retrieve, "reranked": reranked_retrieve}


def retrieval_metrics(golden_set: list[dict], top_k: int, retrieve_fn) -> dict:
    hits, reciprocal_ranks = [], []
    for item in golden_set:
        results = retrieve_fn(item["question"], top_k=top_k)
        retrieved_ids = [r["chunk_id"] for r in results]
        relevant = set(item["source_chunk_ids"])
        rank = next((i + 1 for i, cid in enumerate(retrieved_ids) if cid in relevant), None)
        hits.append(1 if rank else 0)
        reciprocal_ranks.append(1 / rank if rank else 0)

    n = len(golden_set)
    return {
        f"precision_at_{top_k}": sum(hits) / n / top_k,
        f"recall_at_{top_k}": sum(hits) / n,
        "mrr": sum(reciprocal_ranks) / n,
    }


def build_generation_dataset(golden_set: list[dict], retrieve_fn) -> EvaluationDataset:
    samples = []
    for item in golden_set:
        print(f"  generating answer for {item['id']}...")
        result = generate(item["question"])  # generation always uses reranked_retrieve internally
        hits = retrieve_fn(item["question"], top_k=TOP_K)
        samples.append(SingleTurnSample(
            user_input=item["question"],
            response=result["answer"],
            retrieved_contexts=[h["text"] for h in hits],
        ))
    return EvaluationDataset(samples=samples)


def run_crag_eval(golden_set: list[dict], top_k: int) -> tuple[dict, EvaluationDataset]:
    """CRAG's grading/retry changes both what's retrieved (a retry can find a
    better chunk) and what's generated from (only correct/ambiguous chunks),
    so it needs its own pass rather than reusing retrieval_metrics() +
    build_generation_dataset() -- doing so also avoids running the expensive
    multi-call CRAG pipeline twice per question.
    """
    hits, reciprocal_ranks, samples = [], [], []
    for item in golden_set:
        print(f"  CRAG pipeline for {item['id']}...")
        crag_result = retrieve_with_crag(item["question"], top_k=top_k)

        retrieved_ids = [c["chunk_id"] for c in crag_result["chunks"]]
        relevant = set(item["source_chunk_ids"])
        rank = next((i + 1 for i, cid in enumerate(retrieved_ids) if cid in relevant), None)
        hits.append(1 if rank else 0)
        reciprocal_ranks.append(1 / rank if rank else 0)

        if crag_result["insufficient"]:
            answer_text = INSUFFICIENT_RETRIEVAL_MESSAGE
            used_hits = []
        else:
            answer_text = generate_from_hits(item["question"], crag_result["used_chunks"])["answer"]
            used_hits = crag_result["used_chunks"]

        samples.append(SingleTurnSample(
            user_input=item["question"],
            response=answer_text,
            retrieved_contexts=[h["text"] for h in used_hits] or ["(no chunks passed relevance grading)"],
        ))

    n = len(golden_set)
    retrieval_scores = {
        f"precision_at_{top_k}": sum(hits) / n / top_k,
        f"recall_at_{top_k}": sum(hits) / n,
        "mrr": sum(reciprocal_ranks) / n,
    }
    return retrieval_scores, EvaluationDataset(samples=samples)


def _load_all_runs() -> dict:
    if BASELINE_METRICS_PATH.exists():
        data = json.loads(BASELINE_METRICS_PATH.read_text())
        # Migrate the Week 1 flat-file format (no run-name nesting) to the
        # multi-run format used from Week 2 onward.
        if "precision_at_5" in data:
            return {"dense": data}
        return data
    return {}


def _update_readme(all_runs: dict) -> None:
    readme_path = PROJECT_ROOT / "README.md"
    text = readme_path.read_text()

    run_keys = [k for k in ("dense", "hybrid", "reranked", "crag") if k in all_runs]
    header = "| Metric | " + " | ".join(RUN_LABELS[k] for k in run_keys) + " |"
    sep = "|---|" + "|".join("---" for _ in run_keys) + "|"

    def row(label: str, key: str) -> str:
        return "| " + label + " | " + " | ".join(f"{all_runs[k][key]:.3f}" for k in run_keys) + " |"

    table = "\n".join([
        header,
        sep,
        row(f"Recall@{TOP_K}", f"recall_at_{TOP_K}"),
        row("MRR", "mrr"),
        row("Faithfulness (RAGAS)", "faithfulness"),
        row("Answer Relevance (RAGAS)", "answer_relevancy"),
    ]) + "\n"
    note = (
        f"\n_Precision@{TOP_K} omitted from the table above: with exactly one "
        f"relevant chunk per golden question it's mathematically capped at "
        f"1/{TOP_K}, so it doesn't carry independent signal beyond Recall@{TOP_K}._\n"
    )

    marker = "## Baseline Metrics (Week 1)"
    new_heading = "## Metrics: Before / After"
    idx = text.find(marker)
    if idx == -1:
        idx = text.find(new_heading)
    if idx == -1:
        text += f"\n{new_heading}\n\n{table}{note}"
    else:
        end = text.find("\n## ", idx + 1)
        end = end if end != -1 else len(text)
        text = f"{text[:idx]}{new_heading}\n\n{table}{note}\n{text[end:]}"
    readme_path.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(RUN_LABELS), default="hybrid")
    args = parser.parse_args()

    golden_set = json.loads(GOLDEN_QA_PATH.read_text())
    print(f"Loaded {len(golden_set)} golden QA pairs; mode={args.mode}")

    if args.mode == "crag":
        print("\nRunning CRAG pipeline (retrieval + grading + generation) per question...")
        retrieval_scores, dataset = run_crag_eval(golden_set, TOP_K)
        print(retrieval_scores)
    else:
        retrieve_fn = RETRIEVERS[args.mode]
        print("\nComputing retrieval metrics...")
        retrieval_scores = retrieval_metrics(golden_set, TOP_K, retrieve_fn)
        print(retrieval_scores)

        print("\nGenerating answers for RAGAS evaluation...")
        dataset = build_generation_dataset(golden_set, retrieve_fn)

    judge_llm = LangchainLLMWrapper(ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, num_ctx=8192))
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

    print("\nRunning RAGAS faithfulness + answer_relevancy (local LLM judge, may take a while)...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    ragas_scores = result.to_pandas()[["faithfulness", "answer_relevancy"]].mean().to_dict()

    run_metrics = {**retrieval_scores, **ragas_scores}
    print(f"\nFinal metrics ({args.mode}):", json.dumps(run_metrics, indent=2))

    all_runs = _load_all_runs()
    all_runs[args.mode] = run_metrics
    BASELINE_METRICS_PATH.write_text(json.dumps(all_runs, indent=2))
    _update_readme(all_runs)


if __name__ == "__main__":
    main()
