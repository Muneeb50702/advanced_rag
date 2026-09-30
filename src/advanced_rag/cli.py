"""Command-line interface for local, cited document questions and retrieval evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import URLError

from .core import Retriever, extractive_answer, load_chunks, ollama_answer


def evaluate(retriever: Retriever, cases: list[dict], top_k: int) -> dict:
    if not cases:
        raise ValueError("evaluation file contains no cases")
    details = []
    for case in cases:
        hits = retriever.search(case["question"], top_k)
        sources = [hit.source for hit in hits]
        expected = case["source"]
        rank = sources.index(expected) + 1 if expected in sources else None
        details.append({"question": case["question"], "expected_source": expected,
                        "retrieved_sources": sources, "rank": rank})
    return {"count": len(details), "top_k": top_k,
            "recall_at_k": sum(item["rank"] is not None for item in details) / len(details),
            "mrr_at_k": sum(1 / item["rank"] for item in details if item["rank"]) / len(details),
            "cases": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=Path("examples/operations"))
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="retrieve evidence and answer")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=3)
    ask.add_argument("--ollama-model", help="optional local Ollama model for generative answers")
    ev = sub.add_parser("eval", help="measure retrieval on a JSONL question set")
    ev.add_argument("--cases", type=Path, default=Path("eval/questions.jsonl"))
    ev.add_argument("--top-k", type=int, default=3)
    ev.add_argument("--output", type=Path)
    args = parser.parse_args()
    retriever = Retriever(load_chunks(args.docs))
    if args.command == "ask":
        hits = retriever.search(args.question, args.top_k)
        try:
            answer = (ollama_answer(args.question, hits, args.ollama_model)
                      if args.ollama_model else extractive_answer(hits))
        except (URLError, TimeoutError, OSError) as exc:
            parser.error(f"Ollama unavailable: {exc}")
        print(json.dumps({"answer": answer, "evidence": [hit.to_dict() for hit in hits]}, indent=2))
    else:
        with args.cases.open(encoding="utf-8") as handle:
            cases = [json.loads(line) for line in handle if line.strip()]
        report = evaluate(retriever, cases, args.top_k)
        rendered = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)


if __name__ == "__main__":
    main()
