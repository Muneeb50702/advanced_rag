import json
from pathlib import Path

from advanced_rag.cli import evaluate
from advanced_rag.core import Retriever, extractive_answer, load_chunks


ROOT = Path(__file__).resolve().parents[1]


def test_search_returns_ranked_source_and_line():
    retriever = Retriever(load_chunks(ROOT / "examples/operations"))
    hits = retriever.search("How often are database backups made?", 2)
    assert hits[0].source == "backups.md"
    assert hits[0].line > 0
    assert "[backups.md:" in extractive_answer(hits)


def test_no_evidence_abstains():
    retriever = Retriever(load_chunks(ROOT / "examples/operations"))
    assert retriever.search("quantum bananas orbiting Saturn") == []
    assert "cannot answer" in extractive_answer([])


def test_evaluation_reports_recall_and_mrr():
    retriever = Retriever(load_chunks(ROOT / "examples/operations"))
    cases = [json.loads(line) for line in (ROOT / "eval/questions.jsonl").read_text().splitlines()]
    report = evaluate(retriever, cases, top_k=2)
    assert report["count"] == 4
    assert report["recall_at_k"] >= 0.75
    assert 0 <= report["mrr_at_k"] <= 1


def test_paragraphs_preserve_line_number(tmp_path):
    (tmp_path / "note.md").write_text("# Heading\n\nFirst fact.\n\nSecond fact.\n")
    assert [(c.line, c.text) for c in load_chunks(tmp_path)] == [
        (1, "Heading"), (3, "First fact."), (5, "Second fact.")
    ]
