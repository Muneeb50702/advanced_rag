# Advanced RAG

An inspectable local document-QA pipeline: ingest Markdown/text, split into line-aware chunks, rank evidence with TF-IDF, and return answers with source references. It includes a retrieval evaluation command and an optional local Ollama generator. The default mode works offline without a model download or API key.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
rag ask "How often are database backups made?"
rag eval --top-k 2
pytest -q
```

Commands run from the repository root. Supply your own UTF-8 `.md` and `.txt` files with `--docs`:

```bash
rag --docs /path/to/notes ask "What is the rollback procedure?" --top-k 3
rag --docs /path/to/notes eval --cases eval/questions.jsonl --output reports/retrieval.json
```

The `ask` JSON response contains an answer plus every retrieved evidence chunk, similarity score, file path, and starting line. With no matching evidence, it abstains. In offline mode the answer is an extractive preview of the first two hits; read the evidence for the full context.

For generated prose, run [Ollama](https://ollama.com/) locally with a model you already have, then use `--ollama-model MODEL`. The generator receives only retrieved chunks and an instruction to cite them. **Generated citations and factual claims still need human review**; the application does not claim to verify them automatically.

## Evaluation

`eval/questions.jsonl` maps each question to an expected source file. `rag eval` reports recall@k and mean reciprocal rank (MRR@k), plus a per-question ranking trace. This tests retrieval, not answer correctness. The included operations documents are illustrative sample data; replace them and the evaluation set with your own corpus before drawing conclusions about a real domain.

## Design notes

- `src/advanced_rag/core.py` handles paragraph chunking, source lines, retrieval, abstention, and optional Ollama calls.
- `src/advanced_rag/cli.py` provides `ask` and `eval` commands.
- No document content is sent to a hosted service. The optional generator contacts only the configured local Ollama endpoint.
- TF-IDF is a transparent lexical baseline. It can miss paraphrases and does not understand meaning like a dense embedding model.

This is a working retrieval-augmented QA baseline with measured retrieval behavior, not a claim that a language model is bundled or that answers are guaranteed correct.
