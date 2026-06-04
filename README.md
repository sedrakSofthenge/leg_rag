# Armenian Legal RAG (starter)

This is a starter implementation of a local RAG stack for Armenian legislation. It ingests PDFs, segments them into articles/clauses, chunks them, generates embeddings, and stores them in Qdrant. A FastAPI server exposes `/ingest`, `/ask`, and `/preview` endpoints.

## Quickstart

Prerequisites:
- Python 3.10+
- Recommended: virtualenv
- Optional: Local Qdrant (Docker) OR use embedded mode via `qdrant-client` path

Install deps:

```
pip install -r requirements.txt  # If you create one
# or install minimal deps:
pip install pydantic fastapi uvicorn pyyaml pypdf qdrant-client
# Optional: OpenAI for embeddings and answer generation
pip install openai tiktoken
```

Set OpenAI key (choose one):

- Key file (recommended for local dev): create `openai_key.txt` at project root with your key. `cfg.yaml` points to it via `embeddings.key_file`.
- Environment variable: `export OPENAI_API_KEY=sk-...`

Project layout is described in `cfg.yaml`. By default:
- PDFs go in `data_raw/`
- Extracted JSONL in `data_text/`
- Local Qdrant storage at `vdb/` (embedded mode)

### 1) Ingest PDFs

```
python ingest.py --pdf-dir ./data_raw
```

This creates a manifest DB `.manifest.sqlite` to track incremental work.

### 2) Run API server

```
uvicorn query_server:app --reload --port 8000
```

### 3) Ask questions

Armenian:

```
curl -X POST localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Ո՞ր հոդվածն է կարգավորում սեփականության իրավունքը"}'
```

English:

```
curl -X POST localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Which article regulates property rights?"}'

### Interactive CLI (no server required)

Run an interactive prompt from the terminal and ask questions directly against the local vector DB:

```
python ask_cli.py --top-k 10
```

Notes:
- Uses the same config as the server (`cfg.yaml`).
- Answer generation is a concise template with Armenian quotes and citations. Enable OpenAI in config if you later extend it to LLM answers.
```

### Configuration

Edit `cfg.yaml` to adjust chunk sizes, topK, reranker toggle, Qdrant mode (embedded path vs HTTP URL), and `embeddings.key_file`.

### Notes

- If you don’t have an OpenAI key or prefer offline-only setup, set `embeddings.provider: fake` in `cfg.yaml`. This uses deterministic hashed vectors with smaller dimension (defaults to 384). This is good for plumbing and local testing, not for accuracy.
- The segmentation heuristics for Armenian laws are regex-based and may need tuning for your corpus.
- Noise removal: configure ingest-time noise patterns in `cfg.yaml` under `ingest.noise_patterns` (regex, case-insensitive). Common defaults include Google translation watermarks. The parser also removes short header/footer lines repeated across pages.
- Answer generation and translation:
  - By default, answers are template-only (no LLM), so English questions won’t translate Armenian quotes.
  - To enable LLM answers with translation, set in `cfg.yaml`:
    ```
    server:
      llm_answer:
        enabled: true
        model: gpt-5
    ```
    Make sure your `openai_key.txt` or `OPENAI_API_KEY` is set.
