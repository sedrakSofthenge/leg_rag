# Legal RAG — Vectorize & Run

A two-phase system:
1. **Vectorize (ingest)** — convert PDF documents into searchable vectors, once per corpus.
2. **Run (query)** — ask questions against those vectors, via CLI, HTTP server, or Streamlit GUI.

Embeddings use **BGE-M3** (local, multilingual). It produces a 1024-d dense vector
plus a sparse lexical vector per chunk; retrieval fuses both with Reciprocal
Rank Fusion (RRF). An optional **BGE cross-encoder reranker** can be turned on
per-corpus for extra precision.

> Nothing here *trains* a model. "Vectorize" = run each chunk through the frozen
> BGE-M3 model once (inference) and store the result.

---

## 0. One-time setup

```bash
cd /home/sethopar/projects/rag/armenianLeg
pip install -r requirements.txt
```
- First ingest/query auto-downloads **BGE-M3 (~2.3 GB)** to `~/.cache/huggingface`.
- Runs on CPU; GPU is strongly recommended for corpora over a few thousand chunks.
- `openai_key.txt` is **only** required for the optional LLM answer and query expansion.
  Retrieval itself needs no API key.

### Configs (one per corpus)
- [cfg.yaml](cfg.yaml) — Armenian corpus (`vdb/`, `data_raw_orig/`).
- [cfg.en.yaml](cfg.en.yaml) — English corpus (`vdb_en/`, `../selenium_pdfs/`).

Independent vector stores and manifests, so the two coexist and can even be
queried at the same time (one process per store; local Qdrant locks the dir).

Every CLI accepts `--cfg <file>`; the FastAPI server and Streamlit GUI honor
the `RAG_CFG` environment variable as well.

---

## 1. Vectorize (ingest)

```bash
# (a) Fresh build (REQUIRED if you changed embedding model or chunking params):
rm -rf vdb .manifest.sqlite data_text/*.jsonl                # Armenian
rm -rf vdb_en .manifest.en.sqlite data_text_en/*.jsonl       # English

# (b) Run ingest. -u keeps progress unbuffered.
python3 -u ingest.py --cfg cfg.yaml                          # Armenian
python3 -u ingest.py --cfg cfg.en.yaml                       # English
```
The folder of PDFs comes from `paths.data_raw` in the cfg; override with `--pdf-dir <path>` if needed.

Per PDF: extract text → segment into articles/clauses → chunk (~800 tokens) →
embed (dense + sparse) → upsert into Qdrant. Debug chunks are written to the
cfg's `paths.data_text` dir as one JSONL per PDF.

### Ingest is resumable
The manifest sqlite records each completed file. Re-running the same command
skips done files with `Skip unchanged: …` and continues. Safe to Ctrl-C.

### Watch progress (in a second terminal — safe, read-only)
```bash
python3 -c "import sqlite3; c=sqlite3.connect('file:.manifest.sqlite?mode=ro',uri=True); \
r=list(c.execute('SELECT upserted FROM files')); print(sum(x[0] for x in r),'/',len(r))"
# replace .manifest.sqlite with .manifest.en.sqlite for the English corpus
```

### How long?
Wildly corpus-dependent on CPU. As reference points:
- 18 Armenian laws (~3,800 chunks) → ~2–3 h on CPU.
- 2,319 HK ordinances (~60,000 chunks) → ~30–50 h on CPU. **Use a GPU at this scale.**

### Memory tuning (avoid OOM on small boxes)
BGE-M3 attention memory grows roughly as O(n²) in sequence length. On a ~8 GB
RAM machine, lower these in cfg if you OOM:
```yaml
embeddings:
  max_input_tokens: 1024   # caps per-sequence length
  batch_size: 4            # fewer sequences in memory at once
```

### Verify after it finishes
```bash
python3 -c "from qdrant_client import QdrantClient; \
print(QdrantClient(path='vdb').count('armenian_legal_chunks').count)"
# Note: only ONE process can open a local vdb at a time.
```

---

## 2. Run (query)

> Only one process can open a given local `vdb*/` at a time. Stop ingest before
> querying, or use the HTTP server / GUI which manage this for you.

### CLI
```bash
python3 ask_cli.py --cfg cfg.yaml --lang hy             # Armenian
python3 ask_cli.py --cfg cfg.en.yaml                    # English (auto-detected)
```
Type a question at `>>`. The CLI prints the **expanded query** (legal-term
rewrite of your colloquial question, used only for retrieval), the **raw top-N
passages**, and finally the LLM answer (if `server.llm_answer.enabled` and a
key is set). `:q` to quit.

### HTTP server
```bash
RAG_CFG=cfg.yaml    uvicorn query_server:app --port 8000   # Armenian
RAG_CFG=cfg.en.yaml uvicorn query_server:app --port 8001   # English
```
Endpoints: `POST /ask` (retrieval + LLM answer), `POST /preview` (retrieval only),
`POST /ingest` (kicks off ingest for the loaded cfg).

```bash
curl -s localhost:8000/preview -H 'content-type: application/json' \
  -d '{"question":"ամենամյա արձակուրդի տևողությունը","top_k":5}' | python3 -m json.tool
```

### Streamlit GUI
```bash
pip install streamlit
RAG_CFG=cfg.yaml streamlit run gui/app.py
# or pass via CLI:  streamlit run gui/app.py -- --cfg cfg.en.yaml
```
Two pages: **Ask** (question → expansion → retrieval → answer + passages) and
**Settings and Ingest** (edit cfg params via form or raw YAML; start/stop ingest;
live manifest progress; tail of the ingest log).

---

## 3. Optional features

### Cross-encoder reranker (precision boost, query-time)
Bumps the most relevant passages into the top-N sent to the LLM. Flip per-cfg:
```yaml
reranker:
  enabled: true                          # default false
  model: BAAI/bge-reranker-v2-m3
  top_n: 40                              # rerank top-N hybrid candidates
```
First query then downloads the reranker model (~2.3 GB) and adds a few seconds
per query on CPU. Toggle live from the Streamlit sidebar (Force on / Force off).

### Query expansion
Runs automatically whenever an OpenAI key is available. The model rewrites your
question into legal-term search terminology (and pulls in related concepts —
e.g. for "murder attempt" it adds the general criminal-attempt rule). The
original question still drives the final answer prompt. Disable by removing the
key file / `OPENAI_API_KEY`.

### LLM answer
Controlled by `server.llm_answer.enabled` in cfg. If off (or key missing), the
system returns a template answer with the top passages quoted.

---

## 4. Scale notes

- Local Qdrant (`qdrant-client` embedded mode) prints a warning past **20k
  points** in a collection. At ~50k+ it starts pressuring RAM; either add swap
  or run Qdrant as a separate process (the code supports it via
  `qdrant.mode: http` + `url:`).

  ```bash
  docker run -d --name qdrant -p 6333:6333 -v "$PWD/qdrant_storage:/qdrant/storage" qdrant/qdrant
  # then in cfg:
  qdrant:
    mode: http
    url: http://localhost:6333
  ```
  ⚠️ Server and local modes use **different on-disk formats** — switching modes
  requires re-ingesting.

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `XLMRobertaModel ... unexpected keyword argument 'dtype'` | FlagEmbedding/transformers clash. Use `FlagEmbedding==1.3.5` + `transformers<4.56` (pinned in requirements.txt). |
| `Storage folder ... is already accessed by another instance` | Another process holds the local-Qdrant lock (ingest, a CLI, the server, or the GUI's cache). Stop it before opening the same `vdb*/` elsewhere. |
| `Killed` mid-ingest (no traceback) | Linux OOM killer. Lower `max_input_tokens` to 1024 and `batch_size` to 2–4 in the cfg, then re-run (resumable). |
| `maximum request size is 300000 tokens` | Only affects the legacy OpenAI embedder path; the batcher splits by token budget. Not applicable to BGE-M3. |
| Re-running ingest re-does everything | Manifest got wiped or the cfg's `paths.manifest_sqlite` changed. Same path = resume; new path = fresh build. |
| Local Qdrant warning about >20k points | Expected past that size. See **Scale notes** to move to Docker Qdrant. |
