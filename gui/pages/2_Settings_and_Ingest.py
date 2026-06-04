"""Settings editor + ingest control page.

The ingest runs as a detached subprocess so it survives Streamlit reruns and
this page's lifecycle. Progress comes from the manifest sqlite (read-only) so
we don't fight the running ingest for the vdb lock.
"""
from __future__ import annotations

import os
import sys
import time
import sqlite3
import subprocess
import signal

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import yaml


st.set_page_config(page_title="Settings & Ingest", page_icon="⚙️", layout="wide")
st.title("⚙️ Settings & Ingest")


def list_cfgs():
    return sorted(
        f for f in os.listdir(ROOT)
        if f.startswith("cfg") and f.endswith(".yaml")
    )


def launch_default_cfg(available):
    """RAG_CFG env var or `-- --cfg X` after streamlit run."""
    val = os.environ.get("RAG_CFG")
    if not val:
        argv = sys.argv
        for i, a in enumerate(argv):
            if a == "--cfg" and i + 1 < len(argv):
                val = argv[i + 1]; break
            if a.startswith("--cfg="):
                val = a.split("=", 1)[1]; break
    if not val:
        return None
    base = os.path.basename(val)
    return base if base in available else None


# ── pick config ────────────────────────────────────────────────────────
cfgs = list_cfgs()
if not cfgs:
    st.error("No cfg*.yaml files found in the project root.")
    st.stop()

_default = launch_default_cfg(cfgs)
_idx = cfgs.index(_default) if _default in cfgs else 0
cfg_path = st.selectbox("Config file", cfgs, index=_idx)
if _default:
    st.caption(f"Default from launch: `{_default}`")
cfg_full = os.path.join(ROOT, cfg_path)

with open(cfg_full, "r", encoding="utf-8") as f:
    cfg_text = f.read()
cfg_obj = yaml.safe_load(cfg_text) or {}

paths = cfg_obj.get("paths", {}) or {}
emb_cfg = cfg_obj.get("embeddings", {}) or {}
ing_cfg = cfg_obj.get("ingest", {}) or {}
qdr_cfg = cfg_obj.get("qdrant", {}) or {}

st.caption(
    f"Vector store: `{paths.get('vdb_path')}` · "
    f"manifest: `{paths.get('manifest_sqlite')}` · "
    f"collection: `{qdr_cfg.get('collection')}`"
)

# ── key params (curated form) ──────────────────────────────────────────
st.subheader("Ingestion parameters")
c1, c2, c3 = st.columns(3)
with c1:
    new_data_raw = st.text_input("PDF folder (paths.data_raw)", paths.get("data_raw", ""))
    new_vdb = st.text_input("Vector store dir (paths.vdb_path)", paths.get("vdb_path", "vdb"))
    new_manifest = st.text_input(
        "Manifest sqlite (paths.manifest_sqlite)",
        paths.get("manifest_sqlite", ".manifest.sqlite"),
    )
with c2:
    new_chunk = st.number_input(
        "chunk_size_tokens", min_value=64,
        value=int(ing_cfg.get("chunk_size_tokens", 800)),
    )
    new_overlap = st.number_input(
        "chunk_overlap_tokens", min_value=0,
        value=int(ing_cfg.get("chunk_overlap_tokens", 150)),
    )
    new_collection = st.text_input(
        "qdrant.collection", qdr_cfg.get("collection", "legal_chunks")
    )
with c3:
    new_max_in = st.number_input(
        "embeddings.max_input_tokens", min_value=128,
        value=int(emb_cfg.get("max_input_tokens", 1024)),
        help="Caps sequence length. Attention memory grows ~O(n²); lower this if you OOM.",
    )
    new_batch = st.number_input(
        "embeddings.batch_size", min_value=1, max_value=128,
        value=int(emb_cfg.get("batch_size", 4)),
        help="Lower → less RAM, slightly slower throughput.",
    )
    new_model = st.text_input("embeddings.model", emb_cfg.get("model", "BAAI/bge-m3"))

if st.button("💾 Save changes"):
    cfg_obj.setdefault("paths", {})
    cfg_obj["paths"]["data_raw"] = new_data_raw
    cfg_obj["paths"]["vdb_path"] = new_vdb
    cfg_obj["paths"]["manifest_sqlite"] = new_manifest
    cfg_obj.setdefault("ingest", {})["chunk_size_tokens"] = int(new_chunk)
    cfg_obj["ingest"]["chunk_overlap_tokens"] = int(new_overlap)
    cfg_obj.setdefault("embeddings", {})["max_input_tokens"] = int(new_max_in)
    cfg_obj["embeddings"]["batch_size"] = int(new_batch)
    cfg_obj["embeddings"]["model"] = new_model
    cfg_obj.setdefault("qdrant", {})["collection"] = new_collection
    with open(cfg_full, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_obj, f, sort_keys=False, allow_unicode=True)
    st.success(f"Saved → {cfg_path}")
    st.rerun()

# ── advanced raw YAML editor ──────────────────────────────────────────
with st.expander("Advanced: edit raw YAML"):
    edited = st.text_area("YAML", cfg_text, height=400, key=f"raw_{cfg_path}")
    if st.button("Save raw YAML", key=f"save_raw_{cfg_path}"):
        try:
            yaml.safe_load(edited)
        except Exception as e:
            st.error(f"Invalid YAML: {e}")
        else:
            with open(cfg_full, "w", encoding="utf-8") as f:
                f.write(edited)
            st.success("Saved")
            st.rerun()

st.divider()

# ── ingest controls ────────────────────────────────────────────────────
st.subheader("Run ingest")

LOG_PATH = os.path.join(ROOT, f".ingest.{cfg_path}.log")
PID_KEY = f"ingest_pid::{cfg_path}"

pdf_override = st.text_input(
    "PDF dir override (optional — leave empty to use cfg paths.data_raw)", ""
)

pid = st.session_state.get(PID_KEY)
running = bool(pid) and os.path.exists(f"/proc/{pid}")

b1, b2, b3 = st.columns([1, 1, 1])
with b1:
    if st.button("▶ Start ingest", type="primary", disabled=running):
        # Free the vdb lock if the Q&A page cached a QdrantClient on this store.
        st.cache_resource.clear()
        cmd = ["python3", "-u", "ingest.py", "--cfg", cfg_path]
        if pdf_override.strip():
            cmd += ["--pdf-dir", pdf_override.strip()]
        with open(LOG_PATH, "w") as logf:
            proc = subprocess.Popen(
                cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        st.session_state[PID_KEY] = proc.pid
        st.success(f"Started PID {proc.pid}. Streaming → {os.path.basename(LOG_PATH)}")
        time.sleep(0.5)
        st.rerun()

with b2:
    if st.button("■ Stop ingest", disabled=not running):
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.5)
            if os.path.exists(f"/proc/{pid}"):
                os.kill(pid, signal.SIGKILL)
        except Exception as e:
            st.warning(f"Kill failed: {e}")
        st.session_state[PID_KEY] = None
        st.rerun()

with b3:
    if st.button("🔄 Refresh status"):
        st.rerun()

st.caption(f"Ingest is **{'running (PID ' + str(pid) + ')' if running else 'not running'}**.")

# ── progress (manifest, read-only) ────────────────────────────────────
manifest_path = os.path.join(ROOT, new_manifest or ".manifest.sqlite")
if os.path.exists(manifest_path):
    try:
        c = sqlite3.connect(f"file:{manifest_path}?mode=ro", uri=True)
        rows = list(c.execute("SELECT upserted FROM files"))
        if rows:
            done = sum(r[0] for r in rows)
            total = len(rows)
            st.progress(done / max(total, 1), text=f"{done}/{total} files committed")
        else:
            st.info("Manifest exists but has no files yet.")
    except Exception as e:
        st.warning(f"Manifest read error: {e}")
else:
    st.info("No manifest yet — start an ingest to begin.")

# ── log tail ──────────────────────────────────────────────────────────
st.subheader("Ingest log (last lines)")
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 50_000))
        tail = f.read().decode("utf-8", errors="ignore")
    # filter out tqdm progress noise
    lines = [
        ln for ln in tail.splitlines()
        if not any(token in ln for token in ("it/s]", "%|", "it/s,"))
    ]
    st.code("\n".join(lines[-200:]) or "(empty)", language="text")
else:
    st.info("No log yet. Click **Start ingest** to begin.")
