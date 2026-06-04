"""Streamlit Q&A page for the Legal RAG.

Run with:
    streamlit run gui/app.py
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

from ingest import load_cfg, build_embedder, build_reranker
from utils import QdrantConnector, BGEReranker
from utils.query_expand import expand_query


st.set_page_config(page_title="Legal RAG · Ask", page_icon="📚", layout="wide")


def list_cfgs() -> List[str]:
    return sorted(
        f for f in os.listdir(ROOT)
        if f.startswith("cfg") and f.endswith(".yaml")
    ) or ["cfg.yaml"]


def launch_default_cfg() -> Optional[str]:
    """Pick a default cfg at launch from (in order): RAG_CFG env var, or `--cfg X`
    passed after `streamlit run gui/app.py -- --cfg X`. Returns a filename in ROOT
    or None to let the user pick from the sidebar."""
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
    # Accept absolute path or basename; we only care about the basename for the selector.
    base = os.path.basename(val)
    return base if base in list_cfgs() else None


@st.cache_resource(show_spinner="Loading model + index…")
def get_pipeline(cfg_path: str, force_reranker: Optional[bool]) -> Dict[str, Any]:
    """Build the RAG pipeline once per (cfg, reranker setting) tuple."""
    cfg = load_cfg(os.path.join(ROOT, cfg_path))
    emb, dim, hybrid = build_embedder(cfg)
    q = QdrantConnector(
        collection=cfg.qdrant.get("collection", "armenian_legal_chunks"),
        vector_size=dim,
        distance=cfg.qdrant.get("distance", "cosine"),
        mode=cfg.qdrant.get("mode", "local"),
        url=cfg.qdrant.get("url"),
        path=cfg.qdrant.get("path", cfg.paths.get("vdb_path")),
        hybrid=hybrid,
    )
    reranker = build_reranker(cfg)
    if force_reranker is True and reranker is None:
        rc = cfg.reranker or {}
        reranker = BGEReranker(
            model_name=rc.get("model", "BAAI/bge-reranker-v2-m3"),
            top_n=int(rc.get("top_n", 40)),
            enabled=True,
        )
    if force_reranker is False and reranker is not None:
        reranker.enabled = False

    # Optional OpenAI client (query expansion + final LLM answer)
    openai_client = None
    llm_cfg = (cfg.server or {}).get("llm_answer", {}) or {}
    key = None
    kf = cfg.embeddings.get("key_file")
    if kf:
        kfp = kf if os.path.isabs(kf) else os.path.join(ROOT, kf)
        if os.path.exists(kfp):
            with open(kfp, "r", encoding="utf-8") as f:
                key = (f.read() or "").strip() or None
    key = key or os.environ.get("OPENAI_API_KEY")
    if key:
        try:
            from openai import OpenAI
            openai_client = OpenAI(api_key=key)
        except Exception:
            openai_client = None

    return {
        "cfg": cfg,
        "emb": emb,
        "q": q,
        "hybrid": hybrid,
        "reranker": reranker,
        "openai": openai_client,
        "llm_model": llm_cfg.get("model", "gpt-5"),
        "llm_max_ctx": int(llm_cfg.get("max_context_passages", 8)),
        "llm_temp": llm_cfg.get("temperature", 0.2),
        "llm_enabled": bool(llm_cfg.get("enabled", False)) and openai_client is not None,
    }


def is_armenian(text: str) -> bool:
    return any("Ա" <= ch <= "֏" for ch in text)


def make_citation(payload: Dict[str, Any], lang: str = "en") -> str:
    law = payload.get("law_title") or payload.get("source_file")
    art = payload.get("article")
    cl = payload.get("clause")
    pages = payload.get("page_span") or (payload.get("page_start"), payload.get("page_end"))
    pages_str = None
    if isinstance(pages, (list, tuple)):
        try:
            pages_str = f"pp. {int(pages[0])}-{int(pages[1])}"
        except Exception:
            pass
    parts = [str(law)]
    if art:
        parts.append(f"Հոդված {art}" if lang == "hy" else f"Article {art}")
    if cl:
        parts.append(f"կետ {cl}" if lang == "hy" else f"Clause {cl}")
    if pages_str:
        parts.append(pages_str)
    return " → ".join(parts)


def generate_llm_answer(pipe, question: str, lang: str, items: List[Dict[str, Any]]) -> Optional[str]:
    client = pipe["openai"]
    if not pipe["llm_enabled"] or client is None:
        return None
    ctx = items[: pipe["llm_max_ctx"]]
    sys_prompt = (
        "You are a precise legal assistant. Use only the provided passages. "
        "Quote verbatim and include the citation tag in brackets."
    )
    if lang == "hy":
        user_prompt = (
            f"Հարցը: {question}\n\n"
            "Ապավինիր միայն հետևյալ հատվածներին. Յուրաքանչյուր մեջբերման կողքին ավելացրու հղումը [Օրենք → Հոդված → Կետ → էջեր].\n"
            "Պատասխանը ներկայացրու հայերենով։\n\n"
            + "\n\n".join(
                f"- [{make_citation(it.get('payload', {}), 'hy')}]\n{it.get('payload', {}).get('text', '')}"
                for it in ctx
            )
        )
    else:
        user_prompt = (
            f"Question: {question}\n\n"
            "Rely only on the passages below. Provide: (1) a concise answer, "
            "(2) a short quote, (3) citations [Law → Article → Clause → pages].\n\n"
            + "\n\n".join(
                f"- [{make_citation(it.get('payload', {}), 'en')}]\n{it.get('payload', {}).get('text', '')}"
                for it in ctx
            )
        )
    try:
        kwargs = {
            "model": pipe["llm_model"],
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        temp = pipe["llm_temp"]
        if temp is not None and temp != 1:
            kwargs["temperature"] = float(temp)
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            if "temperature" in str(e):
                kwargs.pop("temperature", None)
                resp = client.chat.completions.create(**kwargs)
            else:
                raise
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"LLM error: {e}")
        return None


# ─────────────────────────── UI ───────────────────────────
st.title("📚 Ask the Legal RAG")

with st.sidebar:
    st.header("Configuration")
    _cfgs = list_cfgs()
    _default = launch_default_cfg()
    _idx = _cfgs.index(_default) if _default in _cfgs else 0
    cfg_path = st.selectbox("Corpus / config", _cfgs, index=_idx, key="cfg_select")
    if _default:
        st.caption(f"Default from launch: `{_default}`")
    top_k = st.slider("Passages to retrieve (top_k)", 5, 50, 20)
    rr_choice = st.radio(
        "Cross-encoder reranker",
        ["Use cfg setting", "Force on", "Force off"],
    )
    show_passages = st.checkbox("Show retrieved passages", value=True)
    if st.button("🔄 Reload pipeline"):
        st.cache_resource.clear()
        st.rerun()

force_rr = None if rr_choice == "Use cfg setting" else (rr_choice == "Force on")

try:
    pipe = get_pipeline(cfg_path, force_rr)
except Exception as e:
    st.error(f"Failed to initialize pipeline: {e}")
    st.stop()

cfg = pipe["cfg"]
rr_state = (
    f"ON (top_n={pipe['reranker'].top_n})"
    if pipe["reranker"] is not None and pipe["reranker"].enabled
    else "off"
)
st.caption(
    f"DB: `{cfg.qdrant.get('path')}` · collection `{cfg.qdrant.get('collection')}` "
    f"· embedder `{cfg.embeddings.get('model')}` · reranker: {rr_state} "
    f"· LLM answer: {'on' if pipe['llm_enabled'] else 'off'}"
)

question = st.text_area("Your question", height=90, placeholder="Ask in any language…")
ask = st.button("Ask", type="primary", disabled=not question.strip())

if ask and question.strip():
    lang = "hy" if is_armenian(question) else "en"
    with st.spinner("Searching…"):
        expanded = None
        if pipe["openai"] is not None:
            expanded = expand_query(pipe["openai"], pipe["llm_model"], question)
        search_q = f"{question} {expanded}" if expanded else question

        if pipe["hybrid"]:
            qe = pipe["emb"].embed_one(search_q)
            results = pipe["q"].search_hybrid(
                qe["dense"], qe["sparse"], top_k=max(120, top_k)
            )
        else:
            qv = pipe["emb"].embed_texts([search_q])[0]
            results = pipe["q"].search(
                qv, top_k=max(120, top_k), with_payload=True, with_vectors=False
            )

        items = [{"text": r["payload"].get("text", ""), **r} for r in results]
        if pipe["reranker"] is not None and pipe["reranker"].enabled:
            items = pipe["reranker"].rerank(question, items, text_key="text")
        items = items[:top_k]

        answer = generate_llm_answer(pipe, question, lang, items)

    if expanded:
        with st.expander("Expanded query (used for retrieval)"):
            st.code(expanded)

    if answer:
        st.subheader("Answer")
        st.markdown(answer)
    elif not pipe["llm_enabled"]:
        st.info("LLM answer is disabled (no key or `llm_answer.enabled: false`). Showing top passages only.")

    if show_passages:
        st.subheader("Retrieved passages")
        for i, it in enumerate(items, 1):
            p = it.get("payload", {})
            score = float(it.get("score") or 0.0)
            with st.expander(f"#{i} · score {score:.3f} · {make_citation(p, lang)}"):
                st.write((p.get("text") or "")[:3000])
