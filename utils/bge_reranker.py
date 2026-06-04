from __future__ import annotations

from typing import Any, Dict, List

# Query-time cross-encoder reranker (bge-reranker-v2-m3, same family as BGE-M3,
# strong on Armenian). It re-scores (question, passage) pairs to push the most
# relevant passages into the top-N that get sent to the LLM. This is pure
# query-time work — it never touches ingestion / stored vectors.
#
# The model (~2.3GB) is downloaded lazily on first rerank, so leaving this
# disabled costs nothing.


class BGEReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        top_n: int = 40,
        use_fp16: bool = False,
        enabled: bool = False,
    ) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.top_n = top_n
        self.use_fp16 = use_fp16
        self._model = None

    def _load(self):
        if self._model is None:
            from FlagEmbedding import FlagReranker  # type: ignore
            self._model = FlagReranker(self.model_name, use_fp16=self.use_fp16)
        return self._model

    def rerank(self, query: str, items: List[Dict[str, Any]], text_key: str = "text") -> List[Dict[str, Any]]:
        """Re-score the top_n candidates with the cross-encoder; leave the rest as-is.

        Reranked items get a normalized (0..1) score in both 'rerank_score' and 'score'.
        """
        if not self.enabled or not items:
            return items
        head = items[: self.top_n]
        tail = items[self.top_n:]
        try:
            model = self._load()
            pairs = [[query, (it.get(text_key) or "")] for it in head]
            scores = model.compute_score(pairs, normalize=True)
            if not isinstance(scores, list):
                scores = [scores]
        except Exception:
            # On any failure, fall back to the original hybrid order.
            return items
        for it, s in zip(head, scores):
            it["rerank_score"] = float(s)
            it["score"] = float(s)
        head.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return head + tail
