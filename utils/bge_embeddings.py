from __future__ import annotations

from typing import Any, Dict, List, Optional

# BGE-M3 produces a 1024-dim dense vector plus sparse lexical weights in one pass.
# Unlike OpenAI's text-embedding-3-large, it represents Armenian discriminatively,
# and the sparse side gives exact-term (lexical) matching for legal terminology.
DENSE_DIM = 1024


class BGEM3Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = False,
        batch_size: int = 12,
        max_length: int = 8192,
    ) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "FlagEmbedding is required for BGE-M3. Install with `pip install FlagEmbedding`."
            ) from e
        # use_fp16=False on CPU (fp16 needs CUDA).
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.batch_size = batch_size
        self.max_length = max_length

    def embed(self, texts: List[str]) -> Dict[str, Any]:
        """Return {"dense": [[float]...], "sparse": [{"indices":[int], "values":[float]}...]}."""
        if not texts:
            return {"dense": [], "sparse": []}
        out = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = [[float(x) for x in vec] for vec in out["dense_vecs"]]
        sparse: List[Dict[str, List]] = []
        for lw in out["lexical_weights"]:
            # lw maps token-id (str) -> weight (float); may be empty for trivial input.
            indices = [int(k) for k in lw.keys()]
            values = [float(v) for v in lw.values()]
            sparse.append({"indices": indices, "values": values})
        return {"dense": dense, "sparse": sparse}

    def embed_one(self, text: str) -> Dict[str, Any]:
        r = self.embed([text])
        return {"dense": r["dense"][0], "sparse": r["sparse"][0]}
