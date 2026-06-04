from __future__ import annotations

import os
import math
import hashlib
from typing import Callable, List, Optional

# OpenAI's embeddings endpoint caps total tokens per request at 300_000.
# Stay safely under it to leave headroom for tokenizer estimate drift.
_MAX_REQUEST_TOKENS = 280_000


class Embeddings:
    def __init__(
        self,
        provider: str = "openai",
        model: str = "text-embedding-3-large",
        dim: int = 3072,
        batch_size: int = 64,
        api_key: Optional[str] = None,
        key_file: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self._token_counter: Optional[Callable[[str], int]] = None

        self._client = None
        if provider == "openai":
            try:
                from openai import OpenAI  # type: ignore
                # Resolve API key: explicit arg > key_file > env var
                key = api_key or self._read_key_file(key_file) or os.environ.get("OPENAI_API_KEY")
                if not key:
                    # Allow no key for environments that inject config differently, but warn clearly
                    raise RuntimeError(
                        "OpenAI API key not found. Set OPENAI_API_KEY or provide embeddings.key_file in cfg.yaml."
                    )
                self._client = OpenAI(api_key=key)
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(
                    "OpenAI SDK not available. Install with `pip install openai`."
                ) from e

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "openai":
            return self._embed_openai(texts)
        elif self.provider == "fake":
            return [self._hash_to_vec(t) for t in texts]
        else:
            raise ValueError(f"Unknown embeddings provider: {self.provider}")

    # --- Providers ---
    def _count_tokens(self, text: str) -> int:
        if self._token_counter is None:
            from .tokenization import get_token_counter_for_model
            self._token_counter = get_token_counter_for_model(self.model)
        return self._token_counter(text)

    def _batches(self, texts: List[str]):
        """Yield batches bounded by both batch_size (count) and the per-request
        token cap, so a request never exceeds OpenAI's 300k-token limit."""
        batch: List[str] = []
        batch_tokens = 0
        for t in texts:
            t_tok = self._count_tokens(t)
            if batch and (
                len(batch) >= self.batch_size
                or batch_tokens + t_tok > _MAX_REQUEST_TOKENS
            ):
                yield batch
                batch, batch_tokens = [], 0
            batch.append(t)
            batch_tokens += t_tok
        if batch:
            yield batch

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for chunk in self._batches(texts):
            resp = self._client.embeddings.create(model=self.model, input=chunk)
            # New SDK returns objects with .data list and .embedding arrays
            for d in resp.data:
                results.append(list(d.embedding))
        return results

    def _hash_to_vec(self, text: str) -> List[float]:
        # Deterministic pseudo-embedding using SHA1; dimension = self.dim
        out = []
        for i in range(self.dim):
            h = hashlib.sha1(f"{i}:{text}".encode("utf-8")).digest()
            # Map first 4 bytes to float in [-1, 1]
            val = int.from_bytes(h[:4], "big") / 0xFFFFFFFF
            out.append((val * 2.0) - 1.0)
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in out)) or 1.0
        return [v / norm for v in out]

    def _read_key_file(self, key_file: Optional[str]) -> Optional[str]:
        if not key_file:
            return None
        path = os.path.expanduser(key_file)
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                key = f.read().strip()
                return key or None
        except Exception:
            return None
