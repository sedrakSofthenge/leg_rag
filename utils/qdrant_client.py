from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional


class QdrantConnector:
    # Named-vector keys used in hybrid (dense + sparse) mode.
    DENSE = "dense"
    SPARSE = "lex"

    def __init__(
        self,
        collection: str,
        vector_size: int,
        distance: str = "cosine",
        mode: str = "local",
        url: Optional[str] = None,
        path: Optional[str] = None,
        hybrid: bool = False,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                Distance,
                VectorParams,
                PointStruct,
                SparseVectorParams,
            )
        except Exception as e:
            raise RuntimeError(
                "qdrant-client is required. Install with `pip install qdrant-client`."
            ) from e

        self._Distance = Distance
        self._VectorParams = VectorParams
        self.collection = collection
        self._PointStruct = PointStruct
        self.hybrid = hybrid
        dist = distance.lower()
        if dist == "cosine":
            self._distance = Distance.COSINE
        elif dist in ("dot", "dotproduct"):
            self._distance = Distance.DOT
        else:
            self._distance = Distance.EUCLID

        if mode == "http" and url:
            self.client = QdrantClient(url=url)
        elif mode == "local" and path:
            # Embedded/local storage
            self.client = QdrantClient(path=path)
        else:
            # Fallback to defaults
            self.client = QdrantClient(path=path or ":memory:")

        # Ensure collection exists
        if not self._collection_exists(collection):
            if hybrid:
                self.client.create_collection(
                    collection_name=collection,
                    vectors_config={self.DENSE: VectorParams(size=vector_size, distance=self._distance)},
                    sparse_vectors_config={self.SPARSE: SparseVectorParams()},
                )
            else:
                self.client.recreate_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=vector_size, distance=self._distance),
                )

    def _collection_exists(self, name: str) -> bool:
        coll = self.client.get_collections()
        names = [c.name for c in coll.collections]
        return name in names

    def upsert(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        points = []
        for pid, vec, payload in zip(ids, vectors, payloads):
            qid = self._coerce_point_id(pid)
            points.append(self._PointStruct(id=qid, vector=vec, payload=payload))
        self.client.upsert(
            collection_name=self.collection,
            points=points,
            wait=True,
        )

    def _coerce_point_id(self, pid: Any) -> Any:
        """Qdrant accepts int or UUID for point ids. If we get a non-UUID string,
        convert it deterministically to UUID v5 to satisfy the server.
        """
        if isinstance(pid, int):
            return pid
        if isinstance(pid, uuid.UUID):
            # qdrant-client pydantic PointStruct wants id as int or str
            return str(pid)
        if isinstance(pid, str):
            # Try parse as UUID
            try:
                u = uuid.UUID(pid)
                return str(u)
            except Exception:
                # Deterministic UUID from string name
                return str(uuid.uuid5(uuid.NAMESPACE_URL, pid))
        # Fallback: hash to int range
        try:
            s = str(pid)
            return str(uuid.uuid5(uuid.NAMESPACE_URL, s))
        except Exception:
            return 0

    def upsert_hybrid(
        self,
        ids: List[str],
        dense: List[List[float]],
        sparse: List[Dict[str, List]],
        payloads: List[Dict[str, Any]],
    ):
        from qdrant_client.models import SparseVector

        points = []
        for pid, dv, sv, payload in zip(ids, dense, sparse, payloads):
            qid = self._coerce_point_id(pid)
            vec = {self.DENSE: dv}
            # Skip empty sparse vectors (Qdrant rejects zero-length sparse input).
            if sv and sv.get("indices"):
                vec[self.SPARSE] = SparseVector(indices=sv["indices"], values=sv["values"])
            points.append(self._PointStruct(id=qid, vector=vec, payload=payload))
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search_hybrid(
        self,
        dense: List[float],
        sparse: Dict[str, List],
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        prefetch_limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Dense + sparse retrieval fused with Reciprocal Rank Fusion (RRF)."""
        from qdrant_client.models import (
            Filter,
            FieldCondition,
            MatchValue,
            Prefetch,
            FusionQuery,
            Fusion,
            SparseVector,
        )

        qfilter = None
        if filters:
            must = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
            qfilter = Filter(must=must)

        pf_limit = prefetch_limit or max(top_k * 3, 100)
        prefetch = [Prefetch(query=dense, using=self.DENSE, limit=pf_limit, filter=qfilter)]
        if sparse and sparse.get("indices"):
            prefetch.append(
                Prefetch(
                    query=SparseVector(indices=sparse["indices"], values=sparse["values"]),
                    using=self.SPARSE,
                    limit=pf_limit,
                    filter=qfilter,
                )
            )

        res = self.client.query_points(
            collection_name=self.collection,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [{"id": str(p.id), "score": p.score, "payload": p.payload} for p in res.points]

    def search(
        self,
        vector: List[float],
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        hnsw_ef: Optional[int] = None,
        score_threshold: Optional[float] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        qfilter = None
        if filters:
            must = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
            qfilter = Filter(must=must)

        search_params = {"hnsw_ef": hnsw_ef} if hnsw_ef is not None else None

        res = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            query_filter=qfilter,
            score_threshold=score_threshold,
            with_payload=with_payload,
            with_vectors=with_vectors,
            search_params=search_params,   # <-- NOT "params"
        )
        return [{"id": str(r.id), "score": r.score, "payload": r.payload} for r in res]


