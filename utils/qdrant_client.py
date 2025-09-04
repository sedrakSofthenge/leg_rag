from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional


class QdrantConnector:
    def __init__(
        self,
        collection: str,
        vector_size: int,
        distance: str = "cosine",
        mode: str = "local",
        url: Optional[str] = None,
        path: Optional[str] = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
        except Exception as e:
            raise RuntimeError(
                "qdrant-client is required. Install with `pip install qdrant-client`."
            ) from e

        self._Distance = Distance
        self._VectorParams = VectorParams
        self.collection = collection
        self._PointStruct = PointStruct
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

    def search(
        self,
        vector: List[float],
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        qfilter = None
        if filters:
            must: List[Any] = []
            for key, val in filters.items():
                must.append(FieldCondition(key=key, match=MatchValue(value=val)))
            qfilter = Filter(must=must)

        res = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
            with_vectors=False,
        )
        out = []
        for r in res:
            out.append({
                "id": str(r.id),
                "score": r.score,
                "payload": r.payload,
            })
        return out
