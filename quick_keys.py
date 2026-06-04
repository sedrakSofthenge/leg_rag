# quick_keys.py
from qdrant_client import QdrantClient
from ingest import load_cfg

cfg = load_cfg("cfg.yaml")
client = QdrantClient(path=cfg.qdrant.get("path", cfg.paths["vdb_path"]))
coll = cfg.qdrant.get("collection")

pts, _ = client.scroll(collection_name=coll, with_payload=True, limit=10)
for p in pts:
    print(sorted(p.payload.keys()))
    print((p.payload.get("law_title"), p.payload.get("article"), p.payload.get("clause")))
    print((p.payload.get("text") or "")[:200].replace("\n"," "))
    print("---")
