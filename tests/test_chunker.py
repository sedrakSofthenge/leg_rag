from utils.chunker import build_chunks


def test_chunk_overlap_and_no_split():
    # Create tiny segments that simulate clauses
    segs = []
    for i in range(10):
        segs.append({
            "text": f"Կետ {i+1}. Սա օրինակային տեքստ է {i+1}.",
            "law_title": "Օրենք",
            "article": "1",
            "clause": str(i+1),
            "page_start": 1,
            "page_end": 1,
            "source_file": "x.pdf",
        })
    chunks = build_chunks(segs, chunk_size_tokens=60, chunk_overlap_tokens=20, seed="test")
    # Should create more than one chunk
    assert len(chunks) >= 2
    # Overlap should include trailing clause from previous chunk
    c1 = chunks[0]
    c2 = chunks[1]
    assert any(cl in c1["metadata"]["clauses_included"] for cl in c2["metadata"]["clauses_included"])  # overlap present

