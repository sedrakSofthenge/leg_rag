from utils.pdf_parser import segment_armenian_law


def test_segment_basic_articles_and_clauses():
    pages = [
        "ԳԼՈՒԽ 1 Ընդհանուր դրույթներ\nՀոդված 1 Սահմանում\n1) Սա փորձնական առաջին կետն է.\n2) Սա երկրորդ կետն է.",
        "Հոդված 2 Իրավական կարգավորում\n1) Կարգավորումը վերաբերում է ...\n2) Լրացուցիչ դրույթ",
    ]
    segs = segment_armenian_law(pages, source_file="test_law.pdf")
    assert any(s.get("article") == '1' for s in segs)
    assert any(s.get("article") == '2' for s in segs)
    # There should be clause splits
    a1_clauses = [s for s in segs if s.get("article") == '1']
    assert len(a1_clauses) >= 1

