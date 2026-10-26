from src.mini_rag.citation import analyze_citations, parse_citations, split_sentences


def test_parse_citations_extracts_numbers_in_order():
    assert parse_citations("ادعا اول [1] و ادعا دوم [2].") == [1, 2]


def test_valid_citation_within_range():
    sources = [{"n": 1, "title": "t", "page": 1, "chunk_id": "doc-0001"}]
    r = analyze_citations("ادعایی با منبع معتبر [1].", sources)
    assert r["citations"][0]["valid"] is True
    assert r["invalid_citation_rate"] == 0.0


def test_invalid_citation_out_of_range():
    sources = [{"n": 1, "title": "t", "page": 1, "chunk_id": "doc-0001"}]
    r = analyze_citations("ادعایی با منبع نامعتبر [7].", sources)
    assert r["citations"][0]["valid"] is False
    assert r["invalid_citation_rate"] == 1.0


def test_missing_citation_claim_has_no_marker():
    sources = [{"n": 1, "title": "t", "page": 1, "chunk_id": "doc-0001"}]
    r = analyze_citations("این ادعا اصلا منبعی ندارد و باید میسینگ باشد.", sources)
    assert r["claims"][0]["has_citation"] is False
    assert r["missing_citation_rate"] == 1.0


def test_duplicate_citation_detected():
    sources = [{"n": 1, "title": "t", "page": 1, "chunk_id": "doc-0001"}]
    r = analyze_citations("ادعا اول [1]. ادعا دوم هم به همان منبع [1].", sources)
    assert 1 in r["duplicate_numbers"]


def test_citation_coverage_is_fraction_of_valid_claims():
    sources = [{"n": 1, "title": "t", "page": 1, "chunk_id": "doc-0001"}]
    r = analyze_citations("ادعای معتبر با منبع درست [1]. ادعای بدون منبع هیچ‌چیز ندارد.", sources)
    assert r["citation_coverage"] == 0.5


def test_split_sentences_does_not_split_on_comma():
    sentences = split_sentences("جمله‌ی اول با یک، ویرگول داخلش. جمله‌ی دوم جداست.")
    assert len(sentences) == 2