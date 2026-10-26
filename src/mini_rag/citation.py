import re

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT = re.compile(r"[.\n؟!]")
_MIN_SENTENCE_LEN = 8


def split_sentences(text):
    """Split free text into claim-sized sentences (shared by citation + groundedness)."""
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if len(s.strip()) > _MIN_SENTENCE_LEN]


def parse_citations(text):
    """Return every citation number found in the text, in order (duplicates kept)."""
    return [int(n) for n in _CITATION_PATTERN.findall(text)]


def analyze_citations(answer, sources):
    """Validate every citation marker in `answer` against the real `sources` list.

    `sources` is the list returned by generate_answer (each has n, title, page, chunk_id).
    Returns a dict with per-citation details, per-claim coverage, and summary rates.
    """
    num_sources = len(sources)
    sources_by_n = {s["n"]: s for s in sources}

    all_numbers = parse_citations(answer)
    seen = set()
    duplicate_numbers = set()
    citations = []
    for n in all_numbers:
        if n in seen:
            duplicate_numbers.add(n)
        seen.add(n)
        valid = 1 <= n <= num_sources
        record = {"number": n, "valid": valid}
        if valid and n in sources_by_n:
            src = sources_by_n[n]
            record.update(
                document_id=src.get("chunk_id", "").rsplit("-", 1)[0],
                page=src["page"],
                chunk_id=src["chunk_id"],
            )
        citations.append(record)

    claims = []
    for sentence in split_sentences(answer):
        numbers_here = parse_citations(sentence)
        has_citation = len(numbers_here) > 0
        valid_citation = any(1 <= n <= num_sources for n in numbers_here)
        claims.append(
            {
                "text": sentence,
                "has_citation": has_citation,
                "valid_citation": valid_citation,
            }
        )

    total_claims = len(claims)
    claims_with_valid = sum(c["valid_citation"] for c in claims)
    claims_missing = sum(not c["has_citation"] for c in claims)

    total_citations = len(citations)
    valid_citations = sum(c["valid"] for c in citations)
    invalid_citations = total_citations - valid_citations

    return {
        "citations": citations,
        "duplicate_numbers": sorted(duplicate_numbers),
        "claims": claims,
        "citation_coverage": claims_with_valid / total_claims if total_claims else 0.0,
        "valid_citation_rate": valid_citations / total_citations if total_citations else 0.0,
        "invalid_citation_rate": invalid_citations / total_citations if total_citations else 0.0,
        "missing_citation_rate": claims_missing / total_claims if total_claims else 0.0,
    }