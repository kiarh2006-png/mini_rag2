from src.mini_rag.embedding import Embedder
from src.mini_rag.evaluation import check_required_facts, check_semantic_groundedness, classify_answer

NO_INFO = "در اسناد موجود اطلاعات کافی برای پاسخ به این سؤال پیدا نشد."


def _embedder():
    return Embedder()


def test_classify_abstained_when_answer_equals_no_info():
    e = _embedder()
    status, _ = classify_answer(NO_INFO, ["هر فکتی"], 1.0, e, NO_INFO)
    assert status == "ABSTAINED"


def test_classify_unsupported_when_groundedness_low():
    e = _embedder()
    status, _ = classify_answer("پاسخی کاملا بی‌ربط به context", ["فکت"], 0.1, e, NO_INFO)
    assert status == "UNSUPPORTED"


def test_classify_correct_when_all_facts_present():
    e = _embedder()
    status, fc = classify_answer(
        "مجازات این عمل حبس از یک تا سه سال است.", ["حبس", "یک تا سه سال"], 1.0, e, NO_INFO
    )
    assert status == "CORRECT"
    assert fc["matched_ratio"] == 1.0


def test_classify_partially_correct_when_some_facts_missing():
    e = _embedder()
    status, fc = classify_answer(
        "مجازات این عمل حبس است.", ["حبس", "جزای نقدی"], 1.0, e, NO_INFO
    )
    assert status == "PARTIALLY_CORRECT"
    assert 0 < fc["matched_ratio"] < 1


def test_classify_incorrect_when_no_facts_present():
    e = _embedder()
    status, fc = classify_answer(
        "این موضوع در قانون کار تعریف نشده است.", ["جزای نقدی", "حبس ابد"], 1.0, e, NO_INFO
    )
    assert status == "INCORRECT"
    assert fc["matched_ratio"] == 0.0


def test_groundedness_low_lexical_overlap_high_semantic_support():
    """The exact case the mentor asked to demonstrate: paraphrase with low
    word-overlap but real semantic support should be grounded."""
    e = _embedder()
    context = [{"text": "مجازات این عمل حبس از یک تا سه سال است."}]
    r = check_semantic_groundedness("برای این عمل مجازات حبس در نظر گرفته شده است.", context, e)
    assert r["grounded_ratio"] == 1.0


def test_groundedness_high_lexical_overlap_wrong_claim_limitation():
    """Known limitation: high word-overlap with a factually wrong claim
    (negation) still scores as grounded with the current embedding method."""
    e = _embedder()
    context = [{"text": "مجازات این عمل حبس از یک تا سه سال است."}]
    r = check_semantic_groundedness("مجازات این عمل حبس از یک تا سه سال نیست.", context, e)
    assert r["claims"][0]["score"] > 0.55  # documents the limitation, not a desired behavior