from src.mini_rag.normalization import normalize_text


def test_arabic_letters_converted_to_persian():
    assert normalize_text("كتاب") == "کتاب"
    assert normalize_text("علي") == "علی"


def test_digits_converted_to_english():
    assert normalize_text("۱۲۳") == "123"
    assert normalize_text("١٢٣") == "123"


def test_extra_whitespace_collapsed():
    assert normalize_text("سلام   دنیا") == "سلام دنیا"


def test_diacritics_removed():
    assert normalize_text("مُحَمَّد") == "محمد"


def test_leading_trailing_whitespace_stripped():
    assert normalize_text("  سلام  ") == "سلام"


def test_query_and_document_normalize_the_same_way():
    query = normalize_text("كتاب علي")
    document_text = normalize_text("کتاب علی")
    assert query == document_text