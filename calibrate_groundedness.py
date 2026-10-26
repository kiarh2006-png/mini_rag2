from src.mini_rag.embedding import Embedder
from src.mini_rag.evaluation import check_semantic_groundedness

embedder = Embedder()
context = [{"text": "مجازات این عمل حبس از یک تا سه سال است."}]

cases = [
    ("verbatim", "مجازات این عمل حبس از یک تا سه سال است."),
    ("paraphrase_grounded", "برای این عمل مجازات حبس در نظر گرفته شده است."),
    ("unrelated", "امروز هوا آفتابی و مناسب پیاده‌روی است."),
    ("high_overlap_wrong_claim", "مجازات این عمل اعدام از یک تا سه سال است."),
    ("high_overlap_negation", "مجازات این عمل حبس از یک تا سه سال نیست."),
]

for label, sentence in cases:
    r = check_semantic_groundedness(sentence, context, embedder, threshold=0.0)
    score = r["claims"][0]["score"]
    print(f"{label}: score={score:.3f}  -> {sentence}")