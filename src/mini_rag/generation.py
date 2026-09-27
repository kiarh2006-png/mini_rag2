import ollama

MODEL_NAME = "qwen2.5:3b-instruct"

NO_INFO_MESSAGE = "در اسناد موجود اطلاعات کافی برای پاسخ به این سؤال پیدا نشد."

_SYSTEM_PROMPT = (
    "تو یک دستیار حقوقی هستی که فقط بر اساس متن‌های داده‌شده پاسخ می‌دهی. "
    "قوانین:\n"
    "1. فقط از اطلاعات موجود در متن‌های شماره‌گذاری‌شده استفاده کن، از دانش خودت چیزی اضافه نکن.\n"
     "2. بعد از هر جمله، شماره‌ی متنی که از آن برداشته‌ای را بنویس، مثل [1] یا [2] یا [3].\n"
    "3. اگر پاسخ سؤال در متن‌های داده‌شده نیست، دقیقاً همین جمله را بنویس و چیز دیگری اضافه نکن: "
    f'"{NO_INFO_MESSAGE}"'
)


def _build_context(chunks):
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        source = f"{chunk['metadata']['title']}، صفحه {chunk['metadata']['page']}"
        lines.append(f"[{i}] (منبع: {source})\n{chunk['text']}")
    return "\n\n".join(lines)


def generate_answer(query, chunks, model_name=MODEL_NAME):
    """Generate a grounded answer with citations, or the no-info message.

    `chunks` are the hits returned by Retriever.retrieve (already filtered
    by min_score). If empty, the LLM is never called.
    """
    if not chunks:
        return NO_INFO_MESSAGE, []

    context = _build_context(chunks)
    user_prompt = f"متن‌ها:\n{context}\n\nسؤال: {query}"

    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0},
    )

    sources = [
        {
            "n": i,
            "title": c["metadata"]["title"],
            "page": c["metadata"]["page"],
            "chunk_id": c["chunk_id"],
        }
        for i, c in enumerate(chunks, start=1)
    ]
    return response["message"]["content"], sources