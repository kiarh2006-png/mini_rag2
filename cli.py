from src.mini_rag.embedding import Embedder
from src.mini_rag.vector_store import VectorStore
from src.mini_rag.retrieval import Retriever
from src.mini_rag.generation import generate_answer


def main():
    print("در حال بارگذاری مدل‌ها...")
    embedder = Embedder()
    store = VectorStore()
    retriever = Retriever(embedder, store)
    print(f"آماده. {store.count()} chunk در دسترس است.")
    print("برای خروج Ctrl+C را بزنید.\n")

    while True:
        query = input("سؤال شما: ").strip()
        if not query:
            continue

        hits = retriever.retrieve(query, top_k=5)
        answer, sources = generate_answer(query, hits)

        print("\nپاسخ:")
        print(answer)

        if sources:
            print("\nمنابع:")
            for s in sources:
                print(f"  [{s['n']}] {s['title']}، صفحه {s['page']}")
        print()


if __name__ == "__main__":
    main()