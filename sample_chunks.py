import random
from collections import defaultdict
from pathlib import Path

from src.mini_rag.vector_store import VectorStore

CHUNKS_PER_DOC = 3
MIN_LENGTH = 400
PREVIEW_LENGTH = 450
SEED = 42
OUTPUT_PATH = Path("data/eval/chunk_samples.txt")


def main():
    chunks = VectorStore().get_all()

    by_document = defaultdict(list)
    for chunk in chunks:
        text = chunk["text"]
        if len(text) >= MIN_LENGTH and "http" not in text:
            by_document[chunk["metadata"]["document_id"]].append(chunk)

    rng = random.Random(SEED)
    lines = []
    total = 0
    for document_id in sorted(by_document):
        candidates = sorted(by_document[document_id], key=lambda c: c["chunk_id"])
        picked = rng.sample(candidates, min(CHUNKS_PER_DOC, len(candidates)))
        total += len(picked)
        for chunk in sorted(picked, key=lambda c: c["chunk_id"]):
            meta = chunk["metadata"]
            lines.append(
                f"### {chunk['chunk_id']} | {meta['title']} | page {meta['page']}"
            )
            lines.append(chunk["text"][:PREVIEW_LENGTH])
            lines.append("")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {total} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()