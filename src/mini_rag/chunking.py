from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    title: str
    doc_type: str
    page: int
    chunk_index: int
    text: str


def _joined_len(units):
    """Length of the units once joined with single spaces."""
    return sum(len(u) for u in units) + max(len(units) - 1, 0)


def _split_units(text, max_len):
    """Split text into non-empty lines; hard-split lines longer than max_len."""
    units = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        while len(line) > max_len:
            cut = line.rfind(" ", 0, max_len)
            if cut <= 0:
                cut = max_len
            units.append(line[:cut])
            line = line[cut:].strip()
        units.append(line)
    return units


def _tail(units, overlap):
    """Last whole units whose total length fits in `overlap`."""
    tail = []
    for unit in reversed(units):
        if _joined_len([unit] + tail) > overlap:
            break
        tail.insert(0, unit)
    return tail


def split_into_chunks(text, chunk_size=1000, overlap=150):
    """Split one text into overlapping chunks (list of strings)."""
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks = []
    current = []
    for unit in _split_units(text, chunk_size):
        if current and _joined_len(current + [unit]) > chunk_size:
            chunks.append(" ".join(current))
            current = _tail(current, overlap)
            while current and _joined_len(current + [unit]) > chunk_size:
                current.pop(0)
        current.append(unit)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(pages, metadata, chunk_size=1000, overlap=150):
    """Chunk every page of a document; chunks never cross page boundaries.

    `pages` must already be normalized (list of PageText).
    """
    chunks = []
    index = 0
    for page in pages:
        for text in split_into_chunks(page.text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{metadata.document_id}-{index:04d}",
                    document_id=metadata.document_id,
                    filename=metadata.filename,
                    title=metadata.title,
                    doc_type=metadata.doc_type,
                    page=page.page,
                    chunk_index=index,
                    text=text,
                )
            )
            index += 1
    return chunks