from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class PageText:
    """Text of one page. page numbers start from 1."""

    page: int
    text: str


def extract_text(path):
    """Extract text from a PDF or TXT file, returned as a list of PageText."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".txt":
        return _extract_txt(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _extract_pdf(path):
    pages = []
    with pymupdf.open(path) as doc:
        for number, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                pages.append(PageText(page=number, text=text))
    return pages


def _extract_txt(path):
    # utf-8-sig also handles files saved with a BOM by Notepad
    text = path.read_text(encoding="utf-8-sig")
    return [PageText(page=1, text=text)] if text.strip() else []