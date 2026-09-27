import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentMetadata:
    document_id: str
    filename: str
    file_type: str  # "pdf" or "txt"
    doc_type: str  # "law", "regulation", ... or "unknown"
    title: str
    source: str


def load_catalog(path):
    """Load the catalog file; return an empty dict if it does not exist."""
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_metadata(path, catalog):
    """Build document-level metadata from the file and the catalog."""
    path = Path(path)
    entry = catalog.get(path.name, {})
    return DocumentMetadata(
        document_id=path.stem,
        filename=path.name,
        file_type=path.suffix.lower().lstrip("."),
        doc_type=entry.get("doc_type", "unknown"),
        title=entry.get("title", path.stem),
        source=entry.get("source", ""),
    )