import json

from src.mini_rag.metadata import build_metadata, load_catalog


def test_load_catalog_returns_empty_dict_if_file_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    assert load_catalog(missing_path) == {}


def test_load_catalog_reads_real_file(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps({"a.pdf": {"title": "سند الف", "doc_type": "law", "source": "x"}}),
        encoding="utf-8",
    )
    catalog = load_catalog(catalog_path)
    assert catalog["a.pdf"]["title"] == "سند الف"


def test_build_metadata_uses_catalog_entry(tmp_path):
    file_path = tmp_path / "penal_code.pdf"
    file_path.write_bytes(b"")
    catalog = {
        "penal_code.pdf": {
            "title": "قانون مجازات اسلامی",
            "doc_type": "law",
            "source": "rc.majlis.ir",
        }
    }
    meta = build_metadata(file_path, catalog)

    assert meta.document_id == "penal_code"
    assert meta.filename == "penal_code.pdf"
    assert meta.file_type == "pdf"
    assert meta.doc_type == "law"
    assert meta.title == "قانون مجازات اسلامی"
    assert meta.source == "rc.majlis.ir"


def test_build_metadata_falls_back_when_not_in_catalog(tmp_path):
    file_path = tmp_path / "unknown.txt"
    file_path.write_bytes(b"")
    meta = build_metadata(file_path, catalog={})

    assert meta.document_id == "unknown"
    assert meta.doc_type == "unknown"
    assert meta.title == "unknown"
    assert meta.source == ""