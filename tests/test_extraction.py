import pymupdf
import pytest

from src.mini_rag.extraction import extract_text


def _make_pdf(path, page_texts):
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_extract_txt_returns_single_page(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("سلام دنیا", encoding="utf-8")

    pages = extract_text(file_path)

    assert len(pages) == 1
    assert pages[0].page == 1
    assert pages[0].text == "سلام دنیا"


def test_extract_txt_handles_bom(tmp_path):
    file_path = tmp_path / "sample_bom.txt"
    file_path.write_text("سلام دنیا", encoding="utf-8-sig")

    pages = extract_text(file_path)

    assert pages[0].text == "سلام دنیا"


def test_extract_pdf_returns_one_entry_per_nonempty_page(tmp_path):
    file_path = tmp_path / "sample.pdf"
    _make_pdf(file_path, ["Page one text", "Page two text"])

    pages = extract_text(file_path)

    assert len(pages) == 2
    assert pages[0].page == 1
    assert pages[1].page == 2
    assert "Page one" in pages[0].text
    assert "Page two" in pages[1].text


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_text(tmp_path / "does_not_exist.pdf")


def test_unsupported_extension_raises(tmp_path):
    file_path = tmp_path / "sample.docx"
    file_path.write_bytes(b"")
    with pytest.raises(ValueError):
        extract_text(file_path)