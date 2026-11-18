from io import BytesIO

import fitz
from docx import Document

from rag.document_loader import DocumentLoadError, extract_document


def test_extract_txt():
    result = extract_document("notes.txt", b"Python is readable.\n\nIt is dynamically typed.")
    assert len(result) == 1
    assert "Python is readable." in result[0]["text"]


def test_extract_markdown():
    result = extract_document("notes.md", b"# Python\n\nPython is popular.")
    assert "Python is popular." in result[0]["text"]


def test_extract_pdf_pages():
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Python is a programming language.")
    data = pdf.tobytes()
    pdf.close()
    result = extract_document("notes.pdf", data)
    assert len(result) == 1
    assert result[0]["locator"] == "page 1"
    assert "Python is a programming language" in result[0]["text"]


def test_extract_docx_paragraph_and_table():
    document = Document()
    document.add_paragraph("Django is a Python web framework.")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "ORM"
    table.cell(0, 1).text = "Object Relational Mapper"
    output = BytesIO()
    document.save(output)
    result = extract_document("notes.docx", output.getvalue())
    text = result[0]["text"]
    assert "Django is a Python web framework" in text
    assert "Object Relational Mapper" in text


def test_unsupported_legacy_doc_is_explained():
    try:
        extract_document("legacy.doc", b"not a real doc")
    except DocumentLoadError as exc:
        assert ".doc" in str(exc)
    else:
        raise AssertionError("Expected DocumentLoadError")
