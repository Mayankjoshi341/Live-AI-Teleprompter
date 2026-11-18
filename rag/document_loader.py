from __future__ import annotations

import io
from pathlib import Path


class DocumentLoadError(Exception):
    """Raised when an uploaded document cannot be read as text."""


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentLoadError("The text file could not be decoded. Save it as UTF-8 and try again.")


def _clean(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def extract_document(file_name: str, data: bytes) -> list[dict[str, str]]:
    """Extract a document into text segments with a source locator."""
    suffix = Path(file_name).suffix.lower()
    if not data:
        raise DocumentLoadError("The uploaded file is empty.")

    if suffix in {".txt", ".md"}:
        text = _clean(_decode_text(data))
        if not text:
            raise DocumentLoadError("No text was found in this file.")
        return [{"text": text, "locator": "text file"}]

    if suffix == ".pdf":
        try:
            import fitz

            with fitz.open(stream=data, filetype="pdf") as pdf:
                segments = []
                for page_number, page in enumerate(pdf, start=1):
                    page_text = _clean(page.get_text("text"))
                    if page_text:
                        segments.append({"text": page_text, "locator": f"page {page_number}"})
        except Exception as exc:
            raise DocumentLoadError(
                "The PDF could not be opened. It may be invalid, password-protected, or damaged."
            ) from exc
        if not segments:
            raise DocumentLoadError(
                "No selectable text was found in this PDF. It may be scanned/image-only; OCR is not enabled yet."
            )
        return segments

    if suffix == ".docx":
        try:
            from docx import Document

            document = Document(io.BytesIO(data))
            parts: list[str] = []
            for paragraph in document.paragraphs:
                value = _clean(paragraph.text)
                if value:
                    parts.append(value)
            for table_index, table in enumerate(document.tables, start=1):
                for row in table.rows:
                    cells = [_clean(cell.text) for cell in row.cells]
                    row_text = " | ".join(cell for cell in cells if cell)
                    if row_text:
                        parts.append(f"[Table {table_index}] {row_text}")
            for section_index, section in enumerate(document.sections, start=1):
                for paragraph in section.header.paragraphs:
                    value = _clean(paragraph.text)
                    if value:
                        parts.append(f"[Header {section_index}] {value}")
                for paragraph in section.footer.paragraphs:
                    value = _clean(paragraph.text)
                    if value:
                        parts.append(f"[Footer {section_index}] {value}")
            text = "\n".join(parts).strip()
        except Exception as exc:
            raise DocumentLoadError(
                "The DOCX file could not be opened. Save it as a .docx file from Word and try again. Legacy .doc files are not supported."
            ) from exc
        if not text:
            raise DocumentLoadError("No readable text was found in this DOCX file.")
        return [{"text": text, "locator": "document"}]

    raise DocumentLoadError(
        f"Unsupported file type: {suffix or 'unknown'}. Upload PDF, DOCX, TXT or Markdown. Legacy .doc files are not supported."
    )
