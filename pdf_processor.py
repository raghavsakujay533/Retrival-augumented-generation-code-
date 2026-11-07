from io import BytesIO
from pypdf import PdfReader


def extract_pages(pdf_bytes: bytes, filename: str) -> list[dict]:
    """Extract text page-by-page so page citations can be shown in answers."""
    reader = PdfReader(BytesIO(pdf_bytes))

    if reader.is_encrypted:
        raise ValueError(f"{filename} is password-protected.")

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({
                "text": text,
                "source": filename,
                "page": page_number
            })

    if not pages:
        raise ValueError(
            f"No selectable text found in {filename}. "
            "If it is a scanned PDF, run OCR first."
        )
    return pages
