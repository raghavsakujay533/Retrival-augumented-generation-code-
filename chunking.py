import re
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_pages_into_chunks(pages: list[dict]) -> list[dict]:
    """Simple character-window chunking while retaining source/page metadata."""
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for page in pages:
        text = re.sub(r"\s+", " ", page["text"]).strip()
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": page["source"],
                    "page": page["page"]
                })
            if end >= len(text):
                break
            start += step

    return chunks
