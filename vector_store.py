import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL

STORE_DIR = Path("vector_store")
INDEX_PATH = STORE_DIR / "documents.faiss"
METADATA_PATH = STORE_DIR / "metadata.json"


class LocalVectorStore:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build(self, chunks: list[dict]):
        if not chunks:
            raise ValueError("There is no text to index.")

        texts = [item["text"] for item in chunks]
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks = chunks
        self.save()

    def save(self):
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_PATH))
        METADATA_PATH.write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def load(self) -> bool:
        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            return False
        self.index = faiss.read_index(str(INDEX_PATH))
        self.chunks = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        return True

    def search(self, question: str, top_k: int = 4) -> list[dict]:
        if self.index is None:
            raise RuntimeError("Build or load the vector index first.")

        query_vector = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(
            query_vector, min(top_k, len(self.chunks))
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = dict(self.chunks[int(idx)])
            item["score"] = float(score)
            results.append(item)
        return results
