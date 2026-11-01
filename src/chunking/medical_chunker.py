from typing import Any

from src.chunking.base_chunker import BaseChunker


class MedicalChunker(BaseChunker):

    def split( self, documents: list[dict[str, Any]] ) -> list[dict[str, Any]]:

        chunks = []

        for document in documents:

            text = document["text"].strip()
            metadata = document["metadata"].copy()

            if not text:
                continue

            chunk = {
                "text": text,
                "metadata": metadata
            }

            chunks.append(chunk)

        return chunks