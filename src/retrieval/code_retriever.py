
import re

from src.vectordb.chroma_client import ChromaClient
from src.utils.logger import Logger


class CodeRetriever:

    def __init__(self):
        self.logger = Logger.get_logger("CodeRetriever")

        self.chroma_client = ChromaClient()
        self.collection = self.chroma_client.get_collection()

        self.logger.info("CodeRetriever initialisé.")

    def extract_code(self, query: str) -> str | None:
        """
        Extrait un code de nomenclature depuis une requête.
        Exemple : 'Quel est le tarif de A105 ?' -> 'A105'
        """
        if not isinstance(query, str):
            raise TypeError("La requête doit être une chaîne.")

        pattern = r"\b[A-Z]{1,3}-?\d{1,4}\b"

        match = re.search(pattern, query.upper())

        if match:
            return match.group(0)

        return None

    def search(self, query: str) -> list[dict]:
        """
        Recherche un document directement par son code.
        """
        if not isinstance(query, str):
            raise TypeError("La requête doit être une chaîne.")

        query = query.strip()

        if not query:
            raise ValueError("La requête ne peut pas être vide.")

        code = self.extract_code(query)

        if code is None:
            self.logger.info("Aucun code détecté dans la requête.")
            return []

        self.logger.info(f"Recherche par code : {code}")

        results = self.collection.get(
            ids=[code],
            include=["documents", "metadatas"]
        )

        if not results["ids"]:
            self.logger.warning(f"Code introuvable : {code}")
            return []

        retrieved_documents = []

        for document_id, document, metadata in zip(
            results["ids"],
            results["documents"],
            results["metadatas"]
        ):
            retrieved_documents.append({
                "id": document_id,
                "text": document,
                "metadata": metadata
            })

        self.logger.info(
            f"{len(retrieved_documents)} document(s) récupéré(s)."
        )

        return retrieved_documents
