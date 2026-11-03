
import re
from rank_bm25 import BM25Okapi

from src.vectordb.chroma_client import ChromaClient
from src.utils.config import Config
from src.utils.logger import Logger


class BM25Retriever:

    STOPWORDS = {
        "le", "la", "les", "un", "une", "des", "du", "de",
        "d", "au", "aux", "en", "et", "ou", "pour", "par",
        "avec", "sans", "sur", "dans", "est", "sont",
        "quel", "quelle", "quels", "quelles",
        "ce", "cette", "ces", "qui", "que", "quoi",
        "mon", "ma", "mes", "son", "sa", "ses",
        "leur", "leurs", "se", "ne", "pas"
    }

    def __init__(self):
        self.logger = Logger.get_logger("BM25Retriever")

        self.chroma_client = ChromaClient()
        self.collection = self.chroma_client.get_collection()

        self.documents = []
        self.bm25 = None

        self._build_index()

    def tokenize(self, text: str) -> list[str]:
        """
        Nettoie et découpe un texte en tokens
        en supprimant les mots vides.
        """
        if not isinstance(text, str):
            raise TypeError("Le texte doit être une chaîne.")

        text = text.lower()

        tokens = re.findall(r"\b\w+\b", text)

        tokens = [
            token for token in tokens
            if token not in self.STOPWORDS
        ]

        return tokens

    def _build_index(self):
        """
        Récupère les documents ChromaDB et construit
        l'index BM25.
        """
        collection_count = self.collection.count()

        if collection_count == 0:
            self.logger.warning("La collection ChromaDB est vide.")
            return

        results = self.collection.get(
            include=["documents", "metadatas"]
        )

        ids = results["ids"]
        texts = results["documents"]
        metadatas = results["metadatas"]

        tokenized_corpus = []

        for document_id, text, metadata in zip(
            ids, texts, metadatas
        ):
            tokens = self.tokenize(text)

            if not tokens:
                continue

            self.documents.append({
                "id": document_id,
                "text": text,
                "metadata": metadata
            })

            tokenized_corpus.append(tokens)

        if not tokenized_corpus:
            self.logger.warning("Aucun document exploitable.")
            return

        self.bm25 = BM25Okapi(tokenized_corpus)

        self.logger.info(
            f"Index BM25 construit avec {len(self.documents)} documents."
        )

    def search(self, query: str, top_k: int = None) -> list[dict]:
        """
        Recherche les documents les plus pertinents
        avec BM25.
        """
        if not isinstance(query, str):
            raise TypeError("La requête doit être une chaîne.")

        query = query.strip()

        if not query:
            raise ValueError("La requête ne peut pas être vide.")

        if self.bm25 is None:
            self.logger.warning("L'index BM25 n'est pas disponible.")
            return []

        if top_k is None:
            top_k = Config.RETRIEVAL_TOP_K

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(
                "top_k doit être un entier strictement positif."
            )

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = scores.argsort()[::-1][:top_k]

        retrieved_documents = []

        for index in ranked_indices:
            score = float(scores[index])

            if score <= 0:
                continue

            document = self.documents[index]

            retrieved_documents.append({
                "id": document["id"],
                "text": document["text"],
                "metadata": document["metadata"],
                "score": score
            })

        self.logger.info(
            f"{len(retrieved_documents)} documents récupérés avec BM25."
        )

        return retrieved_documents

