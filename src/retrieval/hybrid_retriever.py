
from src.retrieval.retriever import SemanticRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.code_retriever import CodeRetriever
from src.utils.config import Config
from src.utils.logger import Logger


class HybridRetriever:

    def __init__(self):
        self.logger = Logger.get_logger("HybridRetriever")

        self.semantic_retriever = SemanticRetriever()
        self.bm25_retriever = BM25Retriever()
        self.code_retriever = CodeRetriever()

        self.rrf_k = 60

        self.logger.info("HybridRetriever initialisé.")

    def search(self, query: str, top_k: int = None) -> list[dict]:

        if not isinstance(query, str):
            raise TypeError("La requête doit être une chaîne de caractères.")

        query = query.strip()

        if not query:
            raise ValueError("La requête ne peut pas être vide.")

        if top_k is None:
            top_k = Config.RETRIEVAL_TOP_K

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k doit être un entier strictement positif.")

        self.logger.info(f"Recherche hybride : {query}")

        # 1. Vérifier la présence d'un code explicite
        code = self.code_retriever.extract_code(query)

        if code is not None:

            self.logger.info(f"Code détecté : {code}")

            code_results = self.code_retriever.search(query)

            if code_results:
                self.logger.info(
                    "Correspondance exacte trouvée. "
                    "La recherche hybride n'est pas nécessaire."
                )

                for document in code_results:
                    document["retrieval_method"] = "code"

                return code_results[:top_k]

            self.logger.warning(
                f"Aucun document trouvé pour le code {code}."
            )

            return []

        # 2. Recherche sémantique
        semantic_results = self.semantic_retriever.search(
            query,
            top_k=top_k
        )

        # 3. Recherche lexicale BM25
        bm25_results = self.bm25_retriever.search(
            query,
            top_k=top_k
        )

        # 4. Fusion des résultats avec RRF
        fused_results = {}

        retriever_results = [
            semantic_results,
            bm25_results
        ]

        for results in retriever_results:

            for rank, document in enumerate(results, start=1):

                document_id = document["id"]

                if document_id not in fused_results:

                    fused_results[document_id] = {
                        "id": document_id,
                        "text": document["text"],
                        "metadata": document["metadata"],
                        "rrf_score": 0.0,
                        "retrieval_method": "hybrid"
                    }

                fused_results[document_id]["rrf_score"] += (
                    1 / (self.rrf_k + rank)
                )

        # 5. Classement final
        ranked_results = sorted(
            fused_results.values(),
            key=lambda document: document["rrf_score"],
            reverse=True
        )

        final_results = ranked_results[:top_k]

        self.logger.info(
            f"{len(final_results)} documents récupérés après fusion."
        )

        return final_results
