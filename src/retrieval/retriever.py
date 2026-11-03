from src.embeddings.embedding_service import EmbeddingService
from src.vectordb.chroma_client import ChromaClient
from src.utils.config import Config
from src.utils.logger import Logger


class SemanticRetriever:

    def __init__(self):

        self.logger = Logger.get_logger("SemanticRetriever")

        # Service de génération des embeddings
        self.embedding_service = EmbeddingService()

        # Connexion à ChromaDB
        self.chroma_client = ChromaClient()

        self.collection = self.chroma_client.get_collection()

        self.logger.info(
            "SemanticRetriever initialisé."
        )

    def search(
        self,
        query: str,
        top_k: int = None
    ) -> list[dict]:

        if not isinstance(query, str):
            raise TypeError(
                "La requête doit être une chaîne de caractères."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "La requête ne peut pas être vide."
            )

        # Nombre de résultats à retourner
        top_k = top_k or Config.RETRIEVAL_TOP_K

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(
                "top_k doit être un entier strictement positif."
            )

        collection_count = self.collection.count()

        if collection_count == 0:
            self.logger.warning(
                "La collection ChromaDB est vide."
            )
            return []

        top_k = min(top_k, collection_count)

        self.logger.info(
            f"Recherche sémantique : {query}"
        )

        # 1. Transformer la question en embedding
        query_embedding = self.embedding_service.encode(
            [query]
        )

        # 2. Recherche vectorielle dans ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # 3. Structurer les résultats
        retrieved_documents = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for document_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances
        ):

            retrieved_documents.append({

                "id": document_id,

                "text": document,

                "metadata": metadata,

                "distance": distance,

                # Distance cosinus -> similarité
                "similarity": 1 - distance

            })

        self.logger.info(
            f"{len(retrieved_documents)} documents récupérés."
        )

        return retrieved_documents
