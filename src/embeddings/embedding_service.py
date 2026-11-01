from sentence_transformers import SentenceTransformer

from src.utils.config import Config


class EmbeddingService:

    def __init__(self, model_name: str = None):

        self.model_name = model_name or Config.EMBEDDING_MODEL

        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:

        if not texts:
            return []

        if not all(isinstance(text, str) for text in texts):
            raise TypeError(
                "Tous les éléments doivent être des chaînes de caractères."
            )

        embeddings = self.model.encode(
            texts,
            batch_size=Config.EMBEDDING_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        return embeddings.tolist()

    def get_dimension(self) -> int:

        return self.model.get_embedding_dimension()


if __name__ == "__main__":

    service = EmbeddingService()

    print("Modèle :", service.model_name)
    print("Dimension :", service.get_dimension())