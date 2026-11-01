from pathlib import Path
import chromadb

from src.utils.config import Config


class ChromaClient:

    def __init__(self, collection_name: str = None):

        self.persist_directory = Config.CHROMA_DIR

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.collection_name = (
            collection_name or Config.CHROMA_COLLECTION
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": Config.CHROMA_DISTANCE}
        )

    def get_collection(self):
        return self.collection

    def get_collection_count(self) -> int:
        return self.collection.count()

    def add_documents(
        self,
        chunks: list[dict],
        embeddings: list[list[float]]
    ):

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Le nombre de chunks doit correspondre "
                "au nombre d'embeddings."
            )

        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:

            metadata = chunk["metadata"].copy()

            document_id = metadata.get("code")

            if not document_id:
                raise ValueError(
                    "Chaque chunk doit posséder un code unique."
                )

            clean_metadata = {}

            for key, value in metadata.items():

                if value is None:
                    continue

                if isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                else:
                    clean_metadata[key] = str(value)

            ids.append(document_id)
            documents.append(chunk["text"])
            metadatas.append(clean_metadata)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )


if __name__ == "__main__":

    chroma = ChromaClient()

    print("Dossier :", chroma.persist_directory)
    print("Collection :", chroma.collection_name)
    print("Nombre de documents :", chroma.get_collection_count())