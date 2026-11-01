from pathlib import Path

from src.ingestion.loaders.docx_loader import DocxLoader
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.chunking.medical_chunker import MedicalChunker
from src.embeddings.embedding_service import EmbeddingService


def test_embedding_service():

    # 1. Chargement des documents
    base_dir = Path(__file__).resolve().parent.parent
    document_path = base_dir / "src" / "documents" / "nomenclatures.docx"

    loader = DocxLoader()
    documents = loader.load(str(document_path))

    # 2. Preprocessing
    preprocessing = PreprocessingPipeline()
    processed_documents = preprocessing.process(documents)

    # 3. Chunking
    chunker = MedicalChunker()
    chunks = chunker.split(processed_documents)

    # 4. Génération des embeddings
    embedding_service = EmbeddingService()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_service.encode(texts)

    # 5. Vérifications
    print("\n========== TEST EMBEDDINGS ==========")

    print("Nombre de chunks :", len(chunks))
    print("Nombre de vecteurs :", len(embeddings))
    print("Dimension :", embedding_service.get_dimension())

    print("\nPremier embedding :")
    print(embeddings[0][:10])

    assert len(embeddings) == len(chunks)
    assert embedding_service.get_dimension() == 384
    assert len(embeddings[0]) == 384

    print("\nTest EmbeddingService réussi !")


if __name__ == "__main__":
    test_embedding_service()