from pathlib import Path

from src.ingestion.loaders.docx_loader import DocxLoader
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.chunking.medical_chunker import MedicalChunker
from src.embeddings.embedding_service import EmbeddingService
from src.vectordb.chroma_client import ChromaClient


def test_chroma_insertion():

    # 1. Ingestion
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

    # 4. Embeddings
    embedding_service = EmbeddingService()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_service.encode(texts)

    # 5. Insertion ChromaDB
    chroma = ChromaClient()

    chroma.add_documents(chunks, embeddings)

    # 6. Vérification
    collection = chroma.get_collection()

    print("\n========== TEST INSERTION CHROMADB ==========")
    print("Nombre de chunks :", len(chunks))
    print("Nombre d'embeddings :", len(embeddings))
    print("Documents dans ChromaDB :", collection.count())

    # Vérification de A100
    result = collection.get(
        ids=["A100"],
        include=["documents", "metadatas"]
    )

    print("\n========== VERIFICATION A100 ==========")

    print("Document :", result["documents"][0])
    print("Métadonnées :", result["metadatas"][0])

    assert collection.count() == 736
    assert len(result["ids"]) == 1
    assert result["metadatas"][0]["code"] == "A100"

    print("\nTest insertion ChromaDB réussi !")


if __name__ == "__main__":
    test_chroma_insertion()