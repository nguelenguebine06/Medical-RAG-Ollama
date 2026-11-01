from pathlib import Path

from src.ingestion.loaders.docx_loader import DocxLoader
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline


def test_preprocessing_pipeline():

    # Chemin du document
    base_dir = Path(__file__).resolve().parent.parent
    document_path = base_dir / "src" / "documents" / "nomenclatures.docx"

    # 1. Ingestion
    loader = DocxLoader()
    documents = loader.load(str(document_path))

    # 2. Preprocessing
    pipeline = PreprocessingPipeline()
    processed_documents = pipeline.process(documents)

    # Affichage des résultats
    print("Documents extraits :", len(documents))
    print("Documents nettoyés :", len(processed_documents))

    print("\nPremier document après preprocessing :")
    print(processed_documents[0]["text"])

    print("\nMétadonnées conservées :")
    print(processed_documents[0]["metadata"])

    # Vérifications
    assert len(documents) == len(processed_documents)
    assert len(processed_documents) > 0

    for document in processed_documents:
        assert "text" in document
        assert "metadata" in document
        assert isinstance(document["text"], str)
        assert isinstance(document["metadata"], dict)

    # Vérifier la conservation des métadonnées du premier document
    assert (
        processed_documents[0]["metadata"]["code"]
        == documents[0]["metadata"]["code"]
    )

    print("\nTest PreprocessingPipeline réussi !")


if __name__ == "__main__":
    test_preprocessing_pipeline()