from src.utils.config import Config


if __name__ == "__main__":

    print("Nom du projet :", Config.PROJECT_NAME)
    print("Modèle d'embedding :", Config.EMBEDDING_MODEL)
    print("Dimension :", Config.EMBEDDING_DIMENSION)
    print("Collection ChromaDB :", Config.CHROMA_COLLECTION)
    print("Modèle Ollama :", Config.OLLAMA_MODEL)
    print("Dossier ChromaDB :", Config.CHROMA_DIR)

    assert Config.NOMENCLATURE_FILE.exists()
    assert Config.EMBEDDING_DIMENSION == 384
    assert Config.CHROMA_COLLECTION == "rag_documents"

    print("\nConfiguration validée.")