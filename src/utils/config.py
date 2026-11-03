from pathlib import Path


class Config:

    # =====================================
    # GENERAL
    # =====================================
    PROJECT_NAME = "Projet_RAG_Ollama"

    BASE_DIR = Path(__file__).resolve().parents[2]

    # =====================================
    # DOCUMENTS
    # =====================================
    DOCUMENTS_DIR = BASE_DIR / "src" / "documents"
    NOMENCLATURE_FILE = DOCUMENTS_DIR / "nomenclatures.docx"

    # =====================================
    # EMBEDDINGS
    # =====================================
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    EMBEDDING_BATCH_SIZE = 32

    # =====================================
    # VECTOR DATABASE
    # =====================================
    CHROMA_DIR = BASE_DIR / "chroma_data"
    CHROMA_COLLECTION = "rag_documents"
    CHROMA_DISTANCE = "cosine"

    # =====================================
    # LLM - OLLAMA
    # =====================================
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "qwen2.5:3b"

    # =====================================
    # RETRIEVAL
    # =====================================
    RETRIEVAL_TOP_K = 5