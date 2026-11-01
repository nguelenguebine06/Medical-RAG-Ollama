from pathlib import Path
from src.ingestion.loaders.docx_loader import DocxLoader
from src.chunking.medical_chunker import MedicalChunker


BASE_DIR = Path(__file__).resolve().parent.parent

document_path = (
    BASE_DIR / "src" / "documents" / "nomenclatures.docx"
)

# Ingestion
loader = DocxLoader()
documents = loader.load(str(document_path))

# Chunking
chunker = MedicalChunker()
chunks = chunker.split(documents)

print("Nombre de documents :", len(documents))
print("Nombre de chunks :", len(chunks))

print("\nPremier chunk :")
print(chunks[0])