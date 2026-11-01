
from pathlib import Path

from src.ingestion.loaders.docx_loader import DocxLoader


def test_docx_loader():

    base_dir = Path(__file__).resolve().parent.parent

    document_path = (
        base_dir
        / "src"
        / "documents"
        / "nomenclatures.docx"
    )

    loader = DocxLoader()

    documents = loader.load(str(document_path))

    print("\n========== RESULTAT EXTRACTION ==========")

    print("Nombre d'actes :", len(documents))

    assert len(documents) == 736

    print("\n========== VERIFICATION A100 ==========")

    act = next(
        (
            document
            for document in documents
            if document["metadata"]["code"] == "A104"
        ),
        None
    )

    assert act is not None

    print("\nTEXTE COMPLET :")
    print(act["text"])

    print("\nMETADONNEES :")

    for key, value in act["metadata"].items():
        print(f"{key} : {value}")

    assert act["metadata"]["chapter"]
    assert act["metadata"]["note"]
    assert act["metadata"]["article"]
    assert act["metadata"]["article_description"]

    print("\nTest DocxLoader réussi !")

    return documents


if __name__ == "__main__":
    test_docx_loader()
