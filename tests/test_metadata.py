from src.vectordb.chroma_client import ChromaClient


def test_metadata():

    client = ChromaClient()

    result = client.collection.get(
        ids=["A105"],
        include=["metadatas"]
    )

    print("\n========== MÉTADONNÉES CHROMADB ==========")

    if result["metadatas"]:
        print(result["metadatas"][0])
    else:
        print("Document A105 introuvable.")


if __name__ == "__main__":
    test_metadata()