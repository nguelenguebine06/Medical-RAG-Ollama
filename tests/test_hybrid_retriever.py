
from src.retrieval.hybrid_retriever import HybridRetriever


def test_hybrid_retriever():

    retriever = HybridRetriever()

    query = "Quel est le tarif pour le traitement d'une fracture ?"

    results = retriever.search(query, top_k=5)

    print("\n========== RÉSULTATS HYBRIDES ==========")

    for rank, document in enumerate(results, start=1):

        print(f"\nRang : {rank}")
        print(f"Code : {document['id']}")
        print(f"Titre : {document['metadata'].get('title', 'Non disponible')}")
        print(f"Score RRF : {document['rrf_score']:.6f}")

    assert len(results) > 0
    assert len(results) <= 5

    # Vérifier que les documents sont classés par score décroissant
    scores = [document["rrf_score"] for document in results]

    assert scores == sorted(scores, reverse=True)

    print("\nTous les tests HybridRetriever sont validés.")


if __name__ == "__main__":
    test_hybrid_retriever()
