
from src.pipeline.rag_pipeline import RAGPipeline


def test_rag_integration():

    print("\n========== TEST RAG INTEGRATION ==========")

    # Initialisation du véritable pipeline
    pipeline = RAGPipeline()

    # Question réelle
    question = "Quel est le tarif de l'acte A105 ?"

    print(f"\nQUESTION : {question}")

    # Exécution complète du RAG
    result = pipeline.run(
        question=question,
        top_k=5
    )

    # Affichage de la réponse
    print("\n========== RÉPONSE DU LLM ==========")
    print(result["answer"])

    # Affichage des sources
    print("\n========== SOURCES RETROUVÉES ==========")

    for index, source in enumerate(result["sources"], start=1):

        print(f"\nSOURCE {index}")
        print(f"Code : {source['code']}")
        print(f"Titre : {source['title']}")
        print(f"Chapitre : {source['chapter']}")
        print(f"Article : {source['article']}")
        print(f"Description : {source['description']}")
        print(f"Tarif : {source['montant']} DH")
        print(f"Score RRF : {source['rrf_score']}")

    print("\n========== STATISTIQUES ==========")
    print(f"Documents récupérés : {result['retrieval_count']}")

    # Vérifications structurelles
    assert isinstance(result, dict)
    assert isinstance(result["answer"], str)
    assert result["answer"].strip() != ""
    assert isinstance(result["sources"], list)
    assert result["retrieval_count"] == len(result["sources"])

    print("\nTEST D'INTÉGRATION RÉUSSI")


if __name__ == "__main__":
    test_rag_integration()
