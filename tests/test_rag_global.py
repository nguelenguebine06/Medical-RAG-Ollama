
from src.pipeline.rag_pipeline import RAGPipeline


def test_rag_robustness():

    print("\n========== TEST DE ROBUSTESSE RAG ==========")

    pipeline = RAGPipeline()

    # -----------------------------------------
    # TEST 1 : Recherche exacte par code
    # -----------------------------------------

    print("\n========== TEST 1 : RECHERCHE PAR CODE ==========")

    question_1 = "Quel est le tarif de l'acte A105 ?"

    result_1 = pipeline.run(question_1)

    print(f"\nQuestion : {question_1}")
    print(f"Réponse : {result_1['answer']}")

    assert len(result_1["sources"]) == 1
    assert result_1["sources"][0]["code"] == "A105"
    assert result_1["sources"][0]["montant"] == "60"

    print("TEST 1 RÉUSSI")

    # -----------------------------------------
    # TEST 2 : Recherche sémantique
    # -----------------------------------------

    print("\n========== TEST 2 : RECHERCHE SÉMANTIQUE ==========")

    question_2 = "Quel est le tarif du traitement orthopédique d'une fracture ?"

    result_2 = pipeline.run(question_2)

    print(f"\nQuestion : {question_2}")
    print(f"Réponse : {result_2['answer']}")

    print("\nDocuments retrouvés :")

    for source in result_2["sources"]:
        print(
            f"Code : {source['code']} | "
            f"Tarif : {source['montant']} DH"
        )

    assert len(result_2["sources"]) > 0
    assert result_2["retrieval_count"] == len(result_2["sources"])

    print("TEST 2 RÉUSSI")

    # -----------------------------------------
    # TEST 3 : Code inexistant
    # -----------------------------------------

    print("\n========== TEST 3 : CODE INEXISTANT ==========")

    question_3 = "Quel est le tarif de l'acte Z999 ?"

    result_3 = pipeline.run(question_3)

    print(f"\nQuestion : {question_3}")
    print(f"Réponse : {result_3['answer']}")

    assert result_3["sources"] == []
    assert result_3["retrieval_count"] == 0

    print("TEST 3 RÉUSSI")

    # -----------------------------------------
    # BILAN
    # -----------------------------------------

    print("\n========== BILAN ==========")
    print("Les trois scénarios ont été exécutés.")


if __name__ == "__main__":
    test_rag_robustness()
