
from src.pipeline.rag_pipeline import RAGPipeline
from src.utils.logger import Logger

def test_rag_pipeline():

    print("\n========== TEST RAG PIPELINE ==========")

    # Création du pipeline sans initialiser les vrais composants
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.logger = Logger.get_logger("RAGPipeline")
    # Simulation du retriever
    class MockRetriever:

        def search(self, query, top_k):
            return [
                {
                    "id": "A105",
                    "text": "Fracture des deux os de l'avant bras.",
                    "metadata": {
                        "title": "Titre premier : actes de traitement des traumatismes",
                        "chapter": "A - Chapitre premier : Fractures",
                        "article": "Article 2:",
                        "article_description": "Traitement orthopédique complet d'une fracture fermée.",
                        "code": "A105",
                        "description": "Fracture des deux os de l'avant bras",
                        "montant": "60",
                        "note": "Des clichés radiographiques doivent être fournis."
                    },
                    "rrf_score": 0.032522
                }
            ]

    # Simulation du ContextBuilder
    class MockContextBuilder:

        def build(self, documents):
            return "Code A105 : fracture des deux os de l'avant bras. Tarif : 60 DH."

        def build_sources(self, documents):
            return [
                {
                    "id": "A105",
                    "code": "A105",
                    "title": "Titre premier",
                    "chapter": "A - Chapitre premier : Fractures",
                    "article": "Article 2:",
                    "description": "Fracture des deux os de l'avant bras",
                    "montant": "60",
                    "rrf_score": 0.032522
                }
            ]

    # Simulation du PromptBuilder
    class MockPromptBuilder:

        def build(self, context, question):
            return f"CONTEXTE : {context}\nQUESTION : {question}"

    # Simulation du LLM
    class MockLLM:

        def generate(self, prompt):
            return "L'acte A105 correspond à une fracture des deux os de l'avant bras. Son tarif est de 60 DH."

    # Injection des composants simulés
    pipeline.retriever = MockRetriever()
    pipeline.context_builder = MockContextBuilder()
    pipeline.prompt_builder = MockPromptBuilder()
    pipeline.llm = MockLLM()

    # Exécution
    result = pipeline.run(
        question="Quel est le tarif de l'acte A105 ?",
        top_k=5
    )

    # Affichage
    print("\nRÉPONSE :")
    print(result["answer"])

    print("\nSOURCES :")
    print(result["sources"])

    print("\nNOMBRE DE DOCUMENTS :")
    print(result["retrieval_count"])

    # Vérifications
    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    assert "retrieval_count" in result

    assert "60 DH" in result["answer"]
    assert result["retrieval_count"] == 1
    assert result["sources"][0]["code"] == "A105"

    print("\nTEST RÉUSSI")


def test_rag_pipeline_no_results():

    print("\n========== TEST RAG PIPELINE - NO RESULTS ==========")

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.logger = Logger.get_logger("RAGPipeline")

    # Retriever qui ne retourne aucun document
    class MockRetriever:

        def search(self, query, top_k):
            return []

    # Composant LLM qui détecte tout appel inattendu
    class MockLLM:

        def generate(self, prompt):
            raise AssertionError(
                "Le LLM ne doit pas être appelé sans document."
            )

    pipeline.retriever = MockRetriever()
    pipeline.llm = MockLLM()

    # Exécution
    result = pipeline.run(
        question="Quel est le tarif de l'acte Z999 ?",
        top_k=5
    )

    print("\nRÉPONSE :")
    print(result["answer"])

    print("\nSOURCES :")
    print(result["sources"])

    print("\nNOMBRE DE DOCUMENTS :")
    print(result["retrieval_count"])

    # Vérifications
    assert isinstance(result, dict)
    assert result["answer"] != ""
    assert result["sources"] == []
    assert result["retrieval_count"] == 0

    print("\nTEST NO RESULTS RÉUSSI")

if __name__ == "__main__":
    test_rag_pipeline()
    test_rag_pipeline_no_results()
