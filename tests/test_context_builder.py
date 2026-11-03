
from src.context.context_builder import ContextBuilder


def test_context_builder():

    print("\n========== TEST CONTEXT BUILDER ==========")

    builder = ContextBuilder()

    documents = [
        {
            "id": "A105",
            "text": (
                "Fracture des deux os de l'avant bras, "
                "ou fracture de l'un et luxation de l'autre."
            ),
            "metadata": {
                "title": "Titre premier :actes de traitement des traumatismes",
                "chapter": "A - Chapitre premier : Fractures",
                "article": "Article 2:",
                "article_description": (
                    "Traitement orthopédique complet, quelle qu'en soit "
                    "la technique, d'une fracture fermée nécessitant "
                    "une réduction avec ou sans anesthésie ou une "
                    "extension continue."
                ),
                "code": "A105",
                "description": (
                    "Fracture des deux os de l'avant bras, "
                    "ou fracture de l'un et luxation de l'autre"
                ),
                "montant": "60",
                "note": (
                    "Les cotations comprennent l'immobilisation ou "
                    "l'appareillage postopératoire éventuel. "
                    "Des clichés radiographiques pris avant et après "
                    "le traitement doivent être fournis."
                )
            },
            "rrf_score": 0.032522
        }
    ]

    # Test 1 : construction du contexte
    context = builder.build(documents)

    print("\nCONTEXTE GÉNÉRÉ :")
    print(context)

    assert "A105" in context
    assert "60 DH" in context
    assert "Titre premier" in context
    assert "Chapitre premier : Fractures" in context
    assert "Article 2:" in context
    assert "article fermé" not in context  # Vérification de la formulation exacte

    # Test 2 : construction des sources
    sources = builder.build_sources(documents)

    print("\nSOURCES GÉNÉRÉES :")
    print(sources)

    assert len(sources) == 1
    assert sources[0]["code"] == "A105"
    assert sources[0]["montant"] == "60"
    assert sources[0]["rrf_score"] == 0.032522
    assert sources[0]["title"].startswith("Titre premier")
    assert sources[0]["chapter"] == "A - Chapitre premier : Fractures"
    assert sources[0]["article"] == "Article 2:"
    assert sources[0]["article_description"] != ""

    print("\nTEST RÉUSSI")


if __name__ == "__main__":
    test_context_builder()
