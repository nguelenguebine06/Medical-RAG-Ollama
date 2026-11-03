
from src.retrieval.code_retriever import CodeRetriever


def test_code_retriever():

    retriever = CodeRetriever()

    # Test 1 : code existant
    results = retriever.search("Quel est le tarif de A105 ?")
    assert len(results) == 1
    assert results[0]["id"] == "A105"

    # Test 2 : autre code existant
    results = retriever.search("Donne-moi les informations de D527")
    assert len(results) == 1
    assert results[0]["id"] == "D527"

    # Test 3 : code inexistant
    results = retriever.search("Quel est le tarif de Z999 ?")
    assert results == []

    # Test 4 : absence de code
    results = retriever.search("Quel est le tarif d'une fracture ?")
    assert results == []

    print("Tous les tests CodeRetriever sont validés.")


if __name__ == "__main__":
    test_code_retriever()
