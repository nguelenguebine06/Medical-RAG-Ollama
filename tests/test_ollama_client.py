from src.llm.ollama_client import OllamaClient


def test_ollama_generation():

    print("\n========== TEST OLLAMA CLIENT ==========")

    client = OllamaClient()

    prompt = """
    Tu es un assistant spécialisé en intelligence artificielle.

    Explique simplement le fonctionnement d'un système RAG.
    Réponds en français en trois phrases maximum.
    """

    print("\nQuestion envoyée :")
    print(prompt)

    response = client.generate(prompt)

    print("\nRéponse de Qwen :")
    print(response)

    assert isinstance(response, str)
    assert len(response) > 0

    print("\nTEST RÉUSSI")


if __name__ == "__main__":
    test_ollama_generation()