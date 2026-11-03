
from src.prompts.prompt_builder import PromptBuilder


def test_prompt_builder():

    print("\n========== TEST PROMPT BUILDER ==========")

    builder = PromptBuilder()

    context = """
Code : A105
Description : Traitement des traumatismes.
Tarif : 250 DH.

Code : A109
Description : Traitement d'une fracture.
Tarif : 350 DH.
"""

    question = "Quel est le tarif de l'acte A105 ?"

    prompt = builder.build(
        context=context,
        question=question
    )

    print("\nPROMPT GÉNÉRÉ :")
    print(prompt)

    assert "A105" in prompt
    assert "250 DH" in prompt
    assert question in prompt
    assert "N'invente jamais" in prompt

    print("\nTEST RÉUSSI")


if __name__ == "__main__":
    test_prompt_builder()
