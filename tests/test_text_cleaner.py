from src.preprocessing.text_cleaner import TextCleaner


def test_text_cleaner():

    cleaner = TextCleaner()

    text = """
    Traitement des fractures vertébrales
    --------------------------------
    Consultation médicale — examen clinique.

    Contact : medecin@example.com
    Site : https://example.com
    """

    cleaned_text = cleaner.clean(text)

    print("TEXTE ORIGINAL :")
    print(text)

    print("\nTEXTE NETTOYÉ :")
    print(cleaned_text)

    # Vérifications
    assert "medecin@example.com" not in cleaned_text
    assert "https://example.com" not in cleaned_text
    assert "fractures vertébrales" in cleaned_text
    assert "Consultation médicale" in cleaned_text

    print("\nTest TextCleaner réussi !")


if __name__ == "__main__":
    test_text_cleaner()