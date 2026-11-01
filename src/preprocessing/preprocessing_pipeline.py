from src.preprocessing.text_cleaner import TextCleaner


class PreprocessingPipeline:

    def __init__(self):
        self.cleaner = TextCleaner()

    def process(self, documents: list[dict]) -> list[dict]:
        processed_documents = []

        for document in documents:

            # Nettoyer le texte
            cleaned_text = self.cleaner.clean(document["text"])

            # Conserver les métadonnées
            processed_documents.append({
                "text": cleaned_text,
                "metadata": document["metadata"].copy()
            })

        return processed_documents