import re

class TextCleaner:

    def clean(self, text: str) -> str:

       # Nettoie un texte extrait d'un document sans altérer son sens.

        if not isinstance(text, str):
            raise TypeError("Le texte doit être une chaîne de caractères.")

        # 1. Supprimer les URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # 2. Supprimer les emails
        text = re.sub(r"\S+@\S+", "", text)

        # 3. Remplacer les caractères de contrôle
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", text)

        # 4. Normaliser les apostrophes
        text = text.replace("\u2019", "'").replace("\u2018", "'")

        # 5. Normaliser les tirets
        text = text.replace("\u2013", "-").replace("\u2014", "-")

        # 6. Supprimer les séparateurs décoratifs
        text = re.sub(r"[-=]{3,}", " ", text)

        # 7. Normaliser les espaces
        text = re.sub(r"[ \t]+", " ", text)

        # 8. Supprimer les lignes trop courtes
        lines = [
            line.strip()
            for line in text.splitlines()
            if len(line.strip()) >= 3
        ]

        return "\n".join(lines).strip()