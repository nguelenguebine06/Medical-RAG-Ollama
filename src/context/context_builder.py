
class ContextBuilder:

    def build(self, documents: list[dict]) -> str:

        if not isinstance(documents, list):
            raise TypeError("Les documents doivent être fournis sous forme de liste.")

        if not documents:
            raise ValueError("Aucun document fourni pour construire le contexte.")

        context_parts = []

        for index, document in enumerate(documents, start=1):

            metadata = document.get("metadata", {})
            content = document.get("text", "")

            if not isinstance(metadata, dict):
                metadata = {}

            if not isinstance(content, str):
                content = ""

            title = metadata.get("title", "Non renseigné")
            chapter = metadata.get("chapter", "Non renseigné")
            article = metadata.get("article", "Non renseigné")
            article_description = metadata.get(
                "article_description", "Non renseignée"
            )

            code = metadata.get("code", document.get("id", "Non renseigné"))
            description = metadata.get("description", "Non renseignée")
            montant = metadata.get("montant", "Non renseigné")
            note = metadata.get("note", "")

            formatted_document = f"""
DOCUMENT {index}

Titre : {title}
Chapitre : {chapter}
Article : {article}
Description de l'article : {article_description}

Code de l'acte : {code}
Description de l'acte : {description}
Tarif : {montant} DH

Note complémentaire : {note}

Contenu :
{content.strip()}
"""

            context_parts.append(formatted_document.strip())

        return "\n\n------------------------------\n\n".join(context_parts)

    def build_sources(self, documents: list[dict]) -> list[dict]:

        if not isinstance(documents, list):
            raise TypeError("Les documents doivent être fournis sous forme de liste.")

        sources = []

        for document in documents:

            metadata = document.get("metadata", {})

            if not isinstance(metadata, dict):
                metadata = {}

            source = {
                "id": document.get("id"),
                "title": metadata.get("title"),
                "chapter": metadata.get("chapter"),
                "article": metadata.get("article"),
                "article_description": metadata.get("article_description"),
                "code": metadata.get("code", document.get("id")),
                "description": metadata.get("description"),
                "montant": metadata.get("montant"),
                "note": metadata.get("note"),
                "rrf_score": document.get("rrf_score")
            }

            sources.append(source)

        return sources
