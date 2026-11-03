
class PromptBuilder:

    SYSTEM_PROMPT = """
Tu es un assistant spécialisé dans la nomenclature médicale.

Ton rôle est de répondre aux questions des utilisateurs
en utilisant exclusivement les informations présentes
dans le contexte documentaire fourni.

Règles à respecter :

1. Utilise uniquement les informations du contexte.
2. N'invente jamais de code, de description ou de tarif.
3. Si une information est absente du contexte, indique-le clairement.
4. Ne complète pas les informations manquantes avec tes connaissances personnelles.
5. Réponds en français, de manière claire et professionnelle.
6. Lorsque c'est pertinent, mentionne le code de l'acte médical
   et son tarif.
7. Si plusieurs actes correspondent à la question,
   distingue-les clairement.
8. Ne présente jamais une information incertaine comme un fait établi.

Si le contexte ne contient aucune information permettant
de répondre, indique que tu ne disposes pas des éléments
nécessaires pour répondre.
"""

    def build(self, context: str, question: str) -> str:

        if not isinstance(context, str) or not context.strip():
            raise ValueError("Le contexte documentaire est vide.")

        if not isinstance(question, str) or not question.strip():
            raise ValueError("La question doit être une chaîne non vide.")

        prompt = f"""
{self.SYSTEM_PROMPT}

CONTEXTE DOCUMENTAIRE :
----------------------
{context}

QUESTION DE L'UTILISATEUR :
--------------------------
{question}

RÉPONSE :
"""

        return prompt.strip()
