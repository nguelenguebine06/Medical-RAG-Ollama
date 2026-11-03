
from src.retrieval.hybrid_retriever import HybridRetriever
from src.context.context_builder import ContextBuilder
from src.prompts.prompt_builder import PromptBuilder
from src.llm.ollama_client import OllamaClient
from src.utils.config import Config
from src.utils.logger import Logger


class RAGPipeline:

    def __init__(self):
        self.logger = Logger.get_logger("Rag_pipeline")

        # Initialisation des composants
        self.retriever = HybridRetriever()
        self.context_builder = ContextBuilder()
        self.prompt_builder = PromptBuilder()
        self.llm = OllamaClient()

        self.logger.info("Rag_pipeline initialisé.")

    def run(self, question: str, top_k: int = None) -> dict:

        # Validation de la question
        if not isinstance(question, str):
            raise TypeError("La question doit être une chaîne de caractères.")

        question = question.strip()

        if not question:
            raise ValueError("La question ne peut pas être vide.")

        if top_k is None:
            top_k = Config.RETRIEVAL_TOP_K

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k doit être un entier strictement positif.")

        self.logger.info(f"Nouvelle question : {question}")

        # 1. Recherche hybride
        documents = self.retriever.search(
            query=question,
            top_k=top_k
        )

        # 2. Gestion de l'absence de résultats
        if not documents:
            self.logger.warning("Aucun document pertinent trouvé.")

            return {
                "answer": (
                    "Je n'ai trouvé aucun document pertinent dans "
                    "la nomenclature pour répondre à votre question."
                ),
                "sources": [],
                "retrieval_count": 0
            }

        # 3. Construction du contexte
        context = self.context_builder.build(documents)

        # 4. Construction du prompt
        prompt = self.prompt_builder.build(
            context=context,
            question=question
        )

        # 5. Génération de la réponse
        answer = self.llm.generate(prompt)

        # 6. Construction des sources
        sources = self.context_builder.build_sources(documents)

        self.logger.info(
            f"Réponse générée avec {len(sources)} source(s)."
        )

        # 7. Résultat final
        return {
            "answer": answer,
            "sources": sources,
            "retrieval_count": len(documents)
        }
