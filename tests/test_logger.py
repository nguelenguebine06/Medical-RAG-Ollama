from src.utils.logger import Logger


if __name__ == "__main__":

    logger = Logger.get_logger("RAG")

    logger.debug("Message de débogage")
    logger.info("Initialisation du système RAG")
    logger.warning("Avertissement de test")
    logger.error("Erreur de test")

    print("\nTest du logger terminé.")