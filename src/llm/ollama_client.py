import ollama

from src.utils.config import Config
from src.utils.logger import Logger


class OllamaClient:

    def __init__(self):
        self.logger = Logger.get_logger("OllamaClient")

        self.client = ollama.Client(
            host=Config.OLLAMA_BASE_URL
        )

        self.model = Config.OLLAMA_MODEL

        self.logger.info(
            f"OllamaClient initialisé avec le modèle {self.model}"
        )

    def generate(self, prompt: str) -> str:

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Le prompt doit être une chaîne non vide.")

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 300
                }
            )

            return response["response"].strip()

        except ollama.ResponseError as error:
            self.logger.error(f"Erreur Ollama : {error}")
            raise

        except Exception as error:
            self.logger.error(f"Erreur de génération : {error}")
            raise