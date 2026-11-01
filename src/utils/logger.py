import logging
from pathlib import Path

from src.utils.config import Config


class Logger:

    @staticmethod
    def get_logger(name: str) -> logging.Logger:

        logger = logging.getLogger(name)

        # Éviter de multiplier les handlers
        if logger.handlers:
            return logger

        logger.setLevel(logging.DEBUG)

        # Format des messages
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Affichage dans le terminal
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

        # Enregistrement dans un fichier
        log_dir = Config.BASE_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_dir / "rag.log",
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        return logger