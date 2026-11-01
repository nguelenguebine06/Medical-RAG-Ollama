from abc import ABC, abstractmethod
from typing import Any


class BaseLoader(ABC):

    @abstractmethod
    def load(self, file_path: str) -> list[dict[str, Any]]:
        """
        Charge un document et retourne une liste
        de documents structurés avec leurs métadonnées.
        """
        pass