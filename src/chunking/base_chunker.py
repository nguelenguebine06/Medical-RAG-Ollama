from abc import ABC, abstractmethod
from typing import Any


class BaseChunker(ABC):

    @abstractmethod
    def split( self, documents: list[dict[str, Any]] ) -> list[dict[str, Any]]:
        """
        Découpe les documents en chunks
        et retourne les chunks avec leurs métadonnées.
        """
        pass