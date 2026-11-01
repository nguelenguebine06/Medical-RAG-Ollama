
import re
from pathlib import Path
from typing import Any

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from src.ingestion.loaders.base_loader import BaseLoader


class DocxLoader(BaseLoader):

    CODE_PATTERN = re.compile(r"^[A-Z]{1,3}-?\d{1,4}$")

    def load(self, file_path: str) -> list[dict[str, Any]]:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Le fichier '{file_path}' n'existe pas."
            )

        if path.suffix.lower() != ".docx":
            raise ValueError("Le fichier doit être au format DOCX.")

        doc = Document(str(path))

        if self._is_nomenclature_docx(doc):
            return self._load_nomenclature_docx(doc, str(path))

        return self._load_generic_docx(doc, str(path))

    # --------------------------------------------------
    # METADONNEES COMMUNES
    # --------------------------------------------------

    def _build_metadata(self, filepath: str, section: str) -> dict:

        return {
            "filename": Path(filepath).name,
            "source": filepath,
            "type": "docx",
            "section": section
        }

    # --------------------------------------------------
    # DETECTION DU TYPE DE DOCUMENT
    # --------------------------------------------------

    def _is_nomenclature_docx(self, doc) -> bool:

        matches = 0

        for table in doc.tables:

            for row in table.rows[:3]:

                if not row.cells:
                    continue

                first_cell = row.cells[0].text.strip()

                if self.CODE_PATTERN.match(first_cell):
                    matches += 1

                if matches >= 3:
                    return True

        return False

    # --------------------------------------------------
    # CLASSIFICATION DES PARAGRAPHES
    # --------------------------------------------------

    def _is_chapter(self, text: str) -> bool:

        normalized = text.lower().strip()

        return ( "chapitre" in normalized and ( normalized.startswith("chapitre") or re.match(r"^[a-z]\s*[-–]\s*chapitre", normalized) )   )

    def _is_title(self, text: str) -> bool:

        normalized = text.lower().strip()

        return normalized.startswith("titre ")

    def _is_article_heading(self, text: str) -> bool:

        normalized = text.lower().strip()

        return bool(
            re.match(
                r"^article\s+(premier|\d+|[ivxlcdm]+)",
                normalized
            )
        )

    # --------------------------------------------------
    # PARCOURS DES ELEMENTS WORD DANS LEUR ORDRE
    # --------------------------------------------------

    def _iter_block_items(self, doc):

        body = doc.element.body

        for child in body.iterchildren():

            if child.tag.endswith("}p"):
                yield Paragraph(child, doc)

            elif child.tag.endswith("}tbl"):
                yield Table(child, doc)

    # --------------------------------------------------
    # DOCUMENT DOCX GENERIQUE
    # --------------------------------------------------

    def _load_generic_docx(
        self,
        doc,
        filepath: str
    ) -> list[dict[str, Any]]:

        raw_documents = []

        for element in self._iter_block_items(doc):

            if isinstance(element, Paragraph):

                text = element.text.strip()

                if len(text) <= 3:
                    continue

                style = element.style.name if element.style else ""

                if style.startswith(("Heading", "Titre")):
                    section = "heading"

                elif self._is_title(text):
                    section = "title"

                elif self._is_chapter(text):
                    section = "chapter"

                elif self._is_article_heading(text):
                    section = "article"

                else:
                    section = "paragraph"

                raw_documents.append({
                    "text": text,
                    "metadata": {
                        **self._build_metadata(filepath, section),
                        "style": style
                    }
                })

            elif isinstance(element, Table):

                for row in element.rows:

                    cells = [
                        cell.text.strip()
                        for cell in row.cells
                    ]

                    text = " | ".join(
                        cell for cell in cells if cell
                    )

                    if len(text) <= 3:
                        continue

                    raw_documents.append({
                        "text": text,
                        "metadata": self._build_metadata(
                            filepath,
                            "table"
                        )
                    })

        return raw_documents

    # --------------------------------------------------
    # CONTEXTE DE LA NOMENCLATURE
    # --------------------------------------------------

    def _create_context(self) -> dict:

        return {
            "title": "",
            "chapter": "",
            "note": "",
            "article": "",
            "article_description": ""
        }

    def _extract_context(self, documents):

        context = self._create_context()
        raw_documents = []

        for document in documents:

            metadata = document["metadata"]

            if metadata["section"] == "title":
                context["title"] = document["text"]

            elif metadata["section"] == "chapter":

                context["chapter"] = document["text"]

                # Un nouveau chapitre implique un nouveau contexte.
                context["note"] = ""
                context["article"] = ""
                context["article_description"] = ""

            elif metadata["section"] == "note":

                context["note"] = document["text"]

            elif metadata["section"] == "article":

                context["article"] = document["text"]
                context["article_description"] = ""

            elif metadata["section"] == "article_description":

                context["article_description"] = document["text"]

            raw_documents.append({
                "text": document["text"],
                "metadata": metadata,
                "context": context.copy()
            })

        return raw_documents

    # --------------------------------------------------
    # EXTRACTION DES ACTES DE LA NOMENCLATURE
    # --------------------------------------------------

    def _load_nomenclature_docx(
        self,
        doc,
        filepath: str
    ) -> list[dict[str, Any]]:

        raw_documents = []
        context = self._create_context()

        pending_article = False
        pending_chapter_note = False

        for element in self._iter_block_items(doc):

            # ------------------------------------------
            # PARAGRAPHES
            # ------------------------------------------

            if isinstance(element, Paragraph):

                text = element.text.strip()

                if not text:
                    continue

                if self._is_title(text):

                    context["title"] = text
                    context["chapter"] = ""
                    context["note"] = ""
                    context["article"] = ""
                    context["article_description"] = ""

                    pending_article = False
                    pending_chapter_note = False

                elif self._is_chapter(text):

                    context["chapter"] = text

                    context["note"] = ""
                    context["article"] = ""
                    context["article_description"] = ""

                    pending_article = False
                    pending_chapter_note = True

                elif self._is_article_heading(text):

                    context["article"] = text
                    context["article_description"] = ""

                    pending_article = True
                    pending_chapter_note = False

                elif pending_article:

                    context["article_description"] = text
                    pending_article = False

                elif pending_chapter_note:

                    context["note"] = text
                    pending_chapter_note = False

                else:

                    # Conserver les paragraphes complémentaires
                    # sans les interpréter automatiquement comme
                    # une note ou une description d'article.
                    pass

            # ------------------------------------------
            # TABLEAUX
            # ------------------------------------------

            elif isinstance(element, Table):

                for row in element.rows:

                    cells = [
                        cell.text.strip()
                        for cell in row.cells
                    ]

                    if not cells:
                        continue

                    first_cell = cells[0]

                    # Ignorer les lignes qui ne correspondent
                    # pas à un code d'acte.
                    if not self.CODE_PATTERN.match(first_cell):
                        continue

                    code = first_cell

                    description = (
                        cells[1] if len(cells) > 1 else ""
                    )

                    montant = (
                        cells[2] if len(cells) > 2 else ""
                    )

                    montant_2 = (
                        cells[3] if len(cells) > 3 else ""
                    )

                    montant_3 = (
                        cells[4] if len(cells) > 4 else ""
                    )

                    # Construire le texte enrichi.
                    context_parts = [
                        context["title"],
                        context["chapter"],
                        context["note"],
                        context["article"],
                        context["article_description"],
                        description
                    ]

                    full_text = "\n".join(
                        part.strip()
                        for part in context_parts
                        if part and part.strip()
                    )

                    raw_documents.append({

                        "text": full_text,

                        "metadata": {
                            **self._build_metadata(
                                filepath,
                                "nomenclature"
                            ),

                            "is_nomenclature": True,

                            "code": code,

                            "title": context["title"],

                            "chapter": context["chapter"],

                            "note": context["note"],

                            "article": context["article"],

                            "article_description":
                                context["article_description"],

                            "description": description,

                            "montant": montant,

                            "montant_2": montant_2,

                            "montant_3": montant_3,

                            "category": context["chapter"]
                        }
                    })

        return raw_documents
