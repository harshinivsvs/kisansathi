"""
Document loader module for KisanSathi RAG.
Parses agricultural advisory files (Markdown, text, structured guidelines)
and extracts rich metadata tags (crop, region, season, trust level, source).
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentSection(BaseModel):
    """Represents a logical section or page within an advisory document."""
    title: str
    page_number: str = "Page 1"
    content: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class LoadedDocument(BaseModel):
    """Represents a fully parsed agronomic advisory document."""
    document_id: str
    document_name: str
    organization: str
    state: str = "Maharashtra"
    crop: str
    season: str = "All Season"
    trust_level: str = "official_icar"
    year: str = "2024"
    file_path: str
    full_text: str
    sections: List[DocumentSection] = Field(default_factory=list)


class DocumentLoader:
    """Loads and standardizes agronomic advisory documents from disk."""

    @staticmethod
    def parse_frontmatter(raw_text: str) -> tuple[Dict[str, str], str]:
        """Extracts YAML-like frontmatter enclosed in triple dashes (---)."""
        metadata: Dict[str, str] = {}
        content = raw_text

        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw_text, re.DOTALL)
        if frontmatter_match:
            frontmatter_raw = frontmatter_match.group(1)
            content = frontmatter_match.group(2)

            for line in frontmatter_raw.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key, val = line.split(":", 1)
                    metadata[key.strip()] = val.strip().strip('"').strip("'")

        return metadata, content

    @classmethod
    def load_file(cls, file_path: str | Path) -> LoadedDocument:
        """Loads a single document file from disk and parses metadata and sections."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Advisory file not found at: {file_path}")

        raw_text = path.read_text(encoding="utf-8")
        metadata, content = cls.parse_frontmatter(raw_text)

        # Fallback metadata from filename
        stem = path.stem.lower()
        crop = metadata.get("crop", "general")
        if not crop or crop == "general":
            if "tomato" in stem:
                crop = "tomato"
            elif "onion" in stem:
                crop = "onion"
            elif "soybean" in stem:
                crop = "soybean"
            elif "cotton" in stem:
                crop = "cotton"

        document_id = metadata.get("document_id", path.stem.upper())
        document_name = metadata.get("document_name", path.stem.replace("_", " ").title())
        organization = metadata.get("organization", "ICAR / State Agri University")
        state = metadata.get("state", "Maharashtra")
        season = metadata.get("season", "Kharif / Rabi")
        trust_level = metadata.get("trust_level", "official_icar")
        year = metadata.get("year", "2024")

        # Parse sections based on markdown headers (## Section ...)
        sections: List[DocumentSection] = []
        raw_sections = re.split(r"\n(?=##\s+)", content)

        for sec in raw_sections:
            sec = sec.strip()
            if not sec:
                continue

            sec_title = "General Overview"
            page_num = "Page 1"

            first_line = sec.splitlines()[0]
            if first_line.startswith("## "):
                sec_title = first_line.replace("## ", "").strip()
                page_match = re.search(r"\(Page\s*(\d+)\)", sec_title, re.IGNORECASE)
                if page_match:
                    page_num = f"Page {page_match.group(1)}"
                sec_content = "\n".join(sec.splitlines()[1:]).strip()
            else:
                sec_content = sec

            sections.append(
                DocumentSection(
                    title=sec_title,
                    page_number=page_num,
                    content=sec_content,
                    metadata={
                        "document_id": document_id,
                        "crop": crop.lower(),
                        "state": state,
                        "trust_level": trust_level,
                    }
                )
            )

        if not sections:
            sections.append(
                DocumentSection(
                    title="Main Document",
                    page_number="Page 1",
                    content=content,
                    metadata={"document_id": document_id, "crop": crop.lower()}
                )
            )

        return LoadedDocument(
            document_id=document_id,
            document_name=document_name,
            organization=organization,
            state=state,
            crop=crop.lower(),
            season=season,
            trust_level=trust_level,
            year=year,
            file_path=str(path.resolve()),
            full_text=content,
            sections=sections
        )

    @classmethod
    def load_directory(cls, dir_path: str | Path) -> List[LoadedDocument]:
        """Loads all Markdown and text advisory documents from a target directory."""
        path = Path(dir_path)
        if not path.exists():
            return []

        documents = []
        for file in sorted(path.glob("*.md")):
            documents.append(cls.load_file(file))
        for file in sorted(path.glob("*.txt")):
            documents.append(cls.load_file(file))

        return documents
