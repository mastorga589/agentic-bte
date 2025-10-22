"""
Biomedical Named Entity Recognition using spaCy/SciSpaCy + LLM classification (minimal, modern)
"""
import logging
from typing import List, Optional
from langchain_openai import ChatOpenAI
from ...config.settings import get_settings
from ...exceptions.entity_errors import EntityRecognitionError

logger = logging.getLogger(__name__)

class BiomedicalEntityRecognizer:
    SUPPORTED_ENTITY_TYPES = [
        "biologicalProcess", "disease", "drug", "gene", "protein", "biologicalEntity", "general"
    ]

    def __init__(self, openai_api_key: Optional[str] = None):
        self.settings = get_settings()
        self.openai_api_key = openai_api_key or self.settings.openai_api_key
        if not self.openai_api_key:
            raise EntityRecognitionError("OpenAI API key is required for entity recognition")
        self.llm = ChatOpenAI(
            temperature=0,
            model=self.settings.openai_model,
            api_key=self.openai_api_key
        )
        # Add any minimal spaCy/scispacy model loader if actively used; otherwise, keep lean

    def classify_entity(self, query: str, entity: str) -> str:
        # Minimal stub: Use LLM or heuristic, return type
        # Fill with most up-to-date working logic as needed
        raise NotImplementedError("LLM-based entity classification needs to be filled in as required.")
