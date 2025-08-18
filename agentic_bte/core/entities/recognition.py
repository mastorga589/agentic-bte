"""
Entity Recognition - Biomedical Named Entity Recognition

This module provides biomedical named entity recognition using multiple strategies:
- spaCy/SciSpaCy models for scientific entities
- LLM-based biological process extraction
- Fallback extraction for missing dependencies
"""

import logging
from typing import List, Optional, Dict, Set
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from ...config.settings import get_settings
from ...exceptions.entity_errors import EntityRecognitionError

# Optional spaCy imports with fallback handling
try:
    import spacy
    from spacy import Language
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
    Language = None

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ExtractedEntity:
    """Represents an extracted entity with metadata"""
    text: str
    start: int
    end: int
    label: str
    confidence: float = 0.0
    source: str = "unknown"


class BiomedicalEntityRecognizer:
    """
    Biomedical named entity recognizer using multiple extraction strategies
    
    This class combines spaCy/SciSpaCy models with LLM-based extraction to identify
    biomedical entities from text queries. It provides fallback mechanisms when
    dependencies are unavailable.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the entity recognizer
        
        Args:
            openai_api_key: OpenAI API key. If not provided, uses settings default
        """
        self.settings = get_settings()
        self.openai_api_key = openai_api_key or self.settings.openai_api_key
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for entity recognition")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=self.settings.openai_temperature,
            model=self.settings.openai_model,
            api_key=self.openai_api_key
        )
        
        # Initialize spaCy models if available
        self.scientific_nlp = None
        self.drug_disease_nlp = None
        self._initialize_spacy_models()
    
    def _initialize_spacy_models(self) -> None:
        """Initialize spaCy models with error handling"""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available. Using LLM-only entity extraction.")
            return
        
        try:
            # Load scientific model
            self.scientific_nlp = spacy.load(self.settings.scispacy_large_model)
            
            # Add entity linker if available
            try:
                self.scientific_nlp.add_pipe(
                    "scispacy_linker", 
                    config={
                        "resolve_abbreviations": True, 
                        "linker_name": "umls"
                    }
                )
            except Exception as e:
                logger.warning(f"Could not add SciSpaCy linker: {e}")
            
            logger.info(f"Loaded scientific spaCy model: {self.settings.scispacy_large_model}")
            
        except Exception as e:
            logger.warning(f"Could not load scientific spaCy model: {e}")
            self.scientific_nlp = None
        
        try:
            # Load drug/disease model
            self.drug_disease_nlp = spacy.load(self.settings.scispacy_drug_disease_model)
            logger.info(f"Loaded drug/disease spaCy model: {self.settings.scispacy_drug_disease_model}")
            
        except Exception as e:
            logger.warning(f"Could not load drug/disease spaCy model: {e}")
            self.drug_disease_nlp = None
    
    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """
        Extract biomedical entities from text using all available strategies
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of extracted entities with metadata
            
        Raises:
            EntityRecognitionError: If entity extraction fails completely
        """
        if not text or not text.strip():
            return []
        
        entities = []
        
        try:
            # Extract using spaCy models if available
            if self.scientific_nlp or self.drug_disease_nlp:
                entities.extend(self._extract_with_spacy(text))
            
            # Extract biological processes using LLM
            entities.extend(self._extract_biological_processes(text))
            
            # If no entities found, try fallback extraction
            if not entities:
                entities.extend(self._fallback_entity_extraction(text))
            
            # Deduplicate entities while preserving order
            entities = self._deduplicate_entities(entities)
            
            logger.info(f"Extracted {len(entities)} entities from text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            raise EntityRecognitionError(f"Failed to extract entities: {str(e)}") from e
    
    def _extract_with_spacy(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using spaCy models"""
        entities = []
        
        # Process with scientific model
        if self.scientific_nlp:
            try:
                doc = self.scientific_nlp(text)
                for ent in doc.ents:
                    entities.append(ExtractedEntity(
                        text=ent.text.strip(),
                        start=ent.start_char,
                        end=ent.end_char,
                        label=ent.label_,
                        confidence=0.8,
                        source="scispacy_scientific"
                    ))
            except Exception as e:
                logger.warning(f"Scientific spaCy model failed: {e}")
        
        # Process with drug/disease model
        if self.drug_disease_nlp:
            try:
                doc = self.drug_disease_nlp(text)
                for ent in doc.ents:
                    entities.append(ExtractedEntity(
                        text=ent.text.strip(),
                        start=ent.start_char,
                        end=ent.end_char,
                        label=ent.label_,
                        confidence=0.8,
                        source="scispacy_drug_disease"
                    ))
            except Exception as e:
                logger.warning(f"Drug/disease spaCy model failed: {e}")
        
        return entities
    
    def _extract_biological_processes(self, text: str) -> List[ExtractedEntity]:
        """Extract biological processes using LLM"""
        try:
            prompt = f"""
            You are a helpful assistant that extracts biological processes from scientific text.
            
            Biological processes include concepts like:
            - "cholesterol biosynthesis"
            - "protein folding" 
            - "cell division"
            - "DNA repair"
            - "apoptosis"
            - "photosynthesis"
            
            Extract all biological processes from this text. Each entity should be a complete noun phrase.
            Return only the biological process terms, one per line.
            If no biological processes are found, return "NONE".
            
            Text: {text}
            """
            
            response = self.llm.invoke(prompt)
            result = response.content.strip()
            
            if result == "NONE" or not result:
                return []
            
            entities = []
            for line in result.split('\n'):
                line = line.strip()
                if line and line != "NONE":
                    # Clean up the line (remove bullets, numbers, etc.)
                    line = line.lstrip('•-*1234567890. ')
                    if line:
                        entities.append(ExtractedEntity(
                            text=line,
                            start=0,  # LLM extraction doesn't provide positions
                            end=0,
                            label="BiologicalProcess",
                            confidence=0.7,
                            source="llm_biological_process"
                        ))
            
            return entities
            
        except Exception as e:
            logger.warning(f"LLM biological process extraction failed: {e}")
            return []
    
    def _fallback_entity_extraction(self, text: str) -> List[ExtractedEntity]:
        """Fallback entity extraction using LLM when spaCy is unavailable"""
        try:
            prompt = f"""
            Extract biomedical entities from the following text. Include:
            - Diseases and disorders
            - Drugs and chemical compounds  
            - Genes and proteins
            - Biological processes
            - Anatomical structures
            
            Return each entity on a new line, with the entity type in parentheses.
            Example format:
            diabetes (disease)
            insulin (protein)
            
            If no entities found, return "NONE".
            
            Text: {text}
            """
            
            response = self.llm.invoke(prompt)
            result = response.content.strip()
            
            if result == "NONE" or not result:
                return []
            
            entities = []
            for line in result.split('\n'):
                line = line.strip()
                if line and '(' in line and line.endswith(')'):
                    # Parse "entity (type)" format
                    parts = line.rsplit('(', 1)
                    if len(parts) == 2:
                        entity_text = parts[0].strip()
                        entity_type = parts[1].rstrip(')').strip()
                        
                        entities.append(ExtractedEntity(
                            text=entity_text,
                            start=0,
                            end=0,
                            label=entity_type,
                            confidence=0.6,
                            source="llm_fallback"
                        ))
            
            return entities
            
        except Exception as e:
            logger.warning(f"Fallback entity extraction failed: {e}")
            return []
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities while preserving order and preferring higher confidence"""
        seen_texts = set()
        deduplicated = []
        
        # Sort by confidence (descending) to prefer higher confidence entities
        entities.sort(key=lambda x: x.confidence, reverse=True)
        
        for entity in entities:
            # Normalize text for comparison
            normalized_text = entity.text.lower().strip()
            
            if normalized_text not in seen_texts:
                seen_texts.add(normalized_text)
                deduplicated.append(entity)
        
        return deduplicated
    
    def get_available_models(self) -> Dict[str, bool]:
        """Get information about available models"""
        return {
            "spacy_available": SPACY_AVAILABLE,
            "scientific_model_loaded": self.scientific_nlp is not None,
            "drug_disease_model_loaded": self.drug_disease_nlp is not None,
            "llm_available": self.llm is not None
        }


# Convenience function for simple entity extraction
def extract_entities(text: str, openai_api_key: Optional[str] = None) -> List[str]:
    """
    Simple function to extract entity texts from input text
    
    Args:
        text: Input text
        openai_api_key: Optional OpenAI API key
        
    Returns:
        List of entity text strings
    """
    recognizer = BiomedicalEntityRecognizer(openai_api_key)
    entities = recognizer.extract_entities(text)
    return [entity.text for entity in entities]