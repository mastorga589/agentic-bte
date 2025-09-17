"""
Biomedical Entity Recognition Module

This module provides comprehensive biomedical entity recognition capabilities
including named entity recognition, entity classification, and fallback strategies.

Migrated and enhanced from original BTE-LLM implementations.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from langchain_openai import ChatOpenAI

from ...config.settings import get_settings

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Biomedical entity types"""
    GENE = "Gene"
    PROTEIN = "Protein"
    DISEASE = "Disease"
    CHEMICAL = "Chemical"
    DRUG = "Drug"
    BIOLOGICAL_PROCESS = "BiologicalProcess"
    MOLECULAR_FUNCTION = "MolecularFunction"
    CELLULAR_COMPONENT = "CellularComponent"
    PHENOTYPE = "Phenotype"
    ANATOMY = "Anatomy"
    ORGANISM = "Organism"
    UNKNOWN = "Unknown"


@dataclass
class ExtractedEntity:
    """
    Represents an extracted biomedical entity
    """
    text: str
    start: int
    end: int
    label: str
    confidence: float = 0.0
    entity_type: Optional[str] = None
    cui: Optional[str] = None
    entity_id: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass 
class LinkedEntity(ExtractedEntity):
    """
    Represents a linked biomedical entity with knowledge base connections
    """
    linked_concepts: Optional[List[Dict[str, Any]]] = None
    umls_cui: Optional[str] = None
    mesh_id: Optional[str] = None
    ncbi_gene_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class BiomedicalEntityRecognizer:
    """
    Biomedical Entity Recognizer using multiple strategies
    
    This class provides entity recognition using:
    1. spaCy/SciSpaCy models (if available)
    2. LLM-based extraction (fallback)
    3. Rule-based extraction (basic fallback)
    
    Migrated from original BTE-LLM implementations.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the biomedical entity recognizer
        
        Args:
            openai_api_key: OpenAI API key for LLM-based extraction
        """
        self.settings = get_settings()
        self.openai_api_key = openai_api_key or self.settings.openai_api_key
        
        # Initialize components
        self.nlp = None
        self.linker = None
        self.llm = None
        
        # Try to initialize spaCy/SciSpaCy
        self._init_spacy()
        
        # Initialize LLM if API key available
        if self.openai_api_key:
            self._init_llm()
    
    def _init_spacy(self):
        """Initialize spaCy/SciSpaCy models if available"""
        try:
            import spacy
            from scispacy.linking import EntityLinker
            
            # Try to load SciSpaCy biomedical model
            model_names = [
                "en_core_sci_lg",
                "en_ner_bc5cdr_md", 
                "en_core_sci_md",
                "en_core_sci_sm"
            ]
            
            for model_name in model_names:
                try:
                    self.nlp = spacy.load(model_name)
                    logger.info(f"Loaded spaCy model: {model_name}")
                    break
                except OSError:
                    continue
            
            # Try to add entity linker
            if self.nlp:
                try:
                    self.linker = EntityLinker(resolve_abbreviations=True, name="umls")
                    self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
                    logger.info("Added UMLS entity linker")
                except Exception as e:
                    logger.warning(f"Could not add entity linker: {e}")
            
            if not self.nlp:
                logger.warning("No SciSpaCy models found - will use fallback methods")
                
        except ImportError:
            logger.warning("spaCy/SciSpaCy not available - will use fallback methods")
    
    def _init_llm(self):
        """Initialize LLM for entity extraction"""
        try:
            self.llm = ChatOpenAI(
                temperature=0,
                model=self.settings.openai_model,
                api_key=self.openai_api_key
            )
            logger.info("Initialized LLM for entity extraction")
        except Exception as e:
            logger.warning(f"Could not initialize LLM: {e}")
    
    def extract_with_spacy(self, text: str) -> List[ExtractedEntity]:
        """
        Extract entities using spaCy/SciSpaCy
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entity = ExtractedEntity(
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    label=ent.label_,
                    confidence=getattr(ent, 'confidence', 0.8),
                    entity_type=self._map_spacy_label_to_type(ent.label_)
                )
                
                # Add linked concepts if available
                if hasattr(ent, "_.umls_ents") and ent._.umls_ents:
                    best_match = ent._.umls_ents[0]
                    entity.cui = best_match[0]
                    entity.entity_id = best_match[0]
                
                entities.append(entity)
            
            logger.debug(f"Extracted {len(entities)} entities with spaCy")
            return entities
            
        except Exception as e:
            logger.error(f"Error in spaCy entity extraction: {e}")
            return []
    
    def extract_with_llm(self, text: str) -> List[ExtractedEntity]:
        """
        Extract entities using LLM
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        if not self.llm:
            return []
        
        try:
            prompt = f"""
            Extract biomedical entities from the following text. Identify genes, proteins, diseases, chemicals, drugs, biological processes, and other biomedical concepts.
            
            Text: "{text}"
            
            Return a JSON list of entities with the following structure:
            [
                {{
                    "text": "entity text",
                    "start": start_position,
                    "end": end_position,
                    "entity_type": "Disease|Gene|Chemical|Drug|BiologicalProcess|Other",
                    "confidence": 0.9
                }}
            ]
            
            Only return valid JSON.
            """
            
            response = self.llm.invoke(prompt)
            
            # Parse LLM response
            import json
            try:
                entities_data = json.loads(response.content.strip())
                entities = []
                
                for ent_data in entities_data:
                    if isinstance(ent_data, dict):
                        entity = ExtractedEntity(
                            text=ent_data.get("text", ""),
                            start=ent_data.get("start", 0),
                            end=ent_data.get("end", 0),
                            label=ent_data.get("entity_type", "Unknown"),
                            confidence=ent_data.get("confidence", 0.7),
                            entity_type=ent_data.get("entity_type")
                        )
                        entities.append(entity)
                
                logger.debug(f"Extracted {len(entities)} entities with LLM")
                return entities
                
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM response as JSON")
                return []
            
        except Exception as e:
            logger.error(f"Error in LLM entity extraction: {e}")
            return []
    
    def extract_with_rules(self, text: str) -> List[ExtractedEntity]:
        """
        Extract entities using rule-based patterns (basic fallback)
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Basic patterns for common biomedical entities
        patterns = {
            "Gene": [
                r'\b[A-Z][A-Z0-9]+\b',  # Gene symbols (e.g., TP53, BRCA1)
                r'\b[A-Z][a-z]+[0-9]+\b'  # Gene names with numbers
            ],
            "Disease": [
                r'\b\w+oma\b',  # Diseases ending in -oma
                r'\bdiabetes\b',
                r'\bcancer\b',
                r'\bhypertension\b',
                r'\basthma\b'
            ],
            "Chemical": [
                r'\b\w+ine\b',  # Many chemicals end in -ine
                r'\bacid\b',
                r'\bglucose\b',
                r'\binsulin\b'
            ]
        }
        
        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = ExtractedEntity(
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        label=entity_type,
                        confidence=0.5,
                        entity_type=entity_type
                    )
                    entities.append(entity)
        
        logger.debug(f"Extracted {len(entities)} entities with rules")
        return entities
    
    def extract_entities(self, text: str, method: Optional[str] = None) -> List[ExtractedEntity]:
        """
        Extract entities using the best available method
        
        Args:
            text: Input text
            method: Specific method to use ('spacy', 'llm', 'rules', or None for auto)
            
        Returns:
            List of extracted entities
        """
        if method == "spacy" and self.nlp:
            return self.extract_with_spacy(text)
        elif method == "llm" and self.llm:
            return self.extract_with_llm(text)
        elif method == "rules":
            return self.extract_with_rules(text)
        else:
            # Auto-select best available method
            if self.nlp:
                return self.extract_with_spacy(text)
            elif self.llm:
                return self.extract_with_llm(text)
            else:
                return self.extract_with_rules(text)
    
    def _map_spacy_label_to_type(self, label: str) -> str:
        """Map spaCy entity labels to our entity types"""
        label_mapping = {
            "DISEASE": "Disease",
            "CHEMICAL": "Chemical", 
            "GENE_OR_GENOME": "Gene",
            "SPECIES": "Organism",
            "CELL_TYPE": "CellularComponent",
            "CELL_LINE": "CellularComponent",
            "DNA": "Gene",
            "RNA": "Gene",
            "PROTEIN": "Protein"
        }
        return label_mapping.get(label.upper(), "Unknown")


class BioNERTool:
    """
    High-level BioNER tool interface
    
    This class provides a simplified interface for biomedical entity recognition
    with integrated classification and linking capabilities.
    
    Migrated from original BTE-LLM implementations.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize BioNER tool
        
        Args:
            openai_api_key: OpenAI API key for LLM operations
        """
        self.settings = get_settings()
        self.recognizer = BiomedicalEntityRecognizer(openai_api_key)
        
        # Initialize entity linker if available
        try:
            from .entity_linking import EntityLinker
            self.entity_linker = EntityLinker()
        except ImportError:
            self.entity_linker = None
            logger.warning("Entity linker not available")
    
    def extract_entities(self, text: str, method: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract entities and return structured result
        
        Args:
            text: Input text
            method: Extraction method to use
            
        Returns:
            Dictionary with extracted entities and metadata
        """
        entities = self.recognizer.extract_entities(text, method)
        
        result = {
            "text": text,
            "entities": [entity.to_dict() for entity in entities],
            "entity_count": len(entities),
            "method_used": self._get_method_used(),
            "entity_types": list(set(ent.entity_type or "Unknown" for ent in entities))
        }
        
        return result
    
    def extract_and_link_entities(self, text: str, method: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract entities and link them to knowledge bases
        
        Args:
            text: Input text
            method: Extraction method to use
            
        Returns:
            Dictionary with extracted and linked entities
        """
        # First extract entities
        result = self.extract_entities(text, method)
        
        # Then link them if linker is available
        if self.entity_linker:
            try:
                linked_entities = []
                for entity_dict in result["entities"]:
                    entity = ExtractedEntity(**entity_dict)
                    linked = self.entity_linker.link_entity(entity)
                    linked_entities.append(linked.to_dict() if isinstance(linked, LinkedEntity) else entity_dict)
                
                result["entities"] = linked_entities
                result["linked"] = True
                
            except Exception as e:
                logger.warning(f"Entity linking failed: {e}")
                result["linked"] = False
        else:
            result["linked"] = False
        
        return result
    
    def classify_entity_types(self, entities: List[ExtractedEntity]) -> Dict[str, List[ExtractedEntity]]:
        """
        Classify entities by type
        
        Args:
            entities: List of extracted entities
            
        Returns:
            Dictionary mapping entity types to entity lists
        """
        classified = {}
        for entity in entities:
            entity_type = entity.entity_type or "Unknown"
            if entity_type not in classified:
                classified[entity_type] = []
            classified[entity_type].append(entity)
        
        return classified
    
    def _get_method_used(self) -> str:
        """Determine which method was used for extraction"""
        if self.recognizer.nlp:
            return "spacy"
        elif self.recognizer.llm:
            return "llm" 
        else:
            return "rules"


# Convenience functions
def extract_biomedical_entities(text: str, method: Optional[str] = None,
                               openai_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to extract biomedical entities
    
    Args:
        text: Input text
        method: Extraction method ('spacy', 'llm', 'rules', or None)
        openai_api_key: Optional OpenAI API key
        
    Returns:
        Dictionary with extraction results
    """
    tool = BioNERTool(openai_api_key)
    return tool.extract_entities(text, method)


def classify_entity_types(entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convenience function to classify entities by type
    
    Args:
        entities: List of entity dictionaries
        
    Returns:
        Dictionary mapping types to entity lists
    """
    classified = {}
    for entity in entities:
        entity_type = entity.get("entity_type", "Unknown")
        if entity_type not in classified:
            classified[entity_type] = []
        classified[entity_type].append(entity)
    
    return classified