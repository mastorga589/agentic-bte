"""
Biomedical Entity Recognition Module

This module provides biomedical named entity recognition capabilities using
a combination of spaCy/SciSpaCy models and LLM-based classification.

Migrated and enhanced from the original BTE-LLM implementation.
"""

import os
import re
import logging
from typing import List, Optional, Dict, Any

import spacy
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing_extensions import TypedDict, Literal

from ...config.settings import get_settings
from ...exceptions.entity_errors import EntityRecognitionError

logger = logging.getLogger(__name__)


class BiomedicalEntityRecognizer:
    """
    Biomedical Named Entity Recognition using spaCy/SciSpaCy + LLM classification
    
    This class handles extraction and classification of biomedical entities
    from natural language text using a hybrid approach combining:
    - spaCy scientific models for initial extraction
    - SciSpaCy models for drug/disease recognition  
    - LLM-based classification for entity type determination
    - Fallback extraction for when models are unavailable
    """
    
    # Supported entity types for classification
    SUPPORTED_ENTITY_TYPES = [
        "biologicalProcess", "disease", "drug", "gene", 
        "protein", "biologicalEntity", "general"
    ]
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the biomedical entity recognizer
        
        Args:
            openai_api_key: OpenAI API key for LLM operations
        """
        self.settings = get_settings()
        self.openai_api_key = openai_api_key or self.settings.openai_api_key
        
        if not self.openai_api_key:
            raise EntityRecognitionError("OpenAI API key is required for entity recognition")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=0,
            model=self.settings.openai_model,
            api_key=self.openai_api_key
        )
        
        # Initialize spaCy models with error handling
        self._init_spacy_models()
    
    def _init_spacy_models(self):
        """
        Initialize spaCy models with graceful fallback
        """
        self.nlp = None
        self.drug_disease_nlp = None
        
        try:
            # Load scientific spaCy model
            self.nlp = spacy.load(self.settings.scispacy_large_model)
            
            # Add UMLS entity linker
            self.nlp.add_pipe(
                "scispacy_linker", 
                config={
                    "resolve_abbreviations": True, 
                    "linker_name": "umls"
                }
            )
            logger.info(f"Loaded {self.settings.scispacy_large_model} with UMLS linker")
            
        except Exception as e:
            logger.warning(f"Failed to load {self.settings.scispacy_large_model}: {e}")
        
        try:
            # Load drug/disease-specific model
            self.drug_disease_nlp = spacy.load(self.settings.scispacy_drug_disease_model)
            logger.info(f"Loaded {self.settings.scispacy_drug_disease_model}")
            
        except Exception as e:
            logger.warning(f"Failed to load {self.settings.scispacy_drug_disease_model}: {e}")
        
        # Check if we have at least one model
        if self.nlp is None and self.drug_disease_nlp is None:
            logger.warning("No spaCy models available - using fallback entity extraction")
    
    def extract_entities(self, query: str) -> List[str]:
        """
        Extract biomedical entities from query text
        
        Args:
            query: Input text to extract entities from
            
        Returns:
            List of extracted entity text strings
        """
        entities = []
        
        # Use spaCy models if available
        if self.nlp is not None or self.drug_disease_nlp is not None:
            entities.extend(self._extract_with_spacy(query))
        else:
            # Fallback extraction
            entities.extend(self._fallback_entity_extraction(query))
        
        # Extract biological processes using LLM
        bp_entities = self._extract_biological_processes(query)
        if bp_entities:
            entities.extend(bp_entities)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            entity_lower = entity.lower().strip()
            if entity_lower and entity_lower not in seen:
                seen.add(entity_lower)
                unique_entities.append(entity.strip())
        
        logger.info(f"Extracted {len(unique_entities)} unique entities from query")
        return unique_entities
    
    def _extract_with_spacy(self, query: str) -> List[str]:
        """
        Extract entities using spaCy models
        
        Args:
            query: Input text
            
        Returns:
            List of entity texts from spaCy models
        """
        entities = []
        docs = []
        
        # Process with available models
        if self.nlp is not None:
            docs.append(self.nlp(query))
        
        if self.drug_disease_nlp is not None:
            docs.append(self.drug_disease_nlp(query))
        
        # Extract entities from each doc
        for doc in docs:
            for ent in doc.ents:
                entity_text = ent.text.strip()
                if entity_text:
                    entities.append(entity_text)
        
        return entities
    
    def _extract_biological_processes(self, query: str) -> List[str]:
        """
        Extract biological process entities using LLM
        
        Args:
            query: Input text
            
        Returns:
            List of biological process terms
        """
        try:
            bp_prompt = f"""You are a helpful assistant that can extract biological processes from a given query. 
                        These might include concepts such as "cholesterol biosynthesis", "Aminergic neurotransmitter loading into synaptic vesicle", etc. Each entity should be a noun phrase.

                        You must always return the full phrase/long form of each biomedical entity.
                        
                        Return results as a JSON list of strings. Return [] if no biological processes are in the query.
                        DO NOT INCLUDE YOUR THOUGHTS OR EXPLANATIONS.
                        
                        Here is your query: {query}"""

            response = self.llm.invoke(bp_prompt).content.strip()
            
            # Try to parse as JSON list
            import json
            try:
                bp_list = json.loads(response)
                if isinstance(bp_list, list):
                    return [str(item) for item in bp_list if item]
            except json.JSONDecodeError:
                # If not valid JSON, try to extract from string representation
                if response and response != '""' and response != "[]":
                    # Remove quotes and brackets, split by comma
                    cleaned = response.strip('"[]').replace('"', '')
                    if cleaned:
                        return [item.strip() for item in cleaned.split(',') if item.strip()]
            
            return []
            
        except Exception as e:
            logger.warning(f"Failed to extract biological processes with LLM: {e}")
            return []
    
    def classify_entity(self, query: str, entity: str) -> str:
        """
        Classify entity into biomedical type using LLM
        
        Args:
            query: Original query for context
            entity: Entity text to classify
            
        Returns:
            Entity type classification
        """
        try:
            class EntityType(TypedDict):
                """The most appropriate entity type given a specific entity"""
                entType: Literal[
                    "biologicalProcess", "disease", "drug", "gene", 
                    "protein", "biologicalEntity", "general"
                ]
            
            classify_prompt = f"""
            Classify the following biomedical entity into one of: 
            {self.SUPPORTED_ENTITY_TYPES}

            Guidelines:
            - "biologicalProcess": Biological processes, pathways, molecular functions (e.g., "apoptosis", "DNA repair", "metabolism")
            - "disease": Diseases, disorders, syndromes, medical conditions (e.g., "diabetes", "cancer", "hypertension") 
            - "drug": Drugs, medications, compounds, chemicals used therapeutically (e.g., "aspirin", "insulin", "brinzolamide")
            - "gene": Genes and their variants (e.g., "BRCA1", "TP53", "APOE", "insulin gene")
            - "protein": Proteins and protein complexes (e.g., "p53 protein", "insulin receptor", "hemoglobin", "collagen")
            - "biologicalEntity": Cells, organisms, anatomical structures, viruses (e.g., "liver", "T cell", "HIV", "heart")
            - "general": Any other biomedical entity that doesn't fit the above categories

            Biomedical entity: {entity}

            For context, here is the query that the entity was extracted from: {query}
            """
            
            chosen_type = self.llm.with_structured_output(EntityType).invoke(classify_prompt)
            entity_type = str(chosen_type["entType"])
            
            logger.debug(f"Classified '{entity}' as '{entity_type}'")
            return entity_type
            
        except Exception as e:
            logger.warning(f"Failed to classify entity '{entity}': {e}")
            return "general"  # Default fallback
    
    def _fallback_entity_extraction(self, query: str) -> List[str]:
        """
        Simple fallback entity extraction when spaCy models are unavailable
        
        Args:
            query: Input text
            
        Returns:
            List of potential entities extracted using simple patterns
        """
        entities = []
        
        # Simple patterns for common biomedical entities
        patterns = {
            'diseases': r'\b(?:diabetes|cancer|parkinson|alzheimer|hypertension|asthma|arthritis|pneumonia|tuberculosis|hepatitis|encephalitis)\w*\b',
            'drugs': r'\b(?:aspirin|metformin|insulin|morphine|codeine|ibuprofen|acetaminophen|warfarin|L-[Dd]opa|levodopa|brinzolamide)\b',
            'processes': r'\b(?:metabolic process|metabolism|biosynthesis|signaling|pathway|apoptosis)\b',
            'general': r'\b(?:drug|drugs|medication|treatment|therapy|gene|protein|enzyme)s?\b'
        }
        
        for category, pattern in patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend(matches)
        
        # Extract capitalized words that might be biomedical entities
        cap_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*\b', query)
        entities.extend(cap_words)
        
        # Extract terms that look like gene symbols (all caps, 2-10 characters)
        gene_symbols = re.findall(r'\b[A-Z]{2,10}\d?\b', query)
        entities.extend(gene_symbols)
        
        logger.warning(f"Using fallback extraction - found {len(entities)} potential entities")
        return entities
    
    def get_available_models(self) -> Dict[str, bool]:
        """
        Get information about available spaCy models
        
        Returns:
            Dictionary showing which models are available
        """
        return {
            "large_scientific_model": self.nlp is not None,
            "drug_disease_model": self.drug_disease_nlp is not None,
            "umls_linker_available": self.nlp is not None and "scispacy_linker" in self.nlp.pipe_names
        }

"""
Entity Recognition - Biomedical Named Entity Recognition

This module provides biomedical named entity recognition using multiple strategies:
- spaCy/SciSpaCy models for scientific entities
- LLM-based biological process extraction
- Fallback extraction for missing dependencies

Migrated and enhanced from original BTE-LLM MCP Server implementation.
"""

import os
import re
import json
import logging
from typing import List, Optional, Dict, Set, Any, Literal
from dataclasses import dataclass
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI

from ...config.settings import get_settings
from ...exceptions.entity_errors import EntityRecognitionError, EntityLinkingError, EntityClassificationError

# Optional spaCy imports with fallback handling
try:
    import spacy
    from spacy import Language
    from scispacy.linking import EntityLinker
    SPACY_AVAILABLE = True
    
    # Global spaCy models - loaded once when module is imported
    nlp = None
    drug_disease_nlp = None
    
    def _initialize_spacy_models():
        global nlp, drug_disease_nlp
        try:
            nlp = spacy.load("en_core_sci_lg")
            drug_disease_nlp = spacy.load("en_ner_bc5cdr_md")
            nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        except Exception as e:
            print(f"Warning: SpaCy/SciSpaCy initialization failed: {e}")
            print(f"BioNER will use fallback entity extraction")
            nlp = None
            drug_disease_nlp = None
    
    # Initialize models on import
    _initialize_spacy_models()
    
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
    Language = None
    EntityLinker = None
    nlp = None
    drug_disease_nlp = None

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


@dataclass
class LinkedEntity:
    """Represents a linked entity with ID and type information"""
    text: str
    entity_id: str
    entity_type: str
    confidence: float = 0.0
    source: str = "unknown"


class EntityType(TypedDict):
    """Entity type classification result"""
    entType: Literal["biologicalProcess", "disease", "drug", "gene", "protein", "biologicalEntity", "general"]


class BiomedicalEntityRecognizer:
    """
    Biomedical named entity recognizer using multiple extraction strategies
    
    This class combines spaCy/SciSpaCy models with LLM-based extraction to identify
    biomedical entities from text queries. It provides fallback mechanisms when
    dependencies are unavailable.
    
    Migrated from original BTE-LLM implementation with enhancements.
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
            temperature=0,  # Use 0 for consistent results in entity tasks
            model=self.settings.openai_model,
            api_key=self.openai_api_key
        )
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract biomedical entities from text using all available strategies
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of entity text strings
            
        Raises:
            EntityRecognitionError: If entity extraction fails completely
        """
        if not text or not text.strip():
            return []
        
        entities = []
        
        try:
            # Use spaCy models if available
            if SPACY_AVAILABLE and nlp is not None and drug_disease_nlp is not None:
                entities.extend(self._extract_with_spacy(text))
            else:
                # Use fallback extraction
                entities.extend(self._fallback_entity_extraction(text))
            
            # Extract biological processes using LLM
            entities.extend(self._extract_biological_processes(text))
            
            # Deduplicate entities while preserving order
            entities = self._deduplicate_entities(entities)
            
            logger.info(f"Extracted {len(entities)} entities from text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            raise EntityRecognitionError(f"Failed to extract entities: {str(e)}", input_text=text) from e
    
    def _extract_with_spacy(self, text: str) -> List[str]:
        """Extract entities using spaCy models"""
        entities = []
        
        # Process with scientific model
        if nlp is not None:
            try:
                doc = nlp(text)
                for ent in doc.ents:
                    entities.append(ent.text.strip())
            except Exception as e:
                logger.warning(f"Scientific spaCy model failed: {e}")
        
        # Process with drug/disease model
        if drug_disease_nlp is not None:
            try:
                doc = drug_disease_nlp(text)
                for ent in doc.ents:
                    entities.append(ent.text.strip())
            except Exception as e:
                logger.warning(f"Drug/disease spaCy model failed: {e}")
        
        return entities
    
    def _extract_biological_processes(self, text: str) -> List[str]:
        """Extract biological processes using LLM"""
        try:
            prompt = f"""You are a helpful assistant that can extract biological processes from a given query. 
                    These might include concepts such as "cholesterol biosynthesis", "Aminergic neurotransmitter loading into synaptic vesicle", etc. Each entity should be a noun.

                    You must always return the full phrase/long form of each biomedical entity 
                    
                    Return results as a list. Return "" if no biological processes are in the query. DO NOT INCLUDE YOUR THOUGHTS
                    Here is your query: {text}"""
            
            response = self.llm.invoke(prompt)
            result = response.content.strip()
            
            if result == "" or result == '""' or not result:
                return []
            
            # Parse the result - it might be a list format or newline separated
            entities = []
            if result.startswith('[') and result.endswith(']'):
                try:
                    # Try to parse as JSON list
                    parsed = json.loads(result)
                    if isinstance(parsed, list):
                        entities = [str(item).strip() for item in parsed if item]
                except:
                    # If JSON parsing fails, treat as string
                    entities = [result.strip('[]"')]
            else:
                # Split by common separators
                for separator in ['\n', ',', ';']:
                    if separator in result:
                        entities = [item.strip().strip('"') for item in result.split(separator) if item.strip()]
                        break
                else:
                    # Single entity
                    entities = [result.strip().strip('"')]
            
            # Filter out empty strings and common non-entities
            filtered_entities = []
            for entity in entities:
                if entity and entity.lower() not in ['none', 'no biological processes', '']:
                    filtered_entities.append(entity)
            
            return filtered_entities
            
        except Exception as e:
            logger.warning(f"LLM biological process extraction failed: {e}")
            return []
    
    def classify_entity(self, query: str, entity: str) -> str:
        """
        Classify entity type using LLM
        
        Args:
            query: The original query for context
            entity: The entity to classify
            
        Returns:
            Entity type classification
            
        Raises:
            EntityClassificationError: If classification fails
        """
        try:
            supported_types = ["biologicalProcess", "disease", "drug", "gene", "protein", "biologicalEntity", "general"]
            
            classify_prompt = f"""
            Classify the following biomedical entity into one of: 
            {supported_types}

            Guidelines:
            - "biologicalProcess": Biological processes, pathways, molecular functions (e.g., "apoptosis", "DNA repair", "metabolism")
            - "disease": Diseases, disorders, syndromes, medical conditions (e.g., "diabetes", "cancer", "hypertension") 
            - "drug": Drugs, medications, compounds, chemicals used therapeutically (e.g., "aspirin", "insulin", "brinzolamide")
            - "gene": Genes and their variants (e.g., "BRCA1", "TP53", "APOE", "insulin gene")
            - "protein": Proteins and protein complexes (e.g., "p53 protein", "insulin receptor", "hemoglobin", "collagen")
            - "biologicalEntity": Cells, organisms, anatomical structures, viruses (e.g., "liver", "T cell", "HIV", "heart")
            - "general": Any other biomedical entity that doesn't fit the above categories

            Biomedical entity: {entity}

            For context, here is the query that the entity was extracted from: {query}
            """
            
            chosen_type = self.llm.with_structured_output(EntityType).invoke(classify_prompt)
            result = str(chosen_type["entType"])
            
            print(f"{entity} - {result}")
            logger.debug(f"Classified entity '{entity}' as '{result}'")
            
            return result
            
        except Exception as e:
            logger.error(f"Entity classification failed for '{entity}': {e}")
            raise EntityClassificationError(
                f"Failed to classify entity: {str(e)}", 
                entity_text=entity,
                available_types=supported_types
            ) from e
    
    def _fallback_entity_extraction(self, query: str) -> List[str]:
        """Simple fallback entity extraction when spaCy is not available"""
        entities = []
        
        # Simple patterns for common biomedical entities
        patterns = {
            'diseases': r'\b(?:diabetes|cancer|parkinson|alzheimer|hypertension|asthma|arthritis|pneumonia|tuberculosis|hepatitis|encephalitic)\w*\b',
            'drugs': r'\b(?:aspirin|metformin|insulin|morphine|codeine|ibuprofen|acetaminophen|warfarin|L-[Dd]opa|levodopa)\b',
            'processes': r'\b(?:metabolic process|metabolism|biosynthesis|signaling|pathway)\b',
            'general': r'\b(?:drug|drugs|medication|treatment|therapy|gene|protein|enzyme)s?\b'
        }
        
        for category, pattern in patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend(matches)
        
        # Also extract capitalized words that might be entities
        cap_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*\b', query)
        entities.extend(cap_words)
        
        return entities
    
    def _deduplicate_entities(self, entities: List[str]) -> List[str]:
        """Remove duplicate entities while preserving order"""
        seen = set()
        unique_entities = []
        for entity in entities:
            normalized = entity.lower().strip()
            if normalized not in seen and normalized:
                seen.add(normalized)
                unique_entities.append(entity)
        
        return unique_entities
    
    def get_available_models(self) -> Dict[str, bool]:
        """Get information about available models"""
        return {
            "spacy_available": SPACY_AVAILABLE,
            "scientific_model_loaded": nlp is not None,
            "drug_disease_model_loaded": drug_disease_nlp is not None,
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
    return recognizer.extract_entities(text)