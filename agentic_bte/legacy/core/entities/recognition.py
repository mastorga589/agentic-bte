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
try:
    import scispacy
    from scispacy.linking import EntityLinker
except ImportError:
    scispacy = None
    EntityLinker = None
    
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing_extensions import TypedDict, Literal

from ...config.settings import get_settings
from ...exceptions.entity_errors import EntityRecognitionError

logger = logging.getLogger(__name__)


class ModelCache:
    """
    Singleton class to cache spaCy models and avoid redundant loading
    """
    _instance = None
    _models_loaded = False
    _nlp = None
    _drug_disease_nlp = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.settings = get_settings()
            self.initialized = True
    
    def get_models(self):
        """Get cached models, loading them if necessary"""
        if not ModelCache._models_loaded:
            self._load_models()
            ModelCache._models_loaded = True
        return ModelCache._nlp, ModelCache._drug_disease_nlp
    
    def _load_models(self):
        """Load spaCy models with error handling"""
        logger.info("Loading spaCy models (cached)...")
        
        try:
            # Load scientific spaCy model
            ModelCache._nlp = spacy.load(self.settings.scispacy_large_model)
            
            # Try to add UMLS entity linker with retry logic for network issues
            umls_loaded = False
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    ModelCache._nlp.add_pipe(
                        "scispacy_linker", 
                        config={
                            "resolve_abbreviations": True, 
                            "linker_name": "umls"
                        }
                    )
                    logger.info(f"Loaded {self.settings.scispacy_large_model} with UMLS linker")
                    umls_loaded = True
                    break
                except Exception as linker_e:
                    if attempt < max_retries - 1:
                        logger.warning(f"UMLS linker attempt {attempt + 1}/{max_retries} failed: {linker_e}")
                        # Brief pause before retry
                        import time
                        time.sleep(1)
                    else:
                        logger.warning(f"Failed to load UMLS linker after {max_retries} attempts: {linker_e}")
                        logger.info(f"Loaded {self.settings.scispacy_large_model} without UMLS linker")
            
        except Exception as e:
            logger.warning(f"Failed to load {self.settings.scispacy_large_model}: {e}")
            ModelCache._nlp = None
        
        try:
            # Load drug/disease-specific model
            ModelCache._drug_disease_nlp = spacy.load(self.settings.scispacy_drug_disease_model)
            logger.info(f"Loaded {self.settings.scispacy_drug_disease_model}")
            
        except Exception as e:
            logger.warning(f"Failed to load {self.settings.scispacy_drug_disease_model}: {e}")
        
        # Check if we have at least one model
        if ModelCache._nlp is None and ModelCache._drug_disease_nlp is None:
            logger.warning("No spaCy models available - using fallback entity extraction")


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
        
        # Get cached spaCy models
        self.model_cache = ModelCache()
        self.nlp, self.drug_disease_nlp = self.model_cache.get_models()
    
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
            
            # Clean up any markdown formatting artifacts
            cleaned_response = response
            # Remove markdown code blocks
            if '```' in cleaned_response:
                # Extract content between code blocks
                import re
                code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', cleaned_response, re.IGNORECASE)
                if code_block_match:
                    cleaned_response = code_block_match.group(1).strip()
                else:
                    # Fallback: remove all ``` markers
                    cleaned_response = re.sub(r'```[^\n]*\n?', '', cleaned_response).strip()
            
            # Try to parse as JSON list
            import json
            try:
                bp_list = json.loads(cleaned_response)
                if isinstance(bp_list, list):
                    # Filter out empty strings and clean each item
                    clean_items = []
                    for item in bp_list:
                        if item and isinstance(item, str):
                            clean_item = str(item).strip()
                            # Skip if it contains formatting artifacts
                            if not any(artifact in clean_item.lower() for artifact in ['```', 'json', '\n']):
                                clean_items.append(clean_item)
                    return clean_items
            except json.JSONDecodeError:
                # If not valid JSON, try to extract from string representation
                if cleaned_response and cleaned_response not in ['""', '"[]"', '[]']:
                    # Remove quotes and brackets, split by comma
                    cleaned = cleaned_response.strip('"[]').replace('"', '')
                    if cleaned and '```' not in cleaned:
                        items = [item.strip() for item in cleaned.split(',') if item.strip()]
                        return [item for item in items if item and '```' not in item]
            
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
            - "general": Any other biomedical entity that does not fit the above categories

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
        Enhanced fallback entity extraction when spaCy models are unavailable
        
        Args:
            query: Input text
            
        Returns:
            List of potential entities extracted using comprehensive patterns
        """
        entities = []
        
        # Enhanced patterns for common biomedical entities
        patterns = {
            'diseases': r'\b(?:diabetes|cancer|parkinson|alzheimer\'?s?|hypertension|asthma|arthritis|pneumonia|tuberculosis|hepatitis|encephalitis|disease|disorder|syndrome)\w*\b',
            'drugs': r'\b(?:aspirin|metformin|insulin|morphine|codeine|ibuprofen|acetaminophen|warfarin|L-[Dd]opa|levodopa|brinzolamide|drug|drugs|medication|treatment|therapy|compound)s?\b',
            'processes': r'\b(?:metabolic process|metabolism|biosynthesis|signaling|pathway|apoptosis|cholinergic|dopaminergic|serotonergic|neurotransmitter|synaptic)\w*\b',
            'biological': r'\b(?:gene|protein|enzyme|receptor|neuron|brain|liver|heart|kidney)s?\b'
        }
        
        for category, pattern in patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend(matches)
        
        # Extract disease phrases (e.g., "Alzheimer's disease", "Type 2 diabetes")
        disease_phrases = re.findall(r'\b(?:[A-Z][a-z]+\'?s?\s+)?(?:disease|disorder|syndrome|cancer|diabetes)\b', query, re.IGNORECASE)
        entities.extend(disease_phrases)
        
        # Extract pathway/process phrases (e.g., "cholinergic pathway")
        pathway_phrases = re.findall(r'\b\w+\s+(?:pathway|process|signaling|system)\b', query, re.IGNORECASE)
        entities.extend(pathway_phrases)
        
        # Extract compound medical terms
        medical_terms = re.findall(r'\b(?:[A-Z][a-z]+\s+){1,2}(?:disease|pathway|receptor|protein|gene)\b', query, re.IGNORECASE)
        entities.extend(medical_terms)
        
        # Extract capitalized words that might be biomedical entities
        cap_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*\b', query)
        entities.extend(cap_words)
        
        # Extract terms that look like gene symbols (all caps, 2-10 characters)
        gene_symbols = re.findall(r'\b[A-Z]{2,10}\d?\b', query)
        entities.extend(gene_symbols)
        
        logger.warning(f"Using enhanced fallback extraction - found {len(entities)} potential entities")
        return entities
    
    def get_available_models(self) -> Dict[str, bool]:
        """
        Get information about available spaCy models
        
        Returns:
            Dictionary showing which models are available
        """
        return {
            "spacy_available": True,
            "scientific_model_loaded": self.nlp is not None,
            "drug_disease_model_loaded": self.drug_disease_nlp is not None,
            "llm_available": self.llm is not None
        }