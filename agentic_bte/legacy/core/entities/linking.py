"""
Entity Linking and Resolution Module

This module provides entity linking capabilities using multiple strategies:
- UMLS linking via SciSpaCy
- SRI Name Resolver API
- Entity ID to name resolution

Migrated and enhanced from the original BTE-LLM implementation.
"""

import re
import json
import logging
import requests
from typing import Dict, List, Optional, Any
from urllib.parse import quote

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Literal

from ...config.settings import get_settings
from ...exceptions.entity_errors import EntityLinkingError
from .recognition import BiomedicalEntityRecognizer

logger = logging.getLogger(__name__)


class EntityLinker:
    """
    Entity Linking using multiple strategies for biomedical entities
    
    This class provides entity linking capabilities using:
    - UMLS linking through SciSpaCy for general biomedical entities
    - SRI Name Resolver for comprehensive biomedical entity resolution
    - Context-aware ID selection using LLMs
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize entity linker
        
        Args:
            openai_api_key: OpenAI API key for LLM operations
        """
        self.settings = get_settings()
        self.openai_api_key = openai_api_key or self.settings.openai_api_key
        
        if not self.openai_api_key:
            raise EntityLinkingError("OpenAI API key is required for entity linking")
        
        # Initialize LLM for context-aware selection
        self.llm = ChatOpenAI(
            temperature=0,
            model=self.settings.openai_model,
            api_key=self.openai_api_key
        )
        
        # Initialize entity recognizer for classification
        self.recognizer = BiomedicalEntityRecognizer(openai_api_key)
    
    def link_entities(self, entity_list: List[str], query: str) -> Dict[str, Dict[str, str]]:
        """
        Link entities to their IDs and return entity types
        
        Args:
            entity_list: List of entity texts to link
            query: Original query for context
            
        Returns:
            Dictionary mapping entity text to {id, type}
        """
        entity_data = {}
        
        for entity in entity_list:
            if not entity or entity.strip() == "":
                continue
            
            try:
                # Classify entity type first
                entity_type = self.recognizer.classify_entity(query, entity)
                
                # Link entity based on type
                entity_id = self._link_single_entity(entity, entity_type, query)
                
                if entity_id:
                    entity_data[entity] = {
                        "id": entity_id,
                        "type": entity_type
                    }
                    logger.debug(f"Linked '{entity}' -> {entity_id} ({entity_type})")
                else:
                    logger.warning(f"Could not link entity: {entity}")
                    
            except Exception as e:
                logger.error(f"Error linking entity '{entity}': {e}")
                continue
        
        logger.info(f"Successfully linked {len(entity_data)} entities")
        return entity_data
    
    def _link_single_entity(self, entity: str, entity_type: str, query: str) -> str:
        """
        Link a single entity to its ID using appropriate strategy
        
        Args:
            entity: Entity text to link
            entity_type: Classified entity type
            query: Original query for context
            
        Returns:
            Entity ID or empty string if linking fails
        """
        if entity_type == "biologicalProcess":
            # Use GO-specific linking for biological processes
            return self._link_biological_process(entity, query)
        else:
            # Try UMLS linking first for other entity types
            umls_id = self._link_with_umls(entity, query)
            if umls_id:
                return umls_id
            
            # Fallback to SRI Name Resolver
            return self._link_with_sri_resolver(entity, query, is_bp=False)
    
    def _link_biological_process(self, entity: str, query: str) -> str:
        """
        Link biological process entities using GO-specific SRI Name Resolver
        
        Args:
            entity: Biological process entity text
            query: Original query for context
            
        Returns:
            GO ID or empty string if linking fails
        """
        candidates = self._get_sri_candidates(entity, is_bp=True)
        if not candidates:
            return ""
        
        return self._select_best_candidate(entity, candidates, query)
    
    def _link_with_umls(self, entity: str, query: str) -> str:
        """
        Link entity using UMLS via SciSpaCy
        
        Args:
            entity: Entity text to link
            query: Original query for context
            
        Returns:
            UMLS ID or empty string if linking fails
        """
        if not self.recognizer.nlp:
            logger.debug("UMLS linking not available - spaCy model not loaded")
            return ""
        
        try:
            doc = self.recognizer.nlp(entity)
            linker = self.recognizer.nlp.get_pipe("scispacy_linker")
            
            candidates = {}
            for ent in doc.ents:
                if ent._.kb_ents:  # Check if entity has linked knowledge base IDs
                    for id_info in ent._.kb_ents:
                        cui = id_info[0]
                        definition = self._remove_tui(str(linker.kb.cui_to_entity[cui]))
                        candidates[cui] = definition
            
            if not candidates:
                return ""
            
            # Use LLM to select most appropriate ID
            selected_cui = self._select_umls_id(entity, candidates, query)
            if selected_cui:
                return f"UMLS:{selected_cui}"
                
        except Exception as e:
            logger.warning(f"UMLS linking failed for '{entity}': {e}")
        
        return ""
    
    def _link_with_sri_resolver(self, entity: str, query: str, is_bp: bool = False) -> str:
        """
        Link entity using SRI Name Resolver
        
        Args:
            entity: Entity text to link
            query: Original query for context
            is_bp: Whether this is a biological process
            
        Returns:
            Entity ID or empty string if linking fails
        """
        candidates = self._get_sri_candidates(entity, is_bp=is_bp)
        if not candidates:
            return ""
        
        return self._select_best_candidate(entity, candidates, query)
    
    def _get_sri_candidates(self, entity: str, is_bp: bool = False, k: int = 50) -> List[Dict[str, Any]]:
        """
        Get candidate entities from SRI Name Resolver
        
        Args:
            entity: Entity text to search
            is_bp: Whether to restrict to biological processes
            k: Maximum number of candidates
            
        Returns:
            List of candidate entities with labels, CURIEs, and scores
        """
        try:
            base_url = self.settings.sri_name_resolver_url
            processed_entity = quote(entity)
            
            url = f"{base_url}?string={processed_entity}&autocomplete=true&limit={k}"
            
            if is_bp:
                url += "&only_prefixes=GO&biolink_type=BiologicalProcess"
            
            response = requests.get(url, headers={"accept": "application/json"})
            response.raise_for_status()
            
            candidate_list = response.json()
            candidates = []
            
            for item in candidate_list:
                candidate = {
                    "label": item.get("label", ""),
                    "curie": item.get("curie", ""),
                    "score": item.get("score", 0)
                }
                candidates.append(candidate)
            
            logger.debug(f"Found {len(candidates)} SRI candidates for '{entity}'")
            return candidates
            
        except Exception as e:
            logger.warning(f"SRI Name Resolver failed for '{entity}': {e}")
            return []
    
    def _select_umls_id(self, entity: str, candidates: Dict[str, str], query: str) -> str:
        """
        Select best UMLS ID using LLM
        
        Args:
            entity: Entity text
            candidates: Dictionary of CUI -> definition
            query: Original query for context
            
        Returns:
            Selected CUI or empty string
        """
        try:
            select_prompt = f"""You are a smart biomedical assistant that can understand the context and the intent behind a query. 
                        Be careful when choosing IDs for entities that can refer to different concepts (for example, HIV can refer either to the virus or the disease; you MUST choose the most appropriate concept/definition based on the query). 
                        Use the context and the intent behind the query to choose the most appropriate ID. 
                        
                        Here is the complete query: {query}
                        Select the one most appropriate ID/CUI for "{entity}" from the list below:
                        {candidates}
                        
                        If none of the choices are appropriate, return "".
                        Otherwise, return only the ID/CUI (format: C1234567).
                        """

            # LLM selects most appropriate ID from list
            selected_id = self.llm.invoke(select_prompt).content.strip()
            
            # Extract just the UMLS CUI using regex
            match = re.search(r"C\d{7}", selected_id)
            if match:
                cui = match.group(0)
                definition = candidates.get(cui, "")
                logger.debug(f"Selected UMLS ID for '{entity}': {cui} - {definition[:100]}...")
                return cui
            
        except Exception as e:
            logger.warning(f"UMLS ID selection failed for '{entity}': {e}")
        
        return ""
    
    def _select_best_candidate(self, entity: str, candidates: List[Dict[str, Any]], query: str) -> str:
        """
        Select best candidate using LLM
        
        Args:
            entity: Entity text
            candidates: List of candidate dictionaries
            query: Original query for context
            
        Returns:
            Selected entity ID or empty string
        """
        if not candidates:
            return ""
        
        try:
            # Create choices for structured output
            choices = [""]
            for candidate in candidates:
                curie = candidate.get("curie", "")
                if curie:
                    choices.append(curie)
            
            if len(choices) <= 1:
                return ""
            
            class SelectedID(TypedDict):
                """The most appropriate ID from the given candidates"""
                selectedID: str
            
            select_prompt = f"""You are a smart biomedical assistant that can understand the context and the intent behind a query. 
                        Be careful when choosing IDs for entities that can refer to different concepts (for example, HIV can refer either to the virus or the disease; you MUST choose the most appropriate concept/definition based on the query). 
                        Use the context and the intent behind the query to choose the most appropriate ID. 
                        
                        Here is the complete query: {query}
                        Select the one most appropriate ID for "{entity}" from the candidates below:
                        {json.dumps(candidates, indent=2)}
                        
                        If none of the choices are appropriate, return "".
                        Otherwise, return only the ID/CURIE.
                        """

            # Use simple text response instead of structured output for flexibility
            selected_id = self.llm.invoke(select_prompt).content.strip()
            
            # Find matching candidate
            for candidate in candidates:
                candidate_curie = candidate.get("curie", "")
                if candidate_curie and candidate_curie in selected_id:
                    logger.debug(f"Selected ID for '{entity}': {candidate_curie}")
                    return candidate_curie
            
        except Exception as e:
            logger.warning(f"Candidate selection failed for '{entity}': {e}")
        
        return ""
    
    @staticmethod
    def _remove_tui(text: str) -> str:
        """
        Remove TUI (Type Unique Identifier) from UMLS definition text
        
        Args:
            text: Text that may contain TUI
            
        Returns:
            Text with TUI removed
        """
        parts = text.split("TUI", 1)
        return parts[0].strip()


class EntityResolver:
    """
    Entity ID to Name Resolution using SRI Name Resolver and BTE metadata
    """
    
    def __init__(self):
        """
        Initialize entity resolver
        """
        self.settings = get_settings()
        self._cache = {}  # Simple in-memory cache
    
    def resolve_single(self, entity_id: str) -> Optional[str]:
        """
        Resolve a single entity ID to its human-readable name
        
        Args:
            entity_id: Entity ID to resolve (e.g., UMLS:C0011847)
            
        Returns:
            Human-readable name or None if resolution fails
        """
        # Check cache first
        if entity_id in self._cache:
            return self._cache[entity_id]
        
        try:
            # Use SRI Name Resolver reverse lookup
            base_url = self.settings.sri_name_resolver_url
            url = f"{base_url}?curie={quote(entity_id)}"
            
            response = requests.get(url, headers={"accept": "application/json"})
            response.raise_for_status()
            
            results = response.json()
            if results and len(results) > 0:
                name = results[0].get("label", "")
                if name:
                    self._cache[entity_id] = name
                    logger.debug(f"Resolved {entity_id} -> {name}")
                    return name
            
        except Exception as e:
            logger.warning(f"Failed to resolve entity ID '{entity_id}': {e}")
        
        return None
    
    def resolve_multiple(self, entity_ids: List[str]) -> Dict[str, str]:
        """
        Resolve multiple entity IDs to names
        
        Args:
            entity_ids: List of entity IDs to resolve
            
        Returns:
            Dictionary mapping entity IDs to resolved names
        """
        resolved = {}
        
        for entity_id in entity_ids:
            if entity_id:
                name = self.resolve_single(entity_id)
                if name:
                    resolved[entity_id] = name
        
        logger.info(f"Resolved {len(resolved)}/{len(entity_ids)} entity names")
        return resolved
    
    def clear_cache(self):
        """Clear the resolution cache"""
        self._cache.clear()
        logger.debug("Entity resolution cache cleared")

