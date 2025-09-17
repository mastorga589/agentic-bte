"""
Entity Linking Module

This module provides entity linking capabilities to connect extracted biomedical
entities to knowledge bases like UMLS, MESH, and others.

Migrated and enhanced from original BTE-LLM implementations.
"""

import logging
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import quote

from .entity_recognition import ExtractedEntity, LinkedEntity
from ...config.settings import get_settings
from ...exceptions.base import ExternalServiceError

logger = logging.getLogger(__name__)


@dataclass
class LinkingCandidate:
    """
    Represents a candidate entity from a knowledge base
    """
    entity_id: str
    name: str
    description: Optional[str] = None
    source: Optional[str] = None
    score: float = 0.0
    semantic_types: Optional[List[str]] = None
    attributes: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class EntityLinker:
    """
    Entity Linker for biomedical entities
    
    This class provides entity linking using multiple knowledge bases:
    1. UMLS (Unified Medical Language System) via SciSpaCy
    2. SRI Name Resolver API
    3. BioThings APIs
    4. Fallback string matching
    
    Migrated from original BTE-LLM implementations.
    """
    
    def __init__(self, timeout: int = 30):
        """
        Initialize entity linker
        
        Args:
            timeout: Request timeout in seconds
        """
        self.settings = get_settings()
        self.timeout = timeout
        
        # Initialize SciSpaCy linker if available
        self.umls_linker = None
        self._init_umls_linker()
        
        # Setup session for API requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'agentic-bte/1.0.0',
            'Accept': 'application/json'
        })
    
    def _init_umls_linker(self):
        """Initialize UMLS linker via SciSpaCy"""
        try:
            from scispacy.linking import EntityLinker as SciSpaCyLinker
            self.umls_linker = SciSpaCyLinker(
                resolve_abbreviations=True, 
                name="umls"
            )
            logger.info("Initialized UMLS entity linker")
        except ImportError:
            logger.warning("SciSpaCy not available - UMLS linking disabled")
        except Exception as e:
            logger.warning(f"Could not initialize UMLS linker: {e}")
    
    def link_with_umls(self, entity: ExtractedEntity) -> List[LinkingCandidate]:
        """
        Link entity using UMLS via SciSpaCy
        
        Args:
            entity: Extracted entity to link
            
        Returns:
            List of linking candidates
        """
        if not self.umls_linker:
            return []
        
        try:
            candidates = []
            
            # Get UMLS candidates
            umls_entities = self.umls_linker.get_candidates(entity.text)
            
            for umls_entity in umls_entities:
                candidate = LinkingCandidate(
                    entity_id=umls_entity.concept_id,
                    name=umls_entity.canonical_name,
                    description=umls_entity.definition,
                    source="UMLS",
                    score=umls_entity.score if hasattr(umls_entity, 'score') else 0.8,
                    semantic_types=umls_entity.types if hasattr(umls_entity, 'types') else None
                )
                candidates.append(candidate)
            
            logger.debug(f"Found {len(candidates)} UMLS candidates for '{entity.text}'")
            return candidates
            
        except Exception as e:
            logger.error(f"Error linking with UMLS: {e}")
            return []
    
    def link_with_sri_resolver(self, entity: ExtractedEntity) -> List[LinkingCandidate]:
        """
        Link entity using SRI Name Resolver API
        
        Args:
            entity: Extracted entity to link
            
        Returns:
            List of linking candidates
        """
        try:
            # SRI Name Resolver endpoints
            endpoints = [
                "https://name-resolver-sri.renci.org/lookup",
                "https://nodenormalization-sri.renci.org/get_normalized_nodes"
            ]
            
            candidates = []
            
            for endpoint in endpoints:
                try:
                    params = {
                        'string': entity.text,
                        'limit': 10
                    }
                    
                    response = self.session.get(
                        endpoint,
                        params=params,
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    # Parse response based on endpoint
                    if 'lookup' in endpoint:
                        results = data.get(entity.text, {})
                        for result in results:
                            candidate = LinkingCandidate(
                                entity_id=result.get('curie', ''),
                                name=result.get('label', entity.text),
                                source="SRI_Resolver",
                                score=0.7,
                                semantic_types=result.get('types', [])
                            )
                            candidates.append(candidate)
                    else:
                        # Handle normalization endpoint format
                        for node_id, node_data in data.items():
                            if isinstance(node_data, dict):
                                candidate = LinkingCandidate(
                                    entity_id=node_data.get('id', {}).get('identifier', ''),
                                    name=node_data.get('id', {}).get('label', entity.text),
                                    source="SRI_Normalizer",
                                    score=0.7
                                )
                                candidates.append(candidate)
                    
                    if candidates:
                        break  # Use first successful endpoint
                        
                except requests.RequestException as e:
                    logger.warning(f"SRI resolver request failed for {endpoint}: {e}")
                    continue
            
            logger.debug(f"Found {len(candidates)} SRI candidates for '{entity.text}'")
            return candidates
            
        except Exception as e:
            logger.error(f"Error linking with SRI resolver: {e}")
            return []
    
    def link_with_biothings(self, entity: ExtractedEntity) -> List[LinkingCandidate]:
        """
        Link entity using BioThings APIs
        
        Args:
            entity: Extracted entity to link
            
        Returns:
            List of linking candidates
        """
        try:
            candidates = []
            
            # BioThings API endpoints for different entity types
            endpoints = {
                "Gene": "https://mygene.info/v3/query",
                "Chemical": "https://mychem.info/v1/query", 
                "Disease": "https://mydisease.info/v1/query"
            }
            
            entity_type = entity.entity_type or "Gene"  # Default fallback
            endpoint = endpoints.get(entity_type)
            
            if not endpoint:
                return []
            
            params = {
                'q': entity.text,
                'size': 5,
                'fields': 'name,summary,symbol'
            }
            
            response = self.session.get(
                endpoint,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            hits = data.get('hits', [])
            for hit in hits:
                candidate = LinkingCandidate(
                    entity_id=hit.get('_id', ''),
                    name=hit.get('name') or hit.get('symbol') or entity.text,
                    description=hit.get('summary'),
                    source=f"BioThings_{entity_type}",
                    score=hit.get('_score', 0.6)
                )
                candidates.append(candidate)
            
            logger.debug(f"Found {len(candidates)} BioThings candidates for '{entity.text}'")
            return candidates
            
        except requests.RequestException as e:
            logger.warning(f"BioThings API request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error linking with BioThings: {e}")
            return []
    
    def select_best_candidate(self, candidates: List[LinkingCandidate], 
                            entity: ExtractedEntity) -> Optional[LinkingCandidate]:
        """
        Select the best linking candidate from multiple options
        
        Args:
            candidates: List of linking candidates
            entity: Original extracted entity
            
        Returns:
            Best candidate or None
        """
        if not candidates:
            return None
        
        # Score candidates based on multiple factors
        scored_candidates = []
        
        for candidate in candidates:
            score = candidate.score
            
            # Boost score for exact name matches
            if candidate.name.lower() == entity.text.lower():
                score += 0.3
            
            # Boost score for partial matches
            elif entity.text.lower() in candidate.name.lower():
                score += 0.1
            
            # Boost score for preferred sources
            source_weights = {
                "UMLS": 0.2,
                "SRI_Resolver": 0.15,
                "BioThings_Gene": 0.1,
                "BioThings_Chemical": 0.1,
                "BioThings_Disease": 0.1
            }
            score += source_weights.get(candidate.source, 0.0)
            
            scored_candidates.append((score, candidate))
        
        # Sort by score and return best
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_candidate = scored_candidates[0][1]
        
        logger.debug(f"Selected best candidate for '{entity.text}': {best_candidate.name} ({best_candidate.source})")
        return best_candidate
    
    def link_entity(self, entity: ExtractedEntity) -> LinkedEntity:
        """
        Link entity to knowledge bases
        
        Args:
            entity: Extracted entity to link
            
        Returns:
            LinkedEntity with linking information
        """
        try:
            all_candidates = []
            
            # Try different linking methods
            linking_methods = [
                self.link_with_umls,
                self.link_with_sri_resolver,
                self.link_with_biothings
            ]
            
            for method in linking_methods:
                try:
                    candidates = method(entity)
                    all_candidates.extend(candidates)
                except Exception as e:
                    logger.warning(f"Linking method {method.__name__} failed: {e}")
            
            # Select best candidate
            best_candidate = self.select_best_candidate(all_candidates, entity)
            
            # Create LinkedEntity
            linked_entity = LinkedEntity(
                text=entity.text,
                start=entity.start,
                end=entity.end,
                label=entity.label,
                confidence=entity.confidence,
                entity_type=entity.entity_type,
                cui=entity.cui,
                entity_id=entity.entity_id,
                attributes=entity.attributes,
                linked_concepts=[candidate.to_dict() for candidate in all_candidates]
            )
            
            # Add best candidate information
            if best_candidate:
                linked_entity.entity_id = best_candidate.entity_id
                linked_entity.umls_cui = best_candidate.entity_id if best_candidate.source == "UMLS" else None
                
                # Extract specific IDs based on source
                if "MESH" in best_candidate.entity_id:
                    linked_entity.mesh_id = best_candidate.entity_id
                elif "NCBIGene" in best_candidate.entity_id:
                    linked_entity.ncbi_gene_id = best_candidate.entity_id
            
            logger.debug(f"Successfully linked entity '{entity.text}' with {len(all_candidates)} candidates")
            return linked_entity
            
        except Exception as e:
            logger.error(f"Error linking entity '{entity.text}': {e}")
            # Return original entity as LinkedEntity if linking fails
            return LinkedEntity(
                text=entity.text,
                start=entity.start,
                end=entity.end,
                label=entity.label,
                confidence=entity.confidence,
                entity_type=entity.entity_type,
                cui=entity.cui,
                entity_id=entity.entity_id,
                attributes=entity.attributes,
                linked_concepts=[]
            )
    
    def batch_link_entities(self, entities: List[ExtractedEntity]) -> List[LinkedEntity]:
        """
        Link multiple entities in batch
        
        Args:
            entities: List of extracted entities
            
        Returns:
            List of linked entities
        """
        linked_entities = []
        
        for entity in entities:
            linked = self.link_entity(entity)
            linked_entities.append(linked)
        
        logger.info(f"Batch linked {len(entities)} entities")
        return linked_entities


class EntityResolver:
    """
    Entity Resolver for resolving IDs to human-readable names
    
    This class resolves biomedical entity IDs to their canonical names
    using various APIs and knowledge bases.
    
    Migrated from original BTE-LLM implementations.
    """
    
    def __init__(self, timeout: int = 30):
        """
        Initialize entity resolver
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'agentic-bte/1.0.0',
            'Accept': 'application/json'
        })
        
        # Cache for resolved names
        self._name_cache = {}
    
    def resolve_entity_name(self, entity_id: str) -> Optional[str]:
        """
        Resolve entity ID to human-readable name
        
        Args:
            entity_id: Entity identifier (e.g., MONDO:0005148, CHEMBL:123)
            
        Returns:
            Human-readable name or None
        """
        # Check cache first
        if entity_id in self._name_cache:
            return self._name_cache[entity_id]
        
        try:
            # Try different resolution methods based on ID prefix
            name = None
            
            if entity_id.startswith('MONDO:'):
                name = self._resolve_mondo_id(entity_id)
            elif entity_id.startswith('CHEMBL:') or entity_id.startswith('CHEBI:'):
                name = self._resolve_chemical_id(entity_id)
            elif entity_id.startswith('NCBIGene:') or entity_id.startswith('HGNC:'):
                name = self._resolve_gene_id(entity_id)
            elif entity_id.startswith('HP:'):
                name = self._resolve_phenotype_id(entity_id)
            else:
                # Try generic SRI resolver
                name = self._resolve_with_sri(entity_id)
            
            # Fallback: extract name from ID
            if not name:
                name = self._generate_fallback_name(entity_id)
            
            # Cache the result
            if name:
                self._name_cache[entity_id] = name
            
            return name
            
        except Exception as e:
            logger.warning(f"Error resolving entity ID '{entity_id}': {e}")
            return self._generate_fallback_name(entity_id)
    
    def _resolve_mondo_id(self, mondo_id: str) -> Optional[str]:
        """Resolve MONDO disease ID"""
        try:
            url = f"https://mydisease.info/v1/disease/{mondo_id}"
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                return data.get('name') or data.get('label')
        except Exception as e:
            logger.debug(f"MONDO resolution failed: {e}")
        return None
    
    def _resolve_chemical_id(self, chem_id: str) -> Optional[str]:
        """Resolve chemical ID (CHEMBL, CHEBI, etc.)"""
        try:
            url = f"https://mychem.info/v1/chem/{chem_id}"
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                return (data.get('name') or 
                       data.get('chembl', {}).get('pref_name') or
                       data.get('chebi', {}).get('name'))
        except Exception as e:
            logger.debug(f"Chemical resolution failed: {e}")
        return None
    
    def _resolve_gene_id(self, gene_id: str) -> Optional[str]:
        """Resolve gene ID (NCBIGene, HGNC, etc.)"""
        try:
            url = f"https://mygene.info/v3/gene/{gene_id}"
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                return data.get('name') or data.get('symbol')
        except Exception as e:
            logger.debug(f"Gene resolution failed: {e}")
        return None
    
    def _resolve_phenotype_id(self, hp_id: str) -> Optional[str]:
        """Resolve Human Phenotype Ontology ID"""
        try:
            # Use OLS (Ontology Lookup Service)
            url = f"https://www.ebi.ac.uk/ols/api/ontologies/hp/terms/{quote(hp_id, safe='')}"
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                return data.get('label')
        except Exception as e:
            logger.debug(f"HP resolution failed: {e}")
        return None
    
    def _resolve_with_sri(self, entity_id: str) -> Optional[str]:
        """Resolve using SRI Name Resolver"""
        try:
            url = "https://name-resolver-sri.renci.org/reverse_lookup"
            params = {'curie': entity_id}
            response = self.session.get(url, params=params, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                return data.get(entity_id, {}).get('label')
        except Exception as e:
            logger.debug(f"SRI resolution failed: {e}")
        return None
    
    def _generate_fallback_name(self, entity_id: str) -> str:
        """Generate fallback name from entity ID"""
        # Extract meaningful part from ID
        if ':' in entity_id:
            parts = entity_id.split(':', 1)
            return f"{parts[0]} {parts[1]}"
        return entity_id
    
    def batch_resolve_names(self, entity_ids: List[str]) -> Dict[str, str]:
        """
        Resolve multiple entity IDs to names
        
        Args:
            entity_ids: List of entity IDs
            
        Returns:
            Dictionary mapping IDs to names
        """
        results = {}
        
        for entity_id in entity_ids:
            name = self.resolve_entity_name(entity_id)
            if name:
                results[entity_id] = name
        
        logger.info(f"Resolved {len(results)}/{len(entity_ids)} entity names")
        return results


# Convenience functions
def link_entities_to_kb(entities: List[ExtractedEntity], timeout: int = 30) -> List[LinkedEntity]:
    """
    Convenience function to link entities to knowledge bases
    
    Args:
        entities: List of extracted entities
        timeout: Request timeout in seconds
        
    Returns:
        List of linked entities
    """
    linker = EntityLinker(timeout)
    return linker.batch_link_entities(entities)


def resolve_entity_names(entity_ids: List[str], timeout: int = 30) -> Dict[str, str]:
    """
    Convenience function to resolve entity IDs to names
    
    Args:
        entity_ids: List of entity IDs
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary mapping IDs to names
    """
    resolver = EntityResolver(timeout)
    return resolver.batch_resolve_names(entity_ids)