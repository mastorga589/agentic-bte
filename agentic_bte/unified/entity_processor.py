"""
Unified Entity Processing System

This module consolidates entity extraction, linking, resolution, and context 
management from all existing implementations into a single, comprehensive
entity processing pipeline.
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass

from .config import UnifiedConfig
from .types import (
    BiomedicalEntity, EntityContext, EntityType, ExecutionStep, 
    ExecutionStatus, PerformanceMetrics
)

# Import existing components that we'll consolidate
from ..core.entities.bio_ner import BioNERTool
from ..core.entities.linking import EntityLinker  
from ..core.entities.recognition import BiomedicalEntityRecognizer
from ..core.queries.mcp_integration import call_mcp_tool

logger = logging.getLogger(__name__)


@dataclass
class EntityExtractionResult:
    """Result from entity extraction process"""
    entities: List[BiomedicalEntity]
    extraction_method: str
    confidence: float
    execution_time: float
    metadata: Dict[str, Any]


class GenericEntityResolver:
    """
    Resolves generic entity terms to specific biomedical entities
    
    Handles cases like "drugs" -> specific drug IDs, "genes" -> specific gene IDs
    """
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        
        # Generic term mappings to entity types
        self.generic_mappings = {
            # Drug-related terms
            "drugs": EntityType.DRUG,
            "medications": EntityType.DRUG,
            "compounds": EntityType.CHEMICAL,
            "therapeutics": EntityType.DRUG,
            "treatments": EntityType.DRUG,
            
            # Gene-related terms
            "genes": EntityType.GENE,
            "proteins": EntityType.PROTEIN,
            "polypeptides": EntityType.PROTEIN,
            
            # Disease-related terms
            "diseases": EntityType.DISEASE,
            "disorders": EntityType.DISEASE,
            "conditions": EntityType.DISEASE,
            "syndromes": EntityType.DISEASE,
            
            # Process-related terms
            "processes": EntityType.PROCESS,
            "pathways": EntityType.PATHWAY,
            "mechanisms": EntityType.PROCESS,
            "functions": EntityType.PROCESS
        }
        
        # Knowledge base for common entity resolutions
        self.knowledge_base = self._build_knowledge_base()
    
    def _build_knowledge_base(self) -> Dict[str, List[str]]:
        """Build knowledge base of common generic -> specific mappings"""
        return {
            "antipsychotics": [
                "CHEMBL:CHEMBL54", "DRUGBANK:DB00409",  # haloperidol
                "CHEMBL:CHEMBL85", "DRUGBANK:DB00734",  # risperidone
                "CHEMBL:CHEMBL1201584", "DRUGBANK:DB00013"  # olanzapine
            ],
            "antibiotics": [
                "CHEMBL:CHEMBL180", "DRUGBANK:DB00254",  # doxycycline
                "CHEMBL:CHEMBL1200518", "DRUGBANK:DB01017", # streptomycin
                "CHEMBL:CHEMBL130", "DRUGBANK:DB00446"   # chloramphenicol
            ],
            "neurotransmitter_receptors": [
                "HGNC:3023", "ENSEMBL:ENSG00000149295",  # DRD2
                "HGNC:3024", "ENSEMBL:ENSG00000151577",  # DRD3
                "HGNC:3358", "ENSEMBL:ENSG00000148680"   # HTR2A
            ],
            "metabolic_enzymes": [
                "HGNC:2625", "ENSEMBL:ENSG00000100197",  # CYP2D6
                "HGNC:2640", "ENSEMBL:ENSG00000160868",  # CYP3A4
                "HGNC:2176", "ENSEMBL:ENSG00000116604"   # COMT
            ]
        }
    
    async def resolve_generic_terms(self, query: str, context: EntityContext) -> Dict[str, List[str]]:
        """
        Resolve generic terms in query to specific entity IDs
        
        Args:
            query: Query text to analyze
            context: Current entity context
            
        Returns:
            Dictionary mapping generic terms to specific entity IDs
        """
        resolutions = {}
        query_lower = query.lower()
        
        # Check for generic terms in query
        for generic_term, entity_type in self.generic_mappings.items():
            if generic_term in query_lower:
                logger.info(f"Found generic term: {generic_term}")
                
                # First try knowledge base
                if generic_term in self.knowledge_base:
                    resolutions[generic_term] = self.knowledge_base[generic_term]
                    continue
                
                # Try to resolve from existing entities in context
                matching_entities = context.get_entities_by_type(entity_type)
                if matching_entities:
                    resolutions[generic_term] = [e.entity_id for e in matching_entities[:10]]
                    continue
                
                # Try semantic resolution
                semantic_ids = await self._semantic_resolution(generic_term, entity_type)
                if semantic_ids:
                    resolutions[generic_term] = semantic_ids
        
        logger.info(f"Resolved {len(resolutions)} generic terms")
        return resolutions
    
    async def _semantic_resolution(self, generic_term: str, entity_type: EntityType) -> List[str]:
        """Use semantic search to resolve generic terms"""
        try:
            # This would integrate with a semantic search system
            # For now, return empty list as fallback
            logger.debug(f"Semantic resolution not yet implemented for {generic_term}")
            return []
        except Exception as e:
            logger.error(f"Semantic resolution failed for {generic_term}: {e}")
            return []


class PlaceholderSystem:
    """
    Manages placeholders that reference results from previous subqueries
    
    Enables queries like "What genes do {drugs_from_subquery_1} target?"
    """
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.placeholders: Dict[str, List[str]] = {}
        self.subquery_results: List[Dict[str, Any]] = []
    
    def extract_placeholders(self, query: str) -> List[str]:
        """Extract placeholder patterns from query"""
        import re
        placeholder_pattern = r'\{([^}]+)\}'
        matches = re.findall(placeholder_pattern, query)
        return matches
    
    def resolve_placeholders(self, query: str, entity_context: EntityContext) -> str:
        """Replace placeholders with actual entity references"""
        placeholders = self.extract_placeholders(query)
        resolved_query = query
        
        for placeholder in placeholders:
            if placeholder in entity_context.placeholder_mappings:
                entity_ids = entity_context.placeholder_mappings[placeholder]
                # Replace with first few entity IDs or names
                if entity_ids:
                    replacement = ", ".join(entity_ids[:3])  # Limit to 3 for readability
                    resolved_query = resolved_query.replace(f"{{{placeholder}}}", replacement)
        
        return resolved_query
    
    def create_placeholders_from_results(self, subquery_index: int, results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Create placeholders from subquery results"""
        placeholders = {}
        
        # Extract entities by type from results
        extracted_entities = self._extract_entities_by_type(results)
        
        for entity_type, entity_ids in extracted_entities.items():
            if len(entity_ids) >= 2:  # Only create placeholder if multiple entities
                placeholder_name = f"{entity_type}s_from_subquery_{subquery_index + 1}"
                placeholders[placeholder_name] = entity_ids[:10]  # Limit to 10
        
        return placeholders
    
    def _extract_entities_by_type(self, results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Extract entities organized by type from API results"""
        entities_by_type = {}
        
        for result in results:
            # Extract from knowledge graph nodes
            if 'knowledge_graph' in result and 'nodes' in result['knowledge_graph']:
                nodes = result['knowledge_graph']['nodes']
                for node_id, node_data in nodes.items():
                    categories = node_data.get('categories', [])
                    for category in categories:
                        entity_type = self._biolink_category_to_type(category)
                        if entity_type:
                            if entity_type not in entities_by_type:
                                entities_by_type[entity_type] = []
                            if node_id not in entities_by_type[entity_type]:
                                entities_by_type[entity_type].append(node_id)
        
        return entities_by_type
    
    def _biolink_category_to_type(self, category: str) -> Optional[str]:
        """Map biolink category to entity type string"""
        mapping = {
            'biolink:SmallMolecule': 'drug',
            'biolink:Gene': 'gene',
            'biolink:Protein': 'protein',
            'biolink:Disease': 'disease',
            'biolink:BiologicalProcess': 'process',
            'biolink:Pathway': 'pathway'
        }
        return mapping.get(category)


class UnifiedEntityProcessor:
    """
    Unified entity processing pipeline that consolidates all entity-related
    operations from different implementations
    """
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        
        # Initialize component processors
        self.bio_ner = BioNERTool() if config.domain.enable_entity_extraction else None
        self.entity_linker = EntityLinker() if config.domain.enable_entity_linking else None
        # Initialize entity recognizer with error handling
        if config.domain.enable_entity_extraction:
            try:
                self.entity_recognizer = BiomedicalEntityRecognizer()
            except Exception as e:
                logger.warning(f"Failed to initialize BiomedicalEntityRecognizer: {e}")
                self.entity_recognizer = None
        else:
            self.entity_recognizer = None
        
        # Initialize unified components
        self.generic_resolver = GenericEntityResolver(config)
        self.placeholder_system = PlaceholderSystem(config)
        
        # Entity type hierarchy for consolidation
        self.entity_hierarchies = {
            EntityType.GENE: ['HGNC', 'ENSEMBL', 'NCBIGene'],
            EntityType.PROTEIN: ['UniProtKB', 'PR'],
            EntityType.DRUG: ['DRUGBANK', 'CHEBI', 'PUBCHEM.COMPOUND'],
            EntityType.DISEASE: ['MONDO', 'DOID', 'UMLS', 'HP'],
            EntityType.PATHWAY: ['REACTOME', 'KEGG', 'GO'],
            EntityType.PROCESS: ['GO']
        }
        
        logger.info("Unified entity processor initialized")
    
    async def process_entities(self, query: str, context: Optional[EntityContext] = None) -> EntityContext:
        """
        Main entry point for unified entity processing
        
        Args:
            query: Input query text
            context: Optional existing entity context to build upon
            
        Returns:
            Comprehensive entity context with all processing applied
        """
        start_time = time.time()
        
        # Initialize context if not provided
        if context is None:
            context = EntityContext([])
        
        logger.info(f"Processing entities for query: {query[:100]}...")
        
        # Step 1: Extract entities using multiple methods
        if self.config.domain.enable_entity_extraction:
            extraction_results = await self._extract_entities_multiple_methods(query)
            await self._consolidate_extracted_entities(extraction_results, context)
        
        # Step 2: Link entities to knowledge bases
        if self.config.domain.enable_entity_linking:
            await self._link_entities(context)
        
        # Step 3: Resolve entity names to human-readable format
        if self.config.domain.enable_entity_resolution:
            await self._resolve_entity_names(context)
        
        # Step 4: Resolve generic entity mappings
        if self.config.domain.enable_generic_entity_mapping:
            generic_resolutions = await self.generic_resolver.resolve_generic_terms(query, context)
            context.generic_resolutions.update(generic_resolutions)
        
        # Step 5: Process placeholders if present
        placeholder_query = self.placeholder_system.resolve_placeholders(query, context)
        if placeholder_query != query:
            logger.info(f"Resolved placeholders in query: {placeholder_query}")
        
        # Step 6: Apply entity filtering and ranking
        await self._filter_and_rank_entities(context)
        
        processing_time = time.time() - start_time
        logger.info(f"Entity processing completed in {processing_time:.2f}s with {len(context.entities)} entities")
        
        return context
    
    async def _extract_entities_multiple_methods(self, query: str) -> List[EntityExtractionResult]:
        """Extract entities using multiple methods in parallel"""
        extraction_tasks = []
        
        # Method 1: BioNER via MCP (primary)
        if self.bio_ner:
            extraction_tasks.append(self._extract_via_bio_ner_mcp(query))
        
        # Method 2: Direct BioNER (fallback)
        if self.entity_recognizer:
            extraction_tasks.append(self._extract_via_recognizer(query))
        
        # Method 3: Simple regex-based extraction (fallback)
        extraction_tasks.append(self._extract_via_simple_patterns(query))
        
        # Execute all methods concurrently
        if self.config.performance.enable_parallel_execution:
            results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        else:
            results = []
            for task in extraction_tasks:
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    results.append(e)
        
        # Filter out exceptions and return valid results
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        if not valid_results:
            logger.warning("All entity extraction methods failed")
            return []
        
        return valid_results
    
    async def _extract_via_bio_ner_mcp(self, query: str) -> EntityExtractionResult:
        """Extract entities using BioNER via MCP integration"""
        start_time = time.time()
        
        try:
            response = await call_mcp_tool("bio_ner", query=query)
            
            entities = []
            if 'entities' in response:
                for entity_data in response['entities']:
                    entity = BiomedicalEntity(
                        name=entity_data.get('name', ''),
                        entity_id=entity_data.get('id', ''),
                        entity_type=self._map_entity_type(entity_data.get('type', 'unknown')),
                        confidence=entity_data.get('confidence', 0.8),
                        source="bio_ner_mcp",
                        categories=[entity_data.get('type', 'unknown')]
                    )
                    entities.append(entity)
            
            execution_time = time.time() - start_time
            
            return EntityExtractionResult(
                entities=entities,
                extraction_method="bio_ner_mcp",
                confidence=0.9,
                execution_time=execution_time,
                metadata={"tool": "bio_ner", "via": "mcp"}
            )
            
        except Exception as e:
            logger.error(f"BioNER MCP extraction failed: {e}")
            raise
    
    async def _extract_via_recognizer(self, query: str) -> EntityExtractionResult:
        """Extract entities using direct entity recognizer"""
        start_time = time.time()
        
        try:
            # This would use the direct entity recognizer
            entities = []  # Placeholder implementation
            
            execution_time = time.time() - start_time
            
            return EntityExtractionResult(
                entities=entities,
                extraction_method="direct_recognizer",
                confidence=0.7,
                execution_time=execution_time,
                metadata={"tool": "entity_recognizer"}
            )
            
        except Exception as e:
            logger.error(f"Direct recognizer extraction failed: {e}")
            raise
    
    async def _extract_via_simple_patterns(self, query: str) -> EntityExtractionResult:
        """Extract entities using simple pattern matching as fallback"""
        start_time = time.time()
        
        entities = []
        
        # Simple pattern-based extraction
        import re
        
        # Drug patterns
        drug_patterns = [
            r'\b[A-Za-z]+mycin\b',  # antibiotics ending in -mycin
            r'\b[A-Za-z]+cillin\b', # antibiotics ending in -cillin
            r'\b[A-Za-z]+cycline\b' # antibiotics ending in -cycline
        ]
        
        for pattern in drug_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                entity = BiomedicalEntity(
                    name=match.lower(),
                    entity_id=f"PATTERN:{match.upper()}",
                    entity_type=EntityType.DRUG,
                    confidence=0.5,
                    source="pattern_matching"
                )
                entities.append(entity)
        
        execution_time = time.time() - start_time
        
        return EntityExtractionResult(
            entities=entities,
            extraction_method="simple_patterns",
            confidence=0.5,
            execution_time=execution_time,
            metadata={"patterns_used": len(drug_patterns)}
        )
    
    async def _consolidate_extracted_entities(self, extraction_results: List[EntityExtractionResult], context: EntityContext):
        """Consolidate entities from multiple extraction methods"""
        # Collect all entities
        all_entities = []
        for result in extraction_results:
            all_entities.extend(result.entities)
        
        # Deduplicate entities by name and ID
        consolidated_entities = {}
        
        for entity in all_entities:
            # Create composite key for deduplication
            key = (entity.name.lower(), entity.entity_id.lower())
            
            if key not in consolidated_entities:
                consolidated_entities[key] = entity
            else:
                # Merge entities - take higher confidence
                existing = consolidated_entities[key]
                if entity.confidence > existing.confidence:
                    consolidated_entities[key] = entity
                else:
                    # Merge synonyms and sources
                    existing.synonyms.extend(entity.synonyms)
                    if entity.source not in existing.source:
                        existing.source += f", {entity.source}"
        
        # Add consolidated entities to context
        for entity in consolidated_entities.values():
            context.add_entity(entity)
        
        logger.info(f"Consolidated {len(all_entities)} entities into {len(consolidated_entities)} unique entities")
    
    async def _link_entities(self, context: EntityContext):
        """Link entities to external knowledge bases"""
        if not self.entity_linker:
            return
        
        for entity in context.entities:
            try:
                # Attempt to link entity to external KBs
                # Note: link_entities expects a list of entity names and returns a dict
                linked_data = self.entity_linker.link_entities([entity.name], f"Entity linking for {entity.name}")
                linked_info = linked_data.get(entity.name, None)
                
                if linked_info:
                    # Update entity with linked information
                    if 'id' in linked_info:
                        # Update entity ID if found
                        entity.entity_id = linked_info['id']
                    if 'type' in linked_info:
                        # Update entity type if more specific type found
                        pass  # Keep original type for now
                
            except Exception as e:
                logger.warning(f"Failed to link entity {entity.name}: {e}")
                continue
    
    async def _resolve_entity_names(self, context: EntityContext):
        """Resolve entity IDs to human-readable names"""
        if not self.config.integration.enable_name_resolution:
            return
        
        for entity in context.entities:
            if entity.entity_id and entity.entity_id.startswith(('UMLS:', 'CHEBI:', 'HGNC:')):
                try:
                    # Attempt to resolve entity ID to name
                    resolved_name = await self._resolve_entity_id_to_name(entity.entity_id)
                    
                    if resolved_name and resolved_name != entity.name:
                        entity.synonyms.append(entity.name)  # Keep original as synonym
                        entity.name = resolved_name
                        
                except Exception as e:
                    logger.warning(f"Failed to resolve entity ID {entity.entity_id}: {e}")
                    continue
    
    async def _resolve_entity_id_to_name(self, entity_id: str) -> Optional[str]:
        """Resolve a specific entity ID to its name"""
        try:
            # This would integrate with name resolution service
            import httpx
            
            url = f"{self.config.integration.name_resolver_url}/lookup"
            params = {"curie": entity_id}
            
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get('name')
                elif response.status_code == 422:
                    # Entity ID not found/invalid - this is expected for some IDs
                    logger.debug(f"Entity ID not found: {entity_id}")
                    return None
                else:
                    logger.debug(f"Name resolution failed for {entity_id}: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error resolving entity ID {entity_id}: {e}")
            return None
    
    async def _filter_and_rank_entities(self, context: EntityContext):
        """Filter and rank entities by relevance and confidence"""
        # Filter by confidence threshold
        min_confidence = self.config.quality.entity_confidence_threshold
        filtered_entities = [e for e in context.entities if e.confidence >= min_confidence]
        
        # Sort by confidence (descending)
        filtered_entities.sort(key=lambda e: e.confidence, reverse=True)
        
        # Limit to maximum entities per query
        max_entities = self.config.performance.max_entities_per_query
        if len(filtered_entities) > max_entities:
            filtered_entities = filtered_entities[:max_entities]
        
        # Update context with filtered entities
        context.entities = filtered_entities
        context.entity_mappings = {entity.name: entity.entity_id for entity in filtered_entities}
        
        logger.info(f"Filtered entities: kept {len(filtered_entities)} out of original set")
    
    def _map_entity_type(self, type_string: str) -> EntityType:
        """Map string entity type to EntityType enum"""
        type_mapping = {
            'gene': EntityType.GENE,
            'protein': EntityType.PROTEIN,
            'drug': EntityType.DRUG,
            'chemical': EntityType.CHEMICAL,
            'disease': EntityType.DISEASE,
            'pathway': EntityType.PATHWAY,
            'process': EntityType.PROCESS,
            'biological_process': EntityType.PROCESS,
            'phenotype': EntityType.PHENOTYPE,
            'anatomy': EntityType.ANATOMY,
            'organism': EntityType.ORGANISM
        }
        
        return type_mapping.get(type_string.lower(), EntityType.UNKNOWN)
    
    def get_extraction_statistics(self, context: EntityContext) -> Dict[str, Any]:
        """Get statistics about the entity extraction process"""
        stats = {
            'total_entities': len(context.entities),
            'entities_by_type': {},
            'average_confidence': 0.0,
            'confidence_distribution': {'very_high': 0, 'high': 0, 'medium': 0, 'low': 0, 'very_low': 0},
            'sources': set(),
            'generic_resolutions': len(context.generic_resolutions),
            'placeholder_mappings': len(context.placeholder_mappings)
        }
        
        if context.entities:
            # Count by type
            for entity in context.entities:
                entity_type = entity.entity_type.value
                stats['entities_by_type'][entity_type] = stats['entities_by_type'].get(entity_type, 0) + 1
                
                # Add source
                stats['sources'].add(entity.source)
                
                # Confidence distribution
                conf_level = entity.confidence_level.value
                stats['confidence_distribution'][conf_level] += 1
            
            # Average confidence
            stats['average_confidence'] = sum(e.confidence for e in context.entities) / len(context.entities)
            
            # Convert sources set to list for JSON serialization
            stats['sources'] = list(stats['sources'])
        
        return stats
    
    async def initialize(self) -> None:
        """Initialize the entity processor"""
        logger.info("Initializing UnifiedEntityProcessor...")
        # Components are initialized in __init__, nothing additional needed
        logger.info("UnifiedEntityProcessor initialization completed")
    
    async def extract_entities(
        self,
        text: str,
        context: Optional[EntityContext] = None
    ) -> List[BiomedicalEntity]:
        """Extract entities from text and return as list"""
        entity_context = await self.process_entities(text, context)
        return entity_context.entities
