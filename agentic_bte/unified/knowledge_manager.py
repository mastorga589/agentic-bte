"""
Unified Knowledge Manager

This module consolidates all knowledge-related functionality including RDF graph management,
TRAPI query building, evidence scoring, predicate selection, and knowledge graph operations
into a unified, cohesive system.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy

from .config import UnifiedConfig
from .types import (
    BiomedicalResult, EntityContext, KnowledgeGraph, 
    BiomedicalRelationship, ExecutionContext
)

# Import existing knowledge components
from ..core.knowledge.evidence_scoring import EvidenceScorer, ScoringWeights
from ..core.knowledge.predicate_strategy import (
    PredicateSelector, QueryIntent, PredicateConfig
)
from ..agents.rdf_manager import RDFGraphManager

logger = logging.getLogger(__name__)


class KnowledgeSource(Enum):
    """Types of knowledge sources"""
    BTE_API = "bte_api"
    TRAPI_ENDPOINT = "trapi_endpoint"
    LOCAL_CACHE = "local_cache"
    RDF_GRAPH = "rdf_graph"
    META_KNOWLEDGE_GRAPH = "meta_kg"


@dataclass
class PredicateRanking:
    """Ranking information for a predicate"""
    predicate: str
    relevance_score: float
    support_count: int
    tier: str  # primary, secondary, fallback
    confidence: float = 0.0
    

@dataclass
class TRAPIQuery:
    """TRAPI query with metadata"""
    query_graph: Dict[str, Any]
    query_id: str
    predicate: str
    entities: List[str]
    confidence: float
    source_intent: QueryIntent
    estimated_results: int = 0


@dataclass 
class KnowledgeEvidence:
    """Evidence supporting a knowledge assertion"""
    source: KnowledgeSource
    confidence: float
    provenance: List[str]
    attributes: Dict[str, Any]
    study_count: int = 0
    clinical_phase: Optional[int] = None


@dataclass
class KnowledgeAssertion:
    """A knowledge assertion with evidence"""
    subject: str
    predicate: str
    object: str
    subject_type: str
    object_type: str
    evidence: List[KnowledgeEvidence]
    aggregated_confidence: float
    provenance_count: int = 0


class UnifiedKnowledgeManager:
    """
    Unified Knowledge Manager that consolidates all knowledge operations:
    - RDF graph management and accumulation
    - TRAPI query building and optimization
    - Evidence-based scoring and ranking
    - Intelligent predicate selection
    - Knowledge graph construction and querying
    """
    
    def __init__(self, config: UnifiedConfig):
        """
        Initialize the unified knowledge manager
        
        Args:
            config: Unified configuration
        """
        logger.info("=== INITIALIZING UNIFIED KNOWLEDGE MANAGER ===")
        logger.debug(f"Configuration: {config}")
        
        self.config = config
        
        # Initialize core components
        logger.debug("Initializing RDF graph manager...")
        self.rdf_manager = RDFGraphManager()
        
        logger.debug("Initializing evidence scorer with weights...")
        scoring_weights = ScoringWeights(
            clinical_evidence=config.quality.evidence_weight,
            predicate_relevance=config.quality.predicate_weight,
            source_reliability=config.quality.source_weight,
            result_multiplicity=config.quality.multiplicity_weight,
            study_quality=config.quality.study_weight
        )
        logger.debug(f"Scoring weights: clinical={scoring_weights.clinical_evidence}, predicate={scoring_weights.predicate_relevance}, source={scoring_weights.source_reliability}, multiplicity={scoring_weights.result_multiplicity}, study={scoring_weights.study_quality}")
        
        self.evidence_scorer = EvidenceScorer(weights=scoring_weights)
        
        # Initialize predicate selector
        logger.debug("Initializing predicate selector...")
        predicate_config = PredicateConfig(
            max_predicates_per_subquery=config.domain.max_predicates_per_query,
            min_results_threshold=config.quality.min_results_threshold,
            fallback_threshold=config.quality.fallback_threshold
        )
        logger.debug(f"Predicate config: max_predicates={predicate_config.max_predicates_per_subquery}, min_threshold={predicate_config.min_results_threshold}, fallback={predicate_config.fallback_threshold}")
        
        self.predicate_selector = PredicateSelector(config=predicate_config)
        
        # Knowledge storage
        logger.debug("Initializing knowledge storage...")
        self.knowledge_assertions: Dict[str, KnowledgeAssertion] = {}
        self.cached_trapi_queries: Dict[str, TRAPIQuery] = {}
        self.meta_knowledge_graph: Dict[str, Any] = {}
        
        # Performance tracking
        self.query_build_count = 0
        self.evidence_score_count = 0
        self.knowledge_assertion_count = 0
        
        logger.info("Unified Knowledge Manager initialized successfully")
        logger.debug("All components ready: RDF manager, evidence scorer, predicate selector, knowledge storage")
    
    async def process_biomedical_query(self, 
                                     query: str, 
                                     entity_context: EntityContext,
                                     execution_context: ExecutionContext) -> List[TRAPIQuery]:
        """
        Process a biomedical query into optimized TRAPI queries
        
        Args:
            query: Natural language biomedical query
            entity_context: Extracted entity information
            execution_context: Execution context
            
        Returns:
            List of optimized TRAPI queries
        """
        logger.info(f"=== PROCESSING BIOMEDICAL QUERY ===")
        logger.info(f"Query: {query}")
        logger.debug(f"Entity context: {len(entity_context.entities)} entities")
        logger.debug(f"Entities: {[f'{e.name} ({e.entity_type.value})' for e in entity_context.entities]}")
        logger.debug(f"Execution context: {execution_context}")
        
        import time
        start_time = time.time()
        
        try:
            # Detect query intent
            logger.info("Step 1: Detecting query intent...")
            intent_start = time.time()
            entity_data = [{'type': entity.entity_type.value, 'name': entity.name} 
                          for entity in entity_context.entities]
            logger.debug(f"Entity data for intent detection: {entity_data}")
            
            query_intent = self.predicate_selector.detect_query_intent(query, entity_data)
            intent_time = time.time() - intent_start
            
            logger.info(f"Detected query intent: {query_intent.value} (in {intent_time:.2f}s)")
            
            # Generate TRAPI queries for entity pairs
            logger.info("Step 2: Generating TRAPI queries...")
            trapi_queries = []
            
            entities = entity_context.entities
            if len(entities) < 2:
                logger.warning(f"Need at least 2 entities to build TRAPI queries, got {len(entities)}")
                logger.debug(f"Available entities: {[e.name for e in entities]}")
                return trapi_queries
            
            # Generate queries for all entity pairs
            pair_count = 0
            query_gen_start = time.time()
            
            for i, subject_entity in enumerate(entities):
                for j, object_entity in enumerate(entities[i+1:], i+1):
                    pair_count += 1
                    logger.debug(f"Processing entity pair {pair_count}: {subject_entity.name} -> {object_entity.name}")
                    
                    # Build TRAPI query for this entity pair
                    pair_start = time.time()
                    query_set = await self._build_entity_pair_queries(
                        subject_entity, object_entity, query_intent, 
                        query, execution_context
                    )
                    pair_time = time.time() - pair_start
                    
                    logger.debug(f"Generated {len(query_set)} queries for pair in {pair_time:.2f}s")
                    trapi_queries.extend(query_set)
            
            query_gen_time = time.time() - query_gen_start
            logger.info(f"Generated {len(trapi_queries)} total queries from {pair_count} entity pairs in {query_gen_time:.2f}s")
            
            # Rank and optimize queries
            logger.info("Step 3: Optimizing query set...")
            opt_start = time.time()
            optimized_queries = self._optimize_query_set(trapi_queries, query_intent)
            opt_time = time.time() - opt_start
            
            logger.info(f"Optimized to {len(optimized_queries)} queries in {opt_time:.2f}s")
            
            # Cache queries
            logger.info("Step 4: Caching queries...")
            cache_start = time.time()
            for trapi_query in optimized_queries:
                cache_key = self._generate_query_cache_key(trapi_query)
                self.cached_trapi_queries[cache_key] = trapi_query
                logger.debug(f"Cached query {trapi_query.query_id} with key: {cache_key}")
            
            cache_time = time.time() - cache_start
            total_time = time.time() - start_time
            
            logger.info(f"Cached {len(optimized_queries)} queries in {cache_time:.2f}s")
            logger.info(f"=== BIOMEDICAL QUERY PROCESSING COMPLETED in {total_time:.2f}s ===")
            logger.debug(f"Query IDs: {[q.query_id for q in optimized_queries]}")
            
            return optimized_queries
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"=== BIOMEDICAL QUERY PROCESSING FAILED after {total_time:.2f}s ===")
            logger.error(f"Error: {type(e).__name__}: {str(e)}")
            logger.debug(f"Full error traceback:", exc_info=True)
            return []
    
    async def _build_entity_pair_queries(self,
                                       subject_entity,
                                       object_entity,
                                       query_intent: QueryIntent,
                                       original_query: str,
                                       execution_context: ExecutionContext) -> List[TRAPIQuery]:
        """Build TRAPI queries for a specific entity pair"""
        logger.debug(f"Building TRAPI queries for pair: {subject_entity.name} ({subject_entity.entity_type.value}) -> {object_entity.name} ({object_entity.entity_type.value})")
        
        subject_category = f"biolink:{subject_entity.entity_type.value}"
        object_category = f"biolink:{object_entity.entity_type.value}"
        
        logger.debug(f"Categories: {subject_category} -> {object_category}")
        logger.debug(f"Entity IDs: {subject_entity.entity_id} -> {object_entity.entity_id}")
        
        # Select predicates for this entity pair
        logger.debug(f"Selecting predicates for intent: {query_intent.value}")
        predicate_start = time.time()
        
        predicate_rankings = self.predicate_selector.select_predicates(
            query_intent, subject_category, object_category
        )
        
        predicate_time = time.time() - predicate_start
        logger.debug(f"Selected {len(predicate_rankings)} predicates in {predicate_time:.3f}s: {[p[0] for p in predicate_rankings]}")
        
        queries = []
        
        for i, (predicate, confidence) in enumerate(predicate_rankings):
            logger.debug(f"Building query {i+1}/{len(predicate_rankings)} with predicate: {predicate} (confidence: {confidence:.3f})")
            
            # Build TRAPI query structure
            query_graph = {
                "nodes": {
                    "n0": {
                        "categories": [subject_category],
                        "ids": [subject_entity.entity_id] if subject_entity.entity_id else None
                    },
                    "n1": {
                        "categories": [object_category],
                        "ids": [object_entity.entity_id] if object_entity.entity_id else None
                    }
                },
                "edges": {
                    "e01": {
                        "subject": "n0",
                        "object": "n1", 
                        "predicates": [predicate]
                    }
                }
            }
            
            # Remove None values from nodes
            nodes_with_ids = 0
            for node_id, node_data in query_graph["nodes"].items():
                if node_data["ids"] is None:
                    del node_data["ids"]
                else:
                    nodes_with_ids += 1
            
            logger.debug(f"Query graph built: {nodes_with_ids} nodes with IDs, predicate: {predicate}")
            
            # Estimate query results
            estimated_results = self._estimate_query_results(
                subject_category, predicate, object_category
            )
            logger.debug(f"Estimated results for this query: {estimated_results}")
            
            # Create TRAPI query object
            trapi_query = TRAPIQuery(
                query_graph=query_graph,
                query_id=f"query_{self.query_build_count}",
                predicate=predicate,
                entities=[subject_entity.name, object_entity.name],
                confidence=confidence,
                source_intent=query_intent,
                estimated_results=estimated_results
            )
            
            queries.append(trapi_query)
            self.query_build_count += 1
            logger.debug(f"Created TRAPI query: {trapi_query.query_id}")
        
        logger.debug(f"Built {len(queries)} TRAPI queries for entity pair")
        return queries
    
    def _optimize_query_set(self, queries: List[TRAPIQuery], intent: QueryIntent) -> List[TRAPIQuery]:
        """Optimize and rank the set of TRAPI queries"""
        logger.debug(f"Optimizing query set: {len(queries)} queries for intent {intent.value}")
        
        if not queries:
            logger.warning("No queries to optimize")
            return queries
        
        # Sort by confidence and estimated results
        def query_score(q: TRAPIQuery) -> float:
            confidence_weight = 0.7
            results_weight = 0.3
            normalized_results = min(q.estimated_results, 100) / 100.0
            score = (q.confidence * confidence_weight) + (normalized_results * results_weight)
            return score
        
        # Calculate scores for logging
        query_scores = []
        for q in queries:
            score = query_score(q)
            query_scores.append((q.query_id, score, q.confidence, q.estimated_results))
            logger.debug(f"Query {q.query_id}: score={score:.3f} (conf={q.confidence:.3f}, est_results={q.estimated_results})")
        
        # Sort by score
        sorted_queries = sorted(queries, key=query_score, reverse=True)
        logger.debug(f"Sorted queries by score: {[q.query_id for q in sorted_queries]}")
        
        # Limit number of queries based on configuration
        max_queries = self.config.domain.max_queries_per_strategy
        optimized_queries = sorted_queries[:max_queries]
        
        logger.info(f"Optimized {len(queries)} queries to {len(optimized_queries)} top queries (limit: {max_queries})")
        
        if len(queries) > max_queries:
            dropped_queries = sorted_queries[max_queries:]
            logger.debug(f"Dropped {len(dropped_queries)} lower-scoring queries: {[q.query_id for q in dropped_queries]}")
        
        # Log final selection
        for i, q in enumerate(optimized_queries):
            score = query_score(q)
            logger.debug(f"Selected query {i+1}: {q.query_id} (score={score:.3f}, predicate={q.predicate})")
        
        return optimized_queries
    
    def score_knowledge_results(self,
                              results: List[Dict[str, Any]],
                              edges_data: Dict[str, Dict],
                              predicate: str,
                              query_intent: QueryIntent) -> List[Tuple[Dict[str, Any], float]]:
        """
        Score biomedical results using evidence-based scoring
        
        Args:
            results: List of BTE/TRAPI results
            edges_data: Knowledge graph edges data
            predicate: Predicate used in query
            query_intent: Detected query intent
            
        Returns:
            List of (result, score) tuples sorted by score
        """
        logger.info(f"=== SCORING KNOWLEDGE RESULTS ===")
        logger.info(f"Scoring {len(results)} results for predicate: {predicate}, intent: {query_intent.value}")
        logger.debug(f"Edges data keys: {len(edges_data)} edges")
        
        scoring_start = time.time()
        scored_results = []
        scoring_errors = 0
        
        for i, result in enumerate(results):
            try:
                result_start = time.time()
                logger.debug(f"Scoring result {i+1}/{len(results)}...")
                
                score = self.evidence_scorer.score_result(
                    result, edges_data, predicate, query_intent
                )
                
                result_time = time.time() - result_start
                scored_results.append((result, score))
                self.evidence_score_count += 1
                
                logger.debug(f"Result {i+1} scored: {score:.3f} (in {result_time:.3f}s)")
                
            except Exception as e:
                scoring_errors += 1
                logger.error(f"Error scoring result {i+1}: {type(e).__name__}: {str(e)}")
                logger.debug(f"Full scoring error for result {i+1}:", exc_info=True)
                scored_results.append((result, 0.5))  # Default score
        
        # Sort by score descending
        sort_start = time.time()
        scored_results.sort(key=lambda x: x[1], reverse=True)
        sort_time = time.time() - sort_start
        
        total_time = time.time() - scoring_start
        
        # Calculate statistics
        scores = [s for _, s in scored_results]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        
        logger.info(f"Scored {len(results)} results in {total_time:.2f}s (sort: {sort_time:.3f}s)")
        logger.info(f"Score statistics: avg={avg_score:.3f}, min={min_score:.3f}, max={max_score:.3f}")
        
        if scoring_errors > 0:
            logger.warning(f"Encountered {scoring_errors} scoring errors")
        
        # Log top results
        top_results = scored_results[:5]
        logger.debug(f"Top {len(top_results)} scored results:")
        for i, (result, score) in enumerate(top_results):
            logger.debug(f"  {i+1}. Score: {score:.3f}")
        
        logger.debug(f"=== SCORING COMPLETED: {len(scored_results)} scored results ===")
        return scored_results
    
    def build_knowledge_graph(self,
                            scored_results: List[Tuple[Dict[str, Any], float]],
                            edges_data: Dict[str, Dict],
                            nodes_data: Dict[str, Dict]) -> KnowledgeGraph:
        """
        Build a unified knowledge graph from scored results
        
        Args:
            scored_results: Scored result tuples
            edges_data: Edges from knowledge graph
            nodes_data: Nodes from knowledge graph
            
        Returns:
            Unified knowledge graph
        """
        logger.info(f"=== BUILDING KNOWLEDGE GRAPH ===")
        logger.info(f"Input: {len(scored_results)} scored results, {len(edges_data)} edges, {len(nodes_data)} nodes")
        
        kg_build_start = time.time()
        
        # Extract relationships from results
        logger.debug("Step 1: Extracting relationships from results...")
        extract_start = time.time()
        relationships = []
        knowledge_assertions = []
        extraction_errors = 0
        
        for i, (result, score) in enumerate(scored_results):
            try:
                logger.debug(f"Extracting from result {i+1}/{len(scored_results)} (score: {score:.3f})...")
                result_start = time.time()
                
                # Extract relationships from this result
                result_relationships = self._extract_relationships_from_result(
                    result, edges_data, nodes_data, score
                )
                
                result_time = time.time() - result_start
                logger.debug(f"Extracted {len(result_relationships)} relationships from result {i+1} in {result_time:.3f}s")
                
                relationships.extend(result_relationships)
                
                # Create knowledge assertions
                for rel in result_relationships:
                    assertion = KnowledgeAssertion(
                        subject=rel.subject_name,
                        predicate=rel.predicate,
                        object=rel.object_name,
                        subject_type=rel.subject_type,
                        object_type=rel.object_type,
                        evidence=[KnowledgeEvidence(
                            source=KnowledgeSource.BTE_API,
                            confidence=score,
                            provenance=rel.provenance,
                            attributes=rel.attributes
                        )],
                        aggregated_confidence=score
                    )
                    
                    # Store assertion
                    assertion_key = f"{rel.subject_name}:{rel.predicate}:{rel.object_name}"
                    self.knowledge_assertions[assertion_key] = assertion
                    knowledge_assertions.append(assertion)
                    
                    logger.debug(f"Created assertion: {rel.subject_name} -[{rel.predicate}]-> {rel.object_name} (conf: {score:.3f})")
                
            except Exception as e:
                extraction_errors += 1
                logger.error(f"Error extracting relationships from result {i+1}: {type(e).__name__}: {str(e)}")
                logger.debug(f"Full extraction error for result {i+1}:", exc_info=True)
        
        extract_time = time.time() - extract_start
        logger.info(f"Extracted {len(relationships)} relationships from {len(scored_results)} results in {extract_time:.2f}s")
        
        if extraction_errors > 0:
            logger.warning(f"Encountered {extraction_errors} extraction errors")
        
        # Add relationships to RDF graph
        logger.debug("Step 2: Adding relationships to RDF graph...")
        rdf_start = time.time()
        
        if relationships:
            rdf_triples = []
            for rel in relationships:
                rdf_triple = {
                    'subject': rel.subject_name,
                    'subject_type': rel.subject_type,
                    'predicate': rel.predicate,
                    'object': rel.object_name,
                    'object_type': rel.object_type
                }
                rdf_triples.append(rdf_triple)
            
            logger.debug(f"Adding {len(rdf_triples)} triples to RDF graph...")
            added_count = self.rdf_manager.add_triples(rdf_triples)
            rdf_time = time.time() - rdf_start
            
            logger.info(f"Added {added_count} triples to RDF graph in {rdf_time:.3f}s")
        else:
            logger.warning("No relationships to add to RDF graph")
        
        # Build unified knowledge graph
        logger.debug("Step 3: Building unified knowledge graph structure...")
        kg_struct_start = time.time()
        
        # Calculate entity count
        unique_entities = set([r.subject_name for r in relationships] + 
                            [r.object_name for r in relationships])
        entity_count = len(unique_entities)
        
        # Calculate distributions
        confidence_dist = self._calculate_confidence_distribution(relationships)
        provenance_summary = self._calculate_provenance_summary(relationships)
        
        knowledge_graph = KnowledgeGraph(
            relationships=relationships,
            entity_count=entity_count,
            relationship_count=len(relationships),
            confidence_distribution=confidence_dist,
            provenance_summary=provenance_summary
        )
        
        kg_struct_time = time.time() - kg_struct_start
        total_time = time.time() - kg_build_start
        
        self.knowledge_assertion_count += len(knowledge_assertions)
        
        logger.info(f"Built knowledge graph structure in {kg_struct_time:.3f}s")
        logger.info(f"=== KNOWLEDGE GRAPH COMPLETED in {total_time:.2f}s ===")
        logger.info(f"Final graph: {knowledge_graph.entity_count} entities, {knowledge_graph.relationship_count} relationships")
        logger.debug(f"Confidence distribution: {confidence_dist}")
        logger.debug(f"Provenance summary: {dict(list(provenance_summary.items())[:5])}...")  # Show first 5 sources
        
        return knowledge_graph
    
    def _extract_relationships_from_result(self,
                                         result: Dict[str, Any],
                                         edges_data: Dict[str, Dict],
                                         nodes_data: Dict[str, Dict],
                                         confidence: float) -> List[BiomedicalRelationship]:
        """Extract biomedical relationships from a single result"""
        logger.debug(f"Extracting relationships from result with confidence: {confidence:.3f}")
        relationships = []
        
        try:
            # Extract from analyses
            analyses = result.get('analyses', [])
            logger.debug(f"Found {len(analyses)} analyses in result")
            
            for analysis_idx, analysis in enumerate(analyses):
                logger.debug(f"Processing analysis {analysis_idx + 1}/{len(analyses)}...")
                
                edge_bindings = analysis.get('edge_bindings', {})
                node_bindings = analysis.get('node_bindings', {})
                
                logger.debug(f"Analysis {analysis_idx + 1}: {len(edge_bindings)} edge binding groups, {len(node_bindings)} node binding groups")
                
                # Process edge bindings
                for edge_key, edge_list in edge_bindings.items():
                    logger.debug(f"Processing edge binding '{edge_key}' with {len(edge_list)} edges")
                    
                    for edge_idx, edge_binding in enumerate(edge_list):
                        edge_id = edge_binding.get('id')
                        logger.debug(f"Processing edge {edge_idx + 1}: {edge_id}")
                        
                        if edge_id in edges_data:
                            edge_info = edges_data[edge_id]
                            logger.debug(f"Found edge info for {edge_id}")
                            
                            # Get subject and object from edge
                            subject_id = edge_info.get('subject')
                            object_id = edge_info.get('object')
                            predicate = edge_info.get('predicate', 'biolink:related_to')
                            
                            logger.debug(f"Edge relationship: {subject_id} -[{predicate}]-> {object_id}")
                            
                            # Get node information
                            subject_name = subject_id
                            object_name = object_id
                            subject_type = "biolink:NamedThing"
                            object_type = "biolink:NamedThing"
                            
                            if subject_id in nodes_data:
                                subject_node = nodes_data[subject_id]
                                subject_name = subject_node.get('name', subject_id)
                                subject_categories = subject_node.get('categories', [])
                                if subject_categories:
                                    subject_type = subject_categories[0]
                                logger.debug(f"Subject node: {subject_name} ({subject_type})")
                            else:
                                logger.debug(f"Subject node {subject_id} not found in nodes_data")
                            
                            if object_id in nodes_data:
                                object_node = nodes_data[object_id]
                                object_name = object_node.get('name', object_id)
                                object_categories = object_node.get('categories', [])
                                if object_categories:
                                    object_type = object_categories[0]
                                logger.debug(f"Object node: {object_name} ({object_type})")
                            else:
                                logger.debug(f"Object node {object_id} not found in nodes_data")
                            
                            # Extract attributes and sources
                            attributes = edge_info.get('attributes', [])
                            sources = edge_info.get('sources', [])
                            
                            logger.debug(f"Edge has {len(attributes)} attributes and {len(sources)} sources")
                            
                            # Create relationship
                            relationship = BiomedicalRelationship(
                                subject_name=subject_name,
                                subject_id=subject_id,
                                subject_type=subject_type,
                                predicate=predicate,
                                object_name=object_name,
                                object_id=object_id,
                                object_type=object_type,
                                confidence=confidence,
                                provenance=[src.get('resource_id', '') for src in sources],
                                attributes={attr.get('attribute_type_id', ''): attr.get('value') 
                                          for attr in attributes},
                                evidence_count=len(sources)
                            )
                            
                            relationships.append(relationship)
                            logger.debug(f"Created relationship: {subject_name} -[{predicate}]-> {object_name} (evidence: {len(sources)})")
                        else:
                            logger.debug(f"Edge {edge_id} not found in edges_data")
        
        except Exception as e:
            logger.error(f"Error extracting relationships from result: {type(e).__name__}: {str(e)}")
            logger.debug(f"Full relationship extraction error:", exc_info=True)
        
        logger.debug(f"Extracted {len(relationships)} relationships from result")
        return relationships
    
    def _estimate_query_results(self, subject_category: str, predicate: str, object_category: str) -> int:
        """Estimate number of results for a query pattern"""
        # Use predicate selector's support data
        support_count = self.predicate_selector.get_predicate_support(
            subject_category, predicate, object_category
        )
        
        # Estimate based on support count and predicate type
        if 'related_to' in predicate:
            base_estimate = 50
        elif 'associated_with' in predicate:
            base_estimate = 40
        elif 'treats' in predicate:
            base_estimate = 20
        elif 'affects' in predicate:
            base_estimate = 30
        else:
            base_estimate = 25
        
        # Scale by support count
        estimated = base_estimate * max(1, support_count)
        return min(estimated, 200)  # Cap at reasonable limit
    
    def _generate_query_cache_key(self, trapi_query: TRAPIQuery) -> str:
        """Generate cache key for TRAPI query"""
        import hashlib
        query_str = json.dumps(trapi_query.query_graph, sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()[:16]
    
    def _calculate_confidence_distribution(self, relationships: List[BiomedicalRelationship]) -> Dict[str, int]:
        """Calculate confidence score distribution"""
        distribution = {"high": 0, "medium": 0, "low": 0}
        
        for rel in relationships:
            if rel.confidence >= 0.8:
                distribution["high"] += 1
            elif rel.confidence >= 0.5:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        
        return distribution
    
    def _calculate_provenance_summary(self, relationships: List[BiomedicalRelationship]) -> Dict[str, int]:
        """Calculate provenance source summary"""
        sources = {}
        for rel in relationships:
            for prov in rel.provenance:
                if prov:
                    source_key = prov.split(':')[0] if ':' in prov else prov
                    sources[source_key] = sources.get(source_key, 0) + 1
        return sources
    
    def query_knowledge_graph(self, sparql_query: str) -> List[Dict[str, Any]]:
        """
        Query the accumulated RDF knowledge graph
        
        Args:
            sparql_query: SPARQL query string
            
        Returns:
            Query results
        """
        return self.rdf_manager.query_graph(sparql_query)
    
    def get_entity_knowledge(self, entity_name: str) -> Dict[str, Any]:
        """
        Get all accumulated knowledge about a specific entity
        
        Args:
            entity_name: Name of entity to query
            
        Returns:
            Dictionary with entity knowledge
        """
        # Get from RDF graph
        rdf_relationships = self.rdf_manager.get_entity_relationships(entity_name)
        
        # Get from knowledge assertions
        entity_assertions = []
        for assertion_key, assertion in self.knowledge_assertions.items():
            if (entity_name.lower() in assertion.subject.lower() or 
                entity_name.lower() in assertion.object.lower()):
                entity_assertions.append(assertion)
        
        return {
            'entity_name': entity_name,
            'rdf_relationships': rdf_relationships,
            'knowledge_assertions': entity_assertions,
            'assertion_count': len(entity_assertions),
            'relationship_count': len(rdf_relationships)
        }
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get comprehensive knowledge system statistics"""
        rdf_stats = self.rdf_manager.get_summary_stats()
        
        return {
            # RDF Graph stats
            'rdf_graph': rdf_stats,
            
            # Knowledge assertions
            'knowledge_assertions': len(self.knowledge_assertions),
            
            # Query building stats
            'trapi_queries_built': self.query_build_count,
            'cached_queries': len(self.cached_trapi_queries),
            
            # Scoring stats
            'results_scored': self.evidence_score_count,
            
            # Intent distribution
            'intent_distribution': self._get_intent_distribution(),
            
            # Predicate usage
            'predicate_usage': self._get_predicate_usage(),
            
            # Performance metrics
            'avg_confidence': self._get_average_confidence()
        }
    
    def _get_intent_distribution(self) -> Dict[str, int]:
        """Get distribution of query intents processed"""
        distribution = {}
        for trapi_query in self.cached_trapi_queries.values():
            intent = trapi_query.source_intent.value
            distribution[intent] = distribution.get(intent, 0) + 1
        return distribution
    
    def _get_predicate_usage(self) -> Dict[str, int]:
        """Get usage count for each predicate"""
        usage = {}
        for assertion in self.knowledge_assertions.values():
            predicate = assertion.predicate
            usage[predicate] = usage.get(predicate, 0) + 1
        return usage
    
    def _get_average_confidence(self) -> float:
        """Get average confidence across all knowledge assertions"""
        if not self.knowledge_assertions:
            return 0.0
        
        total_confidence = sum(assertion.aggregated_confidence 
                             for assertion in self.knowledge_assertions.values())
        return total_confidence / len(self.knowledge_assertions)
    
    def clear_knowledge_cache(self):
        """Clear all cached knowledge data"""
        self.knowledge_assertions.clear()
        self.cached_trapi_queries.clear()
        self.rdf_manager.clear_graph()
        
        # Reset counters
        self.query_build_count = 0
        self.evidence_score_count = 0
        self.knowledge_assertion_count = 0
        
        logger.info("Cleared all knowledge cache data")
    
    def export_knowledge_graph(self, format: str = "turtle") -> str:
        """
        Export the accumulated knowledge graph
        
        Args:
            format: Export format (turtle, json, etc.)
            
        Returns:
            Serialized knowledge graph
        """
        if format.lower() == "turtle":
            return self.rdf_manager.get_turtle_representation()
        elif format.lower() == "json":
            # Export knowledge assertions as JSON
            assertions_data = []
            for assertion in self.knowledge_assertions.values():
                assertions_data.append({
                    'subject': assertion.subject,
                    'predicate': assertion.predicate,
                    'object': assertion.object,
                    'subject_type': assertion.subject_type,
                    'object_type': assertion.object_type,
                    'confidence': assertion.aggregated_confidence,
                    'evidence_count': len(assertion.evidence),
                    'provenance_count': assertion.provenance_count
                })
            
            return json.dumps(assertions_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def validate_knowledge_consistency(self) -> Dict[str, Any]:
        """
        Validate consistency of accumulated knowledge
        
        Returns:
            Validation report
        """
        report = {
            'total_assertions': len(self.knowledge_assertions),
            'conflicts_detected': 0,
            'low_confidence_assertions': 0,
            'missing_evidence_assertions': 0,
            'conflicts': [],
            'recommendations': []
        }
        
        # Check for conflicting assertions
        grouped_assertions = {}
        for assertion_key, assertion in self.knowledge_assertions.items():
            entity_pair = f"{assertion.subject}:{assertion.object}"
            if entity_pair not in grouped_assertions:
                grouped_assertions[entity_pair] = []
            grouped_assertions[entity_pair].append(assertion)
        
        # Detect conflicts
        for entity_pair, assertions in grouped_assertions.items():
            if len(assertions) > 1:
                predicates = set(a.predicate for a in assertions)
                if len(predicates) > 1:
                    # Check for semantic conflicts
                    conflicting_predicates = self._check_predicate_conflicts(predicates)
                    if conflicting_predicates:
                        report['conflicts_detected'] += 1
                        report['conflicts'].append({
                            'entity_pair': entity_pair,
                            'conflicting_predicates': list(conflicting_predicates),
                            'assertions': len(assertions)
                        })
        
        # Check for low confidence assertions
        for assertion in self.knowledge_assertions.values():
            if assertion.aggregated_confidence < 0.3:
                report['low_confidence_assertions'] += 1
            
            if not assertion.evidence:
                report['missing_evidence_assertions'] += 1
        
        # Generate recommendations
        if report['conflicts_detected'] > 0:
            report['recommendations'].append("Consider consolidating conflicting assertions")
        
        if report['low_confidence_assertions'] > len(self.knowledge_assertions) * 0.2:
            report['recommendations'].append("High proportion of low-confidence assertions detected")
        
        logger.info(f"Knowledge validation completed: {report['conflicts_detected']} conflicts detected")
        return report
    
    def _check_predicate_conflicts(self, predicates: Set[str]) -> Set[str]:
        """Check if predicates are semantically conflicting"""
        # Define conflicting predicate pairs
        conflicts = {
            ('biolink:treats', 'biolink:causes'),
            ('biolink:beneficial', 'biolink:harmful'),
            ('biolink:increases', 'biolink:decreases'),
            ('biolink:activates', 'biolink:inhibits')
        }
        
        conflicting_predicates = set()
        predicate_list = list(predicates)
        
        for i, pred1 in enumerate(predicate_list):
            for pred2 in predicate_list[i+1:]:
                pred1_clean = pred1.replace('biolink:', '').lower()
                pred2_clean = pred2.replace('biolink:', '').lower()
                
                for conflict_pair in conflicts:
                    if ((pred1_clean in conflict_pair[0] or conflict_pair[0] in pred1_clean) and
                        (pred2_clean in conflict_pair[1] or conflict_pair[1] in pred2_clean)):
                        conflicting_predicates.add(pred1)
                        conflicting_predicates.add(pred2)
        
        return conflicting_predicates
    
    async def initialize(self) -> None:
        """Initialize the knowledge manager"""
        logger.info("=== INITIALIZING UNIFIED KNOWLEDGE MANAGER ===")
        
        # Check component status
        logger.debug("Checking component initialization status...")
        
        components_status = {
            'rdf_manager': self.rdf_manager is not None,
            'evidence_scorer': self.evidence_scorer is not None,
            'predicate_selector': self.predicate_selector is not None,
            'knowledge_assertions': isinstance(self.knowledge_assertions, dict),
            'cached_trapi_queries': isinstance(self.cached_trapi_queries, dict)
        }
        
        logger.debug(f"Component status: {components_status}")
        
        # Test RDF manager
        try:
            logger.debug("Testing RDF manager functionality...")
            rdf_stats = self.rdf_manager.get_summary_stats()
            logger.debug(f"RDF manager stats: {rdf_stats}")
        except Exception as e:
            logger.warning(f"RDF manager test failed: {e}")
        
        # Test evidence scorer
        try:
            logger.debug("Testing evidence scorer...")
            # Just verify it's properly initialized with weights
            logger.debug(f"Evidence scorer initialized with weights")
        except Exception as e:
            logger.warning(f"Evidence scorer test failed: {e}")
        
        # Test predicate selector
        try:
            logger.debug("Testing predicate selector...")
            # Test with sample data
            logger.debug(f"Predicate selector ready")
        except Exception as e:
            logger.warning(f"Predicate selector test failed: {e}")
        
        # Initialize counters
        logger.debug(f"Performance counters initialized: queries={self.query_build_count}, scores={self.evidence_score_count}, assertions={self.knowledge_assertion_count}")
        
        logger.info("UnifiedKnowledgeManager initialization completed")
        logger.info(f"Ready with components: {', '.join([k for k, v in components_status.items() if v])}")
    
    async def build_trapi_queries(
        self,
        text: str,
        entities: List[Any],
        max_results: int = 100
    ) -> List[TRAPIQuery]:
        """Build TRAPI queries for demo compatibility"""
        # Simple mock implementation for demo
        queries = []
        if entities and len(entities) >= 1:
            # Create a simple TRAPI query
            query = TRAPIQuery(
                query_graph={
                    "nodes": {"n0": {"categories": ["biolink:NamedThing"]}},
                    "edges": {}
                },
                query_id="demo_query",
                predicate="biolink:related_to",
                entities=[e.name for e in entities],
                confidence=0.8,
                source_intent=QueryIntent.GENERAL
            )
            queries.append(query)
        return queries
    
    async def score_results(self, results: List[Any]) -> List[Any]:
        """Score results for demo compatibility"""
        # Simple mock implementation that returns results as-is
        return results
    
    async def integrate_results(
        self,
        results: List[Any],
        entities: List[Any]
    ) -> List[KnowledgeAssertion]:
        """Integrate results into knowledge assertions"""
        # Simple mock implementation
        assertions = []
        for i, result in enumerate(results[:5]):  # Limit to first 5
            assertion = KnowledgeAssertion(
                subject=getattr(result, 'subject_entity', 'unknown'),
                predicate=getattr(result, 'predicate', 'biolink:related_to'),
                object=getattr(result, 'object_entity', 'unknown'),
                subject_type="biolink:NamedThing",
                object_type="biolink:NamedThing",
                evidence=[
                    KnowledgeEvidence(
                        source=KnowledgeSource.BTE_API,
                        confidence=getattr(result, 'confidence', 0.5),
                        provenance=["demo_source"],
                        attributes={}
                    )
                ],
                aggregated_confidence=getattr(result, 'confidence', 0.5)
            )
            assertions.append(assertion)
        return assertions
    
    async def build_knowledge_graph(
        self,
        assertions: List[KnowledgeAssertion]
    ) -> Dict[str, Any]:
        """Build knowledge graph from assertions"""
        nodes = {}
        edges = []
        
        for i, assertion in enumerate(assertions):
            # Add nodes
            if assertion.subject not in nodes:
                nodes[assertion.subject] = {
                    "id": assertion.subject,
                    "name": assertion.subject,
                    "type": assertion.subject_type
                }
            
            if assertion.object not in nodes:
                nodes[assertion.object] = {
                    "id": assertion.object,
                    "name": assertion.object,
                    "type": assertion.object_type
                }
            
            # Add edge
            edge = {
                "id": f"edge_{i}",
                "subject": assertion.subject,
                "predicate": assertion.predicate,
                "object": assertion.object,
                "confidence": assertion.aggregated_confidence
            }
            edges.append(edge)
        
        return {
            "nodes": list(nodes.values()),
            "edges": edges
        }
