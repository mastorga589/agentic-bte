"""
Graph of Thoughts (GoT) Framework for Biomedical Query Optimization

Implementation inspired by "Graph of Thoughts: Solving Elaborate Problems with Large Language Models"
by Besta et al. This framework enables graph-based reasoning where LLM thoughts are vertices
and dependencies between thoughts are edges.

Key Features:
- Graph-based query decomposition and planning
- Thought transformations: aggregation, refinement, branching
- Biomedical-specific aggregation and scoring
- Volume and latency optimization
- Performance metrics from the paper
"""

import logging
import time
import asyncio
import networkx as nx
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import hashlib

from agentic_bte.core.queries.interfaces import OptimizationResult, OptimizationStrategy, OptimizationMetrics
# from call_mcp_tool import call_mcp_tool  # Not needed - use mcp_integration instead

logger = logging.getLogger(__name__)


class ThoughtType(Enum):
    """Types of thoughts in the GoT framework"""
    ENTITY_EXTRACTION = "entity_extraction"
    QUERY_BUILDING = "query_building"
    API_EXECUTION = "api_execution"
    RESULT_AGGREGATION = "result_aggregation"
    REFINEMENT = "refinement"
    VALIDATION = "validation"


class TransformationType(Enum):
    """Types of thought transformations"""
    GENERATE = "generate"      # Generate new thoughts from existing
    AGGREGATE = "aggregate"    # Combine multiple thoughts
    REFINE = "refine"         # Improve existing thought
    BRANCH = "branch"         # Create multiple variants
    MERGE = "merge"           # Combine similar thoughts
    VALIDATE = "validate"     # Check thought quality


@dataclass
class GoTThought:
    """Represents a single thought in the Graph of Thoughts"""
    id: str
    thought_type: ThoughtType
    content: Dict[str, Any]
    confidence: float = 0.0
    execution_time: float = 0.0
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Ensure ID is unique if not provided"""
        if not self.id:
            content_hash = hashlib.md5(
                json.dumps(self.content, sort_keys=True).encode()
            ).hexdigest()[:8]
            self.id = f"{self.thought_type.value}_{content_hash}"


@dataclass
class GoTMetrics:
    """Extended metrics for GoT framework based on the paper"""
    # Core metrics from the paper
    volume: int = 0           # Number of thoughts that could impact final result
    latency: int = 0          # Number of hops to reach final thought
    total_thoughts: int = 0   # Total thoughts generated
    aggregation_count: int = 0  # Number of aggregation operations
    refinement_count: int = 0   # Number of refinement operations
    
    # Performance metrics
    parallel_executions: int = 0
    dependency_depth: int = 0
    graph_complexity: float = 0.0
    
    # Cost metrics
    cost_reduction_factor: float = 0.0  # Cost reduction compared to baseline
    quality_improvement: float = 0.0    # Quality improvement vs baseline


class ThoughtTransformation(ABC):
    """Abstract base class for thought transformations"""
    
    @abstractmethod
    async def apply(self, graph: nx.DiGraph, input_thoughts: List[GoTThought], 
                   context: Dict[str, Any]) -> List[GoTThought]:
        """Apply the transformation to generate new thoughts"""
        pass
    
    @abstractmethod
    def get_type(self) -> TransformationType:
        """Return the transformation type"""
        pass


class GenerateTransformation(ThoughtTransformation):
    """Generate new thoughts from existing ones"""
    
    def get_type(self) -> TransformationType:
        return TransformationType.GENERATE
    
    async def apply(self, graph: nx.DiGraph, input_thoughts: List[GoTThought], 
                   context: Dict[str, Any]) -> List[GoTThought]:
        """Generate new thoughts based on input thoughts"""
        logger.debug(f"=== GENERATE TRANSFORMATION ===")
        logger.debug(f"Processing {len(input_thoughts)} input thoughts")
        logger.debug(f"Input thought types: {[t.thought_type.value for t in input_thoughts]}")
        logger.debug(f"Context keys: {list(context.keys())}")
        
        new_thoughts = []
        generation_start = time.time()
        
        for i, thought in enumerate(input_thoughts):
            logger.debug(f"Processing thought {i+1}/{len(input_thoughts)}: {thought.id} ({thought.thought_type.value})")
            
            if thought.thought_type == ThoughtType.ENTITY_EXTRACTION:
                logger.debug(f"Generating query building thought from entity extraction: {thought.id}")
                entities = thought.content.get("entities", {})
                logger.debug(f"Extracted entities: {list(entities.keys())}")
                
                # Generate query building thoughts from entity extraction
                query_thought = GoTThought(
                    id="",  # Will be auto-generated
                    thought_type=ThoughtType.QUERY_BUILDING,
                    content={
                        "entities": entities,
                        "original_query": context.get("query", ""),
                        "source_thought": thought.id
                    },
                    dependencies={thought.id}
                )
                new_thoughts.append(query_thought)
                logger.debug(f"Created query building thought: {query_thought.id} with {len(entities)} entities")
                
            elif thought.thought_type == ThoughtType.QUERY_BUILDING:
                logger.debug(f"Generating API execution thought from query building: {thought.id}")
                trapi_query = thought.content.get("trapi_query", {})
                logger.debug(f"TRAPI query available: {bool(trapi_query)}")
                
                # Generate API execution thoughts from query building
                api_thought = GoTThought(
                    id="",
                    thought_type=ThoughtType.API_EXECUTION,
                    content={
                        "trapi_query": trapi_query,
                        "source_thought": thought.id
                    },
                    dependencies={thought.id}
                )
                new_thoughts.append(api_thought)
                logger.debug(f"Created API execution thought: {api_thought.id}")
            else:
                logger.debug(f"No generation rule for thought type: {thought.thought_type.value}")
        
        generation_time = time.time() - generation_start
        logger.debug(f"Generated {len(new_thoughts)} new thoughts in {generation_time:.3f}s")
        logger.debug(f"New thought types: {[t.thought_type.value for t in new_thoughts]}")
        
        return new_thoughts


class AggregateTransformation(ThoughtTransformation):
    """Aggregate multiple thoughts into a single thought"""
    
    def get_type(self) -> TransformationType:
        return TransformationType.AGGREGATE
    
    async def apply(self, graph: nx.DiGraph, input_thoughts: List[GoTThought], 
                   context: Dict[str, Any]) -> List[GoTThought]:
        """Aggregate multiple thoughts using biomedical-specific logic"""
        logger.debug(f"=== AGGREGATE TRANSFORMATION ===")
        logger.debug(f"Aggregating {len(input_thoughts)} input thoughts")
        
        if len(input_thoughts) < 2:
            logger.debug("Less than 2 thoughts - no aggregation needed")
            return input_thoughts
        
        aggregation_start = time.time()
        
        # Group thoughts by type
        thoughts_by_type = {}
        for thought in input_thoughts:
            if thought.thought_type not in thoughts_by_type:
                thoughts_by_type[thought.thought_type] = []
            thoughts_by_type[thought.thought_type].append(thought)
        
        logger.debug(f"Grouped thoughts by type: {[(t.value, len(thoughts)) for t, thoughts in thoughts_by_type.items()]}")
        
        aggregated_thoughts = []
        
        for thought_type, thoughts in thoughts_by_type.items():
            logger.debug(f"Processing {len(thoughts)} thoughts of type: {thought_type.value}")
            
            if len(thoughts) > 1:
                logger.debug(f"Aggregating {len(thoughts)} {thought_type.value} thoughts...")
                agg_start = time.time()
                
                aggregated = await self._aggregate_by_type(thought_type, thoughts)
                agg_time = time.time() - agg_start
                
                if aggregated:
                    aggregated_thoughts.append(aggregated)
                    logger.debug(f"Aggregated {len(thoughts)} -> 1 {thought_type.value} thought in {agg_time:.3f}s")
                else:
                    logger.warning(f"Aggregation failed for type: {thought_type.value}")
                    aggregated_thoughts.extend(thoughts)
            else:
                logger.debug(f"Single {thought_type.value} thought - no aggregation needed")
                aggregated_thoughts.extend(thoughts)
        
        total_time = time.time() - aggregation_start
        logger.debug(f"Aggregation completed: {len(input_thoughts)} -> {len(aggregated_thoughts)} thoughts in {total_time:.3f}s")
        
        return aggregated_thoughts
    
    async def _aggregate_by_type(self, thought_type: ThoughtType, 
                               thoughts: List[GoTThought]) -> Optional[GoTThought]:
        """Aggregate thoughts of the same type"""
        if thought_type == ThoughtType.ENTITY_EXTRACTION:
            return self._aggregate_entities(thoughts)
        elif thought_type == ThoughtType.API_EXECUTION:
            return self._aggregate_results(thoughts)
        elif thought_type == ThoughtType.RESULT_AGGREGATION:
            return self._aggregate_final_results(thoughts)
        
        return None
    
    def _aggregate_entities(self, thoughts: List[GoTThought]) -> GoTThought:
        """Aggregate entity extraction thoughts"""
        logger.debug(f"Aggregating {len(thoughts)} entity extraction thoughts")
        
        combined_entities = {}
        all_dependencies = set()
        total_confidence = 0.0
        
        for i, thought in enumerate(thoughts):
            entities = thought.content.get("entities", {})
            logger.debug(f"Thought {i+1}: {len(entities)} entities, confidence: {thought.confidence:.3f}")
            
            combined_entities.update(entities)
            all_dependencies.update(thought.dependencies)
            total_confidence += thought.confidence
        
        avg_confidence = total_confidence / len(thoughts) if thoughts else 0.0
        logger.debug(f"Combined {len(combined_entities)} unique entities, avg confidence: {avg_confidence:.3f}")
        
        aggregated_thought = GoTThought(
            id="",
            thought_type=ThoughtType.ENTITY_EXTRACTION,
            content={
                "entities": combined_entities,
                "aggregated_from": [t.id for t in thoughts]
            },
            confidence=avg_confidence,
            dependencies=all_dependencies,
            metadata={"aggregation_count": len(thoughts)}
        )
        
        logger.debug(f"Created aggregated entity thought: {aggregated_thought.id}")
        return aggregated_thought
    
    def _aggregate_results(self, thoughts: List[GoTThought]) -> GoTThought:
        """Aggregate API execution results with confidence weighting"""
        logger.debug(f"Aggregating results from {len(thoughts)} API execution thoughts")
        
        combined_results = []
        all_dependencies = set()
        confidence_sum = 0.0
        
        for i, thought in enumerate(thoughts):
            results = thought.content.get("results", [])
            weight = thought.confidence if thought.confidence > 0 else 1.0
            logger.debug(f"Thought {i+1}: {len(results)} results, confidence: {thought.confidence:.3f}, weight: {weight:.3f}")
            
            # Weight results by thought confidence
            for result in results:
                result_copy = result.copy()
                original_score = result_copy.get("score", 0.0)
                result_copy["weighted_score"] = original_score * weight
                result_copy["source_thought"] = thought.id
                combined_results.append(result_copy)
            
            all_dependencies.update(thought.dependencies)
            confidence_sum += thought.confidence
        
        logger.debug(f"Combined {len(combined_results)} total results")
        
        # Sort by weighted score and deduplicate
        sort_start = time.time()
        combined_results.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
        sort_time = time.time() - sort_start
        
        dedup_start = time.time()
        deduped_results = self._deduplicate_results(combined_results)
        dedup_time = time.time() - dedup_start
        
        avg_confidence = confidence_sum / len(thoughts) if thoughts else 0.0
        
        logger.debug(f"Sorted results in {sort_time:.3f}s, deduplicated in {dedup_time:.3f}s")
        logger.debug(f"Final: {len(deduped_results)} unique results, avg confidence: {avg_confidence:.3f}")
        
        aggregated_thought = GoTThought(
            id="",
            thought_type=ThoughtType.RESULT_AGGREGATION,
            content={
                "results": deduped_results,
                "aggregated_from": [t.id for t in thoughts]
            },
            confidence=avg_confidence,
            dependencies=all_dependencies,
            metadata={"original_count": len(combined_results), "deduped_count": len(deduped_results)}
        )
        
        logger.debug(f"Created aggregated result thought: {aggregated_thought.id}")
        return aggregated_thought
    
    def _aggregate_final_results(self, thoughts: List[GoTThought]) -> GoTThought:
        """Aggregate final result thoughts"""
        all_results = []
        all_dependencies = set()
        
        for thought in thoughts:
            results = thought.content.get("results", [])
            all_results.extend(results)
            all_dependencies.update(thought.dependencies)
        
        # Remove duplicates and rank by confidence
        final_results = self._deduplicate_results(all_results)
        final_results.sort(key=lambda x: x.get("weighted_score", x.get("score", 0)), reverse=True)
        
        return GoTThought(
            id="",
            thought_type=ThoughtType.RESULT_AGGREGATION,
            content={
                "results": final_results,
                "final_answer": self._generate_final_answer(final_results),
                "aggregated_from": [t.id for t in thoughts]
            },
            confidence=min(1.0, len(final_results) / 10.0),  # Confidence based on result count
            dependencies=all_dependencies
        )
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on content similarity"""
        seen_hashes = set()
        deduped = []
        
        for result in results:
            # Create hash based on key result properties
            result_str = json.dumps({
                k: v for k, v in result.items() 
                if k not in ["weighted_score", "source_thought"]
            }, sort_keys=True)
            result_hash = hashlib.md5(result_str.encode()).hexdigest()
            
            if result_hash not in seen_hashes:
                seen_hashes.add(result_hash)
                deduped.append(result)
        
        return deduped
    
    def _generate_final_answer(self, results: List[Dict[str, Any]]) -> str:
        """Generate a final answer from aggregated results"""
        if not results:
            return "No results found."
        
        top_results = results[:5]  # Focus on top 5 results
        answer = f"Found {len(results)} total results. "
        
        if top_results:
            best_score = top_results[0].get("weighted_score", top_results[0].get("score", 0))
            answer += f"Top result scored {best_score:.3f}. "
        
        return answer


class RefineTransformation(ThoughtTransformation):
    """Refine existing thoughts by improving their quality"""
    
    def get_type(self) -> TransformationType:
        return TransformationType.REFINE
    
    async def apply(self, graph: nx.DiGraph, input_thoughts: List[GoTThought], 
                   context: Dict[str, Any]) -> List[GoTThought]:
        """Refine thoughts by improving their content and confidence"""
        refined_thoughts = []
        
        for thought in input_thoughts:
            if thought.confidence < 0.7:  # Only refine low-confidence thoughts
                refined = await self._refine_thought(thought, context)
                refined_thoughts.append(refined)
            else:
                refined_thoughts.append(thought)
        
        return refined_thoughts
    
    async def _refine_thought(self, thought: GoTThought, context: Dict[str, Any]) -> GoTThought:
        """Refine a single thought"""
        if thought.thought_type == ThoughtType.ENTITY_EXTRACTION:
            return await self._refine_entity_extraction(thought, context)
        elif thought.thought_type == ThoughtType.QUERY_BUILDING:
            return await self._refine_query_building(thought, context)
        
        return thought  # Return unchanged if can't refine
    
    async def _refine_entity_extraction(self, thought: GoTThought, context: Dict[str, Any]) -> GoTThought:
        """Refine entity extraction by re-running with different parameters"""
        try:
            # Re-extract with more specific focus
            ner_response = call_mcp_tool("bio_ner", query=context.get("query", ""))
            refined_entities = ner_response.get("entities", {})
            
            # Merge with existing entities
            existing_entities = thought.content.get("entities", {})
            combined_entities = {**existing_entities, **refined_entities}
            
            refined_thought = GoTThought(
                id=f"{thought.id}_refined",
                thought_type=thought.thought_type,
                content={
                    "entities": combined_entities,
                    "refined_from": thought.id
                },
                confidence=min(1.0, thought.confidence + 0.2),
                dependencies=thought.dependencies.copy(),
                metadata={**thought.metadata, "refinement_count": 1}
            )
            
            return refined_thought
            
        except Exception as e:
            logger.warning(f"Failed to refine entity extraction: {e}")
            return thought
    
    async def _refine_query_building(self, thought: GoTThought, context: Dict[str, Any]) -> GoTThought:
        """Refine query building with additional context"""
        try:
            # Rebuild query with refined entities
            entities = thought.content.get("entities", {})
            original_query = thought.content.get("original_query", context.get("query", ""))
            
            trapi_response = call_mcp_tool(
                "build_trapi_query",
                query=original_query,
                entity_data=entities
            )
            
            refined_query = trapi_response.get("query", {})
            
            refined_thought = GoTThought(
                id=f"{thought.id}_refined",
                thought_type=thought.thought_type,
                content={
                    "trapi_query": refined_query,
                    "entities": entities,
                    "original_query": original_query,
                    "refined_from": thought.id
                },
                confidence=min(1.0, thought.confidence + 0.1),
                dependencies=thought.dependencies.copy()
            )
            
            return refined_thought
            
        except Exception as e:
            logger.warning(f"Failed to refine query building: {e}")
            return thought


class GoTBiomedicalPlanner:
    """
    Graph of Thoughts planner for biomedical queries
    
    This class implements the core GoT framework for biomedical query optimization,
    managing the graph of thoughts and orchestrating transformations.
    """
    
    def __init__(self, max_iterations: int = 5, enable_parallel: bool = True):
        """
        Initialize the GoT planner
        
        Args:
            max_iterations: Maximum iterations for refinement loops
            enable_parallel: Enable parallel execution of independent thoughts
        """
        logger.info("=== INITIALIZING GOT BIOMEDICAL PLANNER ===")
        logger.debug(f"Configuration: max_iterations={max_iterations}, enable_parallel={enable_parallel}")
        
        self.graph = nx.DiGraph()
        self.thoughts: Dict[str, GoTThought] = {}
        
        # Initialize transformations
        logger.debug("Initializing transformation components...")
        self.transformations = {
            TransformationType.GENERATE: GenerateTransformation(),
            TransformationType.AGGREGATE: AggregateTransformation(),
            TransformationType.REFINE: RefineTransformation()
        }
        logger.debug(f"Initialized transformations: {list(self.transformations.keys())}")
        
        self.max_iterations = max_iterations
        self.enable_parallel = enable_parallel
        self.metrics = GoTMetrics()
        
        logger.info(f"GoT Planner initialized with {len(self.transformations)} transformations")
        logger.debug(f"Ready for parallel execution: {enable_parallel}")
    
    async def plan_and_execute(self, query: str, entities: Optional[Dict[str, str]] = None,
                              config: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """
        Plan and execute a biomedical query using GoT framework
        
        Args:
            query: The biomedical query to process
            entities: Optional pre-extracted entities
            config: Optional configuration parameters
            
        Returns:
            OptimizationResult with GoT execution details
        """
        logger.info(f"=== GOT PLAN AND EXECUTE ===")
        logger.info(f"Query: {query}")
        logger.debug(f"Pre-extracted entities: {len(entities) if entities else 0}")
        logger.debug(f"Config: {config}")
        
        start_time = time.time()
        
        result = OptimizationResult(
            query=query,
            strategy=OptimizationStrategy.BASIC_ADAPTIVE,  # Will be updated by caller
            start_time=start_time
        )
        
        try:
            logger.info(f"Starting GoT planning and execution for query: {query[:100]}...")
            result.reasoning_chain.append("Initializing Graph of Thoughts framework")
            
            # Initialize graph with root thought
            logger.debug("Creating root thought...")
            root_thought = GoTThought(
                id="root",
                thought_type=ThoughtType.ENTITY_EXTRACTION,
                content={"query": query, "entities": entities or {}},
                confidence=1.0
            )
            
            self._add_thought(root_thought)
            result.reasoning_chain.append("Created root thought for entity extraction")
            logger.debug(f"Root thought created: {root_thought.id}")
            
            # Execute GoT planning phases
            logger.info("=== EXECUTING GOT PHASES ===")
            
            phase1_start = time.time()
            await self._execute_entity_extraction_phase(query, entities, result)
            phase1_time = time.time() - phase1_start
            logger.info(f"Phase 1 completed in {phase1_time:.2f}s")
            
            phase2_start = time.time()
            await self._execute_query_building_phase(result)
            phase2_time = time.time() - phase2_start
            logger.info(f"Phase 2 completed in {phase2_time:.2f}s")
            
            phase3_start = time.time()
            await self._execute_api_execution_phase(config or {}, result)
            phase3_time = time.time() - phase3_start
            logger.info(f"Phase 3 completed in {phase3_time:.2f}s")
            
            phase4_start = time.time()
            await self._execute_aggregation_phase(result)
            phase4_time = time.time() - phase4_start
            logger.info(f"Phase 4 completed in {phase4_time:.2f}s")
            
            # Calculate final metrics
            logger.debug("Calculating GoT metrics...")
            metrics_start = time.time()
            self._calculate_got_metrics()
            metrics_time = time.time() - metrics_start
            logger.debug(f"Metrics calculated in {metrics_time:.3f}s")
            
            # Extract final results
            logger.debug("Extracting final results...")
            final_thoughts = [t for t in self.thoughts.values() 
                            if t.thought_type == ThoughtType.RESULT_AGGREGATION]
            
            logger.debug(f"Found {len(final_thoughts)} final aggregation thoughts")
            
            if final_thoughts:
                best_final = max(final_thoughts, key=lambda x: x.confidence)
                logger.debug(f"Best final thought: {best_final.id} (confidence: {best_final.confidence:.3f})")
                
                result.results = best_final.content.get("results", [])
                result.final_answer = best_final.content.get("final_answer", "")
                result.success = len(result.results) > 0
                
                logger.debug(f"Extracted {len(result.results)} final results")
                
                # Extract entities from all entity extraction thoughts
                entity_thoughts = [t for t in self.thoughts.values() 
                                 if t.thought_type == ThoughtType.ENTITY_EXTRACTION]
                
                logger.debug(f"Found {len(entity_thoughts)} entity extraction thoughts")
                for et in entity_thoughts:
                    entities_found = et.content.get("entities", {})
                    result.entities.update(entities_found)
                    logger.debug(f"Added {len(entities_found)} entities from thought {et.id}")
            else:
                logger.warning("No final aggregation thoughts found")
                result.success = False
            
            # Update result with GoT metrics
            result.metrics.subqueries_executed = self.metrics.total_thoughts
            result.metrics.concurrent_batches = self.metrics.parallel_executions
            result.metrics.quality_score = self._calculate_quality_score(result)
            
            logger.debug(f"GoT Metrics: thoughts={self.metrics.total_thoughts}, volume={self.metrics.volume}, latency={self.metrics.latency}")
            
            result.reasoning_chain.append(f"GoT execution completed: {self.metrics.total_thoughts} thoughts, "
                                        f"volume={self.metrics.volume}, latency={self.metrics.latency}")
            
            total_time = time.time() - start_time
            logger.info(f"=== GOT EXECUTION COMPLETED in {total_time:.2f}s ===")
            logger.info(f"Success: {result.success}, Results: {len(result.results)}, Entities: {len(result.entities)}")
            logger.info(f"GoT Metrics - Volume: {self.metrics.volume}, Latency: {self.metrics.latency}, Total thoughts: {self.metrics.total_thoughts}")
            
        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"GoT execution failed: {str(e)}"
            result.errors.append(error_msg)
            result.reasoning_chain.append(error_msg)
            logger.error(f"=== GOT EXECUTION FAILED after {total_time:.2f}s ===")
            logger.error(f"Error: {type(e).__name__}: {str(e)}")
            logger.debug(f"Full GoT execution error:", exc_info=True)
        
        result.finalize()
        return result
    
    def _add_thought(self, thought: GoTThought):
        """Add a thought to the graph"""
        logger.debug(f"Adding thought to graph: {thought.id} ({thought.thought_type.value})")
        logger.debug(f"Dependencies: {thought.dependencies}")
        logger.debug(f"Confidence: {thought.confidence:.3f}")
        
        self.thoughts[thought.id] = thought
        self.graph.add_node(thought.id, thought=thought)
        
        # Add edges for dependencies
        edges_added = 0
        for dep_id in thought.dependencies:
            if dep_id in self.graph:
                self.graph.add_edge(dep_id, thought.id)
                edges_added += 1
                logger.debug(f"Added edge: {dep_id} -> {thought.id}")
            else:
                logger.warning(f"Dependency {dep_id} not found in graph for thought {thought.id}")
        
        self.metrics.total_thoughts += 1
        logger.debug(f"Thought added successfully. Total thoughts: {self.metrics.total_thoughts}, Edges added: {edges_added}")
    
    async def _execute_entity_extraction_phase(self, query: str, entities: Optional[Dict[str, str]], 
                                             result: OptimizationResult):
        """Execute entity extraction phase"""
        logger.info(f"=== PHASE 1: ENTITY EXTRACTION ===")
        logger.debug(f"Query: {query}")
        logger.debug(f"Pre-provided entities: {entities}")
        
        result.reasoning_chain.append("Phase 1: Entity extraction")
        
        extraction_start = time.time()
        
        if entities is None:
            logger.debug("No pre-extracted entities - calling bio_ner tool...")
            try:
                ner_start = time.time()
                from ..mcp_integration import call_mcp_tool  # Import here to avoid circular imports
                ner_response = call_mcp_tool("bio_ner", query=query)
                ner_time = time.time() - ner_start
                
                extracted_entities = ner_response.get("entities", {})
                logger.debug(f"NER tool returned {len(extracted_entities)} entities in {ner_time:.2f}s")
                logger.debug(f"Extracted entity types: {list(extracted_entities.keys())}")
                
            except Exception as e:
                logger.warning(f"Entity extraction failed: {type(e).__name__}: {str(e)}")
                logger.debug(f"Full NER error:", exc_info=True)
                extracted_entities = {}
        else:
            logger.debug(f"Using pre-provided entities: {len(entities)} entities")
            extracted_entities = entities
        
        # Determine confidence based on extraction results
        confidence = 0.8 if extracted_entities else 0.3
        logger.debug(f"Entity extraction confidence: {confidence:.3f}")
        
        # Create entity extraction thought
        entity_thought = GoTThought(
            id="entity_extraction_1",
            thought_type=ThoughtType.ENTITY_EXTRACTION,
            content={"entities": extracted_entities, "query": query},
            confidence=confidence,
            dependencies={"root"}
        )
        
        self._add_thought(entity_thought)
        
        extraction_time = time.time() - extraction_start
        logger.info(f"Entity extraction phase completed in {extraction_time:.2f}s")
        logger.info(f"Extracted {len(extracted_entities)} entities with confidence {confidence:.3f}")
        
        result.reasoning_chain.append(f"Extracted {len(extracted_entities)} entities")
    
    async def _execute_query_building_phase(self, result: OptimizationResult):
        """Execute query building phase"""
        logger.info(f"=== PHASE 2: QUERY BUILDING ===")
        
        result.reasoning_chain.append("Phase 2: Query building")
        
        # Find all entity extraction thoughts
        entity_thoughts = [t for t in self.thoughts.values() 
                          if t.thought_type == ThoughtType.ENTITY_EXTRACTION]
        
        logger.debug(f"Found {len(entity_thoughts)} entity extraction thoughts for query building")
        
        query_thoughts = []
        building_start = time.time()
        
        for i, entity_thought in enumerate(entity_thoughts):
            logger.debug(f"Building queries from entity thought {i+1}/{len(entity_thoughts)}: {entity_thought.id}")
            
            try:
                entities = entity_thought.content.get("entities", {})
                original_query = entity_thought.content.get("query", "")
                
                logger.debug(f"Entity thought {entity_thought.id}: {len(entities)} entities")
                
                if not entities:
                    logger.warning(f"No entities available in thought {entity_thought.id} - skipping query building")
                    continue
                
                query_start = time.time()
                from ..mcp_integration import call_mcp_tool  # Import here to avoid circular imports
                
                trapi_response = call_mcp_tool(
                    "build_trapi_query",
                    query=original_query,
                    entity_data=entities
                )
                
                query_time = time.time() - query_start
                trapi_query = trapi_response.get("query", {})
                
                logger.debug(f"TRAPI query built in {query_time:.2f}s, query available: {bool(trapi_query)}")
                
                # Determine confidence based on query building success
                confidence = 0.8 if trapi_query else 0.2
                
                query_thought = GoTThought(
                    id=f"query_building_{i+1}",
                    thought_type=ThoughtType.QUERY_BUILDING,
                    content={
                        "trapi_query": trapi_query,
                        "entities": entities,
                        "original_query": original_query
                    },
                    confidence=confidence,
                    dependencies={entity_thought.id}
                )
                
                self._add_thought(query_thought)
                query_thoughts.append(query_thought)
                
                logger.debug(f"Created query building thought: {query_thought.id} (confidence: {confidence:.3f})")
                
            except Exception as e:
                logger.warning(f"Query building failed for entity thought {entity_thought.id}: {type(e).__name__}: {str(e)}")
                logger.debug(f"Full query building error for {entity_thought.id}:", exc_info=True)
        
        building_time = time.time() - building_start
        logger.info(f"Query building phase completed in {building_time:.2f}s")
        logger.info(f"Built {len(query_thoughts)} TRAPI queries from {len(entity_thoughts)} entity thoughts")
        
        result.reasoning_chain.append(f"Built {len(query_thoughts)} TRAPI queries")
    
    async def _execute_api_execution_phase(self, config: Dict[str, Any], result: OptimizationResult):
        """Execute API execution phase"""
        logger.info(f"=== PHASE 3: API EXECUTION ===")
        logger.debug(f"Config: {config}")
        
        result.reasoning_chain.append("Phase 3: API execution")
        
        # Find all query building thoughts
        query_thoughts = [t for t in self.thoughts.values() 
                         if t.thought_type == ThoughtType.QUERY_BUILDING]
        
        logger.debug(f"Found {len(query_thoughts)} query building thoughts for API execution")
        
        if not query_thoughts:
            logger.warning("No query building thoughts found - skipping API execution")
            return
        
        api_thoughts = []
        execution_start = time.time()
        
        # Decide execution strategy
        use_parallel = self.enable_parallel and len(query_thoughts) > 1
        logger.info(f"Using {'parallel' if use_parallel else 'sequential'} execution for {len(query_thoughts)} queries")
        
        if use_parallel:
            # Parallel execution
            logger.debug("Starting parallel API execution...")
            tasks = []
            for i, query_thought in enumerate(query_thoughts):
                logger.debug(f"Creating task {i+1} for query thought: {query_thought.id}")
                task = self._execute_single_api_call(query_thought, i+1, config)
                tasks.append(task)
            
            parallel_start = time.time()
            api_results = await asyncio.gather(*tasks, return_exceptions=True)
            parallel_time = time.time() - parallel_start
            
            logger.debug(f"Parallel execution completed in {parallel_time:.2f}s")
            
            success_count = 0
            error_count = 0
            
            for i, api_result in enumerate(api_results):
                if isinstance(api_result, GoTThought):
                    self._add_thought(api_result)
                    api_thoughts.append(api_result)
                    success_count += 1
                    logger.debug(f"Task {i+1} succeeded: {api_result.id}")
                elif isinstance(api_result, Exception):
                    error_count += 1
                    logger.warning(f"Task {i+1} failed: {type(api_result).__name__}: {str(api_result)}")
                    logger.debug(f"Full parallel task {i+1} error:", exc_info=api_result)
            
            logger.info(f"Parallel execution: {success_count} succeeded, {error_count} failed")
            self.metrics.parallel_executions += 1
            
        else:
            # Sequential execution
            logger.debug("Starting sequential API execution...")
            for i, query_thought in enumerate(query_thoughts):
                logger.debug(f"Executing API call {i+1}/{len(query_thoughts)} for thought: {query_thought.id}")
                
                try:
                    call_start = time.time()
                    api_thought = await self._execute_single_api_call(query_thought, i+1, config)
                    call_time = time.time() - call_start
                    
                    self._add_thought(api_thought)
                    api_thoughts.append(api_thought)
                    
                    logger.debug(f"API call {i+1} completed in {call_time:.2f}s: {api_thought.id}")
                    
                except Exception as e:
                    logger.warning(f"API execution failed for query thought {query_thought.id}: {type(e).__name__}: {str(e)}")
                    logger.debug(f"Full sequential API error for {query_thought.id}:", exc_info=True)
        
        execution_time = time.time() - execution_start
        logger.info(f"API execution phase completed in {execution_time:.2f}s")
        logger.info(f"Successfully executed {len(api_thoughts)} API calls from {len(query_thoughts)} queries")
        
        result.reasoning_chain.append(f"Executed {len(api_thoughts)} API calls")
    
    async def _execute_single_api_call(self, query_thought: GoTThought, index: int, 
                                     config: Dict[str, Any]) -> GoTThought:
        """Execute a single API call"""
        logger.debug(f"Executing single API call {index} for thought: {query_thought.id}")
        
        trapi_query = query_thought.content.get("trapi_query", {})
        
        if not trapi_query:
            error_msg = f"No TRAPI query available in thought {query_thought.id}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        k = config.get("k", 5)
        max_results = config.get("maxresults", 50)
        
        logger.debug(f"API call {index} parameters: k={k}, max_results={max_results}")
        logger.debug(f"TRAPI query structure: {list(trapi_query.keys()) if isinstance(trapi_query, dict) else type(trapi_query)}")
        
        try:
            api_start = time.time()
            from ..mcp_integration import call_mcp_tool  # Import here to avoid circular imports
            
            bte_response = call_mcp_tool(
                "call_bte_api",
                json_query=trapi_query,
                k=k,
                maxresults=max_results
            )
            
            api_time = time.time() - api_start
            logger.debug(f"BTE API call {index} completed in {api_time:.2f}s")
            
            message = bte_response.get("message", {})
            api_results = message.get("results", [])
            
            logger.debug(f"API call {index} returned {len(api_results)} results")
            
            # Calculate confidence based on result count
            confidence = min(1.0, len(api_results) / 10.0)
            logger.debug(f"API call {index} confidence: {confidence:.3f}")
            
            api_thought = GoTThought(
                id=f"api_execution_{index}",
                thought_type=ThoughtType.API_EXECUTION,
                content={
                    "results": api_results,
                    "trapi_query": trapi_query,
                    "result_count": len(api_results),
                    "execution_time": api_time
                },
                confidence=confidence,
                dependencies={query_thought.id}
            )
            
            logger.debug(f"Created API execution thought: {api_thought.id}")
            return api_thought
            
        except Exception as e:
            logger.error(f"API call {index} failed for thought {query_thought.id}: {type(e).__name__}: {str(e)}")
            logger.debug(f"Full API call error:", exc_info=True)
            raise
    
    async def _execute_aggregation_phase(self, result: OptimizationResult):
        """Execute result aggregation phase"""
        logger.info(f"=== PHASE 4: RESULT AGGREGATION ===")
        
        result.reasoning_chain.append("Phase 4: Result aggregation")
        
        # Find all API execution thoughts
        api_thoughts = [t for t in self.thoughts.values() 
                       if t.thought_type == ThoughtType.API_EXECUTION]
        
        logger.debug(f"Found {len(api_thoughts)} API execution thoughts for aggregation")
        
        if not api_thoughts:
            logger.warning("No API execution thoughts found - skipping aggregation")
            return
        
        aggregation_start = time.time()
        
        if len(api_thoughts) > 1:
            logger.info(f"Aggregating {len(api_thoughts)} API execution results...")
            
            # Apply aggregation transformation
            aggregate_transform = self.transformations[TransformationType.AGGREGATE]
            
            try:
                transform_start = time.time()
                aggregated = await aggregate_transform.apply(
                    self.graph, 
                    api_thoughts, 
                    {"query": result.query}
                )
                transform_time = time.time() - transform_start
                
                logger.debug(f"Aggregation transformation completed in {transform_time:.3f}s")
                logger.debug(f"Transformation produced {len(aggregated)} aggregated thoughts")
                
                for i, agg_thought in enumerate(aggregated):
                    self._add_thought(agg_thought)
                    self.metrics.aggregation_count += 1
                    logger.debug(f"Added aggregated thought {i+1}: {agg_thought.id} ({agg_thought.thought_type.value})")
                    
                    # Log aggregation details
                    if agg_thought.thought_type == ThoughtType.RESULT_AGGREGATION:
                        results_count = len(agg_thought.content.get("results", []))
                        aggregated_from = agg_thought.content.get("aggregated_from", [])
                        logger.debug(f"Aggregated result has {results_count} results from {len(aggregated_from)} thoughts")
                
            except Exception as e:
                logger.error(f"Aggregation transformation failed: {type(e).__name__}: {str(e)}")
                logger.debug(f"Full aggregation error:", exc_info=True)
                # Add individual thoughts if aggregation fails
                for thought in api_thoughts:
                    if thought.id not in self.thoughts:
                        self._add_thought(thought)
        
        else:
            logger.info(f"Only 1 API execution thought - no aggregation needed")
            # Single thought - no aggregation needed, but ensure it's in the graph
            if api_thoughts[0].id not in self.thoughts:
                self._add_thought(api_thoughts[0])
        
        aggregation_time = time.time() - aggregation_start
        logger.info(f"Aggregation phase completed in {aggregation_time:.2f}s")
        logger.info(f"Aggregation count: {self.metrics.aggregation_count}")
        
        result.reasoning_chain.append(f"Aggregated {len(api_thoughts)} API results")
    
    def _calculate_got_metrics(self):
        """Calculate GoT-specific metrics from the paper"""
        logger.debug(f"Calculating GoT metrics for graph with {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        
        if not self.graph.nodes:
            logger.warning("Empty graph - cannot calculate metrics")
            return
        
        metrics_start = time.time()
        
        # Calculate volume: number of thoughts that could impact final results
        final_thoughts = [t for t in self.thoughts.values() 
                         if t.thought_type == ThoughtType.RESULT_AGGREGATION]
        
        logger.debug(f"Found {len(final_thoughts)} final aggregation thoughts")
        
        if final_thoughts:
            final_thought = final_thoughts[0]  # Take the first final thought
            logger.debug(f"Using final thought for metrics: {final_thought.id}")
            
            # Volume = number of ancestors in the graph
            try:
                ancestors = nx.ancestors(self.graph, final_thought.id)
                self.metrics.volume = len(ancestors) + 1  # Include the final thought itself
                logger.debug(f"Volume calculation: {len(ancestors)} ancestors + 1 = {self.metrics.volume}")
            except Exception as e:
                logger.warning(f"Failed to calculate volume: {e}")
                self.metrics.volume = len(self.graph.nodes)
            
            # Latency = shortest path from root to final thought
            try:
                self.metrics.latency = nx.shortest_path_length(self.graph, "root", final_thought.id)
                logger.debug(f"Latency: {self.metrics.latency} hops from root to final")
            except nx.NetworkXNoPath:
                logger.warning(f"No path from root to final thought {final_thought.id}")
                self.metrics.latency = 0
            except Exception as e:
                logger.warning(f"Failed to calculate latency: {e}")
                self.metrics.latency = 0
        else:
            logger.warning("No final thoughts found - using total nodes for volume")
            self.metrics.volume = len(self.graph.nodes)
            self.metrics.latency = 0
        
        # Dependency depth = maximum depth in the graph
        if "root" in self.graph:
            try:
                depths = []
                for node in self.graph.nodes:
                    if node != "root":
                        try:
                            depth = nx.shortest_path_length(self.graph, "root", node)
                            depths.append(depth)
                        except nx.NetworkXNoPath:
                            pass  # Skip unreachable nodes
                
                self.metrics.dependency_depth = max(depths) if depths else 0
                logger.debug(f"Dependency depth: {self.metrics.dependency_depth} (from {len(depths)} reachable nodes)")
                
            except Exception as e:
                logger.warning(f"Failed to calculate dependency depth: {e}")
                self.metrics.dependency_depth = 0
        else:
            logger.warning("No root node found in graph")
            self.metrics.dependency_depth = 0
        
        # Graph complexity = edges / nodes ratio
        if self.graph.nodes:
            self.metrics.graph_complexity = len(self.graph.edges) / len(self.graph.nodes)
            logger.debug(f"Graph complexity: {len(self.graph.edges)} edges / {len(self.graph.nodes)} nodes = {self.metrics.graph_complexity:.3f}")
        
        metrics_time = time.time() - metrics_start
        logger.debug(f"GoT metrics calculated in {metrics_time:.3f}s")
        logger.info(f"Final GoT metrics - Volume: {self.metrics.volume}, Latency: {self.metrics.latency}, Depth: {self.metrics.dependency_depth}, Complexity: {self.metrics.graph_complexity:.3f}")
    
    def _calculate_quality_score(self, result: OptimizationResult) -> float:
        """Calculate quality score based on GoT execution"""
        factors = []
        
        # Result diversity (0-0.3)
        if result.results:
            unique_types = set()
            for r in result.results:
                node_bindings = r.get("node_bindings", {})
                for key in node_bindings.keys():
                    unique_types.add(key)
            diversity_score = min(0.3, len(unique_types) / 10.0 * 0.3)
            factors.append(diversity_score)
        
        # Graph complexity bonus (0-0.2)
        complexity_bonus = min(0.2, self.metrics.graph_complexity * 0.1)
        factors.append(complexity_bonus)
        
        # Volume efficiency (0-0.3)
        if self.metrics.volume > 0:
            volume_efficiency = min(0.3, self.metrics.volume / self.metrics.total_thoughts * 0.3)
            factors.append(volume_efficiency)
        
        # Aggregation bonus (0-0.2)
        if self.metrics.aggregation_count > 0:
            aggregation_bonus = min(0.2, self.metrics.aggregation_count / 5.0 * 0.2)
            factors.append(aggregation_bonus)
        
        return sum(factors)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of GoT execution"""
        return {
            "total_thoughts": self.metrics.total_thoughts,
            "volume": self.metrics.volume,
            "latency": self.metrics.latency,
            "aggregation_count": self.metrics.aggregation_count,
            "refinement_count": self.metrics.refinement_count,
            "parallel_executions": self.metrics.parallel_executions,
            "dependency_depth": self.metrics.dependency_depth,
            "graph_complexity": self.metrics.graph_complexity,
            "thoughts_by_type": {
                thought_type.value: len([t for t in self.thoughts.values() if t.thought_type == thought_type])
                for thought_type in ThoughtType
            }
        }
    
    def visualize_graph(self) -> Dict[str, Any]:
        """Return graph structure for visualization"""
        nodes = []
        edges = []
        
        for thought_id, thought in self.thoughts.items():
            nodes.append({
                "id": thought_id,
                "type": thought.thought_type.value,
                "confidence": thought.confidence,
                "created_at": thought.created_at
            })
        
        for source, target in self.graph.edges:
            edges.append({"source": source, "target": target})
        
        return {"nodes": nodes, "edges": edges}


class GoTOptimizer:
    """
    Integration wrapper for using GoT with existing optimizer interface
    """
    
    def __init__(self, max_iterations: int = 5, enable_parallel: bool = True):
        self.planner = GoTBiomedicalPlanner(max_iterations, enable_parallel)
    
    async def optimize_with_got(self, query: str, strategy: OptimizationStrategy,
                               entities: Optional[Dict[str, str]] = None,
                               config: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """
        Optimize a query using the GoT framework
        
        Args:
            query: Biomedical query to optimize
            strategy: Optimization strategy (affects configuration)
            entities: Optional pre-extracted entities
            config: Optional configuration parameters
            
        Returns:
            OptimizationResult enhanced with GoT metrics
        """
        result = await self.planner.plan_and_execute(query, entities, config)
        result.strategy = strategy
        
        # Add GoT-specific metrics to reasoning chain
        summary = self.planner.get_execution_summary()
        result.reasoning_chain.append(f"GoT Summary: {summary}")
        
        return result
    
    def get_metrics(self) -> GoTMetrics:
        """Get GoT metrics"""
        return self.planner.metrics
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        return self.planner.get_execution_summary()