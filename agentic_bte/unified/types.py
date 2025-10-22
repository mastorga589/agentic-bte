"""
Unified Data Structures

This module provides standardized data structures that work across all
execution strategies, ensuring consistent data formats throughout the system.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Union
from enum import Enum
import json
import uuid



class EntityType(Enum):
    """Biomedical entity types"""
    GENE = "gene"
    PROTEIN = "protein"  
    DISEASE = "disease"
    DRUG = "drug"
    CHEMICAL = "chemical"
    PATHWAY = "pathway"
    PROCESS = "process"
    PHENOTYPE = "phenotype"
    ANATOMY = "anatomy"
    ORGANISM = "organism"
    UNKNOWN = "unknown"


class ConfidenceLevel(Enum):
    """Confidence levels for results and entities"""
    VERY_LOW = "very_low"     # 0.0 - 0.2
    LOW = "low"               # 0.2 - 0.4
    MEDIUM = "medium"         # 0.4 - 0.6
    HIGH = "high"             # 0.6 - 0.8
    VERY_HIGH = "very_high"   # 0.8 - 1.0


class ExecutionStatus(Enum):
    """Execution status for steps and overall results"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ExecutionStrategy(Enum):
    """Execution strategies for biomedical queries"""
    SIMPLE = "simple"
    GOT_FRAMEWORK = "got_framework"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    COMPREHENSIVE = "comprehensive"


@dataclass
class BiomedicalEntity:
    """Standardized biomedical entity representation"""
    name: str
    entity_id: str
    entity_type: EntityType
    confidence: float
    source: str = "unknown"
    synonyms: List[str] = field(default_factory=list)
    description: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate entity data"""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        if not self.name:
            raise ValueError("Entity name cannot be empty")
        
        if not self.entity_id:
            raise ValueError("Entity ID cannot be empty")
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level enum"""
        if self.confidence >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.6:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.4:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "confidence": self.confidence,
            "source": self.source,
            "synonyms": self.synonyms,
            "description": self.description,
            "categories": self.categories,
            "attributes": self.attributes
        }


# Alias for compatibility
Entity = BiomedicalEntity


@dataclass
class EntityContext:
    """Context container for all entity-related information"""
    entities: List[BiomedicalEntity]
    extracted_at: float = field(default_factory=time.time)
    extraction_method: str = "unified"
    entity_mappings: Dict[str, str] = field(default_factory=dict)  # name -> ID
    placeholder_mappings: Dict[str, List[str]] = field(default_factory=dict)  # placeholder -> entity_ids
    generic_resolutions: Dict[str, List[str]] = field(default_factory=dict)  # "drugs" -> specific IDs
    
    def __post_init__(self):
        """Build entity mappings"""
        self.entity_mappings = {entity.name: entity.entity_id for entity in self.entities}
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[BiomedicalEntity]:
        """Get entities filtered by type"""
        return [e for e in self.entities if e.entity_type == entity_type]
    
    def get_entity_by_id(self, entity_id: str) -> Optional[BiomedicalEntity]:
        """Get entity by ID"""
        for entity in self.entities:
            if entity.entity_id == entity_id:
                return entity
        return None
    
    def get_entity_by_name(self, name: str) -> Optional[BiomedicalEntity]:
        """Get entity by name"""
        for entity in self.entities:
            if entity.name.lower() == name.lower():
                return entity
            if name.lower() in [s.lower() for s in entity.synonyms]:
                return entity
        return None
    
    def add_entity(self, entity: BiomedicalEntity):
        """Add entity to context"""
        self.entities.append(entity)
        self.entity_mappings[entity.name] = entity.entity_id
    
    def get_high_confidence_entities(self, threshold: float = 0.7) -> List[BiomedicalEntity]:
        """Get entities above confidence threshold"""
        return [e for e in self.entities if e.confidence >= threshold]


@dataclass
class BiomedicalRelationship:
    """Standardized biomedical relationship representation"""
    subject: BiomedicalEntity
    predicate: str
    object: BiomedicalEntity
    confidence: float
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate relationship data"""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        if not self.predicate:
            raise ValueError("Predicate cannot be empty")
    
    def to_triple(self) -> tuple:
        """Convert to simple subject-predicate-object triple"""
        return (self.subject.entity_id, self.predicate, self.object.entity_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "subject": self.subject.to_dict(),
            "predicate": self.predicate,
            "object": self.object.to_dict(),
            "confidence": self.confidence,
            "evidence": self.evidence,
            "sources": self.sources,
            "attributes": self.attributes
        }


@dataclass  
class KnowledgeGraph:
    """Container for biomedical knowledge graph data"""
    relationships: List[BiomedicalRelationship]
    entities: List[BiomedicalEntity]
    created_at: float = field(default_factory=time.time)
    source: str = "unified_system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Build entity index"""
        self._entity_index = {entity.entity_id: entity for entity in self.entities}
    
    def get_entity_relationships(self, entity_id: str) -> List[BiomedicalRelationship]:
        """Get all relationships involving an entity"""
        return [
            rel for rel in self.relationships 
            if rel.subject.entity_id == entity_id or rel.object.entity_id == entity_id
        ]
    
    def get_relationships_by_predicate(self, predicate: str) -> List[BiomedicalRelationship]:
        """Get relationships filtered by predicate"""
        return [rel for rel in self.relationships if rel.predicate == predicate]
    
    def add_relationship(self, relationship: BiomedicalRelationship):
        """Add relationship to graph"""
        self.relationships.append(relationship)
        
        # Add entities if not already present
        if relationship.subject.entity_id not in self._entity_index:
            self.entities.append(relationship.subject)
            self._entity_index[relationship.subject.entity_id] = relationship.subject
        
        if relationship.object.entity_id not in self._entity_index:
            self.entities.append(relationship.object)
            self._entity_index[relationship.object.entity_id] = relationship.object
    
    def merge(self, other: 'KnowledgeGraph') -> 'KnowledgeGraph':
        """Merge with another knowledge graph"""
        merged_entities = list(self.entities)
        merged_relationships = list(self.relationships)
        
        # Add entities from other graph
        for entity in other.entities:
            if entity.entity_id not in self._entity_index:
                merged_entities.append(entity)
        
        # Add relationships from other graph 
        existing_triples = {rel.to_triple() for rel in self.relationships}
        for relationship in other.relationships:
            if relationship.to_triple() not in existing_triples:
                merged_relationships.append(relationship)
        
        return KnowledgeGraph(
            relationships=merged_relationships,
            entities=merged_entities,
            source=f"merged_{self.source}_{other.source}",
            metadata={**self.metadata, **other.metadata}
        )
    
    def to_rdf_triples(self) -> List[tuple]:
        """Convert to RDF triple format"""
        return [rel.to_triple() for rel in self.relationships]
    
    def get_stats(self) -> Dict[str, int]:
        """Get knowledge graph statistics"""
        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "unique_predicates": len(set(rel.predicate for rel in self.relationships)),
            "entity_types": len(set(entity.entity_type for entity in self.entities))
        }


@dataclass
class ExecutionStep:
    """Individual execution step with timing and results"""
    step_id: str
    step_type: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    execution_time: Optional[float] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate ID if not provided"""
        if not self.step_id:
            self.step_id = f"{self.step_type}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    def mark_success(self, output_data: Dict[str, Any] = None):
        """Mark step as successful"""
        self.status = ExecutionStatus.SUCCESS
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        if output_data:
            self.output_data.update(output_data)
    
    def mark_failure(self, error_message: str):
        """Mark step as failed"""
        self.status = ExecutionStatus.FAILED
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        self.error_message = error_message
        self.confidence = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "execution_time": self.execution_time,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error_message": self.error_message,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class ExecutionContext:
    """Context for executing a biomedical query"""
    query: str
    strategy: str
    entity_context: EntityContext
    knowledge_graph: KnowledgeGraph
    config: Any  # UnifiedConfig - avoiding circular import
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "strategy": self.strategy.value,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "entity_count": len(self.entity_context.entities),
            "knowledge_graph_stats": self.knowledge_graph.get_stats(),
            "metadata": self.metadata
        }


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_execution_time: float = 0.0
    entity_extraction_time: float = 0.0
    query_building_time: float = 0.0
    api_execution_time: float = 0.0
    result_processing_time: float = 0.0
    
    # Parallel execution metrics
    parallel_operations: int = 0
    concurrent_api_calls: int = 0
    parallelization_speedup: float = 1.0
    
    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Quality metrics
    confidence_scores: List[float] = field(default_factory=list)
    evidence_scores: List[float] = field(default_factory=list)
    
    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Error metrics
    error_count: int = 0
    retry_count: int = 0
    timeout_count: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return self.cache_hits / total_requests
    
    @property
    def average_confidence(self) -> float:
        """Calculate average confidence score"""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)
    
    @property
    def average_evidence_score(self) -> float:
        """Calculate average evidence score"""
        if not self.evidence_scores:
            return 0.0
        return sum(self.evidence_scores) / len(self.evidence_scores)
    
    def add_confidence_score(self, score: float):
        """Add confidence score"""
        if 0.0 <= score <= 1.0:
            self.confidence_scores.append(score)
    
    def add_evidence_score(self, score: float):
        """Add evidence score"""
        if 0.0 <= score <= 1.0:
            self.evidence_scores.append(score)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_execution_time": self.total_execution_time,
            "entity_extraction_time": self.entity_extraction_time,
            "query_building_time": self.query_building_time,
            "api_execution_time": self.api_execution_time,
            "result_processing_time": self.result_processing_time,
            "parallel_operations": self.parallel_operations,
            "concurrent_api_calls": self.concurrent_api_calls,
            "parallelization_speedup": self.parallelization_speedup,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "average_confidence": self.average_confidence,
            "average_evidence_score": self.average_evidence_score,
            "cache_hit_rate": self.cache_hit_rate,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "timeout_count": self.timeout_count
        }


@dataclass
class BiomedicalResult:
    """Complete result from biomedical query processing"""
    query: str
    strategy_used: str
    final_answer: str
    knowledge_graph: KnowledgeGraph
    entity_context: EntityContext
    execution_steps: List[ExecutionStep]
    performance_metrics: PerformanceMetrics
    
    # Result metadata
    success: bool = True
    confidence: float = 0.0
    quality_score: float = 0.0
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    
    # Debugging information
    trapi_queries: List[Dict[str, Any]] = field(default_factory=list)
    raw_results: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate overall metrics"""
        if self.execution_steps:
            # Calculate overall confidence from steps
            step_confidences = [step.confidence for step in self.execution_steps if step.confidence > 0]
            if step_confidences:
                self.confidence = sum(step_confidences) / len(step_confidences)
        
        # Calculate quality score based on various factors
        self.quality_score = self._calculate_quality_score()
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall quality score"""
        factors = []
        
        # Entity extraction quality
        if self.entity_context.entities:
            avg_entity_confidence = sum(e.confidence for e in self.entity_context.entities) / len(self.entity_context.entities)
            factors.append(avg_entity_confidence)
        
        # Knowledge graph richness
        if self.knowledge_graph:
            kg_stats = self.knowledge_graph.get_stats()
            if kg_stats["total_relationships"] > 0:
                # More relationships and entities generally indicate better coverage
                richness_score = min(1.0, (kg_stats["total_relationships"] + kg_stats["total_entities"]) / 100)
                factors.append(richness_score)
        
        # Execution success rate
        successful_steps = len([s for s in self.execution_steps if s.status == ExecutionStatus.SUCCESS])
        if self.execution_steps:
            success_rate = successful_steps / len(self.execution_steps)
            factors.append(success_rate)
        
        # Performance factors
        if self.performance_metrics.average_confidence > 0:
            factors.append(self.performance_metrics.average_confidence)
        
        if self.performance_metrics.average_evidence_score > 0:
            factors.append(self.performance_metrics.average_evidence_score)
        
        # Error penalty
        error_penalty = min(0.5, len(self.errors) * 0.1)
        
        if factors:
            base_quality = sum(factors) / len(factors)
            return max(0.0, base_quality - error_penalty)
        else:
            return 0.0
    
    def get_successful_steps(self) -> List[ExecutionStep]:
        """Get all successful execution steps"""
        return [step for step in self.execution_steps if step.status == ExecutionStatus.SUCCESS]
    
    def get_failed_steps(self) -> List[ExecutionStep]:
        """Get all failed execution steps"""
        return [step for step in self.execution_steps if step.status == ExecutionStatus.FAILED]
    
    def add_warning(self, warning: str):
        """Add warning message"""
        self.warnings.append(warning)
    
    def add_error(self, error: str):
        """Add error message"""
        self.errors.append(error)
        self.success = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "query": self.query,
            "strategy_used": str(self.strategy_used),
            "final_answer": self.final_answer,
            "success": self.success,
            "confidence": self.confidence,
            "quality_score": self.quality_score,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "knowledge_graph_stats": self.knowledge_graph.get_stats() if self.knowledge_graph else {},
            "entity_count": len(self.entity_context.entities),
            "execution_steps": [step.to_dict() for step in self.execution_steps],
            "performance_metrics": self.performance_metrics.to_dict(),
            "warnings": self.warnings,
            "errors": self.errors
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def save_to_file(self, filepath: str):
        """Save result to JSON file"""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BiomedicalResult':
        """Create result from dictionary"""
        # This would need detailed implementation to reconstruct all objects
        # For now, create a minimal result
        result = cls(
            query=data.get("query", ""),
            strategy_used=data.get("strategy_used", "simple"),
            final_answer=data.get("final_answer", ""),
            knowledge_graph=KnowledgeGraph([], []),  # Would need to reconstruct
            entity_context=EntityContext([]),  # Would need to reconstruct
            execution_steps=[],  # Would need to reconstruct
            performance_metrics=PerformanceMetrics()  # Would need to reconstruct
        )
        
        result.success = data.get("success", False)
        result.confidence = data.get("confidence", 0.0)
        result.quality_score = data.get("quality_score", 0.0)
        result.warnings = data.get("warnings", [])
        result.errors = data.get("errors", [])
        
        return result


# Utility functions
def create_error_result(query: str, error_message: str, strategy: str = "simple") -> BiomedicalResult:
    """Create an error result"""
    result = BiomedicalResult(
        query=query,
        strategy_used=strategy,
        final_answer=f"Error processing query: {error_message}",
        knowledge_graph=KnowledgeGraph([], []),
        entity_context=EntityContext([]),
        execution_steps=[],
        performance_metrics=PerformanceMetrics(),
        success=False,
        confidence=0.0,
        quality_score=0.0
    )
    result.add_error(error_message)
    return result


def merge_results(results: List[BiomedicalResult]) -> BiomedicalResult:
    """Merge multiple results into a single comprehensive result"""
    if not results:
        raise ValueError("Cannot merge empty list of results")
    
    if len(results) == 1:
        return results[0]
    
    # Use first result as base
    merged = results[0]
    
    # Merge knowledge graphs
    for result in results[1:]:
        merged.knowledge_graph = merged.knowledge_graph.merge(result.knowledge_graph)
    
    # Merge entity contexts
    all_entities = []
    for result in results:
        all_entities.extend(result.entity_context.entities)
    
    # Remove duplicate entities
    unique_entities = []
    seen_ids = set()
    for entity in all_entities:
        if entity.entity_id not in seen_ids:
            unique_entities.append(entity)
            seen_ids.add(entity.entity_id)
    
    merged.entity_context = EntityContext(unique_entities)
    
    # Merge execution steps
    all_steps = []
    for result in results:
        all_steps.extend(result.execution_steps)
    merged.execution_steps = all_steps
    
    # Combine warnings and errors
    all_warnings = []
    all_errors = []
    for result in results:
        all_warnings.extend(result.warnings)
        all_errors.extend(result.errors)
    
    merged.warnings = list(set(all_warnings))  # Remove duplicates
    merged.errors = list(set(all_errors))
    
    # Recalculate quality score
    merged.quality_score = merged._calculate_quality_score()
    
    return merged


@dataclass
class ExecutionContext:
    """Context for query execution"""
    query: str
    strategy: Optional['ExecutionStrategy'] = None
    entity_context: Optional[EntityContext] = None
    knowledge_graph: Optional[KnowledgeGraph] = None
    config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_error_result(error_message: str, error_code: str = "GENERAL_ERROR") -> 'BiomedicalResult':
    """Create a BiomedicalResult representing an error"""
    return BiomedicalResult(
        success=False,
        total_results=0,
        results=[],
        error_message=error_message,
        error_code=error_code,
        execution_time=0.0,
        stages_completed=[],
        metadata={"error": True}
    )
