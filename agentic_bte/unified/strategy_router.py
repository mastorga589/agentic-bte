"""
(REMOVED) Strategy Router

This module implements intelligent strategy selection that chooses the optimal
execution approach based on query complexity, resource availability, historical
performance, and system constraints.
"""

import logging
import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .config import UnifiedConfig, ExecutionStrategy
from .types import EntityContext, ExecutionContext, BiomedicalEntity
from .performance import StrategyPerformanceTracker, ResourceMonitor

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels"""
    VERY_LOW = "very_low"      # Simple entity lookup
    LOW = "low"                # Single-hop relationships
    MEDIUM = "medium"          # Multi-entity queries
    HIGH = "high"              # Complex multi-hop queries
    VERY_HIGH = "very_high"    # Highly complex research queries


@dataclass
class QueryAnalysis:
    """Analysis results for a biomedical query"""
    complexity: QueryComplexity
    estimated_entities: int
    estimated_relationships: int
    estimated_execution_time: float
    requires_domain_expertise: bool
    requires_parallel_execution: bool
    requires_iterative_refinement: bool
    confidence_factors: List[str]
    reasoning: List[str]
    metadata: Dict[str, Any]


@dataclass
class ResourceConstraints:
    """Current system resource constraints"""
    available_memory_mb: float
    cpu_utilization: float
    concurrent_capacity: int
    cache_availability: bool
    api_rate_limits: Dict[str, int]
    timeout_constraints: float


@dataclass 
class StrategyRecommendation:
    """Strategy recommendation with justification"""
    primary_strategy: ExecutionStrategy
    fallback_strategies: List[ExecutionStrategy]
    confidence: float
    reasoning: List[str]
    estimated_performance: Dict[str, float]
    resource_requirements: Dict[str, Any]


class QueryComplexityAnalyzer:
    """Analyzes query complexity using multiple approaches"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        
        # Complexity indicators
        self.entity_complexity_patterns = {
            'drug_patterns': [
                r'\b(?:drug|medication|compound|therapeutic|treatment)s?\b',
                r'\b\w+mycin\b', r'\b\w+cillin\b', r'\b\w+cycline\b'
            ],
            'disease_patterns': [
                r'\b(?:disease|disorder|condition|syndrome|cancer|tumor)s?\b',
                r'\b\w+osis\b', r'\b\w+itis\b', r'\b\w+emia\b'
            ],
            'gene_patterns': [
                r'\b(?:gene|protein|enzyme|receptor)s?\b',
                r'\b[A-Z]{2,8}[0-9]{1,3}\b',  # Gene symbols
                r'\b(?:CYP|DRD|HTR|COMT)\w*\b'
            ],
            'process_patterns': [
                r'\b(?:pathway|mechanism|process|function)s?\b',
                r'\b(?:metabolism|synthesis|degradation|signaling)\b'
            ]
        }
        
        # Complexity keywords
        self.complexity_indicators = {
            'simple': ['what is', 'define', 'explain', 'describe'],
            'moderate': ['relationship', 'association', 'interaction', 'effect'],
            'complex': ['mechanism', 'pathway', 'cascade', 'network', 'system'],
            'very_complex': ['comprehensive', 'systematic', 'multi-factorial', 'personalized']
        }
        
        # Multi-entity indicators
        self.multi_entity_patterns = [
            r'\band\b', r'\bor\b', r'\bbetween\b', r'\bamong\b',
            r'\bmultiple\b', r'\bvarious\b', r'\bdifferent\b',
            r'\bseveral\b', r'\bmany\b', r'\bnumerous\b'
        ]
        
        # Temporal/causal complexity
        self.temporal_patterns = [
            r'\bafter\b', r'\bbefore\b', r'\bduring\b', r'\bfollowing\b',
            r'\bcause\b', r'\blead to\b', r'\bresult in\b', r'\btrigger\b'
        ]
    
    async def analyze_query_complexity(self, query: str, entity_context: Optional[EntityContext] = None) -> QueryAnalysis:
        """
        Analyze query complexity using multiple approaches
        
        Args:
            query: Natural language query
            entity_context: Optional entity context for additional analysis
            
        Returns:
            Comprehensive query analysis
        """
        logger.info(f"Analyzing query complexity: {query[:100]}...")
        
        # Initialize analysis
        analysis = QueryAnalysis(
            complexity=QueryComplexity.LOW,
            estimated_entities=0,
            estimated_relationships=0,
            estimated_execution_time=30.0,
            requires_domain_expertise=False,
            requires_parallel_execution=False,
            requires_iterative_refinement=False,
            confidence_factors=[],
            reasoning=[],
            metadata={}
        )
        
        query_lower = query.lower()
        
        # 1. Entity count analysis
        entity_count = self._count_entities_in_query(query)
        analysis.estimated_entities = entity_count
        analysis.reasoning.append(f"Estimated {entity_count} entities in query")
        
        if entity_context:
            actual_entities = len(entity_context.entities)
            analysis.estimated_entities = max(entity_count, actual_entities)
            analysis.reasoning.append(f"Context contains {actual_entities} actual entities")
        
        # 2. Keyword-based complexity
        keyword_complexity = self._analyze_keyword_complexity(query_lower)
        analysis.reasoning.append(f"Keyword complexity: {keyword_complexity}")
        
        # 3. Multi-entity relationship analysis
        multi_entity_score = self._analyze_multi_entity_patterns(query_lower)
        if multi_entity_score > 0:
            analysis.requires_parallel_execution = True
            analysis.reasoning.append("Multi-entity patterns detected - parallel execution recommended")
        
        # 4. Temporal/causal complexity
        temporal_score = self._analyze_temporal_complexity(query_lower)
        if temporal_score > 0:
            analysis.requires_iterative_refinement = True
            analysis.reasoning.append("Temporal/causal patterns detected - iterative refinement recommended")
        
        # 5. Domain expertise requirements
        domain_score = self._analyze_domain_expertise_requirements(query_lower)
        if domain_score > 0.7:
            analysis.requires_domain_expertise = True
            analysis.reasoning.append("High domain expertise requirements detected")
        
        # 6. Estimate relationships
        analysis.estimated_relationships = max(1, entity_count * 2)  # Conservative estimate
        
        # 7. Overall complexity determination
        complexity_score = (
            min(entity_count / 10, 1.0) * 0.3 +
            keyword_complexity * 0.3 +
            multi_entity_score * 0.2 +
            temporal_score * 0.1 +
            domain_score * 0.1
        )
        
        if complexity_score >= 0.8:
            analysis.complexity = QueryComplexity.VERY_HIGH
            analysis.estimated_execution_time = 120.0
        elif complexity_score >= 0.6:
            analysis.complexity = QueryComplexity.HIGH
            analysis.estimated_execution_time = 90.0
        elif complexity_score >= 0.4:
            analysis.complexity = QueryComplexity.MEDIUM
            analysis.estimated_execution_time = 60.0
        elif complexity_score >= 0.2:
            analysis.complexity = QueryComplexity.LOW
            analysis.estimated_execution_time = 30.0
        else:
            analysis.complexity = QueryComplexity.VERY_LOW
            analysis.estimated_execution_time = 15.0
        
        analysis.reasoning.append(f"Overall complexity score: {complexity_score:.2f} -> {analysis.complexity.value}")
        
        # 8. Confidence factors
        self._determine_confidence_factors(analysis, query)
        
        logger.info(f"Query complexity analysis complete: {analysis.complexity.value}")
        return analysis
    
    def _count_entities_in_query(self, query: str) -> int:
        """Count potential entities in query"""
        entity_count = 0
        query_lower = query.lower()
        
        for category, patterns in self.entity_complexity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query_lower)
                entity_count += len(matches)
        
        # Also count capitalized words (potential entity names)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
        entity_count += len(capitalized_words)
        
        return max(entity_count, 1)  # At least 1 entity
    
    def _analyze_keyword_complexity(self, query_lower: str) -> float:
        """Analyze keyword-based complexity indicators"""
        max_complexity = 0.0
        
        for complexity_level, keywords in self.complexity_indicators.items():
            if any(keyword in query_lower for keyword in keywords):
                if complexity_level == 'simple':
                    max_complexity = max(max_complexity, 0.2)
                elif complexity_level == 'moderate':
                    max_complexity = max(max_complexity, 0.4)
                elif complexity_level == 'complex':
                    max_complexity = max(max_complexity, 0.7)
                elif complexity_level == 'very_complex':
                    max_complexity = max(max_complexity, 0.9)
        
        return max_complexity
    
    def _analyze_multi_entity_patterns(self, query_lower: str) -> float:
        """Analyze patterns indicating multi-entity queries"""
        score = 0.0
        
        for pattern in self.multi_entity_patterns:
            if re.search(pattern, query_lower):
                score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_temporal_complexity(self, query_lower: str) -> float:
        """Analyze temporal and causal complexity patterns"""
        score = 0.0
        
        for pattern in self.temporal_patterns:
            if re.search(pattern, query_lower):
                score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_domain_expertise_requirements(self, query_lower: str) -> float:
        """Analyze requirements for domain expertise"""
        expertise_keywords = [
            'mechanism', 'pharmacokinetics', 'pharmacodynamics', 'clinical',
            'therapeutic', 'personalized', 'precision', 'biomarker',
            'efficacy', 'toxicity', 'adverse', 'contraindication'
        ]
        
        score = 0.0
        for keyword in expertise_keywords:
            if keyword in query_lower:
                score += 0.15
        
        return min(score, 1.0)
    
    def _determine_confidence_factors(self, analysis: QueryAnalysis, query: str):
        """Determine factors affecting confidence in the analysis"""
        factors = []
        
        if analysis.estimated_entities > 5:
            factors.append("high_entity_count")
        
        if len(query) > 200:
            factors.append("long_query")
        elif len(query) < 20:
            factors.append("short_query")
        
        if analysis.requires_domain_expertise:
            factors.append("domain_expertise_required")
        
        if analysis.requires_parallel_execution:
            factors.append("parallel_execution_recommended")
        
        if analysis.requires_iterative_refinement:
            factors.append("iterative_refinement_recommended")
        
        analysis.confidence_factors = factors


class ResourceConstraintAnalyzer:
    """Analyzes current system resource constraints"""
    
    def __init__(self, config: UnifiedConfig, resource_monitor: ResourceMonitor):
        self.config = config
        self.resource_monitor = resource_monitor
    
    async def analyze_resource_constraints(self) -> ResourceConstraints:
        """Analyze current resource constraints"""
        current_usage = self.resource_monitor.get_current_usage()
        
        # Calculate available resources
        available_memory = max(0, self.config.performance.memory_limit_mb - current_usage.memory_usage_mb)
        
        # Estimate concurrent capacity based on current load
        base_capacity = self.config.performance.max_concurrent_calls
        cpu_factor = max(0.1, 1.0 - (current_usage.cpu_percent / 100))
        memory_factor = max(0.1, available_memory / self.config.performance.memory_limit_mb)
        
        concurrent_capacity = int(base_capacity * min(cpu_factor, memory_factor))
        
        return ResourceConstraints(
            available_memory_mb=available_memory,
            cpu_utilization=current_usage.cpu_percent,
            concurrent_capacity=concurrent_capacity,
            cache_availability=self.config.caching.enable_caching,
            api_rate_limits={},  # Would be populated from actual API limits
            timeout_constraints=self.config.performance.query_timeout_seconds
        )


# STRATEGY ROUTER REMOVED
    
    def _find_compatible_strategies(self, 
                                  query_analysis: QueryAnalysis, 
                                  resource_constraints: ResourceConstraints) -> List[ExecutionStrategy]:
        """Find strategies compatible with query requirements and constraints"""
        compatible = []
        
        for strategy, capabilities in self.strategy_capabilities.items():
            # Check complexity compatibility
            if query_analysis.complexity.value > capabilities['max_complexity'].value:
                continue
            
            # Check feature requirements
            if query_analysis.requires_parallel_execution and not capabilities['supports_parallel']:
                continue
            
            if query_analysis.requires_domain_expertise and not capabilities['supports_domain_expertise']:
                continue
            
            if query_analysis.requires_iterative_refinement and not capabilities['supports_iterative']:
                continue
            
            # Check resource constraints
            if resource_constraints.concurrent_capacity < 2 and not capabilities['resource_efficient']:
                continue
            
            if resource_constraints.available_memory_mb < 500 and not capabilities['resource_efficient']:
                continue
            
            compatible.append(strategy)
        
        # Always include hybrid as fallback
        if ExecutionStrategy.HYBRID_ADAPTIVE not in compatible:
            compatible.append(ExecutionStrategy.HYBRID_ADAPTIVE)
        
        return compatible
    
    def _rank_strategies(self, 
                        compatible_strategies: List[ExecutionStrategy],
                        query_analysis: QueryAnalysis,
                        resource_constraints: ResourceConstraints,
                        performance_rankings: List[Tuple[ExecutionStrategy, float]]) -> List[Tuple[ExecutionStrategy, float]]:
        """Rank compatible strategies by suitability score"""
        strategy_scores = []
        
        # Create performance lookup
        performance_lookup = {strategy: score for strategy, score in performance_rankings}
        
        for strategy in compatible_strategies:
            score = 0.0
            capabilities = self.strategy_capabilities[strategy]
            
            # Performance history score (30%)
            historical_performance = performance_lookup.get(strategy, 0.0)
            score += 0.3 * historical_performance
            
            # Capability match score (40%)
            capability_score = 0.0
            
            # Complexity handling
            if query_analysis.complexity == QueryComplexity.VERY_LOW and capabilities['fast_execution']:
                capability_score += 0.3
            elif query_analysis.complexity in [QueryComplexity.HIGH, QueryComplexity.VERY_HIGH]:
                if capabilities['max_complexity'] in [QueryComplexity.HIGH, QueryComplexity.VERY_HIGH]:
                    capability_score += 0.3
            
            # Feature requirements
            if query_analysis.requires_parallel_execution and capabilities['supports_parallel']:
                capability_score += 0.2
            
            if query_analysis.requires_domain_expertise and capabilities['supports_domain_expertise']:
                capability_score += 0.2
            
            if query_analysis.requires_iterative_refinement and capabilities['supports_iterative']:
                capability_score += 0.2
            
            score += 0.4 * capability_score
            
            # Resource efficiency score (20%)
            resource_score = 0.0
            if resource_constraints.available_memory_mb < 1000 and capabilities['resource_efficient']:
                resource_score += 0.5
            
            if resource_constraints.cpu_utilization > 70 and capabilities['fast_execution']:
                resource_score += 0.3
            
            if resource_constraints.concurrent_capacity < 3 and capabilities['resource_efficient']:
                resource_score += 0.2
            
            score += 0.2 * resource_score
            
            # Execution time estimate bonus (10%)
            if query_analysis.estimated_execution_time > 60 and capabilities['fast_execution']:
                score += 0.1
            
            strategy_scores.append((strategy, score))
        
        # Sort by score descending
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        return strategy_scores
    
    def _create_recommendation(self,
                              ranked_strategies: List[Tuple[ExecutionStrategy, float]],
                              query_analysis: QueryAnalysis,
                              resource_constraints: ResourceConstraints) -> StrategyRecommendation:
        """Create final strategy recommendation"""
        if not ranked_strategies:
            # Emergency fallback
            return StrategyRecommendation(
                primary_strategy=ExecutionStrategy.SIMPLE,
                fallback_strategies=[ExecutionStrategy.HYBRID_ADAPTIVE],
                confidence=0.1,
                reasoning=["No compatible strategies found - using simple fallback"],
                estimated_performance={'execution_time': 60.0, 'success_rate': 0.5},
                resource_requirements={'memory_mb': 100, 'cpu_percent': 10}
            )
        
        primary_strategy, primary_score = ranked_strategies[0]
        fallback_strategies = [strategy for strategy, _ in ranked_strategies[1:3]]  # Top 2 fallbacks
        
        # Calculate confidence based on score and analysis certainty
        confidence = min(0.95, primary_score * 0.8 + 0.2)  # Scale to reasonable range
        
        # Build reasoning
        reasoning = [
            f"Query complexity: {query_analysis.complexity.value}",
            f"Estimated entities: {query_analysis.estimated_entities}",
            f"Strategy score: {primary_score:.2f}"
        ]
        reasoning.extend(query_analysis.reasoning[:3])  # Add key analysis points
        
        # Estimate performance
        capabilities = self.strategy_capabilities[primary_strategy]
        estimated_performance = {
            'execution_time': query_analysis.estimated_execution_time * (1.2 if not capabilities['fast_execution'] else 0.8),
            'success_rate': 0.9 if primary_score > 0.7 else 0.7,
            'quality_score': 0.8 if capabilities['supports_domain_expertise'] else 0.6
        }
        
        # Estimate resource requirements
        base_memory = 200 if capabilities['resource_efficient'] else 500
        base_cpu = 20 if capabilities['fast_execution'] else 40
        
        resource_requirements = {
            'memory_mb': base_memory * query_analysis.estimated_entities,
            'cpu_percent': base_cpu,
            'concurrent_calls': 2 if capabilities['supports_parallel'] else 1
        }
        
        return StrategyRecommendation(
            primary_strategy=primary_strategy,
            fallback_strategies=fallback_strategies,
            confidence=confidence,
            reasoning=reasoning,
            estimated_performance=estimated_performance,
            resource_requirements=resource_requirements
        )
    
    def get_strategy_explanation(self, strategy: ExecutionStrategy) -> Dict[str, Any]:
        """Get detailed explanation of a strategy's capabilities"""
        if strategy not in self.strategy_capabilities:
            return {}
        
        capabilities = self.strategy_capabilities[strategy]
        performance = self.performance_tracker.get_strategy_performance(strategy)
        
        return {
            'strategy': strategy.value,
            'capabilities': capabilities,
            'historical_performance': {
                'success_rate': performance.success_rate,
                'average_execution_time': performance.average_execution_time,
                'average_quality_score': performance.average_quality_score,
                'total_uses': performance.total_uses
            },
            'best_for': self._get_strategy_best_use_cases(strategy),
            'limitations': self._get_strategy_limitations(strategy)
        }
    
    def _get_strategy_best_use_cases(self, strategy: ExecutionStrategy) -> List[str]:
        """Get best use cases for a strategy"""
        use_cases = {
            ExecutionStrategy.SIMPLE: [
                "Simple entity lookups",
                "Basic relationship queries", 
                "Quick exploratory queries"
            ],
            ExecutionStrategy.GOT_FRAMEWORK: [
                "Complex multi-step reasoning",
                "Graph-based query decomposition",
                "High-quality result aggregation"
            ],
            ExecutionStrategy.LANGGRAPH_AGENTS: [
                "Research-grade comprehensive analysis",
                "Domain expertise integration",
                "Multi-turn iterative refinement"
            ],
            ExecutionStrategy.PRODUCTION_GOT: [
                "Production biomedical queries",
                "Parallel predicate execution",
                "Evidence-weighted results"
            ],
            ExecutionStrategy.ENHANCED_GOT: [
                "Queries requiring pharmaceutical expertise",
                "Mechanism-based analysis",
                "Drug discovery research"
            ],
            ExecutionStrategy.STATEFUL_ITERATIVE: [
                "Long-running research processes",
                "Knowledge accumulation queries",
                "Context-dependent analysis"
            ],
            ExecutionStrategy.HYBRID_ADAPTIVE: [
                "General purpose queries",
                "Unknown complexity queries",
                "Fallback for other strategies"
            ]
        }
        return use_cases.get(strategy, [])
    
    def _get_strategy_limitations(self, strategy: ExecutionStrategy) -> List[str]:
        """Get limitations of a strategy"""
        limitations = {
            ExecutionStrategy.SIMPLE: [
                "Limited to simple relationships",
                "No parallel execution",
                "Basic result quality"
            ],
            ExecutionStrategy.GOT_FRAMEWORK: [
                "High computational overhead",
                "Complex implementation",
                "Slower execution"
            ],
            ExecutionStrategy.LANGGRAPH_AGENTS: [
                "Resource intensive",
                "Complex state management",
                "Potential performance issues"
            ],
            ExecutionStrategy.PRODUCTION_GOT: [
                "Tightly coupled components",
                "Limited domain expertise",
                "Complex configuration"
            ],
            ExecutionStrategy.ENHANCED_GOT: [
                "Hard-coded knowledge bases",
                "Limited scalability",
                "Pharmaceutical focus only"
            ],
            ExecutionStrategy.STATEFUL_ITERATIVE: [
                "Early termination issues",
                "Less battle-tested",
                "Context management complexity"
            ],
            ExecutionStrategy.HYBRID_ADAPTIVE: [
                "Strategy selection overhead",
                "Potentially suboptimal heuristics",
                "Complexity in debugging"
            ]
        }
        return limitations.get(strategy, [])
    
    async def initialize(self) -> None:
        """Initialize the strategy router"""
        logger.info("Initializing UnifiedStrategyRouter...")
        # Components are initialized in __init__, nothing additional needed  
        logger.info("UnifiedStrategyRouter initialization completed")
