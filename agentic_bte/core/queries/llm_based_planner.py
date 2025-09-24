"""
Advanced LLM-Based Query Planner

This module implements sophisticated query planning using only LLMs without hardcoded templates.
Features include:
- Complex dependency analysis between subqueries
- Bidirectional search strategies (forward, backward, convergence)
- Dynamic replanning based on intermediate results
- Multi-strategy query decomposition
- Parallel execution opportunity identification

This approach uses structured LLM reasoning to replicate the benefits of template-based
planning while maintaining the flexibility of pure natural language processing.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
from collections import defaultdict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from rdflib import Graph, URIRef, Namespace

from ..entities.bio_ner import BioNERTool
from ..knowledge.trapi import TRAPIQueryBuilder
from ..knowledge.bte_client import BTEClient
from ...config.settings import get_settings
from ...exceptions.base import ExternalServiceError

logger = logging.getLogger(__name__)


@dataclass
class MetaKGEdge:
    """Represents an edge in the BTE meta-knowledge graph"""
    subject: str
    predicate: str
    object: str
    api_name: str = ""
    provided_by: str = ""
    
    def __str__(self) -> str:
        return f"{self.subject} -{self.predicate}-> {self.object}"


class SearchDirection(Enum):
    """Search direction for bidirectional queries"""
    FORWARD = "forward"
    BACKWARD = "backward" 
    CONVERGENT = "convergent"


class DependencyType(Enum):
    """Types of dependencies between subqueries"""
    SEQUENTIAL = "sequential"  # Q2 needs results from Q1
    PARALLEL = "parallel"     # Q1 and Q2 can run simultaneously
    CONDITIONAL = "conditional"  # Q2 runs only if Q1 meets criteria
    CONVERGENT = "convergent"   # Q3 combines results from Q1 and Q2


class QueryStrategy(Enum):
    """Strategic approaches for query decomposition"""
    MECHANISTIC = "mechanistic"        # drug -> target -> pathway -> disease
    FUNCTIONAL = "functional"          # entity -> function -> related entities
    BIDIRECTIONAL = "bidirectional"   # forward + backward + convergence
    EXPLORATORY = "exploratory"       # broad search then narrow down
    COMPARATIVE = "comparative"        # compare multiple entities/pathways


@dataclass
class DependencyRelation:
    """Represents a dependency relationship between subqueries"""
    dependent_id: str
    prerequisite_id: str
    dependency_type: DependencyType
    confidence: float = 1.0
    reasoning: str = ""


@dataclass
class AdvancedSubqueryNode:
    """Enhanced subquery node with advanced planning features"""
    id: str = field(default_factory=lambda: str(uuid4()))
    query: str = ""
    search_direction: SearchDirection = SearchDirection.FORWARD
    strategy: QueryStrategy = QueryStrategy.MECHANISTIC
    priority: int = 1
    estimated_cost: float = 1.0
    confidence_threshold: float = 0.7
    expected_entity_types: Set[str] = field(default_factory=set)
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    suggested_edge: Optional[MetaKGEdge] = None  # The meta-KG edge that inspired this subquery
    
    # Execution tracking
    results: List[Dict[str, Any]] = field(default_factory=list)
    actual_cost: float = 0.0
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class AdvancedExecutionPlan:
    """Sophisticated execution plan with dependency management"""
    id: str = field(default_factory=lambda: str(uuid4()))
    original_query: str = ""
    strategy: QueryStrategy = QueryStrategy.MECHANISTIC
    
    # Subqueries and dependencies
    subqueries: List[AdvancedSubqueryNode] = field(default_factory=list)
    dependencies: List[DependencyRelation] = field(default_factory=list)
    
    # Execution organization
    parallel_groups: List[List[str]] = field(default_factory=list)
    execution_phases: List[List[str]] = field(default_factory=list)
    contingency_plans: Dict[str, List[str]] = field(default_factory=dict)
    
    # Planning metadata
    total_estimated_cost: float = 0.0
    expected_completion_time: float = 0.0
    confidence_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    planning_reasoning: str = ""


class LLMBasedQueryPlanner:
    """
    Advanced query planner using sophisticated LLM reasoning
    
    This planner uses structured LLM prompts to:
    1. Analyze query intent and complexity
    2. Choose optimal decomposition strategy
    3. Generate dependent subqueries with reasoning
    4. Identify parallel execution opportunities
    5. Create contingency plans for failures
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.BIOLINK = None
        self.EX = None
        self.GENE = None
        self.DISEASE = None
        self.PHYSPROCESS = None
        self.BIOENT = None
        self.PATHPROCESS = None
        self.SMALLMOL = None
        self.PHENFEATURE = None
        self.POLYPEPTIDE = None
        self.ENTITY_NAMESPACE_MAP = {}
        """
        Initialize the advanced LLM-based query planner
        
        Args:
            openai_api_key: OpenAI API key for LLM operations
        """
        self.settings = get_settings()
        self.openai_api_key = openai_api_key or self.settings.openai_api_key
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for LLM-based planning")
        
        # Initialize components
        self.llm = ChatOpenAI(
            temperature=0.1,  # Slight temperature for creative planning
            model=self.settings.openai_model,
            api_key=self.openai_api_key
        )
        self.bio_ner = BioNERTool(openai_api_key)
        self.trapi_builder = TRAPIQueryBuilder(openai_api_key)
        self.bte_client = BTEClient()
        
        # Planning cache and history
        self._plan_cache: Dict[str, AdvancedExecutionPlan] = {}
        self._planning_history: List[Dict[str, Any]] = []
        
        # RDF namespace setup
        self._setup_rdf_namespaces()
        
        # Meta-KG cache
        self._meta_kg_edges: List[MetaKGEdge] = []
        self._edges_by_subject: Dict[str, List[MetaKGEdge]] = defaultdict(list)
        self._edges_by_object: Dict[str, List[MetaKGEdge]] = defaultdict(list)
        self._initialize_meta_kg()
    
    def create_advanced_plan(self, 
                           query: str,
                           entities: Dict[str, str] = None,
                           max_subqueries: int = 12,
                           enable_bidirectional: bool = True,
                           confidence_threshold: float = 0.7) -> AdvancedExecutionPlan:
        """
        Create an advanced execution plan using sophisticated LLM reasoning
        
        Args:
            query: Natural language biomedical query
            entities: Optional pre-extracted entities
            max_subqueries: Maximum number of subqueries
            enable_bidirectional: Enable bidirectional search strategies
            confidence_threshold: Minimum confidence for results
            
        Returns:
            Advanced execution plan with dependencies and strategies
        """
        logger.info(f"Creating advanced execution plan for: {query[:100]}...")
        
        try:
            # Extract entities if needed
            if entities is None:
                entity_result = self.bio_ner.extract_and_link(query)
                entities = entity_result.get("entity_ids", {})
            
            # Analyze query and determine strategy
            strategy_analysis = self._analyze_query_strategy(query, entities)
            logger.info(f"Selected strategy: {strategy_analysis['strategy']} with confidence {strategy_analysis['confidence']}")
            
            # Create base plan
            plan = AdvancedExecutionPlan(
                original_query=query,
                strategy=QueryStrategy(strategy_analysis["strategy"]),
                confidence_score=strategy_analysis["confidence"],
                planning_reasoning=strategy_analysis["reasoning"]
            )
            
            # Generate subqueries using the selected strategy
            subqueries = self._generate_strategic_subqueries(
                query, entities, strategy_analysis, max_subqueries, enable_bidirectional
            )
            
            # Create subquery nodes
            for subquery_data in subqueries:
                # Validate search direction
                direction = subquery_data.get("direction", "forward")
                valid_directions = [d.value for d in SearchDirection]
                if direction not in valid_directions:
                    direction = "forward"  # Default fallback
                
                node = AdvancedSubqueryNode(
                    query=subquery_data["query"],
                    search_direction=SearchDirection(direction),
                    strategy=QueryStrategy(strategy_analysis["strategy"]),
                    priority=subquery_data.get("priority", 1),
                    estimated_cost=subquery_data.get("estimated_cost", 1.0),
                    confidence_threshold=confidence_threshold,
                    expected_entity_types=set(subquery_data.get("expected_entities", [])),
                    reasoning=subquery_data.get("reasoning", ""),
                    metadata=subquery_data.get("metadata", {}),
                    suggested_edge=subquery_data.get("suggested_edge")  # Add meta-KG edge if available
                )
                plan.subqueries.append(node)
            
            # Analyze dependencies between subqueries
            dependencies = self._analyze_subquery_dependencies(plan.subqueries, strategy_analysis)
            plan.dependencies = dependencies
            
            # Create execution phases and parallel groups
            self._organize_execution_phases(plan)
            
            # Generate contingency plans
            plan.contingency_plans = self._create_contingency_plans(plan)
            
            # Calculate estimates
            plan.total_estimated_cost = sum(node.estimated_cost for node in plan.subqueries)
            plan.expected_completion_time = self._estimate_completion_time(plan)
            
            # Cache the plan
            self._plan_cache[plan.id] = plan
            
            # Record planning history
            self._planning_history.append({
                "timestamp": time.time(),
                "query": query,
                "strategy": strategy_analysis["strategy"],
                "subquery_count": len(plan.subqueries),
                "plan_id": plan.id
            })
            
            logger.info(f"Created advanced plan with {len(plan.subqueries)} subqueries in {len(plan.execution_phases)} phases")
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create advanced plan: {e}")
            # Return simple fallback plan
            return self._create_fallback_plan(query, entities, str(e))
    
    def _setup_rdf_namespaces(self):
        """Setup RDF namespaces for knowledge graph context"""
        self.BIOLINK = Namespace("https://w3id.org/biolink/vocab/")
        self.EX = Namespace("http://example.org/entity/")
        self.GENE = Namespace("https://biolink.github.io/biolink-model/Gene/")
        self.DISEASE = Namespace("https://biolink.github.io/biolink-model/Disease/")
        self.PHYSPROCESS = Namespace("https://biolink.github.io/biolink-model/PhysiologicalProcess/")
        self.BIOENT = Namespace("https://biolink.github.io/biolink-model/BiologicalEntity/")
        self.PATHPROCESS = Namespace("https://biolink.github.io/biolink-model/PathologicalProcess/")
        self.SMALLMOL = Namespace("https://biolink.github.io/biolink-model/SmallMolecule/")
        self.PHENFEATURE = Namespace("https://biolink.github.io/biolink-model/PhenotypicFeature/")
        self.POLYPEPTIDE = Namespace("https://biolink.github.io/biolink-model/Polypeptide/")
        
        # Entity type mapping
        self.ENTITY_NAMESPACE_MAP = {
            "biolink:Gene": self.GENE,
            "biolink:Disease": self.DISEASE,
            "biolink:PhysiologicalProcess": self.PHYSPROCESS,
            "biolink:BiologicalEntity": self.BIOENT,
            "biolink:PathologicalProcess": self.PATHPROCESS,
            "biolink:SmallMolecule": self.SMALLMOL,
            "biolink:PhenotypicFeature": self.PHENFEATURE,
            "biolink:Polypeptide": self.POLYPEPTIDE,
        }
    
    def _initialize_meta_kg(self):
        """Initialize and parse the BTE meta-knowledge graph"""
        try:
            logger.info("Loading BTE meta-knowledge graph for edge analysis")
            meta_kg_data = self.bte_client.get_meta_knowledge_graph()
            
            # Parse edges into structured format
            for edge_data in meta_kg_data.get("edges", []):
                edge = MetaKGEdge(
                    subject=edge_data.get("subject", ""),
                    predicate=edge_data.get("predicate", ""),
                    object=edge_data.get("object", ""),
                    api_name=edge_data.get("api", {}).get("name", ""),
                    provided_by=edge_data.get("provided_by", "")
                )
                
                self._meta_kg_edges.append(edge)
                self._edges_by_subject[edge.subject].append(edge)
                self._edges_by_object[edge.object].append(edge)
            
            logger.info(f"Loaded {len(self._meta_kg_edges)} meta-KG edges for informed planning")
            
        except Exception as e:
            logger.error(f"Error loading meta-KG: {e}")
            # Continue with empty meta-KG - will fall back to basic approach
    
    def _get_available_edges_from_entities(self, entities: Dict[str, str]) -> List[MetaKGEdge]:
        """
        Get available edges that can extend from current entities
        
        Args:
            entities: Dictionary of entity names to IDs
            
        Returns:
            List of meta-KG edges that can be explored from current entities
        """
        # Determine entity types from entity IDs
        entity_types = set()
        for entity_id in entities.values():
            if ':' in entity_id:
                id_prefix = entity_id.split(':')[0]
                # Map common prefixes to biolink types
                type_mapping = {
                    'UMLS': ['biolink:Disease', 'biolink:SmallMolecule', 'biolink:PhysiologicalProcess'],
                    'CHEBI': ['biolink:SmallMolecule'],
                    'NCBIGene': ['biolink:Gene'],
                    'MONDO': ['biolink:Disease'],
                    'GO': ['biolink:PhysiologicalProcess'],
                    'UNII': ['biolink:SmallMolecule']
                }
                
                if id_prefix in type_mapping:
                    entity_types.update(type_mapping[id_prefix])
                else:
                    # Default to common types
                    entity_types.update(['biolink:Disease', 'biolink:SmallMolecule', 'biolink:Gene'])
        
        # If no types identified, use common starting types
        if not entity_types:
            entity_types = {'biolink:Disease', 'biolink:SmallMolecule', 'biolink:Gene'}
        
        # Get edges from these entity types
        available_edges = []
        for entity_type in entity_types:
            available_edges.extend(self._edges_by_subject.get(entity_type, []))
        
        return available_edges
    
    def _score_edges_for_query(self, edges: List[MetaKGEdge], query: str) -> List[Tuple[float, MetaKGEdge]]:
        """
        Score meta-KG edges based on relevance to the query
        
        Args:
            edges: List of meta-KG edges to score
            query: Original query for context
            
        Returns:
            List of tuples (score, edge) sorted by score descending
        """
        scored_edges = []
        query_lower = query.lower()
        
        for edge in edges:
            score = 0
            edge_pred_lower = edge.predicate.lower()
            edge_obj_lower = edge.object.lower()
            edge_subj_lower = edge.subject.lower()
            
            # High priority for treatment relationships
            if "treat" in query_lower and "treated_by" in edge_pred_lower:
                score += 15
            
            # Mechanism of action specific scoring
            if any(term in query_lower for term in ["mechanism", "activity", "pathway", "function"]):
                if "gene" in edge_obj_lower or "polypeptide" in edge_obj_lower:
                    score += 12  # Drug-protein interactions are key for mechanisms
                if "physiologicalprocess" in edge_obj_lower or "pathway" in edge_obj_lower:
                    score += 10  # Pathway involvement
                if "interacts_with" in edge_pred_lower or "affects" in edge_pred_lower:
                    score += 8   # Interaction predicates important for mechanisms
            
            # Antioxidant activity specific
            if "antioxidant" in query_lower:
                if "gene" in edge_obj_lower or "polypeptide" in edge_obj_lower:
                    score += 15  # Antioxidant proteins/genes are highly relevant
                if "physiologicalprocess" in edge_obj_lower:
                    score += 12  # Antioxidant processes
                if "associated_with" in edge_pred_lower or "participates_in" in edge_pred_lower:
                    score += 10
            
            # General scoring for key concepts
            if "gene" in query_lower and "gene" in edge_obj_lower:
                score += 8
            if "drug" in query_lower and "smallmolecule" in edge_obj_lower:
                score += 8
            if "disease" in query_lower and "disease" in edge_subj_lower:
                score += 8
            
            # Boost reliable, informative predicates
            high_value_predicates = ["interacts_with", "affects", "regulates", "participates_in", "has_phenotype"]
            if any(pred in edge_pred_lower for pred in high_value_predicates):
                score += 6
                
            reliable_predicates = ["treated_by", "associated_with", "related_to", "causes"]
            if any(pred in edge_pred_lower for pred in reliable_predicates):
                score += 4
                
            scored_edges.append((score, edge))
        
        # Sort by score descending
        scored_edges.sort(key=lambda x: x[0], reverse=True)
        return scored_edges
    def _analyze_query_strategy(self, query: str, entities: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze query to determine optimal decomposition strategy
        
        Args:
            query: Original query
            entities: Extracted entities
            
        Returns:
            Strategy analysis with reasoning
        """
        try:
            entity_list = list(entities.keys())[:15]
            entity_types = list(set([ent.split(':')[0] for ent in entities.values() if ':' in ent]))
            
            strategy_prompt = f"""You are an expert biomedical query strategist. Analyze this research question and determine the optimal decomposition strategy.

Query: "{query}"
Entities: {entity_list}
Entity types: {entity_types}

Choose the best strategy:
1. mechanistic - Follow causal chains (drug → target → pathway → disease)
2. functional - Explore entity functions (entity → function → related entities)
3. bidirectional - Search both forward and backward, find convergence points
4. exploratory - Broad search first, then narrow down to specifics
5. comparative - Compare multiple entities, pathways, or mechanisms

Respond with ONLY a valid JSON object:
{{
  "strategy": "mechanistic",
  "confidence": 0.8,
  "reasoning": "This strategy is optimal because...",
  "key_factors": ["treatment_focus", "drug_mechanism"],
  "expected_benefits": ["targeted_results", "systematic_approach"],
  "potential_challenges": ["data_availability", "complexity"]
}}"""
            
            response = self.llm.invoke([HumanMessage(content=strategy_prompt)])
            
            # Parse JSON response
            try:
                response_text = response.content.strip()
                
                # Clean up response - remove any markdown formatting
                if response_text.startswith('```json'):
                    response_text = response_text.replace('```json', '').replace('```', '')
                if response_text.startswith('```'):
                    response_text = response_text.replace('```', '')
                
                # Try to extract JSON from response
                response_text = response_text.strip()
                
                # Look for JSON object boundaries
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    strategy_data = json.loads(json_str)
                    
                    # Validate strategy
                    valid_strategies = [s.value for s in QueryStrategy]
                    if strategy_data.get("strategy") not in valid_strategies:
                        strategy_data["strategy"] = "mechanistic"  # Default fallback
                        
                    return strategy_data
                else:
                    raise json.JSONDecodeError("No JSON object found", response_text, 0)
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse strategy analysis JSON: {e}")
                return {
                    "strategy": "mechanistic",
                    "confidence": 0.6,
                    "reasoning": "Fallback to mechanistic strategy due to parsing error",
                    "key_factors": ["parsing_error"],
                    "expected_benefits": ["systematic_approach"],
                    "potential_challenges": ["limited_analysis"]
                }
                
        except Exception as e:
            logger.error(f"Strategy analysis failed: {e}")
            return {
                "strategy": "mechanistic", 
                "confidence": 0.3,
                "reasoning": f"Error in analysis: {e}",
                "key_factors": ["error"],
                "expected_benefits": [],
                "potential_challenges": ["analysis_failure"]
            }
    
    def _generate_strategic_subqueries(self, 
                                     query: str,
                                     entities: Dict[str, str],
                                     strategy_analysis: Dict[str, Any],
                                     max_subqueries: int,
                                     enable_bidirectional: bool) -> List[Dict[str, Any]]:
        """
        Generate subqueries using the selected strategy and meta-KG edges
        
        Args:
            query: Original query
            entities: Extracted entities
            strategy_analysis: Strategy analysis results
            max_subqueries: Maximum subqueries to generate
            enable_bidirectional: Whether to enable bidirectional searching
            
        Returns:
            List of subquery specifications
        """
        try:
            # Get available meta-KG edges for current entities
            available_edges = self._get_available_edges_from_entities(entities)
            scored_edges = self._score_edges_for_query(available_edges, query)
            
            # Take top edges for subquery generation
            top_edges = [edge for _, edge in scored_edges[:max_subqueries * 2]]  # Get more edges than needed
            
            logger.info(f"Found {len(available_edges)} available meta-KG edges, using top {len(top_edges)} for subquery generation")
            
            # Generate meta-KG informed subqueries
            meta_kg_subqueries = self._generate_metakg_informed_subqueries(
                query, entities, top_edges, strategy_analysis, max_subqueries
            )
            
            # If meta-KG approach fails or produces too few results, fall back to strategic approach
            if len(meta_kg_subqueries) < max_subqueries // 2:
                logger.info("Meta-KG approach produced insufficient results, supplementing with strategic approach")
                strategic_subqueries = self._generate_strategic_approach_subqueries(
                    query, entities, strategy_analysis, max_subqueries - len(meta_kg_subqueries)
                )
                meta_kg_subqueries.extend(strategic_subqueries)
            
            return meta_kg_subqueries[:max_subqueries]
                
        except Exception as e:
            logger.error(f"Strategic subquery generation failed: {e}")
            return self._generate_fallback_subqueries(query, entities)
    
    def _generate_metakg_informed_subqueries(self, 
                                           query: str,
                                           entities: Dict[str, str], 
                                           top_edges: List[MetaKGEdge],
                                           strategy_analysis: Dict[str, Any],
                                           max_subqueries: int) -> List[Dict[str, Any]]:
        """
        Generate subqueries informed by meta-KG edges for discrete relationships
        
        Args:
            query: Original query
            entities: Extracted entities
            top_edges: Top-scored meta-KG edges
            strategy_analysis: Strategy analysis results
            max_subqueries: Maximum subqueries to generate
            
        Returns:
            List of meta-KG informed subquery specifications
        """
        try:
            entity_list = list(entities.keys())[:10]
            edges_summary = []
            for i, edge in enumerate(top_edges[:8]):
                edges_summary.append(f"{i+1}. {edge.subject} --{edge.predicate}--> {edge.object}")
            
            metakg_prompt = f"""Create biomedical subqueries based on meta-KG relationship edges. Each subquery targets ONE discrete relationship.

Original query: "{query}"
Key entities: {entity_list}
Strategy: {strategy_analysis['strategy']}

Top meta-KG edges:
{chr(10).join(edges_summary[:6])}

Create {min(max_subqueries, 4)} subqueries. Each must:
- Target exactly ONE edge relationship (single-hop)
- Be answerable by biomedical knowledge graph
- Use natural language for the specific predicate
- Use only these direction values: "forward", "backward", or "convergent"

Examples:
- "What drugs treat diabetes?" (Disease -treated_by-> SmallMolecule)
- "What genes does metformin interact with?" (SmallMolecule -interacts_with-> Gene)

Respond with ONLY a JSON array:
[
  {{
    "query": "What diseases are treated by small molecules?",
    "edge_relationship": "Disease -treated_by-> SmallMolecule",
    "direction": "forward",
    "priority": 1,
    "estimated_cost": 0.8,
    "reasoning": "Treatment relationships are core to this query",
    "expected_entities": ["biolink:Disease", "biolink:SmallMolecule"],
    "metadata": {{"metakg_edge": true, "relationship_type": "treated_by"}}  
  }}
]"""
            response = self.llm.invoke([HumanMessage(content=metakg_prompt)])
            
            try:
                response_text = response.content.strip()
                
                # Clean response - remove markdown formatting
                if response_text.startswith('```json'):
                    response_text = response_text.replace('```json', '').replace('```', '')
                if response_text.startswith('```'):
                    response_text = response_text.replace('```', '')
                
                response_text = response_text.strip()
                
                # Look for JSON array boundaries
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    subqueries = json.loads(json_str)
                else:
                    # If no array found, look for single object and wrap in array
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = response_text[start_idx:end_idx]
                        single_query = json.loads(json_str)
                        subqueries = [single_query]
                    else:
                        raise json.JSONDecodeError("No JSON found", response_text, 0)
                
                if not isinstance(subqueries, list):
                    subqueries = [subqueries] if isinstance(subqueries, dict) else []
                
                validated_subqueries = []
                for i, sq in enumerate(subqueries[:max_subqueries]):
                    if isinstance(sq, dict) and "query" in sq:
                        # Set defaults and associate with meta-KG edge
                        sq.setdefault("direction", "forward")
                        sq.setdefault("priority", i + 1)
                        sq.setdefault("estimated_cost", 0.8)
                        sq.setdefault("reasoning", "Meta-KG informed subquery")
                        sq.setdefault("expected_entities", [])
                        sq.setdefault("metadata", {"metakg_edge": True})
                        
                        # Associate with corresponding edge if available
                        if i < len(top_edges):
                            sq["suggested_edge"] = top_edges[i]
                            sq["metadata"]["edge_subject"] = top_edges[i].subject
                            sq["metadata"]["edge_predicate"] = top_edges[i].predicate
                            sq["metadata"]["edge_object"] = top_edges[i].object
                        
                        validated_subqueries.append(sq)
                
                logger.info(f"Generated {len(validated_subqueries)} meta-KG informed subqueries")
                return validated_subqueries
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse meta-KG subquery JSON: {e}")
                logger.debug(f"Response content: {response.content[:200]}...")
                return []
                
        except Exception as e:
            logger.error(f"Meta-KG subquery generation failed: {e}")
            return []
    
    def _generate_strategic_approach_subqueries(self,
                                               query: str,
                                               entities: Dict[str, str],
                                               strategy_analysis: Dict[str, Any],
                                               max_subqueries: int) -> List[Dict[str, Any]]:
        """
        Generate subqueries using strategic approach (fallback from meta-KG)
        
        Args:
            query: Original query
            entities: Extracted entities
            strategy_analysis: Strategy analysis results
            max_subqueries: Maximum subqueries to generate
            
        Returns:
            List of strategic subquery specifications
        """
        try:
            strategy = strategy_analysis["strategy"]
            entity_list = list(entities.keys())[:12]
            
            # Strategy-specific prompts
            strategy_prompts = {
                "mechanistic": self._get_mechanistic_prompt(),
                "functional": self._get_functional_prompt(),
                "bidirectional": self._get_bidirectional_prompt(),
                "exploratory": self._get_exploratory_prompt(),
                "comparative": self._get_comparative_prompt()
            }
            
            base_prompt = strategy_prompts.get(strategy, strategy_prompts["mechanistic"])
            
            full_prompt = f"""{base_prompt}

Original query: "{query}"
Key entities: {entity_list}
Strategy reasoning: {strategy_analysis['reasoning']}

Generate {min(max_subqueries, 4)} strategic subqueries.

Requirements:
- Each subquery answerable by single biomedical knowledge graph query
- Build logical progression from general to specific
- Single-hop relationships only

Respond with ONLY a JSON array:
[
  {{
    "query": "What drugs treat macular degeneration?",
    "direction": "forward",
    "priority": 1,
    "estimated_cost": 1.0,
    "reasoning": "Direct treatment relationship query",
    "expected_entities": ["biolink:Disease", "biolink:SmallMolecule"],
    "metadata": {{"phase": "exploration"}}  
  }}
]"""
            
            response = self.llm.invoke([HumanMessage(content=full_prompt)])
            
            try:
                response_text = response.content.strip()
                
                # Clean response - remove markdown formatting
                if response_text.startswith('```json'):
                    response_text = response_text.replace('```json', '').replace('```', '')
                if response_text.startswith('```'):
                    response_text = response_text.replace('```', '')
                
                response_text = response_text.strip()
                
                # Look for JSON array boundaries
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    subqueries = json.loads(json_str)
                else:
                    # If no array found, look for single object and wrap in array
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = response_text[start_idx:end_idx]
                        single_query = json.loads(json_str)
                        subqueries = [single_query]
                    else:
                        raise json.JSONDecodeError("No JSON found", response_text, 0)
                
                if not isinstance(subqueries, list):
                    subqueries = [subqueries] if isinstance(subqueries, dict) else []
                
                # Validate and clean subqueries
                validated_subqueries = []
                for sq in subqueries[:max_subqueries]:
                    if isinstance(sq, dict) and "query" in sq:
                        # Set defaults for missing fields
                        sq.setdefault("direction", "forward")
                        sq.setdefault("priority", 5)
                        sq.setdefault("estimated_cost", 1.0)
                        sq.setdefault("reasoning", "Strategic subquery")
                        sq.setdefault("expected_entities", [])
                        sq.setdefault("metadata", {"strategic": True})
                        
                        validated_subqueries.append(sq)
                
                logger.info(f"Generated {len(validated_subqueries)} strategic subqueries")
                return validated_subqueries
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse strategic subquery JSON: {e}")
                logger.debug(f"Response content: {response.content[:200]}...")
                return []
                
        except Exception as e:
            logger.error(f"Strategic approach subquery generation failed: {e}")
            return []
    
    def _get_mechanistic_prompt(self) -> str:
        """Get prompt for mechanistic strategy"""
        return """
        Use a MECHANISTIC strategy to decompose this biomedical query.
        
        Follow causal chains and biological mechanisms:
        - Start with direct relationships (drug-target, gene-disease)
        - Progress through intermediate mechanisms (pathways, processes)
        - End with ultimate outcomes or phenotypes
        - Consider both molecular and systems-level mechanisms
        
        Example flow: Drug → Protein Target → Biological Process → Pathway → Disease
        """
    
    def _get_functional_prompt(self) -> str:
        """Get prompt for functional strategy"""
        return """
        Use a FUNCTIONAL strategy to decompose this biomedical query.
        
        Focus on biological functions and their relationships:
        - Identify core entities and their primary functions
        - Explore functional relationships between entities
        - Investigate functional consequences and effects
        - Consider both normal and pathological functions
        
        Example flow: Entity → Primary Function → Functional Partners → Functional Outcomes
        """
    
    def _get_bidirectional_prompt(self) -> str:
        """Get prompt for bidirectional strategy"""
        return """
        Use a BIDIRECTIONAL strategy to decompose this biomedical query.
        
        Search in multiple directions to find convergence points:
        - FORWARD: Start from causes, progress to effects
        - BACKWARD: Start from effects, trace back to causes  
        - CONVERGENT: Find where forward and backward searches meet
        - Include queries that explore both directions
        
        This is especially useful for finding indirect relationships and alternative pathways.
        """
    
    def _get_exploratory_prompt(self) -> str:
        """Get prompt for exploratory strategy"""
        return """
        Use an EXPLORATORY strategy to decompose this biomedical query.
        
        Cast a wide net initially, then narrow down:
        - Begin with broad exploratory queries
        - Use initial results to guide more specific investigations
        - Progress from general associations to specific mechanisms
        - Include discovery-oriented queries
        
        Example flow: Broad Associations → Specific Relationships → Mechanistic Details
        """
    
    def _get_comparative_prompt(self) -> str:
        """Get prompt for comparative strategy"""
        return """
        Use a COMPARATIVE strategy to decompose this biomedical query.
        
        Focus on comparing multiple entities, pathways, or mechanisms:
        - Identify entities/mechanisms to compare
        - Generate parallel queries for each comparison target
        - Include queries that directly compare results
        - Consider comparative effectiveness, mechanisms, or outcomes
        
        Example flow: Query A for Entity 1 → Query B for Entity 2 → Comparative Analysis
        """
    
    def _analyze_subquery_dependencies(self, 
                                     subqueries: List[AdvancedSubqueryNode],
                                     strategy_analysis: Dict[str, Any]) -> List[DependencyRelation]:
        """
        Analyze dependencies between subqueries using LLM reasoning
        
        Args:
            subqueries: List of subquery nodes
            strategy_analysis: Strategy analysis results
            
        Returns:
            List of dependency relationships
        """
        try:
            if len(subqueries) <= 1:
                return []
            
            # Prepare subquery data for analysis
            subquery_data = []
            for i, sq in enumerate(subqueries):
                subquery_data.append({
                    "id": sq.id,
                    "index": i,
                    "query": sq.query,
                    "direction": sq.search_direction.value,
                    "priority": sq.priority,
                    "reasoning": sq.reasoning
                })
            
            dependency_prompt = f"""
            You are an expert in biomedical research workflow design. Analyze these subqueries 
            to identify dependencies and execution order requirements.
            
            Strategy: {strategy_analysis['strategy']}
            Strategy reasoning: {strategy_analysis['reasoning']}
            
            Subqueries:
            {json.dumps(subquery_data, indent=2)}
            
            For each subquery, determine:
            1. Which other subqueries (if any) it depends on
            2. What type of dependency exists
            3. How confident you are in this dependency
            4. Why this dependency exists
            
            Dependency types:
            - SEQUENTIAL: Must run after prerequisite completes
            - PARALLEL: Can run simultaneously with others
            - CONDITIONAL: Only run if prerequisite meets certain criteria
            - CONVERGENT: Combines results from multiple prerequisites
            
            Respond with a JSON array of dependencies:
            [
                {{
                    "dependent_id": "uuid_of_dependent_subquery",
                    "prerequisite_id": "uuid_of_prerequisite_subquery", 
                    "dependency_type": "sequential|parallel|conditional|convergent",
                    "confidence": 0.0-1.0,
                    "reasoning": "explanation of why this dependency exists"
                }}
            ]
            
            Only include dependencies where you are >0.6 confident.
            """
            
            response = self.llm.invoke([HumanMessage(content=dependency_prompt)])
            
            try:
                dependency_data = json.loads(response.content.strip())
                
                dependencies = []
                for dep_data in dependency_data:
                    if isinstance(dep_data, dict) and dep_data.get("confidence", 0) > 0.6:
                        try:
                            dep = DependencyRelation(
                                dependent_id=dep_data["dependent_id"],
                                prerequisite_id=dep_data["prerequisite_id"],
                                dependency_type=DependencyType(dep_data["dependency_type"]),
                                confidence=dep_data.get("confidence", 0.8),
                                reasoning=dep_data.get("reasoning", "")
                            )
                            dependencies.append(dep)
                        except (KeyError, ValueError) as e:
                            logger.warning(f"Invalid dependency data: {dep_data}, error: {e}")
                
                logger.info(f"Identified {len(dependencies)} subquery dependencies")
                return dependencies
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse dependency JSON")
                return self._infer_simple_dependencies(subqueries)
                
        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            return self._infer_simple_dependencies(subqueries)
    
    def _organize_execution_phases(self, plan: AdvancedExecutionPlan):
        """
        Organize subqueries into execution phases based on dependencies
        
        Args:
            plan: Execution plan to organize
        """
        # Create dependency graph
        prereq_map = {}  # dependent_id -> set of prerequisite_ids
        for dep in plan.dependencies:
            if dep.dependent_id not in prereq_map:
                prereq_map[dep.dependent_id] = set()
            prereq_map[dep.dependent_id].add(dep.prerequisite_id)
        
        # Organize into phases using topological sort
        remaining_subqueries = {sq.id: sq for sq in plan.subqueries}
        phases = []
        
        while remaining_subqueries:
            # Find subqueries with no unmet dependencies
            ready_subqueries = []
            for sq_id, sq in remaining_subqueries.items():
                unmet_deps = prereq_map.get(sq_id, set()) & set(remaining_subqueries.keys())
                if not unmet_deps:
                    ready_subqueries.append(sq_id)
            
            if not ready_subqueries:
                # Break cycles by picking highest priority remaining subquery
                highest_priority_id = max(remaining_subqueries.keys(), 
                                        key=lambda x: remaining_subqueries[x].priority)
                ready_subqueries = [highest_priority_id]
                logger.warning("Breaking dependency cycle, proceeding with highest priority subquery")
            
            # Add ready subqueries to current phase
            phases.append(ready_subqueries)
            
            # Remove completed subqueries
            for sq_id in ready_subqueries:
                del remaining_subqueries[sq_id]
        
        plan.execution_phases = phases
        
        # Identify parallel groups within each phase
        parallel_groups = []
        for phase in phases:
            # Group subqueries that can run in parallel
            parallel_groups.append(phase)  # For now, all subqueries in a phase can run in parallel
        
        plan.parallel_groups = parallel_groups
    
    def _create_contingency_plans(self, plan: AdvancedExecutionPlan) -> Dict[str, List[str]]:
        """
        Create contingency plans for subquery failures
        
        Args:
            plan: Execution plan
            
        Returns:
            Dictionary mapping subquery IDs to alternative approaches
        """
        contingencies = {}
        
        try:
            for sq in plan.subqueries:
                # Create alternative queries for critical subqueries
                if sq.priority <= 3:  # High priority subqueries get contingencies
                    alt_prompt = f"""
                    Create 2-3 alternative approaches for this biomedical subquery in case the original fails:
                    
                    Original: "{sq.query}"
                    Reasoning: {sq.reasoning}
                    Strategy: {plan.strategy.value}
                    
                    Generate alternatives that:
                    - Ask similar questions using different entity relationships
                    - Use broader or narrower scope as appropriate
                    - Employ different biomedical perspectives
                    
                    Return as JSON array of alternative query strings:
                    ["alternative query 1", "alternative query 2", "alternative query 3"]
                    """
                    
                    response = self.llm.invoke([HumanMessage(content=alt_prompt)])
                    
                    try:
                        alternatives = json.loads(response.content.strip())
                        if isinstance(alternatives, list):
                            contingencies[sq.id] = alternatives[:3]  # Limit to 3 alternatives
                    except json.JSONDecodeError:
                        # Simple fallback alternatives
                        contingencies[sq.id] = [
                            sq.query.replace("What", "Which"),
                            sq.query.replace("does", "can"),
                            f"What are the main relationships involving {sq.query.split()[-1]}?"
                        ]
        
        except Exception as e:
            logger.warning(f"Contingency planning failed: {e}")
        
        return contingencies
    
    def _estimate_completion_time(self, plan: AdvancedExecutionPlan) -> float:
        """
        Estimate total completion time for the execution plan
        
        Args:
            plan: Execution plan
            
        Returns:
            Estimated completion time in seconds
        """
        # Base time per query (API call + processing)
        base_time_per_query = 3.0  # seconds
        
        # Calculate time per phase (accounting for parallelization)
        total_time = 0.0
        for phase in plan.execution_phases:
            # Time for this phase is the maximum time of any subquery in the phase
            phase_costs = [sq.estimated_cost for sq in plan.subqueries if sq.id in phase]
            max_phase_cost = max(phase_costs) if phase_costs else 1.0
            phase_time = max_phase_cost * base_time_per_query
            total_time += phase_time
        
        return total_time
    
    def _generate_fallback_subqueries(self, query: str, entities: Dict[str, str]) -> List[Dict[str, Any]]:
        """Generate simple fallback subqueries"""
        entity_list = list(entities.keys())[:5]
        
        fallback_subqueries = [
            {
                "query": query,
                "direction": "forward",
                "priority": 1,
                "estimated_cost": 1.0,
                "reasoning": "Original query as fallback",
                "expected_entities": [],
                "metadata": {"fallback": True}
            }
        ]
        
        # Add simple entity-based queries
        for i, entity in enumerate(entity_list[:3]):
            fallback_subqueries.append({
                "query": f"What is related to {entity}?",
                "direction": "forward",
                "priority": i + 2,
                "estimated_cost": 0.8,
                "reasoning": f"Explore relationships for {entity}",
                "expected_entities": ["biolink:NamedThing"],
                "metadata": {"fallback": True, "entity_focus": entity}
            })
        
        return fallback_subqueries
    
    def _infer_simple_dependencies(self, subqueries: List[AdvancedSubqueryNode]) -> List[DependencyRelation]:
        """Create simple sequential dependencies as fallback"""
        dependencies = []
        
        for i in range(1, len(subqueries)):
            dep = DependencyRelation(
                dependent_id=subqueries[i].id,
                prerequisite_id=subqueries[i-1].id,
                dependency_type=DependencyType.SEQUENTIAL,
                confidence=0.7,
                reasoning="Simple sequential dependency (fallback)"
            )
            dependencies.append(dep)
        
        return dependencies
    
    def _create_fallback_plan(self, query: str, entities: Dict[str, str], error: str) -> AdvancedExecutionPlan:
        """Create simple fallback plan in case of errors"""
        plan = AdvancedExecutionPlan(
            original_query=query,
            strategy=QueryStrategy.MECHANISTIC,
            confidence_score=0.3,
            planning_reasoning=f"Fallback plan due to error: {error}"
        )
        
        # Add single subquery
        node = AdvancedSubqueryNode(
            query=query,
            search_direction=SearchDirection.FORWARD,
            strategy=QueryStrategy.MECHANISTIC,
            priority=1,
            estimated_cost=1.0,
            reasoning="Fallback to original query"
        )
        plan.subqueries = [node]
        plan.execution_phases = [[node.id]]
        plan.parallel_groups = [[node.id]]
        plan.total_estimated_cost = 1.0
        plan.expected_completion_time = 3.0
        
        return plan
    
    def get_plan_summary(self, plan: AdvancedExecutionPlan) -> Dict[str, Any]:
        """
        Get a comprehensive summary of an execution plan
        
        Args:
            plan: Execution plan to summarize
            
        Returns:
            Detailed plan summary
        """
        return {
            "plan_id": plan.id,
            "original_query": plan.original_query,
            "strategy": plan.strategy.value,
            "confidence_score": plan.confidence_score,
            "planning_reasoning": plan.planning_reasoning,
            
            # Subquery information
            "total_subqueries": len(plan.subqueries),
            "execution_phases": len(plan.execution_phases),
            "parallel_opportunities": sum(len(group) for group in plan.parallel_groups),
            
            # Dependencies
            "total_dependencies": len(plan.dependencies),
            "dependency_types": {
                dtype.value: len([d for d in plan.dependencies if d.dependency_type == dtype])
                for dtype in DependencyType
            },
            
            # Estimates
            "estimated_cost": plan.total_estimated_cost,
            "estimated_time": plan.expected_completion_time,
            
            # Contingencies
            "contingency_plans": len(plan.contingency_plans),
            
            # Detailed subqueries
            "subqueries": [
                {
                    "id": sq.id,
                    "query": sq.query,
                    "direction": sq.search_direction.value,
                    "priority": sq.priority,
                    "estimated_cost": sq.estimated_cost,
                    "reasoning": sq.reasoning,
                    "expected_entities": list(sq.expected_entity_types),
                    "metadata": sq.metadata
                }
                for sq in plan.subqueries
            ],
            
            # Dependencies detail
            "dependencies": [
                {
                    "dependent_id": dep.dependent_id,
                    "prerequisite_id": dep.prerequisite_id,
                    "type": dep.dependency_type.value,
                    "confidence": dep.confidence,
                    "reasoning": dep.reasoning
                }
                for dep in plan.dependencies
            ],
            
            # Execution organization
            "execution_phases": [
                {"phase": i, "subquery_ids": phase}
                for i, phase in enumerate(plan.execution_phases)
            ]
        }


# Convenience functions
def create_advanced_plan(query: str, 
                        entities: Dict[str, str] = None,
                        openai_api_key: Optional[str] = None,
                        **kwargs) -> AdvancedExecutionPlan:
    """
    Convenience function to create an advanced execution plan
    
    Args:
        query: Natural language query
        entities: Optional pre-extracted entities
        openai_api_key: Optional OpenAI API key
        **kwargs: Additional parameters for plan creation
        
    Returns:
        Advanced execution plan
    """
    planner = LLMBasedQueryPlanner(openai_api_key)
    return planner.create_advanced_plan(query, entities, **kwargs)


def analyze_plan_complexity(plan: AdvancedExecutionPlan) -> Dict[str, Any]:
    """
    Analyze the complexity of an execution plan
    
    Args:
        plan: Execution plan to analyze
        
    Returns:
        Complexity analysis
    """
    return {
        "subquery_count": len(plan.subqueries),
        "dependency_count": len(plan.dependencies),
        "max_phase_parallelism": max(len(phase) for phase in plan.execution_phases) if plan.execution_phases else 0,
        "strategy_complexity": {
            "mechanistic": 1.0,
            "functional": 1.2,
            "bidirectional": 1.8,
            "exploratory": 1.5,
            "comparative": 1.6
        }.get(plan.strategy.value, 1.0),
        "estimated_complexity_score": plan.total_estimated_cost * len(plan.execution_phases)
    }