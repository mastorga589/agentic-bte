"""
LangGraph-inspired Context-Driven Adaptive Query Optimizer

This module implements the true LangGraph approach for adaptive biomedical query 
optimization, based on the actual BTE-LLM prototype implementation. Unlike the 
previous strategy-based approach, this uses pure contextual reasoning and 
mechanistic decomposition.

Key features:
- Context-driven subquery generation (no predefined strategies)  
- RDF/Turtle-based knowledge accumulation
- Mechanistic reasoning chains
- Single-hop query focus  
- Iterative refinement based on intermediate results
- Pure LLM-based planning decisions
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from uuid import uuid4
from copy import deepcopy

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from rdflib import Graph, URIRef, Namespace
from rdflib.namespace import RDF, RDFS

from ..entities.bio_ner import BioNERTool
from ..knowledge.trapi import TRAPIQueryBuilder
from ..knowledge.bte_client import BTEClient
from ...config.settings import get_settings
from ...exceptions.base import ExternalServiceError

logger = logging.getLogger(__name__)


@dataclass
class ContextualSubquery:
    """Context-driven subquery without predefined strategies"""
    id: str = field(default_factory=lambda: str(uuid4()))
    query: str = ""
    reasoning_context: str = ""  # Why this query was generated
    results: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = False
    execution_time: float = 0.0
    iteration_number: int = 0
    
    
@dataclass 
class AdaptivePlan:
    """Context-driven execution plan using RDF knowledge accumulation"""
    id: str = field(default_factory=lambda: str(uuid4()))
    original_query: str = ""
    knowledge_graph: Graph = field(default_factory=Graph)  # RDF graph for context
    executed_subqueries: List[ContextualSubquery] = field(default_factory=list)
    entity_data: Dict[str, str] = field(default_factory=dict)
    accumulated_results: List[Dict[str, Any]] = field(default_factory=list)
    current_iteration: int = 0
    max_iterations: int = 10
    total_execution_time: float = 0.0
    final_answer: str = ""
    completion_reason: str = ""  # Why the plan finished


class LangGraphAdaptiveOptimizer:
    """
    LangGraph-inspired adaptive query optimizer using context-driven planning
    
    This optimizer creates adaptive execution plans that use contextual reasoning
    and mechanistic decomposition, without relying on predefined strategies. 
    It follows the actual LangGraph planner node implementation.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the LangGraph-style adaptive optimizer
        
        Args:
            openai_api_key: OpenAI API key for LLM operations
        """
        self.settings = get_settings()
        self.openai_api_key = openai_api_key or self.settings.openai_api_key
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for adaptive query optimization")
        
        # Initialize components
        self.llm = ChatOpenAI(
            temperature=0,
            model=self.settings.openai_model,
            api_key=self.openai_api_key
        )
        self.bio_ner = BioNERTool(openai_api_key)
        self.trapi_builder = TRAPIQueryBuilder(openai_api_key)
        self.bte_client = BTEClient()
        
        # RDF namespace setup (same as LangGraph implementation)
        self._setup_rdf_namespaces()
    
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

    def _update_knowledge_graph(self, results: List[Dict[str, Any]], graph: Graph) -> None:
        """
        Update RDF knowledge graph with new results (same as LangGraph RDFgraphUpdater)
        
        Args:
            results: New relationship results to add
            graph: RDF graph to update
        """
        # Bind namespaces
        graph.bind("biolink", self.BIOLINK)
        graph.bind("gene", self.GENE)
        graph.bind("disease", self.DISEASE)
        graph.bind("physprocess", self.PHYSPROCESS)
        graph.bind("bioent", self.BIOENT)
        graph.bind("pathprocess", self.PATHPROCESS)
        graph.bind("smallmol", self.SMALLMOL)
        graph.bind("polypeptide", self.POLYPEPTIDE)
        graph.bind("phenfeature", self.PHENFEATURE)
        
        def make_entity_uri(name: str, entity_type: str):
            ns = self.ENTITY_NAMESPACE_MAP.get(entity_type, self.EX)
            return ns[name.replace(" ", "_").replace(":", "-").lower()]
        
        # Add triples to graph
        for result in results:
            if all(key in result for key in ['subject', 'predicate', 'object']):
                subj = make_entity_uri(result['subject'], result.get("subject_type", ""))
                pred = URIRef(self.BIOLINK + result['predicate'].split(":")[1])
                obj = make_entity_uri(result['object'], result.get("object_type", ""))
                graph.add((subj, pred, obj))

    def _generate_contextual_subquery(self, 
                                    original_query: str,
                                    current_graph: Graph,
                                    entity_data: Dict[str, str],
                                    iteration: int) -> Optional[str]:
        """
        Generate next subquery based on current knowledge state (true LangGraph approach)
        
        Args:
            original_query: The original user query
            current_graph: Current RDF knowledge graph
            entity_data: Available entity mappings
            iteration: Current iteration number
            
        Returns:
            Next single-hop subquery or None if complete
        """
        # LangGraph planner prompt (adapted from actual implementation)
        planner_prompt = f"""Your role is to create a plan to answer the main query by dividing it into several simpler single hop subqueries. 
        The system has access to a knowledge graph where necessary data will be retrieved from to answer the query.

        This will be an iterative process where answers to previous subqueries might be relevant in constructing future queries. 
        Initial subqueries should encompass the overall query path; each subsequent subquery should be more refined. A mechanistic approach should be used in developing the query plan.
        For example, for "Which drugs can treat Crohn's disease by targeting the inflammatory response?", 
        the first subquery might be "Which drugs can treat Crohn's disease?" followed by "Which genes do these drugs target?", 
        then "Which of these genes are related to the inflammatory response?".
        
        Here are the node types present in the knowledge graph: Disease, PhysiologicalProcess, BiologicalEntity, Gene, PathologicalProcess, Polypeptide, SmallMolecule, PhenotypicFeature

        The predicates in the knowledge graph can be grouped into causality, treatment, genetics & biomarkers, interactions, phenotypes & diagnostics, responses & effects, associations, and structural/hierarchical relationships.
        Restrict your queries to within these relationships.

        Here are the current results so far expressed in Turtle:
        {current_graph.serialize(format="turtle")}

        Make sure that each subquery interrogates discrete relationships between node types. 
        For example, instead of directly asking "What are the mechanisms of action of these drugs?", 
        you must create subqueries that can help form a reasoning chain to answer the question ("What genes do these drugs interact with?" and "Which physiological processes are these genes involved in?)

        If a subquery results in "No results found", rephrase/reframe the question or explore the question from a different angle.

        Do NOT prescribe which nodes to use in your subquery, the nodes and predicates are only for your reference.
        You need to determine what the next single-hop subquery is and formulate it into a natural language subquestion. 

        Original query: {original_query}
        Current iteration: {iteration}
        
        If you believe enough information has been gathered to answer the original query, respond with "COMPLETE".
        Otherwise, provide ONLY the single-hop natural language subquestion (for example, "What genes does doxorubicin target?"). 
        Please DO NOT include your thoughts or anything else as it will interfere with downstream processes.
        """
        
        try:
            response = self.llm.invoke([{"role": "system", "content": planner_prompt}])
            subquery = response.content.strip()
            
            if subquery == "COMPLETE" or iteration >= 10:  # Max iterations check
                return None
            
            logger.info(f"Generated contextual subquery: {subquery}")
            return subquery
            
        except Exception as e:
            logger.error(f"Error generating contextual subquery: {e}")
            return None

    def _execute_subquery_with_bte(self, subquery: str, entity_data: Dict[str, str]) -> Tuple[List[Dict[str, Any]], float]:
        """
        Execute a single subquery using BTE API
        
        Args:
            subquery: The subquery to execute
            entity_data: Available entity mappings
            
        Returns:
            Tuple of (results, execution_time)
        """
        start_time = time.time()
        
        try:
            # Build TRAPI query
            trapi_query = self.trapi_builder.build_trapi_query(
                query=subquery,
                entity_data=entity_data
            )
            
            if not trapi_query:
                logger.warning(f"Failed to build TRAPI query for: {subquery}")
                return [], time.time() - start_time
            
            # Execute with BTE
            bte_response = self.bte_client.execute_trapi_query(trapi_query)
            
            # Parse results
            results, _ = self.bte_client.parse_bte_results(bte_response, k=5, max_results=50)
            
            logger.info(f"Subquery executed successfully: {len(results)} results found")
            return results, time.time() - start_time
            
        except Exception as e:
            logger.error(f"Error executing subquery '{subquery}': {e}")
            return [], time.time() - start_time

    def _synthesize_final_answer(self, 
                                original_query: str,
                                knowledge_graph: Graph,
                                entity_data: Dict[str, str],
                                executed_subqueries: List[ContextualSubquery]) -> str:
        """
        Synthesize final answer from accumulated knowledge (same as LangGraph summary_node)
        
        Args:
            original_query: Original user query
            knowledge_graph: Accumulated RDF knowledge graph  
            entity_data: Entity mappings
            executed_subqueries: All executed subqueries
            
        Returns:
            Comprehensive final answer
        """
        summary_prompt = f"""You are an expert proficient in the pharmaceutical sciences, medicinal chemistry and biomedical research.
        Your team has access to a biomedical knowledge graph, and have provided you with the following results based on the user's query.
        However, the knowledge graph might not be perfect and have gaps in its data, and some relationships between the entities might be implicit.
        Your job is to answer the user's query using your team's results and your own biomedical expertise.
        Your summary of the results MUST be comprehensive while still avoiding redundancy.
        Expound on the logical steps you took to form your final answer.

        Make sure to organize your answer around the user query:
        {original_query}

        Here are the findings of your team expressed in Turtle:
        {knowledge_graph.serialize(format="turtle")}

        Here are the entities each ID corresponds to:
        {entity_data}

        You MUST base your answer on the evidence/context included in the prompt; 
        however, you can use your expertise to contextualize the results. 
        For example, if the results only show target genes for dirithromycin, knowing that dirithromycin and erythromycin are part of the same drug class, you can infer that they are likely to share similar properties and targets.
        
        Maintain complete transparency about your problem-solving process; the relationships between each entity should be clear in your answer.
        Do NOT list down all entities; rather, choose to most important ones to illustrate your point (for example, "the BRCA family of proteins is involved in breast cancer").
        Remember, prioritize accuracy, explainability, and user understanding.
        Only include relevant results in the final answer.
        """
        
        try:
            response = self.llm.invoke([{"role": "system", "content": summary_prompt}])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error synthesizing final answer: {e}")
            return f"Error generating final answer. Executed {len(executed_subqueries)} subqueries with mixed results."

    def create_adaptive_plan(self, 
                           query: str, 
                           entities: Dict[str, str] = None,
                           max_iterations: int = 10) -> AdaptivePlan:
        """
        Create initial adaptive plan using pure contextual reasoning
        
        Args:
            query: Original biomedical query
            entities: Optional pre-extracted entities
            max_iterations: Maximum number of iterations
            
        Returns:
            Initial adaptive plan
        """
        logger.info(f"Creating context-driven adaptive plan for query: {query[:100]}...")
        
        # Extract entities if needed
        if entities is None:
            entity_result = self.bio_ner.extract_and_link(query)
            entities = entity_result.get("entity_ids", {})
        
        # Create plan with RDF knowledge graph
        plan = AdaptivePlan(
            original_query=query,
            entity_data=entities,
            max_iterations=max_iterations
        )
        
        logger.info(f"Created context-driven adaptive plan")
        return plan

    def execute_adaptive_plan(self, plan: AdaptivePlan) -> AdaptivePlan:
        """
        Execute adaptive plan using contextual reasoning (LangGraph style)
        
        Args:
            plan: The adaptive plan to execute
            
        Returns:
            Updated plan with results
        """
        logger.info(f"Executing context-driven adaptive plan for query: {plan.original_query[:100]}...")
        start_time = time.time()
        
        for iteration in range(plan.max_iterations):
            plan.current_iteration = iteration + 1
            logger.info(f"Adaptive iteration {plan.current_iteration}")
            
            # Generate next contextual subquery
            subquery = self._generate_contextual_subquery(
                original_query=plan.original_query,
                current_graph=plan.knowledge_graph,
                entity_data=plan.entity_data,
                iteration=iteration
            )
            
            if not subquery:
                plan.completion_reason = "Plan determined complete by contextual reasoning"
                break
                
            logger.info(f"Executing subquery: {subquery}")
            
            # Execute subquery
            results, exec_time = self._execute_subquery_with_bte(subquery, plan.entity_data)
            
            # Create subquery record
            contextual_subquery = ContextualSubquery(
                query=subquery,
                reasoning_context=f"Generated at iteration {iteration + 1} based on current knowledge state",
                results=results,
                success=len(results) > 0,
                execution_time=exec_time,
                iteration_number=iteration + 1
            )
            
            plan.executed_subqueries.append(contextual_subquery)
            plan.accumulated_results.extend(results)
            
            # Update knowledge graph with new results
            if results:
                self._update_knowledge_graph(results, plan.knowledge_graph)
                logger.info(f"Updated knowledge graph with {len(results)} new relationships")
            else:
                logger.warning(f"No results found for subquery: {subquery}")
        
        # Generate final answer
        plan.final_answer = self._synthesize_final_answer(
            original_query=plan.original_query,
            knowledge_graph=plan.knowledge_graph,
            entity_data=plan.entity_data,
            executed_subqueries=plan.executed_subqueries
        )
        
        plan.total_execution_time = time.time() - start_time
        logger.info(f"Context-driven adaptive execution completed in {plan.total_execution_time:.2f}s with {len(plan.accumulated_results)} total results")
        
        return plan

    def compare_with_static(self, query: str) -> Dict[str, Any]:
        """
        Compare context-driven adaptive approach with static approach
        
        Args:
            query: Query to test both approaches on
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing context-driven vs static approaches for: {query[:100]}...")
        
        # Time adaptive approach
        start_time = time.time()
        adaptive_plan = self.create_adaptive_plan(query)
        adaptive_plan = self.execute_adaptive_plan(adaptive_plan)
        adaptive_time = time.time() - start_time
        
        # Time static approach (using existing knowledge system)
        from ..knowledge.knowledge_system import KnowledgeSystem
        static_system = KnowledgeSystem(self.openai_api_key)
        
        start_time = time.time()
        static_results = static_system.process_biomedical_query(query)
        static_time = time.time() - start_time
        
        comparison = {
            "query": query,
            "adaptive_approach": {
                "execution_time": adaptive_time,
                "subqueries_executed": len(adaptive_plan.executed_subqueries),
                "total_results": len(adaptive_plan.accumulated_results),
                "success": len(adaptive_plan.accumulated_results) > 0,
                "final_answer_length": len(adaptive_plan.final_answer),
                "completion_reason": adaptive_plan.completion_reason
            },
            "static_approach": {
                "execution_time": static_time,
                "total_results": len(static_results.get("relationships", [])),
                "success": len(static_results.get("relationships", [])) > 0,
                "query_type": static_results.get("query_info", {}).get("type", "unknown"),
                "classification_confidence": static_results.get("query_info", {}).get("confidence", 0)
            },
            "comparison_metrics": {
                "adaptive_more_results": len(adaptive_plan.accumulated_results) > len(static_results.get("relationships", [])),
                "adaptive_faster": adaptive_time < static_time,
                "adaptive_more_comprehensive": len(adaptive_plan.executed_subqueries) > 1,
                "result_difference": len(adaptive_plan.accumulated_results) - len(static_results.get("relationships", [])),
                "time_difference": adaptive_time - static_time
            }
        }
        
        logger.info(f"Comparison completed - Adaptive: {len(adaptive_plan.accumulated_results)} results in {adaptive_time:.2f}s, Static: {len(static_results.get('relationships', []))} results in {static_time:.2f}s")
        
        return comparison