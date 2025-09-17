"""
Agent Nodes - Individual Agent Implementations

This module implements the individual agent nodes for the LangGraph multi-agent
workflow: Annotator, Planner, BTE Search, and Summary agents for biomedical
research orchestration.

Migrated and enhanced from the original BTE-LLM implementation.
"""

import time
import logging
from typing import Literal, Dict, Any, List
from copy import deepcopy
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from .state import AgentState, SubQueryResult
from .rdf_manager import RDFGraphManager
from ..core.entities.bio_ner import BioNERTool
from ..core.knowledge.trapi import TRAPIQueryBuilder
from ..core.knowledge.bte_client import BTEClient
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class AnnotatorNode:
    """
    Biomedical entity annotation agent
    
    Extracts and links biomedical entities from user queries using
    the enhanced BioNER pipeline.
    """
    
    def __init__(self):
        self.bio_ner = BioNERTool()
        
    def __call__(self, state: AgentState) -> Command[Literal["orchestrator"]]:
        """
        Execute biomedical entity annotation
        
        Args:
            state: Current agent state
            
        Returns:
            Command to transition to orchestrator with entity data
        """
        logger.info(f"Annotator processing query: {state['query'][:100]}...")
        
        try:
            # Extract entities using BioNER
            response = self.bio_ner.extract_and_link(
                state["query"], 
                include_types=True
            )
            
            if "error" in response:
                logger.error(f"BioNER extraction failed: {response['error']}")
                response = {"entities": {}, "entity_ids": {}}
            
            logger.info(f"Extracted {len(response.get('entities', {}))} entities")
            
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=str(response), 
                            name="annotator"
                        )
                    ],
                    "entity_data": [response]
                },
                goto="orchestrator"
            )
            
        except Exception as e:
            logger.error(f"Error in annotator node: {str(e)}")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=f"Annotation error: {str(e)}", 
                            name="annotator"
                        )
                    ],
                    "entity_data": [{"entities": {}, "entity_ids": {}}]
                },
                goto="orchestrator"
            )


class PlannerNode:
    """
    Query decomposition and planning agent
    
    Generates strategic subqueries for iterative biomedical research
    using mechanistic decomposition approaches.
    """
    
    def __init__(self):
        self.settings = get_settings()
        # Import here to avoid circular imports
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(
            temperature=0,
            model=self.settings.openai_model,
            api_key=self.settings.openai_api_key
        )
    
    def __call__(
        self, 
        state: AgentState, 
        rdf_manager: RDFGraphManager
    ) -> Command[Literal["orchestrator"]]:
        """
        Generate next subquery in the research plan
        
        Args:
            state: Current agent state
            rdf_manager: RDF graph manager for context
            
        Returns:
            Command with generated subquery
        """
        logger.info("Planner generating next subquery...")
        
        try:
            # Get current RDF context
            current_results = rdf_manager.get_turtle_representation()
            
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
            {current_results}

            Make sure that each subquery interrogates discrete relationships between node types. 
            For example, instead of directly asking "What are the mechanisms of action of these drugs?", 
            you must create subqueries that can help form a reasoning chain to answer the question ("What genes do these drugs interact with?" and "Which physiological processes are these genes involved in?)"

            If a subquery results in "No results found", rephrase/reframe the question or explore the question from a different angle.

            Do NOT prescribe which nodes to use in your subquery, the nodes and predicates are only for your reference.
            You need to determine what the next single-hop subquery is and formulate it into a natural language subquestion. 

            Your response MUST ONLY CONTAIN the single-hop natural language subquestion (for example, "What genes does doxorubicin target?"). 
            Please DO NOT include your thoughts or anything else as it will interfere with downstream processes.
            """

            messages = [
                {"role": "assistant", "content": planner_prompt},
            ] + state["messages"]
            
            response = self.llm.invoke(messages)
            subquery = response.content.strip()
            
            logger.info(f"Generated subquery: {subquery}")
            
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=subquery, 
                            name="planner"
                        )
                    ],
                    "subQuery": [subquery]
                },
                goto="orchestrator"
            )
            
        except Exception as e:
            logger.error(f"Error in planner node: {str(e)}")
            fallback_query = "What are the basic relationships related to this query?"
            
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=f"Planning error, using fallback: {fallback_query}", 
                            name="planner"
                        )
                    ],
                    "subQuery": [fallback_query]
                },
                goto="orchestrator"
            )


class BTESearchNode:
    """
    BioThings Explorer search agent
    
    Executes TRAPI queries against the BTE knowledge graph with
    advanced batching, retry logic, and result processing.
    """
    
    def __init__(self):
        self.trapi_builder = TRAPIQueryBuilder()
        self.bte_client = BTEClient()
        self.max_attempts = 3
        
    def __call__(
        self, 
        state: AgentState, 
        rdf_manager: RDFGraphManager
    ) -> Command[Literal["orchestrator"]]:
        """
        Execute BTE search for current subquery
        
        Args:
            state: Current agent state
            rdf_manager: RDF graph manager for result storage
            
        Returns:
            Command with search results and updates
        """
        start_time = time.time()
        subquery = str(state["subQuery"][-1])
        maxresults = state["maxresults"]
        k = state["k"]
        failed_trapis = state.get("failed_trapis", [])
        
        logger.info(f"BTE Search executing subquery: {subquery}")
        
        # Consolidate entity data from all iterations
        entity_data = {}
        for entity_dict in state["entity_data"]:
            if isinstance(entity_dict, dict):
                if "entity_ids" in entity_dict:
                    entity_data.update(entity_dict["entity_ids"])
                else:
                    entity_data.update(entity_dict)
        
        # Execute search with retry logic
        for attempt in range(self.max_attempts):
            logger.info(f"BTE search attempt {attempt + 1}/{self.max_attempts}")
            
            try:
                results, entity_mappings, metadata = self._execute_bte_search(
                    subquery, entity_data, failed_trapis, maxresults, k
                )
                
                if results:
                    # Add results to RDF graph
                    added_triples = rdf_manager.add_triples(results)
                    
                    # Create execution summary
                    execution_time = time.time() - start_time
                    subquery_result = SubQueryResult(
                        subquery=subquery,
                        success=True,
                        results_count=len(results),
                        execution_time=execution_time,
                        confidence=metadata.get("confidence", 0.0),
                        error_message=None
                    )
                    
                    logger.info(f"BTE search successful: {len(results)} results, {added_triples} triples added")
                    
                    return Command(
                        update={
                            "messages": [
                                HumanMessage(
                                    content=f"Search results: {len(results)} relationships found", 
                                    name="BTE_search"
                                )
                            ],
                            "entity_data": [{k: v for k, v in entity_mappings.items()}] if entity_mappings else [],
                            "execution_metadata": {
                                "subquery_results": [subquery_result],
                                "total_api_calls": metadata.get("total_batches", 1),
                                "total_results": len(results)
                            }
                        },
                        goto="orchestrator"
                    )
                else:
                    logger.warning(f"No results found for attempt {attempt + 1}")
                    
            except Exception as e:
                logger.error(f"BTE search attempt {attempt + 1} failed: {str(e)}")
                continue
        
        # All attempts failed
        execution_time = time.time() - start_time
        subquery_result = SubQueryResult(
            subquery=subquery,
            success=False,
            results_count=0,
            execution_time=execution_time,
            confidence=0.0,
            error_message="All search attempts failed"
        )
        
        logger.error(f"BTE search failed after {self.max_attempts} attempts")
        
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content="No results found. Consider replanning.", 
                        name="BTE_search"
                    )
                ],
                "execution_metadata": {
                    "subquery_results": [subquery_result],
                    "total_api_calls": 0,
                    "total_results": 0
                }
            },
            goto="planner"  # Go back to planner for reframing
        )
    
    def _execute_bte_search(
        self, 
        subquery: str, 
        entity_data: Dict[str, str], 
        failed_trapis: List[Dict[str, Any]],
        maxresults: int,
        k: int
    ) -> tuple:
        """Execute TRAPI query building and BTE search"""
        
        # Build TRAPI query
        trapi_query = self.trapi_builder.build_query(
            subquery, entity_data, failed_trapis
        )
        
        if "error" in trapi_query:
            raise Exception(f"TRAPI building failed: {trapi_query['error']}")
        
        # Execute with batching
        results, entity_mappings, metadata = self.bte_client.execute_trapi_with_batching(
            trapi_query, maxresults, k
        )
        
        return results, entity_mappings, metadata
    
    def _filter_existing_entities(
        self, 
        target: Dict[str, str], 
        existing: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Remove entities that already exist in previous iterations"""
        existing_pairs = set()
        for entity_dict in existing:
            if isinstance(entity_dict, dict):
                existing_pairs.update(entity_dict.items())
        
        return {k: v for k, v in target.items() if (k, v) not in existing_pairs}


class SummaryNode:
    """
    Research synthesis and summary agent
    
    Generates comprehensive final answers based on accumulated
    RDF knowledge and biomedical expertise.
    """
    
    def __init__(self):
        self.settings = get_settings()
        # Import here to avoid circular imports
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(
            temperature=0,
            model=self.settings.openai_model,
            api_key=self.settings.openai_api_key
        )
    
    def __call__(
        self, 
        state: AgentState, 
        rdf_manager: RDFGraphManager
    ) -> Command[Literal["__end__"]]:
        """
        Generate final research summary
        
        Args:
            state: Current agent state
            rdf_manager: RDF graph manager with accumulated knowledge
            
        Returns:
            Command to end workflow with final answer
        """
        logger.info("Summary agent generating final answer...")
        
        try:
            # Get accumulated knowledge
            turtle_results = rdf_manager.get_turtle_representation()
            graph_stats = rdf_manager.get_summary_stats()
            
            # Consolidate entity data
            all_entities = {}
            for entity_dict in state["entity_data"]:
                if isinstance(entity_dict, dict):
                    all_entities.update(entity_dict)
            
            final_prompt = f"""You are an expert proficient in the pharmaceutical sciences, medicinal chemistry and biomedical research.
            Your team has access to a biomedical knowledge graph, and have provided you with the following results based on the user's query.
            However, the knowledge graph might not be perfect and have gaps in its data, and some relationships between the entities might be implicit.
            Your job is to answer the user's query using your team's results and your own biomedical expertise.
            Your summary of the results MUST be comprehensive while still avoiding redundancy.
            Expound on the logical steps you took to form your final answer.

            Make sure to organize your answer around the user query:
            {state["query"]}

            Here are the findings of your team expressed in Turtle (total triples: {graph_stats['total_triples']}):
            {turtle_results}

            Here are the entities each ID corresponds to:
            {all_entities}

            You MUST base your answer on the evidence/context included in the prompt; 
            however, you can use your expertise to contextualize the results. 
            For example, if the results only show target genes for dirithromycin, knowing that dirithromycin and erythromycin are part of the same drug class, you can infer that they are likely to share similar properties and targets.
            
            Maintain complete transparency about your problem-solving process; the relationships between each entity should be clear in your answer.
            Do NOT list down all entities; rather, choose to most important ones to illustrate your point (for example, "the BRCA family of proteins is involved in breast cancer").
            Remember, prioritize accuracy, explainability, and user understanding.
            Only include relevant results in the final answer.
            """
            
            messages = [{"role": "system", "content": final_prompt}]
            
            summary = self.llm.invoke(messages)
            final_answer = summary.content
            
            logger.info(f"Generated final answer ({len(final_answer)} chars)")
            
            return Command(
                update={
                    "final_answer": final_answer,
                    "messages": [
                        HumanMessage(
                            content=final_answer,
                            name="summary"
                        )
                    ]
                },
                goto="__end__"
            )
            
        except Exception as e:
            logger.error(f"Error in summary node: {str(e)}")
            fallback_answer = f"I encountered an error while summarizing the results: {str(e)}. Please try rephrasing your query."
            
            return Command(
                update={
                    "final_answer": fallback_answer,
                    "messages": [
                        HumanMessage(
                            content=fallback_answer,
                            name="summary"
                        )
                    ]
                },
                goto="__end__"
            )