#!/usr/bin/env python3
"""
Stateful GoT Optimizer - RDF-Enabled Graph of Thoughts

This implementation follows your insight that GoT should focus on query decomposition
while maintaining stateful access to accumulated knowledge through an RDF graph.

Key Design Principles:
1. GoT excels at query decomposition, not result consolidation
2. Planner has full RDF context access for informed decisions
3. TRAPI builder uses consolidated state entity data
4. Results immediately feed into centralized RDF knowledge base
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from agentic_bte.agents.rdf_manager import RDFGraphManager
from agentic_bte.core.queries.production_got_optimizer import ProductionGoTOptimizer
from agentic_bte.core.knowledge.trapi import TRAPIQueryBuilder
from agentic_bte.core.knowledge.bte_client import BTEClient
from agentic_bte.core.queries.mcp_integration import call_mcp_tool
from agentic_bte.core.queries.result_presenter import QueryStep, QueryResult
from agentic_bte.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class StatefulGoTConfig:
    """Configuration for stateful GoT optimizer"""
    max_subqueries: int = 5
    max_iterations_per_subquery: int = 3
    enable_rdf_context: bool = True
    enable_parallel_predicates: bool = True
    confidence_threshold: float = 0.5


class StatefulGoTOptimizer:
    """
    Stateful Graph of Thoughts Optimizer
    
    Uses RDF graph for persistent knowledge accumulation and provides
    full context access to planning and TRAPI building components.
    """
    
    def __init__(self, config: Optional[StatefulGoTConfig] = None):
        """Initialize stateful GoT optimizer"""
        self.config = config or StatefulGoTConfig()
        self.settings = get_settings()
        
        # Core components with stateful access
        self.rdf_manager = RDFGraphManager()
        self.trapi_builder = TRAPIQueryBuilder()
        self.bte_client = BTEClient()
        
        # Initialize LLM for planning
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(
            temperature=0,
            model=self.settings.openai_model,
            api_key=self.settings.openai_api_key
        )
        
        # State tracking
        self.execution_steps: List[QueryStep] = []
        self.consolidated_entity_data: Dict[str, str] = {}
        self.start_time: float = 0.0
        
        logger.info("Stateful GoT optimizer initialized with RDF context")
    
    async def execute_stateful_query(self, query: str) -> Tuple[QueryResult, str]:
        """
        Execute query using stateful GoT approach
        
        Args:
            query: Natural language biomedical query
            
        Returns:
            Tuple of (QueryResult, presentation_text)
        """
        self.start_time = time.time()
        self.execution_steps = []
        self.rdf_manager.clear_graph()
        
        logger.info(f"Starting stateful GoT execution: {query}")
        
        try:
            # Step 1: Initial entity extraction
            entity_step = await self._extract_entities(query)
            self.execution_steps.append(entity_step)
            
            if not entity_step.success:
                return self._create_error_result(query, "Entity extraction failed")
            
            # Initialize entity data from extraction
            entities = entity_step.output_data.get('entities', [])
            self._update_consolidated_entities(entity_step.output_data.get('entity_ids', {}))
            
            # Step 2: Execute GoT subquery decomposition loop
            subquery_results = await self._execute_stateful_got_loop(query, entities)
            self.execution_steps.extend(subquery_results)
            
            # Step 3: Generate final answer using RDF context
            final_step = await self._generate_final_answer_with_rdf(query)
            self.execution_steps.append(final_step)
            
            # Create final result
            total_time = time.time() - self.start_time
            final_answer = final_step.output_data.get('final_answer', 'No answer generated')
            
            result = QueryResult(
                query=query,
                final_answer=final_answer,
                execution_steps=self.execution_steps,
                total_execution_time=total_time,
                success=True
            )
            
            # Generate presentation
            from agentic_bte.core.queries.result_presenter import ResultPresenter
            presenter = ResultPresenter()
            presentation = presenter.present_results(result)
            
            logger.info(f"Stateful GoT completed successfully in {total_time:.2f}s")
            return result, presentation
            
        except Exception as e:
            logger.error(f"Stateful GoT execution failed: {str(e)}")
            return self._create_error_result(query, "Execution failed", str(e))
    
    async def _execute_stateful_got_loop(self, original_query: str, entities: List[Dict[str, Any]]) -> List[QueryStep]:
        """Execute the main GoT loop with RDF state management"""
        steps = []
        previous_subqueries: List[str] = []
        consecutive_zero_results = 0
        
        for iteration in range(self.config.max_subqueries):
            logger.info(f"GoT iteration {iteration + 1}/{self.config.max_subqueries}")
            
            # Step 1: Generate informed subquery using RDF context
            subquery_step = await self._generate_contextual_subquery(original_query, iteration, previous_subqueries)
            steps.append(subquery_step)
            
            if not subquery_step.success:
                logger.warning(f"Subquery generation failed at iteration {iteration + 1}")
                break
            
            subquery = subquery_step.output_data.get('subquery', '')
            if not subquery:
                logger.info(f"No more subqueries generated at iteration {iteration + 1}")
                break
            
            # Early repetition guard: if we keep asking the same subquery, abort
            if previous_subqueries and subquery.lower().strip() == previous_subqueries[-1].lower().strip():
                logger.info("Detected repeated subquery; applying early termination to avoid loops")
                break
            previous_subqueries.append(subquery)
            
            # Step 2: Execute subquery with stateful TRAPI building
            execution_steps = await self._execute_stateful_subquery(subquery, iteration)
            steps.extend(execution_steps)
            
            # Track results to decide early termination
            last_api_step = next((s for s in reversed(execution_steps) if s.step_type == 'stateful_bte_api'), None)
            if last_api_step and last_api_step.success:
                total_results = last_api_step.output_data.get('total_results', 0)
                if total_results == 0:
                    consecutive_zero_results += 1
                else:
                    consecutive_zero_results = 0
            
            # Early stop if multiple consecutive zero-result subqueries
            if consecutive_zero_results >= 2:
                logger.info("No results from two consecutive subqueries; stopping early")
                break
            
            # Check if we have sufficient information
            rdf_stats = self.rdf_manager.get_summary_stats()
            if rdf_stats['total_triples'] > 100:  # Sufficient knowledge accumulated
                logger.info(f"Sufficient knowledge accumulated ({rdf_stats['total_triples']} triples)")
                break
        
        return steps
    
    async def _generate_contextual_subquery(self, original_query: str, iteration: int, previous_subqueries: List[str]) -> QueryStep:
        """Generate subquery with full RDF context (like LangGraph planner)"""
        step_start = time.time()
        step_id = f"contextual_planning_{int(step_start)}_iter_{iteration}"
        
        logger.info(f"Generating contextual subquery for iteration {iteration}")
        
        try:
            # Get current RDF context (THIS IS THE KEY DIFFERENCE!)
            current_results = self.rdf_manager.get_turtle_representation()
            rdf_stats = self.rdf_manager.get_summary_stats()
            
            # Add signal of previous subqueries to discourage repetition
            prev_block = "\nPrevious subqueries attempted (most recent first):\n- " + "\n- ".join(previous_subqueries[-3:]) if previous_subqueries else "\nNo previous subqueries."
            
            # Create context-aware planning prompt (based on LangGraph planner)
            planning_prompt = f"""Your role is to create a plan to answer the main query by dividing it into several simpler single hop subqueries.
            The system has access to a knowledge graph where necessary data will be retrieved from to answer the query.

            This will be an iterative process where answers to previous subqueries might be relevant in constructing future queries.
            Initial subqueries should encompass the overall query path; each subsequent subquery should be more refined. A mechanistic approach should be used in developing the query plan.
            
            Original Query: "{original_query}"
            Current Iteration: {iteration + 1}
{prev_block}

            Here are the current results so far expressed in Turtle:
            {current_results}
            
            RDF Statistics: {rdf_stats['total_triples']} triples accumulated so far.

            Make sure that each subquery interrogates discrete relationships between node types.
            For example, instead of directly asking \"What are the mechanisms of action of these drugs?\",
            you must create subqueries that can help form a reasoning chain to answer the question 
            (\"What genes do these drugs interact with?\" and \"Which physiological processes are these genes involved in?\")

            If the previous subquery resulted in \"No results found\", rephrase/reframe the question or explore the question from a different angle.
            Avoid repeating the exact same subquery wording as earlier iterations unless it is strictly necessary.

            Based on the accumulated knowledge above, what should be the next single-hop subquery?
            If you believe sufficient information has been gathered to answer the original query, respond with \"COMPLETE\".

            Your response MUST ONLY CONTAIN the single-hop natural language subquestion (for example, \"What genes does doxorubicin target?\") or \"COMPLETE\".
            Please DO NOT include your thoughts or anything else as it will interfere with downstream processes.
            """
            
            response = self.llm.invoke(planning_prompt)
            subquery = response.content.strip()
            
            step_time = time.time() - step_start
            
            if subquery == "COMPLETE":
                logger.info("Planner indicated completion")
                return QueryStep(
                    step_id=step_id,
                    step_type='contextual_planning',
                    input_data={'original_query': original_query, 'iteration': iteration},
                    output_data={'subquery': '', 'status': 'complete'},
                    execution_time=step_time,
                    success=True,
                    confidence=0.9
                )
            
            logger.info(f"Generated contextual subquery: {subquery}")
            
            return QueryStep(
                step_id=step_id,
                step_type='contextual_planning',
                input_data={'original_query': original_query, 'iteration': iteration, 'rdf_context': current_results, 'previous_subqueries': previous_subqueries[-3:]},
                output_data={'subquery': subquery, 'status': 'continue'},
                execution_time=step_time,
                success=True,
                confidence=0.8
            )
            
        except Exception as e:
            step_time = time.time() - step_start
            logger.error(f"Contextual planning failed: {str(e)}")
            
            return QueryStep(
                step_id=step_id,
                step_type='contextual_planning',
                input_data={'original_query': original_query, 'iteration': iteration},
                output_data={},
                execution_time=step_time,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_stateful_subquery(self, subquery: str, iteration: int) -> List[QueryStep]:
        """Execute subquery with stateful entity consolidation"""
        steps = []
        
        # Step 1: Build TRAPI query with consolidated entity data
        trapi_step = await self._build_stateful_trapi_query(subquery, iteration)
        steps.append(trapi_step)
        
        if not trapi_step.success:
            return steps
        
        trapi_query = trapi_step.output_data.get('query', {})
        
        # Step 2: Execute BTE API call
        api_step = await self._execute_bte_api_call(trapi_query, subquery, iteration)
        steps.append(api_step)
        
        # Step 3: Immediately add results to RDF graph (like LangGraph)
        if api_step.success:
            results = api_step.output_data.get('results', [])
            if results:
                added_triples = self.rdf_manager.add_triples(results)
                logger.info(f"Added {added_triples} triples to RDF graph from subquery {iteration + 1}")
                
                # Update consolidated entity data
                entity_mappings = api_step.output_data.get('entity_mappings', {})
                self._update_consolidated_entities(entity_mappings)
        
        return steps
    
    async def _build_stateful_trapi_query(self, subquery: str, iteration: int) -> QueryStep:
        """Build TRAPI query using consolidated entity state"""
        step_start = time.time()
        step_id = f"stateful_trapi_building_{int(step_start)}_iter_{iteration}"
        
        logger.info(f"Building stateful TRAPI query for: {subquery}")
        logger.info(f"Using {len(self.consolidated_entity_data)} consolidated entities")
        
        try:
            # Use TRAPI builder with consolidated entity data (no complex post-processing needed!)
            trapi_query = self.trapi_builder.build_trapi_query(
                subquery, 
                entity_data=self.consolidated_entity_data,  # THIS IS THE FIX!
                failed_trapis=[]
            )
            
            # Post-process: ensure GO IDs map to correct category 'biolink:BiologicalProcess'
            try:
                qg = trapi_query.get('message', {}).get('query_graph', {})
                nodes = qg.get('nodes', {})
                changed = False
                for node_id, node in nodes.items():
                    ids = node.get('ids', [])
                    categories = node.get('categories', [])
                    if any(str(i).startswith('GO:') for i in ids):
                        if 'biolink:BiologicalProcess' not in categories:
                            # Replace PhysiologicalProcess with BiologicalProcess if present, else append
                            if 'biolink:PhysiologicalProcess' in categories:
                                categories = [c for c in categories if c != 'biolink:PhysiologicalProcess']
                            categories.insert(0, 'biolink:BiologicalProcess')
                            node['categories'] = categories
                            changed = True
                if changed:
                    logger.info('Adjusted TRAPI node categories for GO IDs to biolink:BiologicalProcess')
            except Exception as _:
                # Non-fatal; proceed with original query
                pass
            
            step_time = time.time() - step_start
            
            if "error" in trapi_query:
                return QueryStep(
                    step_id=step_id,
                    step_type='stateful_trapi_building',
                    input_data={'subquery': subquery, 'entity_count': len(self.consolidated_entity_data)},
                    output_data={},
                    execution_time=step_time,
                    success=False,
                    error_message=trapi_query["error"]
                )
            
            return QueryStep(
                step_id=step_id,
                step_type='stateful_trapi_building',
                input_data={'subquery': subquery, 'entity_count': len(self.consolidated_entity_data)},
                output_data={'query': trapi_query},
                trapi_query=trapi_query,
                execution_time=step_time,
                success=True,
                confidence=0.8
            )
            
        except Exception as e:
            step_time = time.time() - step_start
            logger.error(f"Stateful TRAPI building failed: {str(e)}")
            
            return QueryStep(
                step_id=step_id,
                step_type='stateful_trapi_building',
                input_data={'subquery': subquery, 'entity_count': len(self.consolidated_entity_data)},
                output_data={},
                execution_time=step_time,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_bte_api_call(self, trapi_query: Dict[str, Any], subquery: str, iteration: int) -> QueryStep:
        """Execute BTE API call"""
        step_start = time.time()
        step_id = f"stateful_bte_api_{int(step_start)}_iter_{iteration}"
        
        logger.info(f"Executing BTE API call for iteration {iteration + 1}")
        
        try:
            # Call BTE API via MCP
            response = await call_mcp_tool(
                "call_bte_api",
                json_query=trapi_query,
                k=10,
                maxresults=100
            )
            
            results = response.get('results', [])
            entity_mappings = response.get('entity_mappings', {})
            metadata = response.get('metadata', {})
            
            step_time = time.time() - step_start
            
            return QueryStep(
                step_id=step_id,
                step_type='stateful_bte_api',
                input_data={'trapi_query': trapi_query, 'subquery': subquery},
                output_data={
                    'results': results,
                    'entity_mappings': entity_mappings,
                    'metadata': metadata,
                    'total_results': len(results)
                },
                execution_time=step_time,
                success=True,
                confidence=metadata.get('confidence', 0.5)
            )
            
        except Exception as e:
            step_time = time.time() - step_start
            logger.error(f"BTE API call failed: {str(e)}")
            
            return QueryStep(
                step_id=step_id,
                step_type='stateful_bte_api',
                input_data={'trapi_query': trapi_query, 'subquery': subquery},
                output_data={},
                execution_time=step_time,
                success=False,
                error_message=str(e)
            )
    
    async def _extract_entities(self, query: str) -> QueryStep:
        """Extract entities using MCP bio_ner tool"""
        step_start = time.time()
        step_id = f"entity_extraction_{int(step_start)}"
        
        logger.info(f"Extracting entities from query: {query}")
        
        try:
            response = await call_mcp_tool("bio_ner", query=query)
            
            entities = response.get('entities', [])
            entity_ids = response.get('entity_ids', {})
            
            step_time = time.time() - step_start
            
            return QueryStep(
                step_id=step_id,
                step_type='entity_extraction',
                input_data={'query': query},
                output_data={
                    'entities': entities,
                    'entity_ids': entity_ids,
                    'total_entities': len(entities)
                },
                execution_time=step_time,
                success=True,
                confidence=0.85
            )
            
        except Exception as e:
            step_time = time.time() - step_start
            logger.error(f"Entity extraction failed: {str(e)}")
            
            return QueryStep(
                step_id=step_id,
                step_type='entity_extraction',
                input_data={'query': query},
                output_data={},
                execution_time=step_time,
                success=False,
                error_message=str(e)
            )
    
    async def _generate_final_answer_with_rdf(self, query: str) -> QueryStep:
        """Generate final answer using RDF context"""
        step_start = time.time()
        step_id = f"rdf_final_answer_{int(step_start)}"
        
        logger.info("Generating final answer with RDF context")
        
        try:
            # Get comprehensive RDF context
            rdf_context = self.rdf_manager.get_turtle_representation()
            rdf_stats = self.rdf_manager.get_summary_stats()
            
            # Generate final answer with domain expertise
            final_answer_prompt = f"""Based on the accumulated biomedical knowledge graph data below, provide a comprehensive, expert-level answer to the research question.

            Research Question: "{query}"

            Accumulated Knowledge Graph (RDF Triples):
            {rdf_context}

            Statistics: {rdf_stats['total_triples']} total knowledge relationships accumulated.

            Please provide a detailed, scientifically accurate answer that:
            1. Integrates the accumulated knowledge effectively
            2. Demonstrates pharmaceutical sciences and medicinal chemistry expertise
            3. Provides mechanistic explanations where relevant
            4. Cites specific drugs, genes, and pathways from the knowledge graph
            5. Offers expert insights and inferences based on the data

            Your response should be comprehensive and demonstrate deep domain knowledge."""
            
            response = self.llm.invoke(final_answer_prompt)
            final_answer = response.content.strip()
            
            step_time = time.time() - step_start
            
            logger.info(f"Generated RDF-informed final answer ({len(final_answer)} characters)")
            
            return QueryStep(
                step_id=step_id,
                step_type='rdf_final_answer',
                input_data={'query': query, 'rdf_triples': rdf_stats['total_triples']},
                output_data={
                    'final_answer': final_answer,
                    'rdf_stats': rdf_stats,
                    'answer_length': len(final_answer)
                },
                execution_time=step_time,
                success=True,
                confidence=0.9
            )
            
        except Exception as e:
            step_time = time.time() - step_start
            logger.error(f"RDF final answer generation failed: {str(e)}")
            
            return QueryStep(
                step_id=step_id,
                step_type='rdf_final_answer',
                input_data={'query': query},
                output_data={},
                execution_time=step_time,
                success=False,
                error_message=str(e)
            )
    
    def _update_consolidated_entities(self, entity_data: Dict[str, str]):
        """Update consolidated entity data (simple merge)"""
        self.consolidated_entity_data.update(entity_data)
        logger.debug(f"Consolidated entity data updated: {len(self.consolidated_entity_data)} total entities")
    
    def _create_error_result(self, query: str, error_type: str, error_message: Optional[str] = None) -> Tuple[QueryResult, str]:
        """Create error result and presentation"""
        total_time = time.time() - self.start_time if self.start_time > 0 else 0.0
        
        result = QueryResult(
            query=query,
            final_answer=f"Query execution failed: {error_type}. {error_message or ''}",
            execution_steps=self.execution_steps,
            total_execution_time=total_time,
            success=False,
            error_message=f"{error_type}: {error_message}" if error_message else error_type
        )
        
        from agentic_bte.core.queries.result_presenter import ResultPresenter
        presenter = ResultPresenter()
        presentation = presenter.present_results(result)
        
        return result, presentation


# Convenience function for easy usage
async def execute_stateful_got_query(query: str, config: Optional[StatefulGoTConfig] = None) -> Tuple[QueryResult, str]:
    """
    Execute a biomedical query using the stateful GoT approach
    
    Args:
        query: Natural language biomedical query
        config: Optional configuration
        
    Returns:
        Tuple of (QueryResult, presentation_text)
    """
    optimizer = StatefulGoTOptimizer(config)
    return await optimizer.execute_stateful_query(query)


if __name__ == "__main__":
    # Example usage
    async def demo():
        query = "What drugs can treat Brucellosis by targeting translation?"
        result, presentation = await execute_stateful_got_query(query)
        
        print("=== STATEFUL GOT RESULT ===")
        print(presentation)
        print(f"Success: {result.success}")
        print(f"Total time: {result.total_execution_time:.2f}s")
    
    asyncio.run(demo())