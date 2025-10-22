"""
Production-Ready Graph of Thoughts (GoT) Optimizer

This module provides a complete, production-ready implementation of the GoT
framework for biomedical query optimization with real MCP integration,
comprehensive result presentation, and debugging capabilities.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json
from copy import deepcopy

# Import our custom modules
from .mcp_integration import call_mcp_tool, get_mcp_integration
from .result_presenter import ResultPresenter, QueryResult, QueryStep, format_final_answer
from .got_framework import GoTBiomedicalPlanner
from .got_aggregation import BiomedicalAggregator, IterativeRefinementEngine
from .got_metrics import GoTMetricsCalculator

logger = logging.getLogger(__name__)


@dataclass
class ProductionConfig:
    """Configuration for production GoT optimizer"""
    # MCP integration settings
    mcp_timeout: int = 60
    mcp_max_retries: int = 3
    
    # GoT framework settings
    max_iterations: int = 5
    confidence_threshold: float = 0.7
    quality_threshold: float = 0.1
    
    # Result presentation settings
    show_debug: bool = True
    show_graphs: bool = False
    save_results: bool = True
    
    # Performance settings
    enable_caching: bool = True
    parallel_execution: bool = True
    max_concurrent: int = 3
    
    # Parallel predicate settings
    enable_parallel_predicates: bool = True
    max_predicates_per_subquery: int = 4
    max_concurrent_predicate_calls: int = 3
    enable_evidence_scoring: bool = True


class ProductionGoTOptimizer:
    """
    Production-ready Graph of Thoughts optimizer for biomedical queries
    
    Integrates all GoT framework components with real MCP tools and provides
    comprehensive result presentation with debugging capabilities.
    """
    
    def __init__(self, config: Optional[ProductionConfig] = None):
        """
        Initialize production GoT optimizer
        
        Args:
            config: Configuration object for the optimizer
        """
        self.config = config or ProductionConfig()
        
        # Initialize core components
        self.got_planner = GoTBiomedicalPlanner(max_iterations=self.config.max_iterations)
        self.aggregator = BiomedicalAggregator(enable_refinement=True)
        self.refinement_engine = IterativeRefinementEngine(
            max_iterations=self.config.max_iterations,
            improvement_threshold=self.config.quality_threshold
        )
        self.metrics_calculator = GoTMetricsCalculator()
        self.result_presenter = ResultPresenter(
            show_debug=self.config.show_debug,
            show_graphs=self.config.show_graphs
        )
        
        # Initialize predicate selector (will be populated with meta-KG later)
        self.predicate_selector = None
        
        # BTE client instance (will be populated with meta-KG when initialized)
        self.bte_client = None
        
        # Execution tracking
        self.execution_steps: List[QueryStep] = []
        self.start_time: float = 0.0
        self.subquery_info: List[Dict[str, Any]] = []
        
        logger.info("Production GoT optimizer initialized")
    
    async def execute_query(self, query: str, **kwargs) -> Tuple[QueryResult, str]:
        """
        Execute a biomedical query using the complete GoT framework
        
        Args:
            query: The biomedical query to execute
            **kwargs: Additional parameters for query execution
            
        Returns:
            Tuple of (QueryResult object, formatted presentation string)
        """
        logger.info(f"Starting production GoT query execution: {query}")
        self.start_time = time.time()
        self.execution_steps = []
        self.subquery_info = []
        
        try:
            # Step 1: Entity Extraction
            entities_step = await self._execute_entity_extraction(query)
            self.execution_steps.append(entities_step)
            
            if not entities_step.success:
                return self._create_error_result(query, "Entity extraction failed", entities_step.error_message)
            
            entities = entities_step.output_data.get('entities', [])
            logger.info(f"Extracted {len(entities)} entities")
            
            # Step 2: Query Building and Execution (GoT-enhanced)
            got_execution_steps = await self._execute_got_framework(query, entities)
            self.execution_steps.extend(got_execution_steps)
            
            # Collect all results from GoT execution with deduplication
            all_results = []
            result_dedup_map = {}  # Key: (subject_id, predicate, object_id) -> best result
            
            for step in got_execution_steps:
                if step.step_type == 'api_execution' and step.success:
                    step_results = step.output_data.get('results', [])
                    predicate_used = step.output_data.get('predicate_used', 'unknown')
                    
                    for result in step_results:
                        # Create deduplication key
                        subject_id = result.get('subject_id', '')
                        object_id = result.get('object_id', '')
                        predicate = result.get('predicate', predicate_used)
                        
                        dedup_key = (subject_id, predicate, object_id)
                        
                        # Keep the result with the highest score
                        current_score = result.get('score', 0.0)
                        
                        if dedup_key not in result_dedup_map:
                            result_dedup_map[dedup_key] = result
                        else:
                            existing_score = result_dedup_map[dedup_key].get('score', 0.0)
                            if current_score > existing_score:
                                result_dedup_map[dedup_key] = result
            
            # Convert deduplicated map back to list
            all_results = list(result_dedup_map.values())
            logger.info(f"After deduplication: {len(all_results)} unique results from {sum(len(step.output_data.get('results', [])) for step in got_execution_steps if step.step_type == 'api_execution' and step.success)} total")
            
            if not all_results:
                return self._create_error_result(query, "No results found", "GoT framework execution produced no results")
            
            # Step 3: Result Aggregation and Refinement
            aggregation_step = await self._execute_aggregation_refinement(all_results, entities)
            self.execution_steps.append(aggregation_step)
            
            final_results = aggregation_step.output_data.get('final_results', all_results)
            
            # Step 4: Generate LLM-based final answer
            from .final_answer_llm import llm_generate_final_answer
            execution_context = {
                'execution_steps': [{
                    'step_type': step.step_type,
                    'step_id': step.step_id,
                    'success': step.success,
                    'confidence': step.confidence,
                    'execution_time': step.execution_time,
                    'output_data': step.output_data if hasattr(step, 'output_data') else {}
                } for step in self.execution_steps],
                'subquery_info': self.subquery_info,
                'entities': entities
            }
            final_answer = await llm_generate_final_answer(query, final_results, entities, execution_context)
            
            # Calculate metrics
            got_metrics = self._calculate_got_metrics()
            
            # Create result object
            total_time = time.time() - self.start_time
            quality_score = self._calculate_quality_score(final_results)
            
            result = QueryResult(
                query=query,
                final_answer=final_answer,
                execution_steps=self.execution_steps,
                total_execution_time=total_time,
                success=True,
                entities_found=entities,
                total_results=len(final_results),
                quality_score=quality_score,
                got_metrics=got_metrics
            )
            
            # Generate presentation
            presentation = self.result_presenter.present_results(result)
            
            # Save results if configured
            if self.config.save_results:
                await self._save_results(result, presentation)
            
            logger.info(f"Query execution completed successfully in {total_time:.3f}s")
            return result, presentation
            
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return self._create_error_result(query, "Execution failed", str(e))
    
    async def _execute_entity_extraction(self, query: str) -> QueryStep:
        """Execute entity extraction step"""
        step_start = time.time()
        step_id = f"entity_extraction_{int(step_start)}"
        
        logger.info("Executing entity extraction")
        
        try:
            # Call MCP bio_ner tool
            response = await call_mcp_tool("bio_ner", query=query)
            
            entities = response.get('entities', [])
            confidence = response.get('confidence', 0.0)
            
            step_time = time.time() - step_start
            
            return QueryStep(
                step_id=step_id,
                step_type='entity_extraction',
                input_data={'query': query},
                output_data={
                    'entities': entities,
                    'confidence': confidence,
                    'source': response.get('source', 'unknown')
                },
                execution_time=step_time,
                success=True,
                confidence=confidence
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
    
    async def _execute_got_framework(self, query: str, entities: List[Dict[str, Any]]) -> List[QueryStep]:
        """Execute the GoT framework for query processing"""
        logger.info("Executing GoT framework")
        
        # Initialize BTE client and predicate selector with meta-KG if not already done
        if self.bte_client is None:
            from ..knowledge.bte_client import BTEClient
            self.bte_client = BTEClient()
            
            # Get meta-KG and store it in the client for propagation
            try:
                meta_kg = self.bte_client.get_meta_knowledge_graph()
                logger.info("Initialized production BTE client with meta-KG cached")
                
                # Initialize predicate selector if parallel predicates are enabled
                if self.config.enable_parallel_predicates and self.predicate_selector is None:
                    from ..knowledge.predicate_strategy import create_predicate_selector
                    self.predicate_selector = create_predicate_selector(meta_kg)
                    logger.info("Initialized predicate selector with meta-KG")
                    
            except Exception as e:
                logger.warning(f"Could not initialize BTE client or predicate selector: {e}")
                self.predicate_selector = None
        
        steps = []
        
        # Create entity data dictionary
        entity_data = {entity.get('name', f'entity_{i}'): entity.get('id', '') 
                      for i, entity in enumerate(entities)}
        
        # For each entity or entity combination, create and execute TRAPI queries
        if len(entities) <= 2:
            # Simple case: single TRAPI query
            steps.extend(await self._execute_single_got_path(query, entity_data, entities))
        else:
            # Complex case: multiple TRAPI queries with GoT planning
            steps.extend(await self._execute_multiple_got_paths(query, entity_data, entities))
        
        return steps
    
    async def _execute_single_got_path(self, query: str, entity_data: Dict[str, str], 
                                     entities: List[Dict[str, Any]]) -> List[QueryStep]:
        """Execute a single GoT path for simple queries"""
        steps = []
        
        # Step 1: Build TRAPI query
        query_building_step = await self._build_trapi_query(query, entity_data, entities)
        steps.append(query_building_step)
        
        if not query_building_step.success:
            return steps
        
        trapi_query = query_building_step.output_data.get('query', {})
        
        # Step 2: Execute API call
        api_execution_step = await self._execute_bte_api(trapi_query, query_building_step.step_id)
        steps.append(api_execution_step)
        
        return steps
    
    async def _execute_multiple_got_paths(self, query: str, entity_data: Dict[str, str],
                                        entities: List[Dict[str, Any]]) -> List[QueryStep]:
        """Execute multiple GoT paths for complex queries with scientifically sound subqueries"""
        steps = []
        
        # Generate scientifically meaningful subqueries based on the original query and entities
        subqueries = await self._generate_scientific_subqueries(query, entities)
        
        # Store subquery information for LLM explainability
        self.subquery_info = subqueries
        
        # Execute each subquery with parallel predicate exploration
        for pair_idx, subquery_info in enumerate(subqueries):
            subquery = subquery_info['query']
            relevant_entities = subquery_info['entities']
            
            # Execute subquery with parallel predicates (if enabled) or standard execution
            subquery_steps = await self._execute_subquery_with_parallel_predicates(
                subquery, relevant_entities, pair_idx
            )
            steps.extend(subquery_steps)
        
        return steps
    
    async def _generate_scientific_subqueries(self, query: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate scientifically meaningful subqueries using LLM-based reasoning
        
        Args:
            query: Original complex query
            entities: List of extracted entities with their types
            
        Returns:
            List of subquery dictionaries with 'query' and 'entities' keys
        """
        try:
            # Import OpenAI for LLM-based reasoning
            from langchain_openai import ChatOpenAI
            from ...config.settings import get_settings
            
            settings = get_settings()
            llm = ChatOpenAI(
                temperature=0.1,  # Low temperature for consistent reasoning
                model=settings.openai_model,
                api_key=settings.openai_api_key
            )
            
            # Prepare entity information for the LLM
            entity_info = []
            for entity in entities:
                name = entity.get('name', 'Unknown')
                entity_type = entity.get('type', 'unknown')
                entity_id = entity.get('id', 'N/A')
                entity_info.append(f"- {name} (Type: {entity_type}, ID: {entity_id})")
            
            entities_text = "\n".join(entity_info)
            
            # Create a comprehensive prompt for biomedical query decomposition
            decomposition_prompt = f"""
            You are an expert biomedical researcher tasked with decomposing a complex biomedical query into scientifically meaningful subqueries that can be answered using knowledge graphs.
            
            Original Query: "{query}"
            
            Extracted Biomedical Entities:
            {entities_text}
            
            Your task is to analyze this query and decompose it into 3-5 focused subqueries that:
            1. Are scientifically sound and answerable using biomedical knowledge graphs
            2. Each focus on a specific biomedical relationship (e.g., drug-disease, gene-disease, protein-pathway)
            3. Together, provide comprehensive coverage of the original query
            4. Include mechanism analysis subqueries when appropriate
            5. Use proper biomedical terminology and concepts
            
            CRITICAL REQUIREMENTS:
            - For drug treatment queries: ALWAYS include both drug-disease queries AND mechanism analysis queries
            - For mechanism queries: Create subqueries that explore drug targets, pathways, and molecular mechanisms
            - Include follow-up subqueries to investigate HOW identified drugs work (gene targets, biological processes)
            - Ensure deep investigation of therapeutic mechanisms and molecular pathways
            
            Guidelines:
            - First subquery: Identify direct therapeutic relationships (e.g., "What drugs treat [disease]?")
            - Second subquery: Explore mechanism targets (e.g., "What genes/proteins do these drugs target?")
            - Third subquery: Investigate biological processes (e.g., "What biological processes are affected by these drugs?")
            - Fourth subquery: Connect mechanisms to the specific biological process mentioned in the query
            - Fifth subquery: Explore pathway relationships if relevant
            
            For each subquery, you should:
            - Formulate a clear, specific biomedical question
            - Identify which entities from the list above are most relevant to that subquery
            - Ensure the subquery can be answered using biomedical knowledge graphs
            - For mechanism subqueries, use terms like "targets", "affects", "modulates", "inhibits"
            
            Respond with a JSON array of subquery objects, where each object has:
            - "query": A clear, specific biomedical question
            - "entities": An array of entity names from the above list that are relevant to this subquery
            - "rationale": A brief explanation of why this subquery is important for answering the original query
            - "query_type": One of ["therapeutic", "mechanism", "pathway", "target", "process"]
            
            Example format for drug treatment query:
            [
                {{
                    "query": "What drugs are used to treat Brucellosis?",
                    "entities": ["Brucellosis", "drugs"],
                    "rationale": "Identifies therapeutic agents for the specific bacterial infection",
                    "query_type": "therapeutic"
                }},
                {{
                    "query": "What genes or proteins do Brucellosis treatment drugs target?",
                    "entities": ["drugs"],
                    "rationale": "Explores the molecular targets and mechanisms of action for identified drugs",
                    "query_type": "target"
                }},
                {{
                    "query": "How do drug targets relate to the translation process?",
                    "entities": ["drugs", "translation"],
                    "rationale": "Connects drug mechanisms to the specific biological process mentioned in the query",
                    "query_type": "mechanism"
                }}
            ]
            
            Focus on scientific accuracy and ensure each subquery addresses a specific aspect of the original complex query.
            """
            
            # Get LLM response
            logger.info("Using LLM to decompose complex query into scientific subqueries...")
            response = llm.invoke(decomposition_prompt)
            
            # Parse the JSON response
            import json
            import re
            
            response_text = response.content.strip()
            
            # Extract JSON from the response (handle markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
            if json_match:
                json_text = json_match.group(1).strip()
            else:
                json_text = response_text
            
            # Try to find JSON array in the text - use a more robust approach
            array_match = re.search(r'\[\s*\{[\s\S]*?\}\s*\]', json_text, re.DOTALL)
            if array_match:
                json_text = array_match.group(0)
            
            try:
                # Clean up the JSON text to handle common LLM formatting issues
                json_text = json_text.replace('\n', ' ').strip()
                # Remove any trailing commas before closing braces or brackets
                json_text = re.sub(r',\s*}', '}', json_text)
                json_text = re.sub(r',\s*\]', ']', json_text)
                
                subquery_data = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM response as JSON: {e}")
                logger.warning(f"Response text: {response_text}")
                # Fallback to simple decomposition
                return self._fallback_subquery_generation(query, entities)
            
            # Convert to the expected format and validate entities
            subqueries = []
            for sq_data in subquery_data:
                if not isinstance(sq_data, dict) or 'query' not in sq_data or 'entities' not in sq_data:
                    logger.warning(f"Invalid subquery format: {sq_data}")
                    continue
                
                subquery_text = sq_data['query']
                entity_names = sq_data['entities']
                rationale = sq_data.get('rationale', '')
                
                # Find matching entities from our extracted entities list
                relevant_entities = []
                for entity_name in entity_names:
                    # Find the entity object that matches this name
                    for entity in entities:
                        if entity.get('name', '').lower() == entity_name.lower():
                            relevant_entities.append(entity)
                            break
                    else:
                        # If exact match not found, try partial match
                        for entity in entities:
                            if (entity_name.lower() in entity.get('name', '').lower() or 
                                entity.get('name', '').lower() in entity_name.lower()):
                                relevant_entities.append(entity)
                                break
                
                # Ensure we have at least some entities for the subquery
                if not relevant_entities and entities:
                    # If no entities matched, use the most relevant ones based on the subquery
                    relevant_entities = entities[:2]  # Use first 2 entities as fallback
                
                subqueries.append({
                    'query': subquery_text,
                    'entities': relevant_entities,
                    'rationale': rationale
                })
            
            # Ensure we have at least one subquery
            if not subqueries:
                logger.warning("LLM generated no valid subqueries, using fallback")
                return self._fallback_subquery_generation(query, entities)
            
            logger.info(f"LLM generated {len(subqueries)} scientific subqueries from: {query}")
            for i, sq in enumerate(subqueries, 1):
                logger.info(f"  Subquery {i}: {sq['query']}")
                logger.info(f"    Entities: {[e.get('name') for e in sq['entities']]}")
                logger.info(f"    Rationale: {sq.get('rationale', 'N/A')}")
            
            return subqueries
            
        except Exception as e:
            logger.error(f"Error in LLM-based subquery generation: {str(e)}")
            return self._fallback_subquery_generation(query, entities)
    
    def _fallback_subquery_generation(self, query: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback method for subquery generation when LLM fails
        
        Args:
            query: Original query
            entities: List of entities
            
        Returns:
            Basic subqueries
        """
        logger.info("Using fallback subquery generation")
        
        if len(entities) <= 2:
            # Simple case: use original query with all entities
            return [{
                'query': query,
                'entities': entities,
                'rationale': 'Direct query with all available entities'
            }]
        
        # Complex case: create subqueries with entity pairs
        subqueries = []
        for i in range(0, len(entities), 2):
            if i + 1 < len(entities):
                entity_pair = [entities[i], entities[i+1]]
                subquery_text = f"How are {entities[i].get('name')} and {entities[i+1].get('name')} related?"
            else:
                entity_pair = [entities[i]]
                subquery_text = f"What is related to {entities[i].get('name')}?"
            
            subqueries.append({
                'query': subquery_text,
                'entities': entity_pair,
                'rationale': 'Fallback pairwise entity relationship query'
            })
        
        return subqueries
    
    async def _execute_subquery_with_parallel_predicates(self, subquery: str, entities: List[Dict[str, Any]], 
                                                       subquery_index: int) -> List[QueryStep]:
        """
        Execute a single subquery with parallel predicate exploration
        
        Args:
            subquery: The subquery text
            entities: Relevant entities for this subquery
            subquery_index: Index for step naming
            
        Returns:
            List of query steps from parallel predicate execution
        """
        steps = []
        
        # CRITICAL FIX: Consolidate entity data from all previous subquery results
        # This enables proper placeholder injection like the LangGraph version
        entity_data = self._consolidate_entity_data_from_all_steps(entities)
        
        logger.info(f"GoT Subquery {subquery_index + 1}: {subquery}")
        logger.info(f"Using original entities: {[e.get('name') for e in entities]}")
        logger.info(f"Total consolidated entities available: {len(entity_data)}")
        
        # Build base TRAPI query to get categories
        base_query_step = await self._build_trapi_query(subquery, entity_data, entities)
        base_query_step.step_id = f"{base_query_step.step_id}_subq_{subquery_index}"
        steps.append(base_query_step)
        
        if not base_query_step.success:
            return steps
        
        base_trapi_query = base_query_step.output_data.get('query', {})
        
        # Determine query intent and categories for predicate selection
        if self.config.enable_parallel_predicates and self.predicate_selector:
            try:
                # Detect query intent
                from ..knowledge.predicate_strategy import QueryIntent
                query_intent = self.predicate_selector.detect_query_intent(subquery, entities)
                
                # Extract categories from TRAPI query
                query_graph = base_trapi_query.get('message', {}).get('query_graph', {})
                nodes = query_graph.get('nodes', {})
                
                subject_category = None
                object_category = None
                
                for node_id, node_data in nodes.items():
                    categories = node_data.get('categories', [])
                    if categories:
                        if subject_category is None:
                            subject_category = categories[0]
                        else:
                            object_category = categories[0]
                            break
                
                if subject_category and object_category:
                    # Select predicates using our strategy
                    selected_predicates = self.predicate_selector.select_predicates(
                        query_intent, subject_category, object_category
                    )
                    
                    logger.info(f"Selected {len(selected_predicates)} predicates for parallel execution: {[p[0] for p in selected_predicates]}")
                    
                    # Execute TRAPI queries in parallel with different predicates
                    parallel_steps = await self._execute_parallel_predicate_queries(
                        base_trapi_query, selected_predicates, query_intent, subquery_index
                    )
                    steps.extend(parallel_steps)
                else:
                    logger.warning("Could not extract subject/object categories for predicate selection")
                    # Fall back to single execution
                    api_step = await self._execute_bte_api(base_trapi_query, base_query_step.step_id)
                    api_step.step_id = f"{api_step.step_id}_subq_{subquery_index}"
                    steps.append(api_step)
                    
            except Exception as e:
                logger.error(f"Error in parallel predicate execution: {e}")
                # Fall back to single execution
                api_step = await self._execute_bte_api(base_trapi_query, base_query_step.step_id)
                api_step.step_id = f"{api_step.step_id}_subq_{subquery_index}"
                steps.append(api_step)
        else:
            # Standard single execution
            api_step = await self._execute_bte_api(base_trapi_query, base_query_step.step_id)
            api_step.step_id = f"{api_step.step_id}_subq_{subquery_index}"
            steps.append(api_step)
        
        return steps
    
    async def _execute_parallel_predicate_queries(self, base_trapi_query: Dict[str, Any], 
                                                 selected_predicates: List[Tuple[str, float]], 
                                                 query_intent, subquery_index: int) -> List[QueryStep]:
        """
        Execute multiple TRAPI queries in parallel with different predicates
        
        Args:
            base_trapi_query: Base TRAPI query structure
            selected_predicates: List of (predicate, priority) tuples
            query_intent: Query intent enum
            subquery_index: Index for step naming
            
        Returns:
            List of API execution steps
        """
        import asyncio
        from copy import deepcopy
        
        # Create TRAPI queries with different predicates
        predicate_queries = []
        for predicate, priority in selected_predicates:
            modified_query = deepcopy(base_trapi_query)
            
            # Update the edge predicate in the query
            query_graph = modified_query.get('message', {}).get('query_graph', {})
            edges = query_graph.get('edges', {})
            
            for edge_id, edge_data in edges.items():
                edge_data['predicates'] = [predicate]
            
            predicate_queries.append((modified_query, predicate, priority))
        
        # Execute queries concurrently with limited concurrency
        semaphore = asyncio.Semaphore(self.config.max_concurrent_predicate_calls)
        
        async def execute_single_predicate_query(query_predicate_priority):
            trapi_query, predicate, priority = query_predicate_priority
            
            async with semaphore:
                try:
                    # Execute with predicate and intent info for scoring
                    response = await call_mcp_tool(
                        "call_bte_api",
                        json_query=trapi_query,
                        k=10,
                        maxresults=100,
                        predicate=predicate,
                        query_intent=query_intent.value
                    )
                    
                    return predicate, priority, response, None
                except Exception as e:
                    logger.warning(f"Predicate {predicate} execution failed: {e}")
                    return predicate, priority, None, str(e)
        
        # Execute all predicate queries concurrently
        logger.info(f"Executing {len(predicate_queries)} predicate queries concurrently")
        results = await asyncio.gather(*[execute_single_predicate_query(pq) for pq in predicate_queries])
        
        # Process results into QuerySteps
        api_steps = []
        total_results = 0
        
        for predicate, priority, response, error in results:
            step_start = time.time()
            step_id = f"api_execution_{int(step_start)}_pred_{predicate.split(':')[-1]}_subq_{subquery_index}"
            
            if error:
                # Failed execution
                step = QueryStep(
                    step_id=step_id,
                    step_type='api_execution',
                    input_data={'predicate': predicate, 'priority': priority},
                    output_data={},
                    execution_time=0.1,
                    success=False,
                    error_message=error
                )
            else:
                # Successful execution
                results_data = response.get('results', [])
                metadata = response.get('metadata', {})
                entity_mappings = response.get('entity_mappings', {})
                step_results = len(results_data)
                total_results += step_results
                
                # Calculate confidence from results (should now have evidence-weighted scores)
                avg_confidence = 0.0
                if results_data:
                    scores = [r.get('score', 0.0) for r in results_data]
                    avg_confidence = sum(scores) / len(scores) if scores else 0.0
                
                step = QueryStep(
                    step_id=step_id,
                    step_type='api_execution',
                    input_data={'predicate': predicate, 'priority': priority, 'query_intent': query_intent.value},
                    output_data={
                        'results': results_data,
                        'total_results': step_results,
                        'metadata': metadata,
                        'entity_mappings': entity_mappings,
                        'predicate_used': predicate,
                        'predicate_priority': priority
                    },
                    execution_time=time.time() - step_start,
                    success=True,
                    confidence=avg_confidence
                )
                
                logger.info(f"Predicate {predicate}: {step_results} results, confidence: {avg_confidence:.3f}")
            
            api_steps.append(step)
        
        logger.info(f"Parallel predicate execution completed: {total_results} total results from {len(api_steps)} predicates")
        return api_steps
    
    async def _build_trapi_query(self, query: str, entity_data: Dict[str, str],
                               entities: List[Dict[str, Any]]) -> QueryStep:
        """Build TRAPI query step"""
        step_start = time.time()
        step_id = f"query_building_{int(step_start)}"
        
        logger.info("Building TRAPI query")
        
        try:
            # Call MCP build_trapi_query tool
            response = await call_mcp_tool(
                "build_trapi_query",
                query=query,
                entity_data=entity_data,
                failed_trapis=[]
            )
            
            trapi_query = response.get('query', {})
            confidence = response.get('confidence', 0.0)
            
            step_time = time.time() - step_start
            
            return QueryStep(
                step_id=step_id,
                step_type='query_building',
                input_data={
                    'query': query,
                    'entity_data': entity_data,
                    'entities': entities
                },
                output_data={
                    'query': trapi_query,
                    'confidence': confidence,
                    'source': response.get('source', 'unknown')
                },
                trapi_query=trapi_query,
                execution_time=step_time,
                success=True,
                confidence=confidence
            )
            
        except Exception as e:
            step_time = time.time() - step_start
            logger.error(f"TRAPI query building failed: {str(e)}")
            
            return QueryStep(
                step_id=step_id,
                step_type='query_building',
                input_data={
                    'query': query,
                    'entity_data': entity_data,
                    'entities': entities
                },
                output_data={},
                execution_time=step_time,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_bte_api(self, trapi_query: Dict[str, Any], parent_step_id: str) -> QueryStep:
        """Execute BTE API call step"""
        step_start = time.time()
        step_id = f"api_execution_{int(step_start)}"
        
        logger.info("Executing BTE API call")
        
        try:
            # Call MCP call_bte_api tool
            response = await call_mcp_tool(
                "call_bte_api",
                json_query=trapi_query,
                k=10,  # Request more results
                maxresults=100
            )
            
            results = response.get('results', [])
            metadata = response.get('metadata', {})
            entity_mappings = response.get('entity_mappings', {})
            total_results = metadata.get('total_results', len(results))
            
            step_time = time.time() - step_start
            
            # Transform BTE results to include knowledge graph structure and scores
            enhanced_results = []
            for result in results:
                # Convert BTE format to enhanced format with knowledge graph structure
                enhanced_result = {
                    'score': 0.8 if result.get('subject') and result.get('object') else 0.0,  # Assign confidence based on completeness
                    'knowledge_graph': {
                        'nodes': {},
                        'edges': {}
                    }
                }
                
                # Add nodes to knowledge graph
                if result.get('subject_id') and result.get('subject'):
                    enhanced_result['knowledge_graph']['nodes'][result['subject_id']] = {
                        'name': result['subject'],
                        'categories': [result.get('subject_type', 'biolink:NamedThing')]
                    }
                
                if result.get('object_id') and result.get('object'):
                    enhanced_result['knowledge_graph']['nodes'][result['object_id']] = {
                        'name': result['object'],
                        'categories': [result.get('object_type', 'biolink:NamedThing')]
                    }
                
                # Add edge to knowledge graph
                if result.get('subject_id') and result.get('object_id') and result.get('predicate'):
                    edge_id = f"edge_{len(enhanced_result['knowledge_graph']['edges'])}"
                    enhanced_result['knowledge_graph']['edges'][edge_id] = {
                        'subject': result['subject_id'],
                        'object': result['object_id'],
                        'predicate': result['predicate']
                    }
                
                # Keep original BTE format for compatibility
                enhanced_result.update(result)
                enhanced_results.append(enhanced_result)
            
            # Calculate average confidence from enhanced results
            avg_confidence = 0.0
            if enhanced_results:
                scores = [r.get('score', 0.0) for r in enhanced_results]
                avg_confidence = sum(scores) / len(scores) if scores else 0.0
            
            return QueryStep(
                step_id=step_id,
                step_type='api_execution',
                input_data={'trapi_query': trapi_query},
                output_data={
                    'results': enhanced_results,
                    'total_results': total_results,
                    'entity_mappings': entity_mappings,
                    'metadata': metadata,
                    'query_used': trapi_query,
                    'source': 'bte_api'
                },
                trapi_query=trapi_query,
                execution_time=step_time,
                success=True,
                confidence=avg_confidence
            )
            
        except Exception as e:
            step_time = time.time() - step_start
            logger.error(f"BTE API execution failed: {str(e)}")
            
            return QueryStep(
                step_id=step_id,
                step_type='api_execution',
                input_data={'trapi_query': trapi_query},
                output_data={},
                trapi_query=trapi_query,
                execution_time=step_time,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_aggregation_refinement(self, all_results: List[Dict[str, Any]],
                                            entities: List[Dict[str, Any]]) -> QueryStep:
        """Execute result aggregation and refinement step"""
        step_start = time.time()
        step_id = f"aggregation_{int(step_start)}"
        
        logger.info(f"Executing aggregation and refinement on {len(all_results)} results")
        
        try:
            # Aggregate results using the biomedical aggregator
            aggregated_results = await self.aggregator.aggregate_results(all_results, entities)
            
            # Apply iterative refinement if quality is below threshold
            initial_quality = self._calculate_quality_score(aggregated_results)
            
            if initial_quality < self.config.quality_threshold:
                logger.info("Applying iterative refinement due to low quality score")
                refined_results = await self.refinement_engine.refine_results(
                    aggregated_results, 
                    entities,
                    max_iterations=3
                )
                final_results = refined_results
            else:
                final_results = aggregated_results
            
            step_time = time.time() - step_start
            
            # Calculate metrics
            final_quality = self._calculate_quality_score(final_results)
            conflicts_resolved = len(all_results) - len(final_results)  # Simplified metric
            
            return QueryStep(
                step_id=step_id,
                step_type='aggregation',
                input_data={
                    'raw_results': all_results,
                    'entities': entities
                },
                output_data={
                    'final_results': final_results,
                    'initial_quality': initial_quality,
                    'final_quality': final_quality,
                    'conflicts_resolved': conflicts_resolved,
                    'duplicates_removed': max(0, len(all_results) - len(final_results)),
                    'ranked_results': final_results  # Already ranked by aggregator
                },
                execution_time=step_time,
                success=True,
                confidence=final_quality
            )
            
        except Exception as e:
            step_time = time.time() - step_start
            logger.error(f"Aggregation and refinement failed: {str(e)}")
            
            return QueryStep(
                step_id=step_id,
                step_type='aggregation',
                input_data={
                    'raw_results': all_results,
                    'entities': entities
                },
                output_data={'final_results': all_results},  # Fallback to original results
                execution_time=step_time,
                success=False,
                error_message=str(e)
            )
    
    def _consolidate_entity_data_from_all_steps(self, original_entities: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Consolidate entity data from all previous execution steps, mirroring LangGraph behavior
        
        This enables proper placeholder injection by making results from previous subqueries
        available to subsequent ones, allowing queries like "What genes do these drugs target?"
        where "these drugs" refers to results from a previous subquery.
        
        Args:
            original_entities: The original entities extracted from the query
            
        Returns:
            Dictionary mapping entity names to IDs from all execution steps
        """
        consolidated_data = {}
        
        # Start with original entities
        for entity in original_entities:
            name = entity.get('name', '')
            entity_id = entity.get('id', '')
            if name and entity_id:
                consolidated_data[name] = entity_id
        
        # Add entities from all previous API execution steps
        for step in self.execution_steps:
            if step.step_type == 'api_execution' and step.success:
                output_data = step.output_data or {}
                
                # Extract entity mappings from BTE API results
                entity_mappings = output_data.get('entity_mappings', {})
                consolidated_data.update(entity_mappings)
                
                # Extract entities from knowledge graph results
                results = output_data.get('results', [])
                for result in results:
                    # Extract subject entities - prioritize name even without ID
                    subject_name = result.get('subject')
                    subject_id = result.get('subject_id')
                    if subject_name:
                        # Use ID if available, otherwise generate a placeholder ID from name
                        if subject_id:
                            consolidated_data[subject_name] = subject_id
                        else:
                            # Create a meaningful placeholder ID for entities without explicit IDs
                            placeholder_id = self._generate_placeholder_id(subject_name)
                            consolidated_data[subject_name] = placeholder_id
                    
                    # Extract object entities - prioritize name even without ID  
                    object_name = result.get('object')
                    object_id = result.get('object_id')
                    if object_name:
                        # Use ID if available, otherwise generate a placeholder ID from name
                        if object_id:
                            consolidated_data[object_name] = object_id
                        else:
                            # Create a meaningful placeholder ID for entities without explicit IDs
                            placeholder_id = self._generate_placeholder_id(object_name)
                            consolidated_data[object_name] = placeholder_id
                    
                    # Extract from knowledge graph nodes if present
                    kg = result.get('knowledge_graph', {})
                    nodes = kg.get('nodes', {})
                    for node_id, node_data in nodes.items():
                        node_name = node_data.get('name')
                        if node_name and node_id:
                            consolidated_data[node_name] = node_id
        
        logger.info(f"Consolidated {len(consolidated_data)} total entities from {len(self.execution_steps)} previous steps")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Available entities: {list(consolidated_data.keys())[:10]}...")  # Show first 10
        
        return consolidated_data
    
    def _generate_placeholder_id(self, entity_name: str) -> str:
        """
        Generate a meaningful placeholder ID for entities without explicit IDs
        
        This enables TRAPI queries to use specific entity names even when BTE results
        don't provide formal IDs, allowing proper cascading between subqueries.
        
        Args:
            entity_name: Name of the entity
            
        Returns:
            Placeholder ID that can be used in TRAPI queries
        """
        import re
        import hashlib
        
        # Clean the entity name for ID generation
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', entity_name.lower())
        clean_name = re.sub(r'_+', '_', clean_name).strip('_')
        
        # Create a short hash for uniqueness while keeping it readable
        name_hash = hashlib.md5(entity_name.encode()).hexdigest()[:8]
        
        # Generate a meaningful placeholder ID
        placeholder_id = f"PLACEHOLDER:{clean_name}_{name_hash}"
        
        return placeholder_id
    
    def _calculate_got_metrics(self) -> Dict[str, Any]:
        """Calculate GoT framework metrics"""
        try:
            # Count thought types
            thought_distribution = {}
            total_thoughts = 0
            
            for step in self.execution_steps:
                step_type = step.step_type
                if step_type not in thought_distribution:
                    thought_distribution[step_type] = 0
                thought_distribution[step_type] += 1
                total_thoughts += 1
            
            # Calculate volume and latency
            volume = len([s for s in self.execution_steps if s.success])
            latency = len(set(s.step_type for s in self.execution_steps))
            
            # Calculate performance metrics
            total_time = sum(s.execution_time for s in self.execution_steps)
            avg_confidence = sum(s.confidence for s in self.execution_steps if s.confidence > 0) / max(1, len([s for s in self.execution_steps if s.confidence > 0]))
            
            # Quality improvement (simplified calculation)
            quality_scores = [s.confidence for s in self.execution_steps if s.confidence > 0]
            quality_improvement = max(quality_scores) / min(quality_scores) if len(quality_scores) > 1 else 1.0
            
            return {
                'volume': volume,
                'latency': latency,
                'total_thoughts': total_thoughts,
                'thought_distribution': thought_distribution,
                'total_execution_time': total_time,
                'average_confidence': avg_confidence,
                'quality_improvement': quality_improvement,
                'cost_reduction': 0.8,  # Estimated based on parallel execution
                'parallel_speedup': 1.2  # Estimated based on concurrent processing
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate GoT metrics: {str(e)}")
            return {}
    
    def _calculate_quality_score(self, results: List[Dict[str, Any]]) -> float:
        """Calculate overall quality score for results"""
        if not results:
            return 0.0
        
        try:
            scores = []
            for result in results:
                score = result.get('score', 0.0)
                if isinstance(score, (int, float)):
                    scores.append(float(score))
            
            if not scores:
                return 0.0
            
            # Calculate weighted average (higher weight for top results)
            weights = [1.0 / (i + 1) for i in range(len(scores))]
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            weight_sum = sum(weights)
            
            return weighted_sum / weight_sum if weight_sum > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate quality score: {str(e)}")
            return 0.0
    
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
        
        presentation = self.result_presenter.present_results(result)
        return result, presentation
    
    async def _save_results(self, result: QueryResult, presentation: str):
        """Save results to file"""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save JSON result
            json_filename = f"got_result_{timestamp}.json"
            with open(json_filename, 'w') as f:
                json.dump({
                    'query': result.query,
                    'final_answer': result.final_answer,
                    'success': result.success,
                    'total_execution_time': result.total_execution_time,
                    'entities_found': result.entities_found,
                    'total_results': result.total_results,
                    'quality_score': result.quality_score,
                    'got_metrics': result.got_metrics,
                    'execution_steps': [
                        {
                            'step_id': step.step_id,
                            'step_type': step.step_type,
                            'execution_time': step.execution_time,
                            'success': step.success,
                            'confidence': step.confidence,
                            'error_message': step.error_message
                        } for step in result.execution_steps
                    ]
                }, f, indent=2)
            
            # Save formatted presentation
            txt_filename = f"got_presentation_{timestamp}.txt"
            with open(txt_filename, 'w') as f:
                f.write(presentation)
            
            logger.info(f"Results saved to {json_filename} and {txt_filename}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
    
    async def close(self):
        """Clean up resources"""
        try:
            mcp_integration = get_mcp_integration()
            mcp_integration.close()
            logger.info("Production GoT optimizer closed")
        except Exception as e:
            logger.error(f"Error closing optimizer: {str(e)}")


# Convenience functions for easy use

async def execute_biomedical_query(query: str, config: Optional[ProductionConfig] = None) -> Tuple[QueryResult, str]:
    """
    Convenience function to execute a biomedical query with GoT optimization
    
    Args:
        query: The biomedical query to execute
        config: Optional configuration object
        
    Returns:
        Tuple of (QueryResult object, formatted presentation string)
    """
    optimizer = ProductionGoTOptimizer(config)
    try:
        return await optimizer.execute_query(query)
    finally:
        await optimizer.close()


def run_biomedical_query(query: str, config: Optional[ProductionConfig] = None) -> Tuple[QueryResult, str]:
    """
    Synchronous wrapper for executing biomedical queries
    
    Args:
        query: The biomedical query to execute
        config: Optional configuration object
        
    Returns:
        Tuple of (QueryResult object, formatted presentation string)
    """
    return asyncio.run(execute_biomedical_query(query, config))