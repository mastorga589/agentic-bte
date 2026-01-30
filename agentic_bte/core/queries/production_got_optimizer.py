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
    max_predicates_per_subquery: int = 2
    max_concurrent_predicate_calls: int = 3
    enable_evidence_scoring: bool = True
    
    # Follow-up subquery controls
    max_seed_drugs_for_followup: int = 20  # limit drugs passed from Q1 into Q2 to reduce blast radius


class ProductionGoTOptimizer:
    """
    Production-ready Graph of Thoughts optimizer for biomedical queries
    
    Integrates all GoT framework components with real MCP tools and provides
    comprehensive result presentation with debugging capabilities.
    """
    
    def __init__(self, config: Optional[ProductionConfig] = None,
                 entity_service: Optional["EntityExtractionService"] = None,
                 trapi_service: Optional["TrapiBuilderService"] = None,
                 bte_service: Optional["BteExecutionService"] = None):
        """
        Initialize production GoT optimizer
        
        Args:
            config: Configuration object for the optimizer
            entity_service: Optional service for entity extraction (decouples MCP)
            trapi_service: Optional service for TRAPI building (decouples MCP)
            bte_service: Optional service for BTE execution (decouples MCP)
        """
        self.config = config or ProductionConfig()
        
        # Optional decoupled services (may be MCP-backed or core-backed)
        self.entity_service = entity_service
        self.trapi_service = trapi_service
        self.bte_service = bte_service
        
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
        
        # Global entity cache to propagate IDs across subqueries
        self.global_entity_data: Dict[str, str] = {}
        # Per-subquery collected results (for seeding later subqueries)
        self.results_by_subquery: Dict[int, List[Dict[str, Any]]] = {}
        
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
            
            # Seed global entity cache with extracted entities
            try:
                for ent in entities:
                    name = ent.get('name')
                    eid = ent.get('id')
                    if name and eid:
                        self.global_entity_data[name] = eid
            except Exception as _:
                pass
            
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
            
            # Step 3: Result Aggregation and Refinement (skip gracefully if no results)
            if not all_results:
                logger.warning("No BTE results found; generating no-results final answer")
                final_results = []
            else:
                aggregation_step = await self._execute_aggregation_refinement(all_results, entities)
                self.execution_steps.append(aggregation_step)
                final_results = aggregation_step.output_data.get('final_results', all_results)
            
            # Step 4: Generate LLM-based final answer (always generate, even with zero results)
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
            
            # Print final answer to stdout (while still returning it)
            try:
                print("\n=== FINAL ANSWER ===\n")
                print(final_answer)
                print("\n====================\n")
            except Exception as e:
                logger.debug(f"Could not print final answer: {e}")
            
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
            # Call entity extraction via injected service if available; fallback to MCP tool
            if getattr(self, 'entity_service', None) is not None:
                entities, confidence, source = await self.entity_service.extract(query)
                response = {"entities": entities, "confidence": confidence, "source": source}
            else:
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
        trapi_queries = query_building_step.output_data.get('queries', [trapi_query] if trapi_query else [])
        
        # Step 2: Execute API call(s) over batches
        batch_results = []
        batch_mappings = {}
        total_time = 0.0
        total_count = 0
        for idx, tq in enumerate(trapi_queries):
            api_execution_step = await self._execute_bte_api(tq, f"{query_building_step.step_id}_batch_{idx+1}")
            steps.append(api_execution_step)
            if api_execution_step.success:
                res = api_execution_step.output_data.get('results', [])
                batch_results.extend(res)
                total_count += api_execution_step.output_data.get('total_results', len(res))
                batch_mappings.update(api_execution_step.output_data.get('entity_mappings', {}))
                total_time += api_execution_step.execution_time
        
        # Replace with a single aggregation-like step summarizing batches (optional)
        if len(trapi_queries) > 1:
            step_id = f"api_execution_{int(time.time())}_batches"
            summary_step = QueryStep(
                step_id=step_id,
                step_type='api_execution',
                input_data={'batch_count': len(trapi_queries)},
                output_data={
                    'results': batch_results,
                    'total_results': total_count,
                    'entity_mappings': batch_mappings,
                    'source': 'bte_api_batched'
                },
                trapi_query=trapi_queries[0],
                execution_time=total_time,
                success=True,
                confidence=(sum(r.get('score',0.0) for r in batch_results)/len(batch_results)) if batch_results else 0.0
            )
            steps.append(summary_step)
        
        return steps
    
    async def _execute_multiple_got_paths(self, query: str, entity_data: Dict[str, str],
                                        entities: List[Dict[str, Any]]) -> List[QueryStep]:
        """Execute multiple GoT paths for complex queries with scientifically sound subqueries"""
        steps = []
        
        # Generate scientifically meaningful subqueries based on the original query and entities
        subqueries = await self._generate_scientific_subqueries(query, entities)
        
        # Store subquery information for LLM explainability
        self.subquery_info = subqueries
        
        # Materialize basic plan JSON with node IDs and dependencies inferred via placeholders
        plan_nodes = []
        producer_indices_by_cat = {}
        for i, sq in enumerate(subqueries):
            node_id = f"Q{i+1}"
            text = sq['query']
            subj_cat = sq.get('subject_category')
            obj_cat = sq.get('object_category')
            deps = []
            # Heuristic dependency: if query references "these drugs", depend on last node producing SmallMolecule
            if 'these drugs' in text.lower() and producer_indices_by_cat.get('biolink:SmallMolecule') is not None:
                deps.append(f"Q{producer_indices_by_cat['biolink:SmallMolecule']+1}")
            if 'these genes' in text.lower() and producer_indices_by_cat.get('biolink:Gene') is not None:
                deps.append(f"Q{producer_indices_by_cat['biolink:Gene']+1}")
            plan_nodes.append({
                'id': node_id,
                'query': text,
                'subject_category': subj_cat,
                'object_category': obj_cat,
                'depends_on': deps,
                'rationale': sq.get('rationale','')
            })
            # Treat object category as produced entity type for downstream use (targets produce Gene, therapies produce SmallMolecule)
            if obj_cat:
                producer_indices_by_cat[obj_cat] = i
        self.current_plan = {'nodes': plan_nodes, 'edges': [
            {'from': n['depends_on'][0], 'to': n['id']} for n in plan_nodes if n.get('depends_on')
        ]}
        
        # User-facing plan summary
        try:
            print("\n=== QUERY PLAN ===")
            for n in plan_nodes:
                deps = ", ".join(n.get('depends_on', [])) or "none"
                print(f"{n['id']}: {n['query']}  [deps: {deps}]")
            print("===================\n")
        except Exception:
            pass
        
        # Execute each subquery with parallel predicate exploration in topo order
        def ready_nodes(nodes, completed):
            for n in nodes:
                if n['id'] in completed:
                    continue
                deps = set(n.get('depends_on', []))
                if deps.issubset(completed):
                    yield n
        completed = set()
        while len(completed) < len(plan_nodes):
            progressed = False
            for node in list(ready_nodes(plan_nodes, completed)):
                subquery = node['query']
                # Start update
                try:
                    print(f"→ Executing {node['id']}: {subquery}")
                except Exception:
                    pass
                # Use canonical subquery index from plan node ID (e.g., Q1 -> 0) to keep step IDs aligned with the original plan order
                try:
                    canonical_idx = int(str(node.get('id','')).lstrip('Q')) - 1
                    if canonical_idx < 0:
                        canonical_idx = len(completed)
                except Exception:
                    canonical_idx = len(completed)
                subquery_steps = await self._execute_subquery_with_parallel_predicates(
                    subquery, entities, canonical_idx
                )
                # Summarize node execution
                try:
                    api_steps = [s for s in subquery_steps if s.step_type == 'api_execution']
                    total_res = sum(s.output_data.get('total_results', 0) for s in api_steps if s.output_data)
                    failures = len([s for s in api_steps if not s.success])
                    print(f"✓ Completed {node['id']}: results={total_res}, api_calls={len(api_steps)}, failures={failures}")
                    # Show brief entity cache size if available
                    if isinstance(getattr(self, 'global_entity_data', None), dict):
                        print(f"  Entities known: {len(self.global_entity_data)}")
                except Exception:
                    pass
                steps.extend(subquery_steps)
                completed.add(node['id'])
                progressed = True
            # Progress update
            try:
                print(f"Plan progress: {len(completed)}/{len(plan_nodes)} nodes done\n")
            except Exception:
                pass
            if not progressed:
                # Break deadlock if any
                break
        
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
            
            # Create a strict prompt enforcing atomic single-hop subqueries
            # Build a dynamic coverage summary from meta-KG
            coverage_text = ""
            try:
                meta = self.bte_client.get_meta_knowledge_graph()
                pairs = {}
                for e in meta.get('edges', []):
                    sc, oc = e.get('subject'), e.get('object')
                    if sc and oc:
                        pairs[(sc, oc)] = pairs.get((sc, oc), 0) + 1
                top = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:10]
                lines = [f"- {sc} -> {oc} (support={cnt})" for (sc, oc), cnt in top]
                coverage_text = "\n".join(lines)
            except Exception:
                coverage_text = "- biolink:SmallMolecule -> biolink:Disease\n- biolink:SmallMolecule -> biolink:Gene\n- biolink:Gene -> biolink:BiologicalProcess\n- biolink:Gene -> biolink:Disease\n- biolink:Disease -> biolink:BiologicalProcess"
            
            # Use placeholder tokens to avoid brace conflicts in f-strings
            decomposition_prompt = """
You are a medicinal chemistry and pharmaceutical sciences expert acting as a biomedical query planner. Think mechanistically and design an execution plan composed of ATOMIC SINGLE-HOP subqueries that a TRAPI-compliant system can answer.

Original Query: "<<QUERY>>"

Extracted Biomedical Entities (context only; do NOT restrict subqueries to these terms):
<<ENTITIES>>

Supported subject→object templates observed in the target knowledge graph (meta-KG):
<<COVERAGE>>

GUIDELINES:
- Each subquery MUST be a single-hop relation between exactly TWO node types.
- Use Biolink categories for nodes; choose from: biolink:SmallMolecule, biolink:Gene, biolink:Protein, biolink:Disease, biolink:BiologicalProcess, biolink:PhenotypicFeature.
- Prefer mechanistic chains when appropriate (therapy → targets → processes), using placeholders like "these drugs" and "these genes" to reference prior results.
- Avoid vague "how" questions. Use concrete forms: "Which X [predicate] Y?" or "Which X are [predicate] of Y?".
- Do NOT include pathway/pathways unless mapped to biolink:BiologicalProcess.
- Generate 3–5 subqueries that together cover therapy, targets, and processes relevant to the original question.

OUTPUT FORMAT (JSON array). Each item MUST have:
- "query": clear single-hop question (no multi-hop, no "how")
- "subject_category": one Biolink category
- "object_category": one Biolink category
- "predicate_hint": optional string (Biolink predicate consistent with the categories)
- "rationale": brief why it helps

Example:
[
  {
    "query": "Which small molecules treat Parkinson's disease?",
    "subject_category": "biolink:SmallMolecule",
    "object_category": "biolink:Disease",
    "predicate_hint": "biolink:treats",
    "rationale": "Therapeutics directly connected to PD"
  },
  {
    "query": "Which genes do these drugs target?",
    "subject_category": "biolink:SmallMolecule",
    "object_category": "biolink:Gene",
    "predicate_hint": "biolink:targets",
    "rationale": "Mechanistic targets of the identified therapeutics"
  },
  {
    "query": "Which biological processes are these genes involved in?",
    "subject_category": "biolink:Gene",
    "object_category": "biolink:BiologicalProcess",
    "predicate_hint": "biolink:participates_in",
    "rationale": "Downstream processes explaining mechanism of action"
  }
]
"""
            decomposition_prompt = decomposition_prompt.replace("<<QUERY>>", query).replace("<<ENTITIES>>", entities_text).replace("<<COVERAGE>>", coverage_text)
            
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
            
            # Convert to expected format and ENFORCE single-hop structure
            allowed_cats = {
                'biolink:SmallMolecule', 'biolink:Gene', 'biolink:Protein', 'biolink:Disease', 'biolink:BiologicalProcess', 'biolink:PhenotypicFeature'
            }
            subqueries = []
            for sq_data in subquery_data:
                if not isinstance(sq_data, dict) or 'query' not in sq_data or 'subject_category' not in sq_data or 'object_category' not in sq_data:
                    logger.warning(f"Invalid subquery format: {sq_data}")
                    continue
                subquery_text = sq_data['query']
                subj_cat = sq_data.get('subject_category')
                obj_cat = sq_data.get('object_category')
                rationale = sq_data.get('rationale', '')
                if subj_cat not in allowed_cats or obj_cat not in allowed_cats:
                    logger.warning(f"Unsupported categories in subquery: {subj_cat} -> {obj_cat}")
                    continue
                # Reject vague 'how' subqueries
                if re.search(r"\bhow\b", subquery_text.lower()):
                    logger.warning(f"Rejecting non-atomic 'how' subquery: {subquery_text}")
                    continue
                subqueries.append({
                    'query': subquery_text,
                    'subject_category': subj_cat,
                    'object_category': obj_cat,
                    'rationale': rationale
                })
            
            # Ensure we have at least one subquery
            if not subqueries:
                logger.warning("LLM generated no valid subqueries, using fallback")
                return self._fallback_subquery_generation(query, entities)
            
            logger.info(f"LLM generated {len(subqueries)} scientific subqueries from: {query}")
            for i, sq in enumerate(subqueries, 1):
                logger.info(f"  Subquery {i}: {sq['query']}")
                logger.info(f"    Types: {sq.get('subject_category')} -> {sq.get('object_category')}")
                logger.info(f"    Rationale: {sq.get('rationale', 'N/A')}")
            
            # FILTER: Keep only subqueries whose subject→object category pair is supported by the meta-KG
            try:
                supported, dropped = [], []
                for sq in subqueries:
                    sc = sq.get('subject_category'); oc = sq.get('object_category')
                    if self._meta_pair_supported(sc, oc):
                        supported.append(sq)
                    else:
                        dropped.append(sq)
                if dropped:
                    logger.warning(f"Filtered {len(dropped)} unsupported subqueries (no meta-KG coverage):")
                    for d in dropped:
                        logger.warning(f"  DROP: {d.get('subject_category')} -> {d.get('object_category')} | {d.get('query')}")
                if not supported:
                    logger.warning("All LLM subqueries lacked meta-KG support; using fallback generator")
                    return self._fallback_subquery_generation(query, entities)
                logger.info(f"Proceeding with {len(supported)} meta-KG-supported subqueries")
                return supported
            except Exception as _:
                # If anything goes wrong, return original set (better to proceed than fail)
                return subqueries
            
        except Exception as e:
            logger.error(f"Error in LLM-based subquery generation: {str(e)}")
            return self._fallback_subquery_generation(query, entities)
    
    def _fallback_subquery_generation(self, query: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback method for subquery generation when LLM fails — generate atomic single-hop subqueries.
        """
        logger.info("Using fallback subquery generation (atomic single-hop)")
        # Helper to check meta-KG coverage
        def has_coverage(subj_cat: str, obj_cat: str, intent: str) -> bool:
            try:
                if not self.predicate_selector:
                    return True
                from ..knowledge.predicate_strategy import QueryIntent
                qi = QueryIntent(intent)
                preds = self.predicate_selector.select_predicates(qi, subj_cat, obj_cat)
                return bool(preds)
            except Exception:
                return True
        
        # Extract convenient lookups
        by_type = {}
        for e in entities:
            by_type.setdefault(e.get('type','Unknown'), []).append(e)
        subs: List[Dict[str, Any]] = []
        
        # If we have a disease, seed with precise disease name
        disease_name = by_type.get('Disease', [{}])[0].get('name', 'the disease') if by_type.get('Disease') else None
        if disease_name and has_coverage('biolink:SmallMolecule','biolink:Disease','therapeutic'):
            subs.append({
                'query': f"Which small molecules treat {disease_name}?",
                'subject_category': 'biolink:SmallMolecule',
                'object_category': 'biolink:Disease',
                'rationale': 'Therapeutics directly connected to the disease'
            })
        
        # Mechanistic chain: drugs -> targets -> processes
        # If a specific small molecule already in entities, use it; else refer to "these drugs" (populated after first subquery)
        sm_entities = by_type.get('SmallMolecule', [])
        if sm_entities:
            sm_name = sm_entities[0].get('name','the drug')
            if has_coverage('biolink:SmallMolecule','biolink:Gene','target'):
                subs.append({
                    'query': f"Which genes does {sm_name} target?",
                    'subject_category': 'biolink:SmallMolecule',
                    'object_category': 'biolink:Gene',
                    'rationale': f'Molecular targets of {sm_name}'
                })
            if has_coverage('biolink:Gene','biolink:BiologicalProcess','mechanism'):
                subs.append({
                    'query': "Which biological processes are these genes involved in?",
                    'subject_category': 'biolink:Gene',
                    'object_category': 'biolink:BiologicalProcess',
                    'rationale': 'Mechanistic processes downstream of targets'
                })
        else:
            # No explicit drug yet; plan to use results from first query
            if has_coverage('biolink:SmallMolecule','biolink:Gene','target'):
                subs.append({
                    'query': "Which genes do these drugs target?",
                    'subject_category': 'biolink:SmallMolecule',
                    'object_category': 'biolink:Gene',
                    'rationale': 'Molecular targets of therapeutics identified in prior step'
                })
            if has_coverage('biolink:Gene','biolink:BiologicalProcess','mechanism'):
                subs.append({
                    'query': "Which biological processes are these genes involved in?",
                    'subject_category': 'biolink:Gene',
                    'object_category': 'biolink:BiologicalProcess',
                    'rationale': 'Mechanistic processes downstream of targets'
                })
        
        # Disease -> BiologicalProcess if covered and still relevant
        if disease_name and has_coverage('biolink:Disease','biolink:BiologicalProcess','mechanism'):
            subs.append({
                'query': f"Which biological processes are associated with {disease_name}?",
                'subject_category': 'biolink:Disease',
                'object_category': 'biolink:BiologicalProcess',
                'rationale': 'Processes implicated in the disease'
            })
        
        # Ensure at least one subquery
        if not subs:
            subs.append({
                'query': "Which small molecules treat the disease?",
                'subject_category': 'biolink:SmallMolecule',
                'object_category': 'biolink:Disease',
                'rationale': 'Generic therapeutic relation'
            })
        return subs
    
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
        # Merge in the global entity_data cache so TRAPI builder LLM sees latest IDs
        if isinstance(self.global_entity_data, dict) and self.global_entity_data:
            entity_data = {**self.global_entity_data, **(entity_data or {})}
        
        # Limit SmallMolecule seeds for follow-up subqueries using top-N from previous results
        try:
            if subquery_index > 0 and self.results_by_subquery.get(subquery_index - 1):
                prev_results = self.results_by_subquery[subquery_index - 1]
                # Gather (name, id, score) for small molecules from previous results
                preferred_prefixes = ("CHEBI:", "ChEMBL:", "DRUGBANK:", "UNII:", "PUBCHEM", "NCIT:")
                triples = []
                for r in prev_results:
                    sid = r.get('subject_id')
                    sname = r.get('subject')
                    stype = (r.get('subject_type') or '').lower()
                    score = r.get('score', 0.0)
                    if sid and sname and (('smallmolecule' in stype) or any(str(sid).startswith(p) for p in preferred_prefixes)):
                        triples.append((sname, sid, score))
                # Sort by score desc and dedupe by id
                triples.sort(key=lambda t: t[2] if isinstance(t[2], (int, float)) else 0.0, reverse=True)
                seen_ids = set()
                top = []
                for name, cid, sc in triples:
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        top.append((name, cid))
                    if len(top) >= self.config.max_seed_drugs_for_followup:
                        break
                if top:
                    # Rebuild entity_data so that only top drug IDs remain for SmallMolecule prefixes; keep all non-drug entries
                    filtered = {}
                    allow_ids = {cid for _, cid in top}
                    for name, cid in (entity_data or {}).items():
                        if not isinstance(cid, str):
                            continue
                        if any(cid.startswith(p) for p in preferred_prefixes):
                            # drug-like id
                            if cid in allow_ids:
                                filtered[name] = cid
                        else:
                            filtered[name] = cid
                    # Ensure we include names for selected top drugs even if name missing in entity_data
                    for name, cid in top:
                        filtered[name] = cid
                    entity_data = filtered
                    logger.info(f"Restricted SmallMolecule seeds to {len(allow_ids)} for subquery {subquery_index+1}")
        except Exception as e:
            logger.debug(f"Unable to restrict follow-up seeds: {e}")
        
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
        # Use pre-batched queries when available (hard cap at 50 IDs per batch)
        base_trapi_queries = base_query_step.output_data.get('queries', [base_trapi_query] if base_trapi_query else [])
        
        # Determine query intent and categories for predicate selection
        if self.config.enable_parallel_predicates and self.predicate_selector:
            try:
                # Detect query intent
                from ..knowledge.predicate_strategy import QueryIntent
                query_intent = self.predicate_selector.detect_query_intent(subquery, entities)
                
                # Extract categories present in TRAPI query
                query_graph = base_trapi_query.get('message', {}).get('query_graph', {})
                nodes = query_graph.get('nodes', {})
                available_categories = [
                    (nid, (ndata.get('categories') or [None])[0], ndata) for nid, ndata in nodes.items()
                ]
                
                # Heuristically choose subject/object categories based on intent and meta-KG coverage
                def choose_category_pair() -> Optional[tuple[str, str]]:
                    cat_set = {c for _, c, _ in available_categories if c}
                    intents = []
                    if query_intent.name.lower() in ('therapeutic',):
                        intents.append(('biolink:SmallMolecule', 'biolink:Disease'))
                    if query_intent.name.lower() in ('mechanism', 'target'):
                        intents.append(('biolink:SmallMolecule', 'biolink:Gene'))
                        intents.append(('biolink:Gene', 'biolink:BiologicalProcess'))
                    if query_intent.name.lower() in ('genetic',):
                        intents.append(('biolink:Gene', 'biolink:Disease'))
                    # Generic fallbacks: try all ordered pairs
                    for sc, oc in intents:
                        if sc in cat_set and oc in cat_set:
                            preds = self.predicate_selector.select_predicates(query_intent, sc, oc)
                            if preds:
                                return sc, oc
                    for sc in cat_set:
                        for oc in cat_set:
                            if sc == oc:
                                continue
                            preds = self.predicate_selector.select_predicates(query_intent, sc, oc)
                            if preds:
                                return sc, oc
                    return None
                
                pair = choose_category_pair()
                if pair:
                    subject_category, object_category = pair
                    # Select predicates using our strategy for the chosen pair
                    selected_predicates = self.predicate_selector.select_predicates(
                        query_intent, subject_category, object_category
                    )
                    # Enforce domain-preferred predicates where applicable
                    try:
                        preferred = []
                        # SmallMolecule -> Gene: prefer targets/affects
                        if subject_category == 'biolink:SmallMolecule' and object_category == 'biolink:Gene':
                            preferred = ['biolink:targets', 'biolink:affects']
                        # Gene -> BiologicalProcess: prefer participates_in
                        if subject_category == 'biolink:Gene' and object_category == 'biolink:BiologicalProcess':
                            preferred = ['biolink:participates_in']
                        if preferred:
                            # Keep only those with meta-KG support
                            supported = []
                            for p in preferred:
                                try:
                                    from ..knowledge.predicate_strategy import PredicateSelector
                                except Exception:
                                    pass
                                # Use existing selector support check
                                if self.predicate_selector.get_predicate_support(subject_category, p, object_category) > 0:
                                    supported.append(p)
                            # Bring supported preferred to the front if present in meta-KG
                            if supported:
                                # Dedup while maintaining priority order
                                existing = [pred for pred, _ in selected_predicates]
                                new_order = supported + [p for p in existing if p not in supported]
                                # Rebuild with original priorities roughly preserved and enforce hard cap here
                                selected_predicates = [(p, 1.0 - 0.05*i) for i, p in enumerate(new_order[:self.config.max_predicates_per_subquery])]
                                logger.info(f"Applied preferred predicates for {subject_category}->{object_category}: {supported}")
                    except Exception as e:
                        logger.debug(f"Predicate preference application skipped: {e}")
                    
                    # Final cap: ensure we try at most max_predicates_per_subquery
                    try:
                        if selected_predicates:
                            # Normalize shape to list[(pred, priority)] and trim
                            norm = []
                            for item in selected_predicates:
                                if isinstance(item, (list, tuple)) and len(item) >= 1:
                                    pred = item[0]
                                    prio = float(item[1]) if len(item) > 1 and isinstance(item[1], (int, float)) else 1.0
                                    norm.append((pred, prio))
                                elif isinstance(item, str):
                                    norm.append((item, 1.0))
                            selected_predicates = norm[: self.config.max_predicates_per_subquery]
                    except Exception:
                        # On any issue, fall back to first N by naive slicing
                        selected_predicates = selected_predicates[: self.config.max_predicates_per_subquery]
                    
                    logger.info(f"Selected {len(selected_predicates)} predicates for {subject_category} → {object_category}: {[p[0] for p in selected_predicates]}")
                    
                    # Execute TRAPI queries in parallel with different predicates, enforcing single-hop
                    parallel_steps = await self._execute_parallel_predicate_queries(
                        base_trapi_queries, selected_predicates, query_intent, subquery_index,
                        subject_category, object_category
                    )
                    steps.extend(parallel_steps)
                else:
                    logger.warning("Could not determine a valid subject/object category pair for predicate selection")
                    # Fall back to single execution with batching support
                    total_time = 0.0
                    batch_results = []
                    batch_mappings = {}
                    for idx, tq in enumerate(base_trapi_queries):
                        api_step = await self._execute_bte_api(tq, base_query_step.step_id)
                        api_step.step_id = f"{api_step.step_id}_subq_{subquery_index}_batch_{idx+1}"
                        steps.append(api_step)
                        if api_step.success:
                            res = api_step.output_data.get('results', [])
                            batch_results.extend(res)
                            batch_mappings.update(api_step.output_data.get('entity_mappings', {}))
                            total_time += api_step.execution_time
                    
            except Exception as e:
                logger.error(f"Error in parallel predicate execution: {e}")
                # Fall back to single execution
                api_step = await self._execute_bte_api(base_trapi_query, base_query_step.step_id)
                api_step.step_id = f"{api_step.step_id}_subq_{subquery_index}"
                steps.append(api_step)
        else:
            # Standard single execution with batching support
            total_time = 0.0
            batch_results = []
            batch_mappings = {}
            if not base_trapi_queries:
                base_trapi_queries = [base_trapi_query] if base_trapi_query else []
            for idx, tq in enumerate(base_trapi_queries):
                api_step = await self._execute_bte_api(tq, base_query_step.step_id)
                api_step.step_id = f"{api_step.step_id}_subq_{subquery_index}_batch_{idx+1}"
                steps.append(api_step)
                if api_step.success:
                    res = api_step.output_data.get('results', [])
                    batch_results.extend(res)
                    batch_mappings.update(api_step.output_data.get('entity_mappings', {}))
                    total_time += api_step.execution_time
        
        # Persist results from this subquery for use by subsequent subqueries
        try:
            all_subq_results = []
            for st in steps:
                if getattr(st, 'step_type', '') == 'api_execution' and getattr(st, 'success', False):
                    all_subq_results.extend((st.output_data or {}).get('results', []) or [])
            self.results_by_subquery[subquery_index] = all_subq_results
            logger.info(f"Stored {len(all_subq_results)} results from subquery {subquery_index+1} for follow-up seeding")
        except Exception:
            pass
        
        return steps
    
    async def _execute_parallel_predicate_queries(self, base_trapi_query: Dict[str, Any], 
                                                 selected_predicates: List[Tuple[str, float]], 
                                                 query_intent, subquery_index: int,
                                                 subject_category: Optional[str] = None,
                                                 object_category: Optional[str] = None) -> List[QueryStep]:
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
        
        # Create TRAPI queries with different predicates, enforcing single-hop
        predicate_queries = []
        
        # Prepare IDs from base query if available and matching categories
        def extract_id_for_category(nodes: Dict[str, Any], category: str) -> Optional[str]:
            for nid, ndata in nodes.items():
                cats = ndata.get('categories', [])
                if cats and cats[0] == category:
                    ids = ndata.get('ids', [])
                    if isinstance(ids, list) and ids:
                        return ids[0]
            return None
        
        # Support list or dict for base query input
        sample_query = (base_trapi_query[0] if isinstance(base_trapi_query, list) and base_trapi_query else base_trapi_query)
        nodes_in_base = (sample_query.get('message', {}).get('query_graph', {}).get('nodes', {})
                         if isinstance(sample_query, dict) else {})
        subj_id = extract_id_for_category(nodes_in_base, subject_category) if subject_category else None
        obj_id = extract_id_for_category(nodes_in_base, object_category) if object_category else None
        
        
        # Helper: collect many IDs from global cache for batch execution
        def collect_ids_for_category(category: Optional[str], max_ids: int = 100) -> list[str]:
            if not category:
                return []
            prefix_map = {
                'biolink:Disease': ['MONDO:', 'DOID:', 'UMLS:'],
                'biolink:SmallMolecule': ['CHEBI:', 'ChEMBL:', 'DRUGBANK:'],
                'biolink:Gene': ['HGNC:', 'NCBIGene:', 'ENSEMBL:'],
                'biolink:Protein': ['PR:', 'UniProtKB:'],
                'biolink:BiologicalProcess': ['GO:']
            }
            prefixes = prefix_map.get(category, [])
            ids = []
            for name, cid in (self.global_entity_data or {}).items():
                if any(str(cid).startswith(p) for p in prefixes):
                    ids.append(str(cid))
                    if len(ids) >= max_ids:
                        break
            # Deduplicate while preserving order
            deduped = []
            seen = set()
            for cid in ids:
                if cid not in seen:
                    seen.add(cid)
                    deduped.append(cid)
            return deduped
        
        # Accept base_trapi_query as single dict or list of batch queries
        base_batches = base_trapi_query if isinstance(base_trapi_query, list) else [base_trapi_query]
        for predicate, priority in selected_predicates:
            modified_batches = []
            for base_q in base_batches:
                mq = deepcopy(base_q)
                qg = mq.get('message', {}).get('query_graph', {})
                edges = qg.get('edges', {})
                if edges:
                    edge_key = next(iter(edges.keys()))
                    edges[edge_key]['predicates'] = [predicate]
                modified_batches.append(mq)
            predicate_queries.append((modified_batches, predicate, priority))
        
        # Execute queries concurrently with limited concurrency
        semaphore = asyncio.Semaphore(self.config.max_concurrent_predicate_calls)
        
        async def execute_single_predicate_query(query_predicate_priority):
            trapi_queries, predicate, priority = query_predicate_priority
            
            # Queue log (outside semaphore)
            try:
                logger.info(f"[parallel] Queued predicate={predicate} batches={len(trapi_queries)} subq={subquery_index+1}")
            except Exception:
                pass
            
            async with semaphore:
                # Start log on acquire
                try:
                    logger.info(f"[parallel] Started predicate={predicate} with {len(trapi_queries)} batch(es) subq={subquery_index+1}")
                except Exception:
                    pass
                try:
                    aggregated_results = []
                    aggregated_mappings = {}
                    messages = []
                    total_time = 0.0
                    for bidx, tq in enumerate(trapi_queries, start=1):
                        t0 = time.time()
                        try:
                            logger.info(f"[parallel] → sending predicate={predicate} batch {bidx}/{len(trapi_queries)} subq={subquery_index+1}")
                        except Exception:
                            pass
                        if getattr(self, 'bte_service', None) is not None:
                            res_list, mappings, metadata = await self.bte_service.execute(
                                trapi_query=tq,
                                k=10,
                                max_results=100,
                                predicate=predicate,
                                query_intent=getattr(query_intent, 'value', None)
                            )
                            response = {"results": res_list, "entity_mappings": mappings, "metadata": metadata}
                        else:
                            response = await call_mcp_tool(
                                "call_bte_api",
                                json_query=tq,
                                k=10,
                                maxresults=100,
                                predicate=predicate,
                                query_intent=query_intent.value
                            )
                        dt = time.time() - t0
                        res = response.get('results', [])
                        aggregated_results.extend(res)
                        aggregated_mappings.update(response.get('entity_mappings', {}))
                        total_time += dt
                        messages.append(response.get('metadata', {}).get('message', 'ok'))
                        try:
                            logger.info(f"[parallel] ← completed predicate={predicate} batch {bidx}/{len(trapi_queries)} results={len(res)} time={dt:.2f}s subq={subquery_index+1}")
                        except Exception:
                            pass
                    try:
                        logger.info(f"[parallel] Finished predicate={predicate} total_results={len(aggregated_results)} total_time={total_time:.2f}s subq={subquery_index+1}")
                    except Exception:
                        pass
                    return predicate, priority, {
                        'results': aggregated_results,
                        'entity_mappings': aggregated_mappings,
                        'metadata': {'message': "; ".join([m for m in messages if m])}
                    }, None
                except Exception as e:
                    logger.warning(f"Predicate {predicate} execution failed: {e}")
                    return predicate, priority, None, str(e)
        
        # Execute all predicate queries concurrently
        logger.info(f"Executing {len(predicate_queries)} predicate queries concurrently (semaphore={self.config.max_concurrent_predicate_calls})")
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
                original_results = response.get('results', [])
                metadata = response.get('metadata', {})
                entity_mappings = response.get('entity_mappings', {})
                # Build quick ID->name map from name->id mappings we received
                id_to_name = {}
                try:
                    id_to_name = {v: k for k, v in entity_mappings.items() if isinstance(k, str) and isinstance(v, str)}
                except Exception:
                    id_to_name = {}
                
# Enhance results to include compact knowledge_graph nodes/edges (consistent with _execute_bte_api)
                results_data = []
                for result in original_results:
                    enhanced = {
                        'score': result.get('score', 0.0),
                        'knowledge_graph': {
                            'nodes': {},
                            'edges': {}
                        },
                        'source_subquery': subquery_index + 1,
                        'source_predicate': predicate
                    }
                    # Nodes
                    if result.get('subject_id'):
                        subj_id = result.get('subject_id')
                        subj_name = id_to_name.get(subj_id, result.get('subject'))
                        enhanced['knowledge_graph']['nodes'][subj_id] = {
                            'name': subj_name,
                            'categories': [result.get('subject_type', 'biolink:NamedThing')]
                        }
                    if result.get('object_id'):
                        obj_id = result.get('object_id')
                        obj_name = id_to_name.get(obj_id, result.get('object'))
                        enhanced['knowledge_graph']['nodes'][obj_id] = {
                            'name': obj_name,
                            'categories': [result.get('object_type', 'biolink:NamedThing')]
                        }
                    # Edge
                    if result.get('subject_id') and result.get('object_id') and (result.get('predicate') or predicate):
                        edge_id = f"edge_{len(enhanced['knowledge_graph']['edges'])}"
                        enhanced['knowledge_graph']['edges'][edge_id] = {
                            'subject': result.get('subject_id'),
                            'object': result.get('object_id'),
                            'predicate': result.get('predicate', predicate)
                        }
                    # Merge original
                    enhanced.update(result)
                    results_data.append(enhanced)
                
                step_results = len(results_data)
                total_results += step_results
                
                # Calculate confidence from results (evidence-weighted scores if present)
                avg_confidence = 0.0
                if results_data:
                    scores = [r.get('score', 0.0) for r in results_data]
                    avg_confidence = sum(scores) / len(scores) if scores else 0.0
                
                # Update global entity cache with mappings and discovered nodes
                try:
                    if isinstance(entity_mappings, dict):
                        for k, v in entity_mappings.items():
                            if k and v:
                                self.global_entity_data[str(k)] = str(v)
                    for r in results_data:
                        if r.get('subject') and r.get('subject_id'):
                            self.global_entity_data[str(r['subject'])] = str(r['subject_id'])
                        if r.get('object') and r.get('object_id'):
                            self.global_entity_data[str(r['object'])] = str(r['object_id'])
                        kg_nodes = (r.get('knowledge_graph') or {}).get('nodes', {})
                        for node_id, node_data in kg_nodes.items():
                            node_name = node_data.get('name')
                            if node_name and node_id:
                                self.global_entity_data[str(node_name)] = str(node_id)
                except Exception:
                    pass
                
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
            # Build TRAPI via injected service if available; fallback to MCP tool
            # Use the global entity ID registry accumulated across steps (avoid lossy per-step filtering)
            if getattr(self, 'trapi_service', None) is not None:
                trapi_query, trapi_queries, confidence, source = await self.trapi_service.build(
                    query=query,
                    entity_data=self.global_entity_data or entity_data,
                    failed_trapis=[]
                )
                response = {"query": trapi_query, "queries": trapi_queries, "confidence": confidence, "source": source}
            else:
                response = await call_mcp_tool(
                    "build_trapi_query",
                    query=query,
                    entity_data=self.global_entity_data or entity_data,
                    failed_trapis=[]
                )
            
            trapi_query = response.get('query', {})
            trapi_queries = response.get('queries', [trapi_query] if trapi_query else [])
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
                    'queries': trapi_queries,
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
            # Execute BTE via injected service if available; fallback to MCP tool
            if getattr(self, 'bte_service', None) is not None:
                results, entity_mappings, metadata = await self.bte_service.execute(
                    trapi_query=trapi_query,
                    k=10,
                    max_results=100,
                )
                response = {"results": results, "entity_mappings": entity_mappings, "metadata": metadata}
            else:
                response = await call_mcp_tool(
                    "call_bte_api",
                    json_query=trapi_query,
                    k=10,  # Request more results
                    maxresults=100
                )
            
            results = response.get('results', [])
            metadata = response.get('metadata', {})
            entity_mappings = response.get('entity_mappings', {})
            # Build quick ID->name map from name->id mappings we received
            id_to_name = {}
            try:
                id_to_name = {v: k for k, v in entity_mappings.items() if isinstance(k, str) and isinstance(v, str)}
            except Exception:
                id_to_name = {}
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
                if result.get('subject_id'):
                    subj_id = result.get('subject_id')
                    subj_name = id_to_name.get(subj_id, result.get('subject'))
                    enhanced_result['knowledge_graph']['nodes'][subj_id] = {
                        'name': subj_name,
                        'categories': [result.get('subject_type', 'biolink:NamedThing')]
                    }
                
                if result.get('object_id'):
                    obj_id = result.get('object_id')
                    obj_name = id_to_name.get(obj_id, result.get('object'))
                    enhanced_result['knowledge_graph']['nodes'][obj_id] = {
                        'name': obj_name,
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
            
            # Update global entity cache with mappings and discovered nodes
            try:
                if isinstance(entity_mappings, dict):
                    for k, v in entity_mappings.items():
                        if k and v:
                            self.global_entity_data[str(k)] = str(v)
                for r in enhanced_results:
                    if r.get('subject') and r.get('subject_id'):
                        self.global_entity_data[str(r['subject'])] = str(r['subject_id'])
                    if r.get('object') and r.get('object_id'):
                        self.global_entity_data[str(r['object'])] = str(r['object_id'])
                    kg_nodes = (r.get('knowledge_graph') or {}).get('nodes', {})
                    for node_id, node_data in kg_nodes.items():
                        node_name = node_data.get('name')
                        if node_name and node_id:
                            self.global_entity_data[str(node_name)] = str(node_id)
            except Exception:
                pass
            
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
        
        # Helper to filter generic/noise names that should not seed TRAPI IDs
        def _is_generic_noise(name: str) -> bool:
            if not isinstance(name, str):
                return True
            n = name.strip().lower()
            generic_terms = {
                'drug', 'drugs', 'treat', 'treatment', 'target', 'targets', 'targeting', 'these', 'those',
                'response', 'phrase', 'assessed', 'entity', 'entities', 'molecule', 'small molecule',
                'enumerate 5 drugs'
            }
            if n in generic_terms:
                return True
            if '**' in n:
                return True
            if n.startswith('["**') or n.startswith('[\"**'):
                return True
            return False

        # Start with original entities
        for entity in original_entities:
            name = entity.get('name', '')
            entity_id = entity.get('id', '')
            if name and entity_id and not _is_generic_noise(name):
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
    
    def _meta_pair_supported(self, subject_category: Optional[str], object_category: Optional[str]) -> bool:
        """Return True if meta-KG lists any edge for subject_category -> object_category."""
        try:
            if not subject_category or not object_category:
                return False
            # Prefer predicate_selector if available; otherwise scan meta-KG directly
            try:
                meta = self.bte_client.meta_kg or self.bte_client.get_meta_knowledge_graph()
            except Exception:
                return True  # Do not block if meta-KG unavailable
            edges = meta.get('edges', []) if isinstance(meta, dict) else []
            for e in edges:
                if e.get('subject') == subject_category and e.get('object') == object_category:
                    return True
            return False
        except Exception:
            return True
    
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