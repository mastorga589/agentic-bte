import asyncio
import networkx as nx
from typing import List, Dict, Optional, Any
import os
import json
import logging
import traceback
from datetime import datetime
from rdflib import Graph, URIRef, Namespace, Literal
from rdflib.namespace import RDF, RDFS

# Setup comprehensive logging for GoT planner
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create formatter for detailed logging
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)

# Add handler if not already present
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from langchain_openai import ChatOpenAI
from agentic_bte.config.settings import get_settings
from agentic_bte.core.knowledge.bte_client import BTEClient
from agentic_bte.unified.execution_engine import UnifiedExecutionEngine
from agentic_bte.unified.config import UnifiedConfig
from agentic_bte.unified.knowledge_manager import UnifiedKnowledgeManager
from agentic_bte.core.knowledge.knowledge_system import BiomedicalKnowledgeSystem
from agentic_bte.core.knowledge.predicate_strategy import QueryIntent

LOG_DIR = os.path.join(os.getcwd(), "debug_output")
os.makedirs(LOG_DIR, exist_ok=True)
def write_llm_log(filename, content):
    path = os.path.join(LOG_DIR, filename)
    with open(path, "w") as f:
        f.write(str(content))
    print(f"[GoTPlanner][DEBUG] Output written to {path}", flush=True)

LANGGRAPH_PLANNING_PROMPT = """
You are an expert biomedical LLM agent. Decompose the following biomedical research question into its atomic reasoning subquestions, expressing the output ONLY in the following valid JSON format—with no explanations or extra commentary.

Your output must be a JSON dictionary with two fields:
{{
  "nodes": [
    {{"id": "Q1", "content": "<first atomic subquestion>", "dependencies": []}},
    {{"id": "Q2", "content": "<second subquestion>", "dependencies": ["Q1"]}} ,
    ...
  ],
  "edges": [{{"from": "Q1", "to": "Q2"}}, ...]
}}

Example:
Q: Which drugs can treat Crohn's disease by modifying the immune response?
Output:
{{
  "nodes": [
    {{"id": "Q1", "content": "Which drugs can treat Crohn's disease?", "dependencies": []}},
    {{"id": "Q2", "content": "Which of these drugs modify the immune response?", "dependencies": ["Q1"]}}
  ],
  "edges": [{{"from": "Q1", "to": "Q2"}}]
}}

Biomedical question to decompose: {query}
"""

LANGGRAPH_ANSWERING_PROMPT = """
You are a biomedical summarizer. Given the following findings/subanswers:
{context}
Provide a single, detailed but concise scientific answer to the main question: "{query}"
Be specific, mechanistic, cite scientific entities if possible, and never invent facts.
"""

class GPT41GoTLLM:
    def __init__(self, model="gpt-4-1106-preview", temperature=0.2):
        settings = get_settings()
        self.llm = ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key,
            temperature=temperature,
            max_tokens=2048
        )
    async def decompose_to_graph(self, user_query):
        user_prompt = LANGGRAPH_PLANNING_PROMPT.format(query=user_query)
        completion = await self.llm.ainvoke([{"role": "user", "content": user_prompt}])
        print("\n[GoTPlanner][ABSOLUTE RAW LLM OUTPUT]:\n", completion.content, flush=True)
        write_llm_log("gotplanner_llm_output_latest.txt", completion.content)
        return completion.content
    async def execute_thought(self, content):
        completion = await self.llm.ainvoke([{"role": "user", "content": content}])
        return completion.content, 1.0
    async def expand_thought(self, content):
        return {"new_nodes": []}
    async def summarize(self, user_query, results):
        context = "\n".join([f"- {r}" for r in results])
        synth_prompt = LANGGRAPH_ANSWERING_PROMPT.format(query=user_query, context=context)
        completion = await self.llm.ainvoke([{"role": "user", "content": synth_prompt}])
        return completion.content
    def _parse_got_decomposition(self, llm_result: str) -> dict:
        """Parse LLM output as JSON, using regex extraction to be robust to markdown/code blocks."""
        import json, re
        print("[GoTPlanner][RAW LLM OUTPUT]:\n", llm_result, flush=True)
        try:
            # Find first valid {...} JSON object in the LLM response
            match = re.search(r'({[\s\S]*})', llm_result)
            if match:
                json_string = match.group(1).strip()
                # Remove any before/after code fences, newlines
                if json_string.startswith('```json'):
                    json_string = json_string[len('```json'):].strip()
                if json_string.startswith('```'):
                    json_string = json_string[len('```'):].strip()
                if json_string.endswith('```'):
                    json_string = json_string[:-len('```')].strip()
                return json.loads(json_string)
        except Exception as e:
            print(f"[GoTPlanner][ERROR] Could not extract/parse JSON: {e}")
        print("[GoTPlanner][ERROR] LLM planning output could not be parsed as a JSON graph!")
        lines = [line for line in llm_result.strip().split("\n") if line]
        nodes = [{
            "id": f"Q{i+1}",
            "content": line,
            "dependencies": []
        } for i, line in enumerate(lines)]
        return {"nodes": nodes, "edges": []}

class ThoughtNode:
    def __init__(self, id: str, content: str, dependencies: Optional[List[str]] = None, status: str = "pending", meta: Optional[dict]=None):
        self.id = id
        self.content = content
        self.dependencies = dependencies if dependencies else []
        self.status = status  # pending, running, complete, retired
        self.meta = meta or {}
        self.result = None
        self.score = None

class GoTPlanner:
    def __init__(self, llm=None, config: Optional[Any]=None):
        start_time = datetime.now()
        logger.info("=== Initializing GoTPlanner ===")
        logger.debug(f"Parameters - llm: {type(llm).__name__ if llm else 'default'}, config: {bool(config)}")
        
        try:
            logger.debug("Setting up LLM component...")
            self.llm = llm or GPT41GoTLLM()
            logger.info(f"LLM configured: {type(self.llm).__name__}")
            
            logger.debug("Initializing NetworkX graph...")
            self.graph = nx.DiGraph()
            logger.debug("Graph initialized")
            
            logger.debug("Setting up BTE client...")
            self.bte_client = BTEClient()
            logger.info("BTE client initialized")
            
            logger.debug("Loading BTE meta knowledge graph...")
            self.metakg = self._load_bte_metakg()
            logger.info(f"Meta KG loaded with {len(self.metakg.get('edges', []))} edges")
            
            logger.debug("Generating BTE constraints for prompts...")
            self.prompt_constraint = self._summarize_metakg_for_prompt(self.metakg)
            logger.debug(f"Prompt constraint generated (length: {len(self.prompt_constraint)} chars)")
            
            logger.debug("Initializing execution engine...")
            self.execution_engine = UnifiedExecutionEngine(config or UnifiedConfig())
            logger.info("Execution engine initialized")
            
            logger.debug("Initializing knowledge manager...")
            self.knowledge_manager = UnifiedKnowledgeManager(config or UnifiedConfig())
            logger.info("Knowledge manager initialized")
            
            # Propagate meta-KG to knowledge manager's predicate selector for proper support filtering
            try:
                self.knowledge_manager.meta_knowledge_graph = self.metakg
                if hasattr(self.knowledge_manager, 'predicate_selector') and self.knowledge_manager.predicate_selector:
                    self.knowledge_manager.predicate_selector.meta_kg = self.metakg
                    # Rebuild support map now that meta-KG is available
                    self.knowledge_manager.predicate_selector._build_predicate_support_map()
                    logger.debug("Propagated meta-KG to predicate selector")
            except Exception as e:
                logger.warning(f"Could not propagate meta-KG to knowledge manager: {e}")
            
            logger.debug("Initializing biomedical knowledge system...")
            self.biomedical_knowledge_system = BiomedicalKnowledgeSystem()
            logger.info("Biomedical knowledge system initialized")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"GoTPlanner initialization completed in {duration:.3f}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize GoTPlanner: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def _load_bte_metakg(self):
        logger.info("=== Loading BTE meta knowledge graph ===")
        try:
            logger.debug("Requesting meta knowledge graph from BTE client...")
            meta = self.bte_client.get_meta_knowledge_graph()
            
            nodes_count = len(meta.get('nodes', []))
            edges_count = len(meta.get('edges', []))
            logger.info(f"BTE meta KG loaded successfully: {nodes_count} nodes, {edges_count} edges")
            
            if logger.isEnabledFor(logging.DEBUG):
                # Log sample of node types and predicates
                edges = meta.get('edges', [])
                if edges:
                    predicates = set(edge.get('predicate', 'unknown') for edge in edges[:20])
                    subjects = set(edge.get('subject', 'unknown') for edge in edges[:20])
                    objects = set(edge.get('object', 'unknown') for edge in edges[:20])
                    logger.debug(f"Sample predicates: {list(predicates)[:5]}")
                    logger.debug(f"Sample subjects: {list(subjects)[:5]}")
                    logger.debug(f"Sample objects: {list(objects)[:5]}")
            
            return meta
        except Exception as ex:
            logger.error(f"Could not load BTE metakg: {ex}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.warning("Using empty meta knowledge graph as fallback")
            return {"nodes": [], "edges": []}

    def _summarize_metakg_for_prompt(self, meta: dict) -> str:
        # Summarize node categories and edge types for prompt injection
        node_types = set()
        allowed_edges = set()
        predicates = set()
        if meta and "edges" in meta:
            for edge in meta["edges"]:
                subj = edge.get("subject")
                pred = edge.get("predicate")
                obj = edge.get("object")
                node_types.add(subj)
                node_types.add(obj)
                predicates.add(pred)
                allowed_edges.add((subj, pred, obj))
        return (
            f"""SYSTEM MESSAGE: You act as a biomedical subquery decomposition planner specialized for BioThings Explorer (BTE).

Your output must ONLY contain atomic scientific subquestions (facts/edges) that correspond to *valid, answerable one-hop paths in the BTE meta knowledge graph*.

STRICT RULES:
- A valid atomic subquestion includes exactly a subject, a predicate, and an object. The subject and object types must match node types, the predicate must match allowed relations.
- Allowed node types (categories): {sorted(node_types)}
- Allowed predicates: {sorted(predicates)}
- Allowed (subject_type, predicate, object_type) triples: {sorted(allowed_edges)}
- You MUST NOT produce subquestions about categories, metadata, rephrasings, or anything not mappable to a direct subject-predicate-object claim. Examples: 'What is the category of X?', 'What is the ID of X?', 'Restate X' and similar must NOT appear.
- You MUST NOT produce subquestions that only return entity IDs, names, synonyms, definitions, or categories—only relational facts, claims, or statements between two biomedical entities.
- If you cannot produce a (subject, predicate, object) atomic subquestion that matches the allowed types and edges above, OMIT IT!
- If the biomedical input question cannot be decomposed into at least one such valid atomic edge, return an empty set of nodes.

POSITIVE EXAMPLES:
- Which drugs treat diabetes? (subject: Diabetes, object: Drug, predicate: treats)
- What genes are associated with Crohn's disease? (subject: Crohn's disease, object: Gene, predicate: associated_with)

NEGATIVE EXAMPLES (MUST be OMITTED):
- What is the category of X?
- What is the identifier (ID) for Y?
- Rephrase/restatement/summary questions.

Your response must be ONLY a valid JSON object of nodes and edges with atomic (subject, predicate, object) questions, following the template. Do not emit explanations, restatements or commentary. Each subquestion should be mappable directly to a valid BTE one-hop execution."""
        )
    async def decompose_initial_query(self, user_query: str):
        start_time = datetime.now()
        logger.info(f"=== Decomposing initial query: '{user_query}' ===")
        logger.debug(f"Query length: {len(user_query)} characters")
        
        try:
            # Reset entity mapping augmentations for each new top-level query
            logger.debug("Resetting entity mappings and RDF graph...")
            self._bte_additional_mappings = {}
            self.rdf_graph = Graph()
            self.rdf_graph.bind("biolink", Namespace("https://w3id.org/biolink/vocab/"))
            self.rdf_graph.bind("ex", Namespace("http://example.org/entity/"))
            logger.debug("RDF graph initialized with biolink and example namespaces")
            
            # Inject the BTE metaKG constraints into the prompt for LLM decomposition
            planning_prompt = self.prompt_constraint + "\n\nBiomedical question to decompose: " + user_query
            logger.debug(f"Planning prompt created (length: {len(planning_prompt)} chars)")
            logger.debug(f"Prompt preview: {planning_prompt[:200]}...")
            
            logger.info("Sending decomposition request to LLM...")
            raw_out = await self.llm.decompose_to_graph(planning_prompt)
            logger.debug(f"LLM raw output: {raw_out}")
            
            logger.debug("Parsing LLM decomposition response...")
            decomposition = self.llm._parse_got_decomposition(raw_out)
            logger.info(f"Decomposed query into {len(decomposition.get('nodes', []))} nodes and {len(decomposition.get('edges', []))} edges")
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Decomposed GoT graph:")
                for node in decomposition.get("nodes", []):
                    logger.debug(f"  Node {node.get('id', 'unknown')}: {node.get('content', 'no content')[:100]}...")
                for edge in decomposition.get("edges", []):
                    logger.debug(f"  Edge: {edge.get('from', 'unknown')} -> {edge.get('to', 'unknown')}")
            
            # Build graph from decomposition
            logger.debug("Building NetworkX graph from decomposition...")
            nodes_added = 0
            for node_dict in decomposition["nodes"]:
                try:
                    node = ThoughtNode(**node_dict)
                    self.graph.add_node(node.id, data=node)
                    nodes_added += 1
                    logger.debug(f"Added node {node.id}: {node.content[:50]}...")
                except Exception as node_error:
                    logger.warning(f"Failed to create node from {node_dict}: {node_error}")
            
            edges_added = 0
            for dep in decomposition.get("edges", []):
                try:
                    self.graph.add_edge(dep["from"], dep["to"])
                    edges_added += 1
                    logger.debug(f"Added edge: {dep['from']} -> {dep['to']}")
                except Exception as edge_error:
                    logger.warning(f"Failed to add edge {dep}: {edge_error}")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Query decomposition completed in {duration:.3f}s: {nodes_added} nodes, {edges_added} edges added")
            
        except Exception as e:
            logger.error(f"Error in decompose_initial_query: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    def get_ready_thoughts(self) -> List[ThoughtNode]:
        ready = []
        for node_id, node_obj in self.graph.nodes.data("data"):
            if node_obj.status == "pending":
                # If all dependencies exist and are done (either complete or retired), mark ready
                if all(self.graph.nodes[dep]["data"].status in ("complete", "retired")
                    for dep in node_obj.dependencies):
                    ready.append(node_obj)
        return ready

    def _retire_unreachable_nodes(self):
        # If a node depends on nodes that are all retired, retire it as well
        for node_id, node_obj in self.graph.nodes.data("data"):
            if node_obj.status == "pending":
                dependencies = node_obj.dependencies
                # If all dependencies are retired, and there are dependencies, retire this as unreachable
                if dependencies and all(self.graph.nodes[dep]["data"].status == "retired" for dep in dependencies):
                    node_obj.status = "retired"
    async def expand_thought(self, node: ThoughtNode):
        expand_results = await self.llm.expand_thought(node.content)
        for new_info in expand_results.get("new_nodes", []):
            if new_info["id"] not in self.graph:
                new_node = ThoughtNode(**new_info)
                self.graph.add_node(new_node.id, data=new_node)
                for dep in new_node.dependencies:
                    self.graph.add_edge(dep, new_node.id)
    def is_graph_complete(self) -> bool:
        return all(node.status in ("retired", "complete") for _, node in self.graph.nodes.data("data"))
    async def execute(self, user_query: str):
        start_time = datetime.now()
        logger.info(f"=== Starting GoTPlanner execution for query: '{user_query}' ===")
        
        try:
            # Step 1: Decompose the initial query
            logger.info("Phase 1: Query decomposition")
            await self.decompose_initial_query(user_query)
            logger.info(f"Graph initialized with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
            # Step 2: Execute thoughts iteratively
            logger.info("Phase 2: Iterative thought execution")
            iteration = 0
            max_iterations = 50  # Safety limit
            
            while not self.is_graph_complete() and iteration < max_iterations:
                iteration += 1
                logger.info(f"=== Execution iteration {iteration} ===")
                
                ready_thoughts = self.get_ready_thoughts()
                logger.info(f"Found {len(ready_thoughts)} ready thoughts for execution")
                
                if not ready_thoughts:
                    logger.warning("No ready thoughts found but graph is not complete - checking for deadlocks")
                    pending_nodes = [n for n in self.graph.nodes if self.graph.nodes[n]['data'].status == 'pending']
                    logger.debug(f"Pending nodes: {[n for n in pending_nodes]}")
                    if pending_nodes:
                        logger.error("Potential deadlock detected - retiring all pending nodes")
                        for node_id in pending_nodes:
                            self.graph.nodes[node_id]['data'].status = 'retired'
                    break
                
                for thought in ready_thoughts:
                    logger.debug(f"Ready thought: {thought.id} - {thought.content[:100]}...")
                
                # Execute ready thoughts in parallel
                logger.debug(f"Executing {len(ready_thoughts)} thoughts in parallel...")
                tasks = [self.execute_thought(t) for t in ready_thoughts]
                await asyncio.gather(*tasks)
                
                # Expand thoughts after execution
                logger.debug("Expanding thoughts after execution...")
                for node in ready_thoughts:
                    try:
                        await self.expand_thought(node)
                    except Exception as expand_error:
                        logger.warning(f"Error expanding thought {node.id}: {expand_error}")
                
                # Log current graph state
                complete_nodes = sum(1 for n in self.graph.nodes if self.graph.nodes[n]['data'].status == 'complete')
                retired_nodes = sum(1 for n in self.graph.nodes if self.graph.nodes[n]['data'].status == 'retired')
                pending_nodes = sum(1 for n in self.graph.nodes if self.graph.nodes[n]['data'].status == 'pending')
                logger.info(f"Iteration {iteration} complete: {complete_nodes} complete, {retired_nodes} retired, {pending_nodes} pending")
            
            if iteration >= max_iterations:
                logger.warning(f"Maximum iterations ({max_iterations}) reached - forcing completion")
            
            # Step 3: Compose final answer
            logger.info("Phase 3: Final answer composition")
            final_answer = await self.compose_final_answer(user_query)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"GoTPlanner execution completed in {duration:.3f}s after {iteration} iterations")
            
            return final_answer
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"GoTPlanner execution failed after {duration:.3f}s: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    async def execute_thought(self, node: ThoughtNode):
        start_time = datetime.now()
        logger.info(f"=== Executing thought {node.id} ===")
        logger.debug(f"Node content: {node.content}")
        logger.debug(f"Node dependencies: {node.dependencies}")
        logger.debug(f"Node status before execution: {node.status}")
        
        try:
            node.status = "running"
            logger.debug(f"Node {node.id} status set to 'running'")
            
            content_to_execute = self._inject_placeholders(node)
            logger.debug(f"Content after placeholder injection: {content_to_execute}")

            # Use TRAPI builder from UnifiedKnowledgeManager or BiomedicalKnowledgeSystem
            entities = []
            
            # 1. Collect robust entity mapping, similar to LangGraph logic
            logger.debug("Collecting entity mappings...")
            entity_data = {}
            
            # Build entity_data strictly from entities actually mentioned in the subquery (LangGraph-style)
            node_content_norm = content_to_execute.lower().replace('-', ' ').replace(',', ' ')
            logger.debug(f"Normalized content: {node_content_norm}")
            
            # Pass all canonical IDs per subquery (BioNER, hop outputs, BTE updates)
            if hasattr(self, 'entity_context') and getattr(self, 'entity_context', None):
                entities = getattr(self.entity_context, 'entities', [])
                logger.debug(f"Found {len(entities)} entities in context")
                for e in entities:
                    # Handle both dict and object entities
                    if isinstance(e, dict):
                        ename = e.get('name')
                        e_id = e.get('id')
                        synonyms = e.get('synonyms', [])
                    else:
                        ename = getattr(e, 'name', None)
                        e_id = getattr(e, 'id', None) or getattr(e, 'entity_id', None)
                        synonyms = getattr(e, 'synonyms', [])
                    
                    if ename and e_id and e_id.strip():
                        entity_data[ename] = e_id
                        logger.debug(f"Added entity mapping: {ename} -> {e_id}")
                        # Add synonyms if available
                        if synonyms:
                            for syn in synonyms:
                                if syn and syn.strip():
                                    entity_data[syn] = e_id
                                    logger.debug(f"Added synonym mapping: {syn} -> {e_id}")
                    else:
                        logger.debug(f"Skipping entity with missing name or ID: {e}")
            else:
                logger.debug("No entity_context found")
                
            if hasattr(self, "_hop_output_ids") and self._hop_output_ids:
                logger.debug(f"Adding hop output IDs: {self._hop_output_ids}")
                entity_data.update(self._hop_output_ids)
                
            if not hasattr(self, '_bte_additional_mappings'):
                self._bte_additional_mappings = {}
            logger.debug(f"Adding BTE additional mappings: {self._bte_additional_mappings}")
            entity_data.update(self._bte_additional_mappings)
            
            logger.info(f"Total entity mappings for {node.id}: {len(entity_data)} entities")
            logger.debug(f"Entity mappings: {entity_data}")

            failed_trapis = []  # Track failed TRAPI attempts for LLM hints

            # 2. Build TRAPI query using identical interface as LangGraph's BTESearchNode
            logger.debug("Building TRAPI query...")
            from agentic_bte.core.knowledge.trapi import TRAPIQueryBuilder
            trapi_builder = TRAPIQueryBuilder()
            trapi_query = trapi_builder.build_trapi_query(content_to_execute, entity_data, failed_trapis)
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"TRAPI query for node {node.id}: {json.dumps(trapi_query, indent=2)}")
            
            # 3. Skip unanswerable subquestions - but allow single-node queries for entity categorization
            query_graph = trapi_query.get('message', {}).get('query_graph', {}) if trapi_query else {}
            has_edges = bool(query_graph.get('edges'))
            has_nodes = bool(query_graph.get('nodes'))
            
            # A valid query must have either:
            # 1. At least one edge (for relationship queries), OR
            # 2. At least one node (for entity categorization queries)
            is_valid_query = has_edges or has_nodes
            
            if not trapi_query or 'error' in trapi_query or not is_valid_query:
                logger.warning(f"Node {node.id} does not map to a valid BTE executable query - retiring node")
                logger.debug(f"TRAPI query validation failed: query={bool(trapi_query)}, has_nodes={has_nodes}, has_edges={has_edges}")
                node.status = "retired"
                node.result = None
                return
            
            # Execute TRAPI with batching and retries (Prototype-aligned)
            def _run_with_retries(base_query: Dict[str, Any]) -> Optional[tuple[list[Dict], Dict[str, str]]]:
                # Helper: extract categories and edge keys
                def _extract_categories(q: Dict[str, Any]) -> tuple[str, str, str]:
                    qg = q.get('message', {}).get('query_graph', {})
                    edges = qg.get('edges', {})
                    nodes = qg.get('nodes', {})
                    if not edges or not nodes:
                        return None, None, None
                    ekey = next(iter(edges.keys()))
                    edge = edges[ekey]
                    skey = edge.get('subject')
                    okey = edge.get('object')
                    s_cat = (nodes.get(skey, {}).get('categories') or [None])[0]
                    o_cat = (nodes.get(okey, {}).get('categories') or [None])[0]
                    return s_cat, o_cat, ekey
                
                # Detect intent via keywords (no entities needed)
                intent = self.knowledge_manager.predicate_selector.detect_query_intent(content_to_execute, [])
                s_cat, o_cat, ekey = _extract_categories(base_query)
                if not ekey:
                    # Single-node queries - execute once
                    results, mappings, meta = self.bte_client.execute_trapi_with_batching(base_query)
                    return (results, mappings) if results else None
                
                # Attempt 1: Execute as-is (batched)
                logger.info(f"Executing TRAPI query for node {node.id}...")
                # Determine current predicate from the query itself
                current_pred = None
                try:
                    current_pred = (base_query.get('message', {}).get('query_graph', {}).get('edges', {}).get(ekey, {}).get('predicates') or [None])[0]
                except Exception:
                    current_pred = None
                results, mappings, meta = self.bte_client.execute_trapi_with_batching(
                    base_query, predicate=current_pred, query_intent=intent.value
                )
                if results:
                    return (results, mappings)
                logger.warning(f"No results for initial TRAPI on node {node.id}; attempting predicate-driven retries")
                
                # Generate candidate predicates for subject/object
                candidates = self.knowledge_manager.predicate_selector.select_predicates(
                    intent, s_cat, o_cat
                )
                candidate_list = [p for p, _ in candidates]
                logger.debug(f"Predicate candidates for {s_cat}->{o_cat}: {candidate_list}")
                
                # Try each predicate
                for pred in candidate_list:
                    alt = json.loads(json.dumps(base_query))  # deep copy
                    alt_qg = alt['message']['query_graph']
                    alt_qg['edges'][ekey]['predicates'] = [pred]
                    logger.info(f"Retrying with predicate '{pred}' for node {node.id}")
                    results, mappings, meta = self.bte_client.execute_trapi_with_batching(
                        alt, predicate=pred, query_intent=intent.value
                    )
                    if results:
                        logger.info(f"Found results with predicate '{pred}' on retry for node {node.id}")
                        return (results, mappings)
                
                # Flip direction if still no results
                logger.warning(f"No results after predicate retries; attempting direction flip for node {node.id}")
                flipped = json.loads(json.dumps(base_query))
                edge = flipped['message']['query_graph']['edges'][ekey]
                edge['subject'], edge['object'] = edge['object'], edge['subject']
                s_cat2, o_cat2, _ = _extract_categories(flipped)
                # Try candidates for flipped direction
                flipped_candidates = self.knowledge_manager.predicate_selector.select_predicates(
                    intent, s_cat2, o_cat2
                )
                flipped_list = [p for p, _ in flipped_candidates] or ['biolink:related_to']
                for pred in flipped_list:
                    alt = json.loads(json.dumps(flipped))
                    alt['message']['query_graph']['edges'][ekey]['predicates'] = [pred]
                    logger.info(f"Retrying with flipped direction and predicate '{pred}' for node {node.id}")
                    results, mappings, meta = self.bte_client.execute_trapi_with_batching(
                        alt, predicate=pred, query_intent=intent.value
                    )
                    if results:
                        logger.info(f"Found results with flipped '{pred}' on retry for node {node.id}")
                        return (results, mappings)
                
                return None
            
            parsed_results_tuple = _run_with_retries(trapi_query)
            
            if parsed_results_tuple:
                # When using execute_trapi_with_batching, we already get parsed results
                logger.debug("Parsing completed via batched executor")
                parsed_results, new_mappings = parsed_results_tuple
                try:
                    if new_mappings and isinstance(new_mappings, dict):
                        self._bte_additional_mappings.update(new_mappings)
                        logger.debug(f"Updated entity mappings with {len(new_mappings)} entries from batched execution")
                except Exception as emap_exc:
                    logger.warning(f"Failed to merge new entity mappings: {emap_exc}")
                # Update entity mappings if available from recent calls is handled inside execute_trapi_with_batching/parse
                
                logger.debug("Adding results to RDF graph...")
                self.add_bte_results_to_rdf(parsed_results)
                
                node.result = parsed_results
                node.score = 1.0
                logger.info(f"Node {node.id} completed successfully with {len(parsed_results)} results")
            else:
                logger.warning(f"BTE did not return valid results for node {node.id} after retries - retiring node")
                node.status = "retired"
                node.result = None
                
        except Exception as bex:
            logger.error(f"Exception during execution of node {node.id}: {bex}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            node.status = "retired"
            node.result = None
        
        # Always set to complete or retired after node execution
        if node.status != "retired":
            node.status = "complete"
            
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Node {node.id} execution completed in {duration:.3f}s with status: {node.status}")
        
        # After each execution, check for any pending nodes with unmet dependencies and retire them if they can't ever complete
        logger.debug("Checking for unreachable nodes...")
        self._retire_unreachable_nodes()

    def _inject_placeholders(self, node: ThoughtNode) -> str:
        content = node.content
        # Gather results for each dependency
        if node.dependencies:
            for dep_id in node.dependencies:
                dep_result = None
                if dep_id in self.graph.nodes:
                    dep_node = self.graph.nodes[dep_id]["data"]
                    dep_result = dep_node.result
                if dep_result:
                    # Convert structured BTE results to readable string format
                    dep_result_str = self._format_bte_results_for_placeholder(dep_result)
                    
                    # Example placeholder: {dep_id}_result or generic {previous_result}
                    content = content.replace(f"{{{dep_id}_result}}", dep_result_str)
                    content = content.replace("{previous_result}", dep_result_str)
        return content
    
    def _format_bte_results_for_placeholder(self, bte_results) -> str:
        """
        Convert BTE results (list of relationship dicts) into a readable string format
        for placeholder injection in dependent subqueries.
        """
        if not bte_results:
            return "[No results from previous step]"
        
        if isinstance(bte_results, str):
            return bte_results
        
        if isinstance(bte_results, list):
            formatted_results = []
            for i, result in enumerate(bte_results[:10]):  # Limit to first 10 for readability
                if isinstance(result, dict):
                    subject = result.get('subject', 'Unknown')
                    predicate = result.get('predicate', 'related_to')
                    obj = result.get('object', 'Unknown')
                    
                    # Clean up predicate for readability
                    predicate_clean = predicate.replace('biolink:', '').replace('_', ' ')
                    
                    formatted_results.append(f"{subject} {predicate_clean} {obj}")
                else:
                    formatted_results.append(str(result))
            
            if len(bte_results) > 10:
                formatted_results.append(f"... and {len(bte_results) - 10} more results")
                
            return "; ".join(formatted_results)
        
        # Fallback for other data types
        return str(bte_results)
    async def compose_final_answer(self, user_query: str) -> str:
        # Collect BTE evidence from ALL completed nodes that have results, not just leaf nodes
        # This fixes the issue where successful intermediate nodes (like Q2) were ignored
        candidate_results = []
        nodes_with_results = []
        
        for node_id in self.graph.nodes:
            node_obj = self.graph.nodes[node_id]["data"]
            if (node_obj.status == "complete" and 
                node_obj.result and 
                isinstance(node_obj.result, list) and 
                len(node_obj.result) > 0):
                candidate_results.extend(node_obj.result)
                nodes_with_results.append(f"{node_id} ({len(node_obj.result)} results)")
        
        print(f"[GoTPlanner] === FINAL ANSWER SYNTHESIS: Using {len(candidate_results)} BTE results from {len(nodes_with_results)} nodes")
        if nodes_with_results:
            print(f"[GoTPlanner] Nodes with results: {', '.join(nodes_with_results)}")
        
        if not candidate_results:
            # Check if we have any results in the RDF graph as a backup
            rdf_triples_count = len(list(self.rdf_graph))
            print(f"[GoTPlanner] No candidate results found, but RDF graph has {rdf_triples_count} triples")
            
            if rdf_triples_count > 0:
                # Use RDF graph content for synthesis even without candidate_results
                turtle_kg = self.rdf_graph.serialize(format="turtle")
                print("[GoTPlanner] === USING RDF GRAPH FOR SYNTHESIS ===\n", turtle_kg[:500], "..." if len(turtle_kg) > 500 else "")
                
                mapped_id_dict = {getattr(e, 'name', str(e)): getattr(e, 'id', str(e)) for e in getattr(self.entity_context, 'entities', [])}
                summary_prompt = f"""
As a biomedical agent, answer the question:
{user_query}

Here is a knowledge graph of evidence (Turtle format):
{turtle_kg}

Entity mappings:
{mapped_id_dict}

Use only the graph and entity map as your evidence. Provide specific relationships found in the graph. If there are no relevant relationships in the graph for the query, say so clearly.
"""
                final = await self.llm.summarize(user_query, [summary_prompt])
                print("[GoTPlanner] === FINAL SYNTHESIZED ANSWER ===\n", final)
                return final
            else:
                return "[No BTE-backed answer available for this query.]"
        
        # We have candidate results - proceed with full synthesis
        turtle_kg = self.rdf_graph.serialize(format="turtle")
        print("[GoTPlanner] === FINAL RDF GRAPH ===\n", turtle_kg[:500], "..." if len(turtle_kg) > 500 else "")
        
        # Also format candidate results as readable evidence for the LLM
        evidence_summary = []
        for i, result in enumerate(candidate_results[:20]):  # Limit to first 20 for readability
            if isinstance(result, dict):
                subject = result.get('subject', 'Unknown')
                predicate = result.get('predicate', 'related_to')
                obj = result.get('object', 'Unknown')
                predicate_clean = predicate.replace('biolink:', '').replace('_', ' ')
                evidence_summary.append(f"- {subject} {predicate_clean} {obj}")
        
        if len(candidate_results) > 20:
            evidence_summary.append(f"... and {len(candidate_results) - 20} more relationships")
        
        evidence_text = "\n".join(evidence_summary)
        
        mapped_id_dict = {getattr(e, 'name', str(e)): getattr(e, 'id', str(e)) for e in getattr(self.entity_context, 'entities', [])}
        summary_prompt = f"""
As a biomedical agent, answer the question:
{user_query}

Here is the evidence found from biomedical knowledge sources:
{evidence_text}

RDF Knowledge Graph (Turtle format):
{turtle_kg}

Entity mappings:
{mapped_id_dict}

Use only this evidence to answer the question. Be specific about the relationships found. If the evidence doesn't directly answer the question, explain what information is available and what might be missing.
"""
        final = await self.llm.summarize(user_query, [summary_prompt])
        print("[GoTPlanner] === FINAL SYNTHESIZED ANSWER ===\n", final)
        return final

    def get_all_results(self) -> List[Dict]:
        """
        Aggregate all parsed BTE results stored on nodes into a single list.
        Deduplicate by (subject_id, predicate, object_id) when available.
        """
        all_results: List[Dict] = []
        seen = set()
        for node_id in self.graph.nodes:
            node_obj = self.graph.nodes[node_id]["data"]
            if node_obj and isinstance(getattr(node_obj, 'result', None), list):
                for rel in node_obj.result:
                    # Build a dedup key if possible
                    key = (
                        rel.get('subject_id'),
                        rel.get('predicate'),
                        rel.get('object_id')
                    )
                    if key not in seen:
                        all_results.append(rel)
                        seen.add(key)
        return all_results

    def add_bte_results_to_rdf(self, parsed_results):
        BL = Namespace("https://w3id.org/biolink/vocab/")
        EX = Namespace("http://example.org/entity/")
        
        def safe_uri_fragment(text):
            """Convert text to a safe URI fragment by replacing problematic characters"""
            import re
            if not text or text == "?":
                return "unknown"
            # Replace spaces and special chars with underscores, keep alphanumeric and common chars
            safe_text = re.sub(r'[^\w\-_.]', '_', str(text))
            # Remove multiple consecutive underscores
            safe_text = re.sub(r'_+', '_', safe_text)
            # Remove leading/trailing underscores
            safe_text = safe_text.strip('_')
            return safe_text or "unknown"
        
        for rel in parsed_results:
            subj_fragment = safe_uri_fragment(rel.get("subject", "?"))
            obj_fragment = safe_uri_fragment(rel.get("object", "?"))
            
            subj_uri = EX[subj_fragment]
            obj_uri = EX[obj_fragment]
            
            pred = str(rel.get("predicate", "related_to")).split(':')[-1]
            pred_fragment = safe_uri_fragment(pred)
            pred_uri = BL[pred_fragment]
            
            self.rdf_graph.add((subj_uri, pred_uri, obj_uri))
            
            # Add types/labels if available
            if rel.get("subject_type"):
                type_fragment = safe_uri_fragment(str(rel["subject_type"]).split(":")[-1])
                self.rdf_graph.add((subj_uri, RDF.type, BL[type_fragment]))
            if rel.get("object_type"):
                type_fragment = safe_uri_fragment(str(rel["object_type"]).split(":")[-1])
                self.rdf_graph.add((obj_uri, RDF.type, BL[type_fragment]))
            if rel.get("subject"):
                self.rdf_graph.add((subj_uri, RDFS.label, Literal(str(rel["subject"]))))
            if rel.get("object"):
                self.rdf_graph.add((obj_uri, RDFS.label, Literal(str(rel["object"]))))
