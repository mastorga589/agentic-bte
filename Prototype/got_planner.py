"""
Graph of Thoughts planner (stand-alone) integrating:
- LLM-based GoT decomposition (nodes/edges)
- Subquery dependency DAG scheduling with parallel execution of independent nodes
- Top-3 predicate concurrent execution for each subquery
- RDF accumulation of results and LLM-based final synthesis
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set

import networkx as nx
from langchain_openai import ChatOpenAI
from rdflib import Graph as RDFGraph, URIRef, Namespace, Literal
from rdflib.namespace import RDF, RDFS

from .settings import get_settings
from .bte_client import BTEClient
from .trapi import TRAPIQueryBuilder
from .predicate_strategy import PredicateSelector, PredicateConfig, QueryIntent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ThoughtNode:
    id: str
    content: str
    dependencies: List[str]
    status: str = "pending"  # pending, running, complete, retired
    result: Optional[List[Dict[str, Any]]] = None


class GoTPlanner:
    def __init__(self, model: str | None = None, temperature: float = 0.2, entity_data: Optional[Dict[str, str]] = None):
        s = get_settings()
        # Determine offline mode (no API key or explicit override)
        self.offline = (not s.openai_api_key) or (os.getenv("PROTOTYPE_OFFLINE", "0") in ("1", "true", "True"))
        if not self.offline:
            self.llm = ChatOpenAI(
                model=model or s.openai_model,
                api_key=s.openai_api_key,
                temperature=temperature,
                max_tokens=s.openai_max_tokens,
            )
        else:
            class _SimpleLLM:
                async def ainvoke(self, messages):
                    # Return a minimal JSON decomposition
                    try:
                        q = None
                        if isinstance(messages, list) and messages:
                            q = messages[-1].get("content") if isinstance(messages[-1], dict) else None
                        plan = {
                            "nodes": [
                                {"id": "Q1", "content": f"Retrieve mechanistic relationships relevant to: {q}", "dependencies": []}
                            ],
                            "edges": []
                        }
                        return type("Resp", (), {"content": json.dumps(plan)})
                    except Exception:
                        return type("Resp", (), {"content": json.dumps({"nodes": [], "edges": []})})
            self.llm = _SimpleLLM()
        self.graph = nx.DiGraph()
        self.rdf_graph = RDFGraph()
        self.rdf_graph.bind("biolink", Namespace("https://w3id.org/biolink/vocab/"))
        self.rdf_graph.bind("ex", Namespace("http://example.org/entity/"))
        self.bte = BTEClient()
        self.trapi_builder = TRAPIQueryBuilder()
        # Predicate selector w/ meta-KG
        meta = self.bte.get_meta_knowledge_graph()
        self.predicate_selector = PredicateSelector(meta, PredicateConfig(max_predicates_per_subquery=3))
        self.debug_dir = s.debug_output_dir
        # Base entity data to seed TRAPI building
        self.entity_data_base: Dict[str, str] = entity_data or {}
        # Execution trace for introspection
        self.execution_trace: List[Dict[str, Any]] = []

    # ---------- LLM Planning ----------
    def _parse_got_decomposition(self, raw: str) -> Dict[str, Any]:
        try:
            m = re.search(r'({[\s\S]*})', raw)
            if m:
                js = m.group(1).strip()
                if js.startswith('```json'):
                    js = js[len('```json'):].strip()
                if js.startswith('```'):
                    js = js[len('```'):].strip()
                if js.endswith('```'):
                    js = js[:-3].strip()
                return json.loads(js)
        except Exception as e:
            logger.warning(f"Could not parse GoT JSON: {e}")
        # Fallback: one node per non-empty line
        lines = [ln for ln in (raw or '').splitlines() if ln.strip()]
        nodes = [{"id": f"Q{i+1}", "content": ln.strip(), "dependencies": []} for i, ln in enumerate(lines)]
        return {"nodes": nodes, "edges": []}

    async def _decompose(self, query: str) -> Dict[str, Any]:
        meta = self.bte.get_meta_knowledge_graph()
        node_types = sorted({e.get('subject') for e in meta.get('edges', [])} | {e.get('object') for e in meta.get('edges', [])})
        predicates = sorted({e.get('predicate') for e in meta.get('edges', [])})
        allowed_triples = sorted({(e.get('subject'), e.get('predicate'), e.get('object')) for e in meta.get('edges', [])})
        prompt = f"""
You are a biomedical LLM planner. Decompose the user's question into a directed acyclic plan of single-hop, actionable subquestions that each correspond to ONE valid BTE one-hop edge.

Output ONLY JSON with fields:
- nodes: list of objects {{id, content, dependencies}} where:
  - id is a string like "Q1", "Q2", ...
  - content is a single-hop NATURAL LANGUAGE QUESTION (not a fragment) that can be translated into one TRAPI one-hop query over the BTE meta-knowledge graph.
    Examples of valid forms (illustrative only):
      - "Which small molecules interact with {{previous_result}}?"
      - "Which genes participate in the PI3K/AKT signaling pathway?"
      - "Which diseases are associated with these genes?"
  - dependencies is a list of node ids whose results should be referenced in this question (use {{Qx_result}} or {{previous_result}} to indicate dependency context when appropriate).
- edges: list of objects {{from, to}} indicating dependency order.

STRICT RULES:
- Each node MUST be a single-hop question that maps to one (subject, predicate, object) triple present in the BTE meta-KG.
- Avoid fragments like "small molecules" or "breast cancer"; always write a complete question with a clear verb.
- Prefer mechanistic relations when relevant (interacts_with, affects, regulates, participates_in, pathway membership, target relationships).
- Keep questions feasible with the allowed schema below.
- Allowed node types: {node_types}
- Allowed predicates: {predicates}
- Allowed (subject,predicate,object) triples (sample): {allowed_triples[:100]}
- Do NOT emit anything except the JSON object.

Biomedical question: {query}
"""
        out = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        try:
            with open(os.path.join(self.debug_dir, "got_decomposition.json"), "w") as f:
                f.write(out.content or "")
        except Exception:
            pass
        return self._parse_got_decomposition(out.content or "{}")
    async def decompose_initial_query(self, query: str) -> None:
        decomp = await self._decompose(query)
        # Nodes
        for nd in decomp.get("nodes", []):
            deps = self._normalize_dependencies(nd.get("dependencies", []))
            node = ThoughtNode(id=nd.get("id"), content=nd.get("content"), dependencies=deps)
            self.graph.add_node(node.id, data=node)
        # Edges
        for ed in decomp.get("edges", []):
            try:
                u = ed.get("from"); v = ed.get("to")
                if u in self.graph.nodes and v in self.graph.nodes:
                    self.graph.add_edge(u, v)
            except Exception:
                pass

    # ----- Planner-only public APIs -----
    async def plan_only(self, query: str) -> Dict[str, Any]:
        """Generate initial GoT plan (nodes/edges) with all nodes in 'pending' status. No execution."""
        # Reset graph for a fresh plan
        self.graph = nx.DiGraph()
        await self.decompose_initial_query(query)
        return self.get_plan()

    def set_plan(self, plan: Dict[str, Any]) -> None:
        """Load an external plan (nodes/edges) into the internal graph, preserving statuses."""
        self.graph = nx.DiGraph()
        if not isinstance(plan, dict):
            return
        for nd in plan.get("nodes", []) or []:
            try:
                nid = nd.get("id")
                content = nd.get("content")
                deps = self._normalize_dependencies(nd.get("dependencies", []))
                status = nd.get("status", "pending")
                node = ThoughtNode(id=nid, content=content, dependencies=deps, status=status)
                self.graph.add_node(nid, data=node)
            except Exception:
                continue
        for ed in plan.get("edges", []) or []:
            try:
                u = ed.get("from"); v = ed.get("to")
                if u in self.graph.nodes and v in self.graph.nodes:
                    self.graph.add_edge(u, v)
            except Exception:
                continue

    def update_statuses(self, status_map: Dict[str, str]) -> None:
        """Update node statuses (pending|running|complete|retired) based on external execution."""
        for nid, st in (status_map or {}).items():
            if nid in self.graph.nodes and st in ("pending", "running", "complete", "retired"):
                data = self.graph.nodes[nid].get('data')
                if data is not None:
                    data.status = st

    def get_ready_node_ids(self) -> List[str]:
        """Return ids of nodes whose dependencies are satisfied and are still pending."""
        out: List[str] = []
        for nid in self.graph.nodes:
            data = self.graph.nodes[nid].get('data')
            if data is None:
                continue
            node: ThoughtNode = data
            if node.status != 'pending':
                continue
            ok = True
            for dep in node.dependencies:
                if dep not in self.graph.nodes:
                    continue
                ddata = self.graph.nodes[dep].get('data')
                if not ddata or ddata.status not in ("complete", "retired"):
                    ok = False
                    break
            if ok:
                out.append(nid)
        return out

    async def refine_with_context(self, user_query: str, known_entity_ids: Dict[str, str], relationships: List[str]) -> Dict[str, Any]:
        """Refine plan using external evidence and IDs; only pending nodes may be updated; may add new nodes."""
        # Update internal entity_data cache
        try:
            for k, v in (known_entity_ids or {}).items():
                if isinstance(k, str) and isinstance(v, str):
                    self.entity_data_base[k] = v
        except Exception:
            pass
        try:
            plan = self.get_plan()
            prompt = (
                "You are refining a biomedical plan of single-hop questions.\n"
                "Rules:\n"
                "- Keep all questions single-hop and aligned to the meta-KG.\n"
                "- Prefer mechanistic relations when possible.\n"
                "- Do not modify nodes that are already complete or retired.\n"
                "- You may add new nodes (use new ids like Q100, Q101, ...).\n"
                "- You may update content/dependencies for PENDING nodes only.\n"
                "- Aim to incorporate new relationships and entity IDs.\n"
                "Return ONLY JSON with fields nodes (id, content, dependencies) and edges (from, to).\n\n"
                f"User question: {user_query}\n"
                f"Current plan: {json.dumps(plan)}\n"
                f"Known entity IDs: {json.dumps(known_entity_ids or {})}\n"
                f"New relationships: {json.dumps(relationships or [])}\n"
            )
            out = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            decomp = self._parse_got_decomposition(out.content or "{}")
            # Apply changes similarly to _refine_plan
            for nd in decomp.get("nodes", []):
                nid = nd.get("id")
                content = nd.get("content")
                deps = self._normalize_dependencies(nd.get("dependencies", []))
                if not isinstance(nid, str) or not isinstance(content, str):
                    continue
                if nid in self.graph.nodes:
                    node: ThoughtNode = self.graph.nodes[nid]['data']
                    if node.status == "pending":
                        node.content = content
                        node.dependencies = deps
                else:
                    node = ThoughtNode(id=nid, content=content, dependencies=deps)
                    self.graph.add_node(nid, data=node)
            for ed in decomp.get("edges", []):
                try:
                    u = ed.get("from"); v = ed.get("to")
                    if u in self.graph.nodes and v in self.graph.nodes:
                        self.graph.add_edge(u, v)
                except Exception:
                    continue
        except Exception:
            pass
        return self.get_plan()

    # ---------- Execution helpers ----------
    def _normalize_dependencies(self, deps: List[str]) -> List[str]:
        out: List[str] = []
        for d in deps or []:
            if not isinstance(d, str):
                continue
            dd = d
            if dd.endswith("_result"):
                dd = dd[: -len("_result")]
            out.append(dd)
        return out
    def _ready_nodes(self) -> List[ThoughtNode]:
        ready: List[ThoughtNode] = []
        for node_id, data in self.graph.nodes.data("data"):
            node: ThoughtNode = data
            if node.status != "pending":
                continue
            if all(self.graph.nodes[dep]["data"].status in ("complete", "retired") for dep in node.dependencies):
                ready.append(node)
        return ready

    def _format_results_for_placeholder(self, res: List[Dict[str, Any]]) -> str:
        if not res:
            return "[No results]"
        out: List[str] = []
        for r in res[:10]:
            if isinstance(r, dict):
                pred = (r.get('predicate') or 'biolink:related_to').replace('biolink:', '').replace('_', ' ')
                out.append(f"{r.get('subject', '?')} {pred} {r.get('object','?')}")
            else:
                out.append(str(r))
        if len(res) > 10:
            out.append(f"... and {len(res)-10} more")
        return "; ".join(out)

    def _inject_dependencies(self, node: ThoughtNode) -> str:
        content = node.content
        for dep in node.dependencies:
            if dep in self.graph.nodes:
                dep_node: ThoughtNode = self.graph.nodes[dep]["data"]
                if dep_node.result:
                    formatted = self._format_results_for_placeholder(dep_node.result)
                    content = content.replace(f"{{{dep}_result}}", formatted)
                    content = content.replace("{previous_result}", formatted)
        return content

    def _add_rdf_triples(self, parsed: List[Dict[str, Any]]) -> None:
        BL = Namespace("https://w3id.org/biolink/vocab/")
        EX = Namespace("http://example.org/entity/")

        def safe(text: Any) -> str:
            import re
            t = str(text or "unknown")
            t = re.sub(r'[^\w\-_.]', '_', t)
            return re.sub(r'_+', '_', t).strip('_') or "unknown"

        for rel in parsed:
            s_uri = EX[safe(rel.get("subject"))]
            o_uri = EX[safe(rel.get("object"))]
            p = str(rel.get("predicate", "related_to")).split(':')[-1]
            p_uri = BL[safe(p)]
            self.rdf_graph.add((s_uri, p_uri, o_uri))
            if rel.get("subject_type"):
                self.rdf_graph.add((s_uri, RDF.type, BL[safe(str(rel['subject_type']).split(':')[-1])]))
            if rel.get("object_type"):
                self.rdf_graph.add((o_uri, RDF.type, BL[safe(str(rel['object_type']).split(':')[-1])]))
            if rel.get("subject"):
                self.rdf_graph.add((s_uri, RDFS.label, Literal(str(rel["subject"]))))
            if rel.get("object"):
                self.rdf_graph.add((o_uri, RDFS.label, Literal(str(rel["object"]))))

    async def _execute_predicate_variants(self, base_query: Dict[str, Any], intent: QueryIntent) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, str], List[Dict[str, Any]]]:
        # Extract a single edge key
        qg = base_query.get('message', {}).get('query_graph', {})
        edges = qg.get('edges', {})
        nodes = qg.get('nodes', {})
        if not edges:
            # single-node query; just run
            results, id_map, _ = self.bte.execute_trapi_with_batching(base_query)
            trapis = [{"predicate": None, "query_graph": base_query.get("message", {}).get("query_graph", {})}]
            return results, [], id_map, trapis
        ekey = next(iter(edges.keys()))
        edge = edges[ekey]
        s_node = nodes.get(edge.get('subject'), {})
        o_node = nodes.get(edge.get('object'), {})
        s_cat = (s_node.get('categories') or [None])[0]
        o_cat = (o_node.get('categories') or [None])[0]
        if not s_cat or not o_cat:
            results, id_map, _ = self.bte.execute_trapi_with_batching(base_query)
            trapis = [{"predicate": None, "query_graph": base_query.get("message", {}).get("query_graph", {})}]
            return results, [], id_map, trapis

        # If offline, synthesize minimal results without calling BTE
        if self.offline:
            tried = ["biolink:interacts_with", "biolink:affects", "biolink:related_to"]
            dummy = [
                {"subject": "DummyEntityA", "subject_id": "EX:A", "subject_type": s_cat or "biolink:BiologicalEntity",
                 "predicate": tried[0],
                 "object": "DummyEntityB", "object_id": "EX:B", "object_type": o_cat or "biolink:BiologicalEntity"}
            ]
            trapis = [{"predicate": tried[0], "query_graph": base_query.get("message", {}).get("query_graph", {})}]
            return dummy, tried, {"DummyEntityA": "EX:A", "DummyEntityB": "EX:B"}, trapis
        # Select top-3 predicates and execute concurrently
        candidates = self.predicate_selector.select_predicates(intent, s_cat, o_cat)
        preds = [p for p, _ in candidates][:3]
        tried: List[str] = []

        async def run_one(pred: str) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, Any]]:
            q = json.loads(json.dumps(base_query))
            q['message']['query_graph']['edges'][ekey]['predicates'] = [pred]
            tried.append(pred)
            res, id_map, _ = self.bte.execute_trapi_with_batching(q, predicate=pred, query_intent=intent.value)
            qg = q.get("message", {}).get("query_graph", {})
            return res, id_map, {"predicate": pred, "query_graph": qg}

        tasks = [run_one(p) for p in preds]
        out_lists = await asyncio.gather(*tasks, return_exceptions=False)
        # Deduplicate by (subject_id, predicate, object_id)
        seen: Set[Tuple[Optional[str], Optional[str], Optional[str]]] = set()
        merged: List[Dict[str, Any]] = []
        id_agg: Dict[str, str] = {}
        trapis: List[Dict[str, Any]] = []
        for lst in out_lists:
            # each item is (results, id_map, trapi_info)
            for r in lst[0]:
                key = (r.get('subject_id'), r.get('predicate'), r.get('object_id'))
                if key not in seen:
                    merged.append(r)
                    seen.add(key)
            if isinstance(lst[1], dict):
                id_agg.update({k: v for k, v in lst[1].items() if isinstance(k, str) and isinstance(v, str)})
            if isinstance(lst[2], dict):
                trapis.append(lst[2])
        return merged, tried, id_agg, trapis

    async def _execute_node(self, node: ThoughtNode) -> Dict[str, Any]:
        node.status = "running"
        text = self._inject_dependencies(node)
        failed: List[Dict] = []
        # Merge base entity data for anchoring IDs
        base_entities = dict(self.entity_data_base) if hasattr(self, "entity_data_base") else {}
        base_trapi = self.trapi_builder.build_trapi_query(text, entity_data=base_entities, failed_trapis=failed)
        # Quick validation
        qg = base_trapi.get('message', {}).get('query_graph', {})
        empty_query = False
        if not qg.get('nodes') and not qg.get('edges'):
            node.status = "retired"
            node.result = None
            empty_query = True
        tried_preds: List[str] = []
        results: List[Dict[str, Any]] = []
        trapi_variants: List[Dict[str, Any]] = []
        if not empty_query:
            # Intent detection
            intent = self.predicate_selector.detect_query_intent(text, [])
            results, tried_preds, id_map, trapi_variants = await self._execute_predicate_variants(base_trapi, intent)
            # Update global entity_data_base with any new IDs learned
            if isinstance(id_map, dict):
                try:
                    for k, v in id_map.items():
                        if isinstance(k, str) and isinstance(v, str):
                            self.entity_data_base[k] = v
                except Exception:
                    pass
            if results:
                node.result = results
                self._add_rdf_triples(results)
                node.status = "complete"
            else:
                node.status = "retired"
                node.result = None
        # Return summary for trace
        sample = []
        for r in (results or [])[:5]:
            if isinstance(r, dict):
                sample.append(f"{r.get('subject')} {r.get('predicate')} {r.get('object')}")
        # Capture compact TRAPI views
        base_qg = base_trapi.get("message", {}).get("query_graph", {}) if not empty_query else {}
        variant_qgs = trapi_variants if not empty_query else []
        return {
            "node_id": node.id,
            "content": node.content,
            "dependencies": node.dependencies,
            "status": node.status,
            "tried_predicates": tried_preds,
            "result_count": len(results or []),
            "result_sample": sample,
            "trapi_base": base_qg,
            "trapi_variants": variant_qgs,
        }

    def _is_complete(self) -> bool:
        return all((self.graph.nodes[n].get('data') is not None) and (self.graph.nodes[n]['data'].status in ("complete", "retired")) for n in self.graph.nodes)

    async def execute(self, query: str, iterative: bool = True, max_rounds: int = 3,
                       initial_parallel: int = 2, per_round_parallel: int = 2) -> str:
        await self.decompose_initial_query(query)
        if not iterative:
            it = 0
            while not self._is_complete() and it < 100:
                it += 1
                ready = self._ready_nodes()
                if not ready:
                    # retire any pending nodes to break deadlock
                    for n in self.graph.nodes:
                        data = self.graph.nodes[n].get('data')
                        if data is not None and data.status == 'pending':
                            data.status = 'retired'
                    break
                summaries = await asyncio.gather(*[self._execute_node(nd) for nd in ready])
                self.execution_trace.append({"iteration": it, "executed": summaries})
            return await self._final_answer(query)

        # Iterative planning mode
        round_no = 0
        while round_no < max_rounds and not self._is_complete():
            round_no += 1
            ready = self._ready_nodes()
            if not ready:
                # attempt refinement even if no ready nodes (may add new nodes)
                if not self.offline:
                    await self._refine_plan(query, round_no)
                # re-check
                ready = self._ready_nodes()
                if not ready:
                    # retire stragglers and stop
                    for n in self.graph.nodes:
                        data = self.graph.nodes[n].get('data')
                        if data is not None and data.status == 'pending':
                            data.status = 'retired'
                    break
            # Limit batch size per round
            cap = initial_parallel if round_no == 1 else per_round_parallel
            batch = ready[:max(1, cap)]
            summaries = await asyncio.gather(*[self._execute_node(nd) for nd in batch])
            self.execution_trace.append({"iteration": round_no, "executed": summaries})
            # Refine plan with new results/IDs
            if not self.offline:
                await self._refine_plan(query, round_no)
        return await self._final_answer(query)

    async def _final_answer(self, query: str) -> str:
        # Collect all results
        all_res: List[Dict[str, Any]] = []
        for n in self.graph.nodes:
            data = self.graph.nodes[n].get('data')
            if not data:
                continue
            node: ThoughtNode = data
            if node.status == 'complete' and node.result:
                all_res.extend(node.result)
        turtle = self.rdf_graph.serialize(format="turtle")
        s = get_settings()
        if self.offline:
            # Simple deterministic summary without LLM
            if not all_res:
                return "Insufficient evidence available offline. No results were generated."
            lines = [f"{r.get('subject')} {r.get('predicate').split(':')[-1]} {r.get('object')}" for r in all_res[:5] if isinstance(r, dict)]
            return "Offline summary based on synthesized evidence: " + "; ".join(lines)
        summarizer = ChatOpenAI(model=s.openai_model, api_key=s.openai_api_key, temperature=0.1, max_tokens=s.openai_max_tokens)
        prompt = f"""
You are a biomedical summarizer.
Question: {query}

Evidence (RDF Turtle graph and extracted relationships):
RDF:
{turtle}

Relationships:
{os.linesep.join([f"- {r.get('subject')} {r.get('predicate','biolink:related_to').split(':')[-1]} {r.get('object')}" for r in all_res[:30]])}

Write a concise, mechanistic, evidence-grounded answer. If evidence is insufficient, say so explicitly.
"""
        out = await summarizer.ainvoke([{"role": "user", "content": prompt}])
        return out.content or "[No answer generated]"

    def _recent_relationships(self, limit: int = 20) -> List[str]:
        # Build compact relationships from RDF for prompting
        BL = str(Namespace("https://w3id.org/biolink/vocab/"))
        out: List[str] = []
        labels: Dict[str, str] = {}
        for s, p, o in self.rdf_graph:
            if str(p).endswith("/label"):
                labels[str(s)] = str(o)
        for s, p, o in self.rdf_graph:
            ps = str(p)
            if ps.startswith(BL):
                pred = ps[len(BL):]
                a = labels.get(str(s), str(s))
                b = labels.get(str(o), str(o))
                out.append(f"{a} --{pred}--> {b}")
            if len(out) >= limit:
                break
        return out

    async def _refine_plan(self, user_query: str, round_no: int) -> None:
        """Ask LLM to refine the plan given current statuses, RDF relationships, and known entity IDs.
        Adds new pending nodes and updates dependencies/content of existing pending nodes.
        Completed/retired nodes are not modified.
        """
        try:
            plan = self.get_plan()
            relationships = self._recent_relationships(limit=20)
            entities = dict(self.entity_data_base)
            prompt = (
                "You are refining a biomedical plan of single-hop questions.\n"
                "Rules:\n"
                "- Keep all questions single-hop and aligned to the meta-KG.\n"
                "- Prefer mechanistic relations when possible.\n"
                "- Do not modify nodes that are already complete or retired.\n"
                "- You may add new nodes (use new ids like Q100, Q101, ...).\n"
                "- You may update content/dependencies for PENDING nodes only.\n"
                "- Aim to incorporate new relationships and entity IDs.\n"
                "Return ONLY JSON with fields nodes (id, content, dependencies) and edges (from, to).\n\n"
                f"User question: {user_query}\n"
                f"Current plan: {json.dumps(plan)}\n"
                f"Known entity IDs: {json.dumps(entities)}\n"
                f"New relationships: {json.dumps(relationships)}\n"
            )
            out = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            decomp = self._parse_got_decomposition(out.content or "{}")
            # Apply changes: update pending nodes, add new nodes, merge edges
            existing_ids: Set[str] = set(self.graph.nodes)
            # Update or add nodes
            for nd in decomp.get("nodes", []):
                nid = nd.get("id")
                content = nd.get("content")
                deps = nd.get("dependencies", [])
                if not isinstance(nid, str) or not isinstance(content, str):
                    continue
                if nid in self.graph.nodes:
                    node: ThoughtNode = self.graph.nodes[nid]['data']
                    if node.status == "pending":
                        node.content = content
                        node.dependencies = self._normalize_dependencies(deps)
                else:
                    # add new pending node
                    node = ThoughtNode(id=nid, content=content, dependencies=self._normalize_dependencies(deps))
                    self.graph.add_node(nid, data=node)
            # Merge edges (ignore invalid)
            for ed in decomp.get("edges", []):
                try:
                    u = ed.get("from"); v = ed.get("to")
                    if u in self.graph.nodes and v in self.graph.nodes:
                        self.graph.add_edge(u, v)
                except Exception:
                    continue
        except Exception:
            # Fail-closed: ignore refinement errors
            return

    def get_rdf_triples(self) -> List[Dict[str, str]]:
        """Return RDF triples as list of dicts (subject, predicate, object)."""
        triples: List[Dict[str, str]] = []
        BL = str(Namespace("https://w3id.org/biolink/vocab/"))
        for s, p, o in self.rdf_graph:
            # Create compact forms when possible
            def compact(uri):
                us = str(uri)
                if us.startswith(BL):
                    return f"biolink:{us[len(BL):]}"
                return us
            triples.append({
                "subject": str(s),
                "predicate": compact(p),
                "object": str(o)
            })
        return triples

    def get_plan(self) -> Dict[str, Any]:
        nodes_info = []
        for nid in self.graph.nodes:
            data = self.graph.nodes[nid].get('data')
            if not data:
                continue
            n: ThoughtNode = data
            nodes_info.append({
                "id": n.id,
                "content": n.content,
                "dependencies": n.dependencies,
                "status": n.status,
                "result_count": len(n.result or []),
            })
        edges_info = [{"from": u, "to": v} for u, v in self.graph.edges]
        return {"nodes": nodes_info, "edges": edges_info}

    def explain_mechanistic_chains(self, max_chains: int = 5, max_hops: int = 4) -> List[str]:
        # Build a directed graph from RDF limited to mechanistic predicates
        G = nx.DiGraph()
        BL = str(Namespace("https://w3id.org/biolink/vocab/"))
        mech_tokens = [
            "interacts_with", "affects", "regulates", "participates_in",
            "disrupts", "influences", "increases", "decreases"
        ]
        types: Dict[str, List[str]] = {}
        labels: Dict[str, str] = {}
        # Collect types and labels
        for s, p, o in self.rdf_graph:
            ps = str(p)
            if ps.endswith("/type"):
                types.setdefault(str(s), []).append(str(o))
            if ps.endswith("/label"):
                labels[str(s)] = str(o)
        # Add mechanistic edges
        for s, p, o in self.rdf_graph:
            pred = str(p)
            if pred.startswith(BL):
                pname = pred[len(BL):]
                if any(tok in pname for tok in mech_tokens):
                    G.add_edge(str(s), str(o), predicate=pname)
        # Identify candidate sources/targets
        def is_type(node: str, keyword: str) -> bool:
            for t in types.get(node, []):
                if t.lower().endswith(keyword.lower()):
                    return True
            return False
        sources = [n for n in G.nodes if is_type(n, "SmallMolecule")]
        targets = [n for n in G.nodes if is_type(n, "Disease")]
        chains: List[str] = []
        for s in sources:
            if len(chains) >= max_chains:
                break
            for t in targets:
                if len(chains) >= max_chains:
                    break
                try:
                    paths = nx.all_simple_paths(G, s, t, cutoff=max_hops)
                    for path in paths:
                        if len(chains) >= max_chains:
                            break
                        # Build textual chain
                        segs = []
                        for i in range(len(path)-1):
                            edge_data = G.get_edge_data(path[i], path[i+1]) or {}
                            pred = edge_data.get('predicate', 'related_to')
                            a = labels.get(path[i], path[i])
                            b = labels.get(path[i+1], path[i+1])
                            segs.append(f"{a} --{pred}--> {b}")
                        if segs:
                            chains.append("; ".join(segs))
                except Exception:
                    continue
        return chains

    def get_execution_report(self) -> Dict[str, Any]:
        return {
            "plan": self.get_plan(),
            "iterations": self.execution_trace,
            "mechanistic_chains": self.explain_mechanistic_chains(),
        }

__all__ = ["GoTPlanner", "ThoughtNode"]
