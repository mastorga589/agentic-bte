"""
TRAPI Query Builder - Build TRAPI queries from natural language

This module provides functionality to convert natural language biomedical
queries into TRAPI (Translator Reasoner API) format using LLMs and knowledge
graph metadata.

Migrated and enhanced from the original BTE-LLM implementation, now with strict batching/context logic.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from copy import deepcopy
from langchain_openai import ChatOpenAI
from ...config.settings import get_settings
from ...exceptions.base import ExternalServiceError
from .bte_client import BTEClient

logger = logging.getLogger(__name__)

def build_trapi_query(query: str, entity_data: Dict[str, str], failed_trapis: List[Dict] = None) -> Dict[str, Any]:
    """Convenience wrapper for class-based builder (for legacy/compat use)"""
    builder = TRAPIQueryBuilder()
    return builder.build_trapi_query(query, entity_data, failed_trapis)

class TRAPIQueryBuilder:
    def __init__(self, openai_api_key: Optional[str] = None):
        self.settings = get_settings()
        self.openai_api_key = openai_api_key or self.settings.openai_api_key
        if not self.openai_api_key:
            raise ExternalServiceError("OpenAI API key is required for TRAPI query building")
        self.llm = ChatOpenAI(
            temperature=0,
            model=self.settings.openai_model,
            api_key=self.openai_api_key
        )
        self.bte_client = BTEClient()
        self._meta_kg = None

    @property
    def meta_kg(self) -> Dict[str, Any]:
        if self._meta_kg is None:
            self._meta_kg = self.bte_client.get_meta_knowledge_graph()
        return self._meta_kg

    def extract_dict(self, raw_string: str) -> Dict[str, Any]:
        raw_string = raw_string.strip()
        if raw_string.startswith("```json"):
            raw_string = raw_string[7:]
            if raw_string.endswith("```"):
                raw_string = raw_string[:-3]
            raw_string = raw_string.strip()
        elif raw_string.startswith("```"):
            lines = raw_string.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            raw_string = '\n'.join(lines).strip()
        start_index = raw_string.find("{")
        if start_index != -1:
            brace_count = 0
            end_index = start_index
            for i, char in enumerate(raw_string[start_index:], start_index):
                if char == '{': brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_index = i
                        break
            if brace_count == 0:
                extracted_dict = raw_string[start_index:end_index + 1].strip()
                try:
                    return json.loads(extracted_dict)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}")
        try:
            return json.loads(raw_string)
        except json.JSONDecodeError:
            logger.error(f"Could not parse as JSON: {raw_string[:200]}...")
            return {"error": "could not parse dict"}

    def find_predicates(self, subject: str, obj: str) -> List[str]:
        predicates = []
        meta_edges = self.meta_kg.get("edges", [])
        excluded_predicates = self.settings.excluded_predicates
        for edge in meta_edges:
            if (edge.get("subject", "").lower() == subject.lower() and edge.get("object", "").lower() == obj.lower()):
                predicate = edge.get("predicate")
                if predicate and predicate not in excluded_predicates:
                    predicates.append(predicate)
        logger.debug(f"Found {len(predicates)} predicates for {subject} -> {obj}")
        return predicates

    def choose_predicate(self, predicate_list: List[str], query: str) -> str:
        if not predicate_list:
            logger.warning("No predicates available for selection")
            return ""
        if len(predicate_list) == 1:
            return predicate_list[0]
        try:
            predicate_prompt = f"""Choose the most specific predicate by examining the query closely and choosing the closest answer.\n\nHere is the query: {query}\nHere are the available predicates: {predicate_list}\nReturn only the chosen predicate string.\n"""
            response = self.llm.invoke(predicate_prompt)
            chosen = response.content.strip()
            for predicate in predicate_list:
                if predicate in chosen:
                    logger.debug(f"Selected predicate '{predicate}' for query: {query}")
                    return predicate
            logger.warning(f"No exact predicate match found, using first: {predicate_list[0]}")
            return predicate_list[0]
        except Exception as e:
            logger.error(f"Error choosing predicate: {e}")
            return predicate_list[0] if predicate_list else ""

    def build_trapi_query(self, query: str, entity_data: Dict[str, str], failed_trapis: List[Dict] = None) -> Dict[str, Any]:
        """
        Direct LangGraph/Prototype TRAPI batch/context builder (LLM-driven with all relevant prompt context).
        Returns a single TRAPI query (first batch). Use build_trapi_query_batches for all batches.
        """
        failed_trapis = failed_trapis or []
        trapi_example = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"categories": ["biolink:Disease"], "ids": ["MONDO:0005148"]},
                        "n1": {"categories": ["biolink:SmallMolecule"]},
                    },
                    "edges": {
                        "e01": {"subject": "n1", "object": "n0", "predicates": ["biolink:treats"]}
                    }
                }
            }
        }
        # Important biolink category mappings for the LLM
        category_guidance = """
IMPORTANT BIOLINK CATEGORY MAPPINGS:
- Drugs, medications, compounds, chemicals → biolink:SmallMolecule (NOT biolink:Drug!)
- Diseases, disorders, conditions → biolink:Disease
- Genes → biolink:Gene  
- Proteins → biolink:Protein
- Biological processes, pathways → biolink:BiologicalProcess
- Phenotypes, symptoms → biolink:PhenotypicFeature
"""
        
        # Compose full LLM prompt for prompt transparency
        prompt = f"""
You are a biomedical knowledge graph assistant and need to appropriately construct a TRAPI batch query.

{category_guidance}

- Query: {query}
- Entities and IDs: {json.dumps(entity_data, indent=2)}
Example TRAPI: {json.dumps(trapi_example, indent=2)}

Rules:
1. Use biolink:SmallMolecule for drugs/chemicals, NOT biolink:Drug
2. Only create one 'ids' field on one node
3. Set IDs for the most informative/relevant node for this subquery
4. Use correct subject-predicate-object direction

Here are failed attempts: {failed_trapis}
Output only a valid JSON TRAPI query (no surrounding text or explanations).
"""
        llm_result = self.llm.invoke(prompt).content.strip()
        trapi_query = self.extract_dict(llm_result)
        # Validate and repair with meta-KG coverage awareness
        try:
            trapi_query = self._validate_and_repair_trapi(trapi_query, query, entity_data)
        except Exception as e:
            logger.warning(f"TRAPI validation/repair failed, using raw LLM output: {e}")
        
        # Sanitize categories and enforce single 'ids' node
        try:
            trapi_query = self._sanitize_trapi_categories_and_ids(trapi_query)
        except Exception as e:
            logger.debug(f"Sanitization skipped: {e}")
        
        # Inject IDs from entity_data when possible (ensure at most one node has ids)
        try:
            self._inject_ids_from_entity_data(trapi_query, entity_data, query)
        except Exception as e:
            logger.debug(f"ID injection skipped: {e}")
        
        # Batching (Prototype pattern, >50 IDs)
        key = None
        if "n0" in trapi_query.get("message", {}).get("query_graph", {}).get("nodes", {}):
            if "ids" in trapi_query["message"]["query_graph"]["nodes"]["n0"]:
                key = "n0"
            elif "ids" in trapi_query["message"]["query_graph"]["nodes"].get("n1", {}):
                key = "n1"
        batch_size = 50
        batched_queries = []
        if key:
            all_ids = trapi_query["message"]["query_graph"]["nodes"][key]["ids"]
            for i in range(0, len(all_ids), batch_size):
                trapi_copy = deepcopy(trapi_query)
                trapi_copy["message"]["query_graph"]["nodes"][key]["ids"] = all_ids[i:i + batch_size]
                batched_queries.append(trapi_copy)
        else:
            batched_queries = [trapi_query]
        logger.info(f"\n====== [TRAPI PROMPT CONTEXT] ======\n{prompt}\n======\n")
        logger.info(f"LLM TRAPI output: {json.dumps(trapi_query,indent=2)}")
        logger.info(f"Batch count: {len(batched_queries)}")
        # Return the first batch; executor handles splitting into multiple requests when needed
        return batched_queries[0] if batched_queries else trapi_query

    def build_trapi_query_batches(self, query: str, entity_data: Dict[str, str], failed_trapis: List[Dict] = None) -> List[Dict[str, Any]]:
        """Build TRAPI queries and return all batches (each <=50 ids on the ids-carrying node)."""
        failed_trapis = failed_trapis or []
        trapi = self.build_trapi_query(query, entity_data, failed_trapis)
        # Recompute batching on sanitized/injected query to produce all batches
        qg = trapi.get("message", {}).get("query_graph", {})
        nodes = qg.get("nodes", {})
        key = None
        if "n0" in nodes and isinstance(nodes["n0"].get("ids"), list):
            key = "n0"
        elif "n1" in nodes and isinstance(nodes["n1"].get("ids"), list):
            key = "n1"
        if not key:
            return [trapi]
        all_ids = nodes[key].get("ids", [])
        batch_size = 50
        batches = []
        for i in range(0, len(all_ids), batch_size):
            t = deepcopy(trapi)
            t["message"]["query_graph"]["nodes"][key]["ids"] = all_ids[i:i+batch_size]
            batches.append(t)
        return batches

    def _validate_and_repair_trapi(self, trapi_query: Dict[str, Any], user_query: str, entity_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Ensure the LLM-produced TRAPI query uses subject/object category pairs and predicates
        that are supported by the BTE meta knowledge graph. If coverage appears weak (no predicates
        available for the selected categories), rewrite categories/predicates using robust fallbacks.
        """
        qg = trapi_query.get("message", {}).get("query_graph", {})
        nodes = qg.get("nodes", {})
        edges = qg.get("edges", {})
        if not nodes or not edges:
            return trapi_query

        # We assume a single edge (common in subqueries)
        edge_key = next(iter(edges.keys()))
        edge = edges[edge_key]
        subj_key = edge.get("subject")
        obj_key = edge.get("object")
        if subj_key not in nodes or obj_key not in nodes:
            return trapi_query

        subj_node = nodes[subj_key]
        obj_node = nodes[obj_key]
        subj_cat = (subj_node.get("categories") or [None])[0]
        obj_cat = (obj_node.get("categories") or [None])[0]

        # Helper to check coverage via meta-KG
        def has_coverage(scat: str, ocat: str) -> List[str]:
            if not scat or not ocat:
                return []
            preds = self.find_predicates(scat, ocat)
            return preds

        # If there are no predicates for the chosen category pair, attempt repairs
        preds = has_coverage(subj_cat, obj_cat)
        if preds:
            # Ensure predicates are set when missing or clearly unsupported
            if not edge.get("predicates"):
                edge["predicates"] = [preds[0]]
            return trapi_query

        # Repair strategies
        repaired = False

        # Strategy A: If the query involves a GO term (BiologicalProcess) on either side,
        # prefer Gene ↔ BiologicalProcess with participates_in/related_to
        def is_biological_process(node):
            cats = node.get("categories", [])
            return any(c for c in cats if isinstance(c, str) and c.endswith(":BiologicalProcess"))

        if is_biological_process(subj_node) or is_biological_process(obj_node):
            # Identify which side is the process
            proc_on_subject = is_biological_process(subj_node)
            proc_node = subj_node if proc_on_subject else obj_node
            other_node = obj_node if proc_on_subject else subj_node

            # Rewrite other side to Gene for robust coverage
            other_node["categories"] = ["biolink:Gene"]
            # Check coverage for Gene -> BiologicalProcess
            subj_cand = "biolink:Gene" if proc_on_subject else proc_node.get("categories")[0]
            obj_cand = proc_node.get("categories")[0] if proc_on_subject else "biolink:Gene"
            cand_preds = has_coverage(subj_cand, obj_cand)
            # Choose a robust predicate if available
            preferred = None
            for p in cand_preds:
                if p.endswith(":participates_in"):
                    preferred = p
                    break
            if not preferred and cand_preds:
                preferred = cand_preds[0]
            if preferred:
                edge["predicates"] = [preferred]
                # Ensure subject/object alignment with the chosen direction
                if proc_on_subject:
                    # We want Gene (subject) -> BiologicalProcess (object)
                    edge["subject"], edge["object"] = obj_key, subj_key
                repaired = True

        # Strategy B: If the query involves a Disease, prefer SmallMolecule ↔ Disease with treats
        def is_disease(node):
            cats = node.get("categories", [])
            return any(c for c in cats if isinstance(c, str) and c.endswith(":Disease"))

        if not repaired and (is_disease(subj_node) or is_disease(obj_node)):
            dis_on_subject = is_disease(subj_node)
            dis_node = subj_node if dis_on_subject else obj_node
            other_node = obj_node if dis_on_subject else subj_node

            # Rewrite other side to SmallMolecule
            other_node["categories"] = ["biolink:SmallMolecule"]
            subj_cand = other_node.get("categories")[0]
            obj_cand = dis_node.get("categories")[0]
            cand_preds = has_coverage(subj_cand, obj_cand)
            preferred = None
            for p in cand_preds:
                if p.endswith(":treats"):
                    preferred = p
                    break
            if not preferred and cand_preds:
                preferred = cand_preds[0]
            if preferred:
                edge["predicates"] = [preferred]
                # Subject should be SmallMolecule, object Disease
                if dis_on_subject:
                    edge["subject"], edge["object"] = obj_key, subj_key
                repaired = True

        # If still no coverage, try flipping direction or generic related_to
        if not repaired:
            flipped_preds = has_coverage(obj_cat, subj_cat)
            if flipped_preds:
                edge["predicates"] = [flipped_preds[0]]
                edge["subject"], edge["object"] = obj_key, subj_key
                repaired = True

        if not repaired:
            # Fallback: set predicate to related_to (if allowed), keep original direction
            # This is a last resort when meta-KG doesn't list the pair (LLM might still be right)
            edge["predicates"] = ["biolink:related_to"]

        return trapi_query

    def _sanitize_trapi_categories_and_ids(self, trapi_query: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure each node has exactly one valid category and only one node carries 'ids'.
        - If multiple categories, prefer more specific ones (drop biolink:NamedThing).
        - If both biolink:Gene and biolink:Protein present, prefer biolink:Gene.
        - If multiple nodes have 'ids', keep on the first and remove from others.
        - If a node has ids, align its category to the ID namespace (CHEBI→SmallMolecule, NCBIGene/HGNC→Gene, GO→BiologicalProcess, MONDO/DOID/UMLS→Disease).
        """
        qg = trapi_query.get("message", {}).get("query_graph", {})
        nodes = qg.get("nodes", {})
        if not isinstance(nodes, dict):
            return trapi_query

        def category_for_id(eid: str) -> Optional[str]:
            if not isinstance(eid, str):
                return None
            if eid.startswith('CHEBI:') or eid.startswith('ChEMBL:') or eid.startswith('DRUGBANK:'):
                return 'biolink:SmallMolecule'
            if eid.startswith('NCBIGene:') or eid.startswith('HGNC:') or eid.startswith('ENSEMBL:'):
                return 'biolink:Gene'
            if eid.startswith('GO:'):
                return 'biolink:BiologicalProcess'
            if eid.startswith('MONDO:') or eid.startswith('DOID:') or eid.startswith('UMLS:'):
                return 'biolink:Disease'
            if eid.startswith('PR:') or eid.startswith('UniProtKB:'):
                return 'biolink:Protein'
            return None

        # Normalize categories and strip non-TRAPI fields; align categories to ids when present
        for nid, ndata in nodes.items():
            # Remove non-standard fields such as 'name'
            if 'name' in ndata:
                ndata.pop('name', None)
            cats = ndata.get("categories", [])
            if isinstance(cats, list) and cats:
                # Remove duplicates while preserving order
                seen = set()
                unique = []
                for c in cats:
                    if c not in seen and isinstance(c, str):
                        seen.add(c)
                        unique.append(c)
                # Drop NamedThing if more specific exists
                if len(unique) > 1 and "biolink:NamedThing" in unique:
                    unique = [c for c in unique if c != "biolink:NamedThing"]
                # Prefer Gene over Protein if both present
                if "biolink:Gene" in unique and "biolink:Protein" in unique:
                    unique = [c for c in unique if c != "biolink:Protein"]
                # Keep only the first category
                ndata["categories"] = [unique[0]] if unique else ["biolink:NamedThing"]
            # Align category to ID namespace if this node already has ids
            ids = ndata.get('ids')
            if isinstance(ids, list) and ids:
                cat = category_for_id(str(ids[0]))
                if cat:
                    ndata['categories'] = [cat]

        # Enforce single 'ids' node
        nodes_with_ids = [nid for nid, ndata in nodes.items() if isinstance(ndata.get("ids"), list) and ndata.get("ids")]
        if len(nodes_with_ids) > 1:
            # Keep IDs on the first node, remove from the rest
            for nid in nodes_with_ids[1:]:
                nodes[nid].pop("ids", None)
        
        # Prune dangling nodes: keep only nodes referenced by edges (single-hop -> exactly two nodes)
        edges = qg.get("edges", {})
        try:
            if isinstance(edges, dict) and edges:
                # collect referenced node ids
                referenced = set()
                for e_id, e_data in edges.items():
                    sk = e_data.get("subject")
                    ok = e_data.get("object")
                    if sk:
                        referenced.add(sk)
                    if ok:
                        referenced.add(ok)
                # keep only referenced nodes
                pruned_nodes = {nid: ndata for nid, ndata in nodes.items() if nid in referenced}
                if pruned_nodes:
                    qg["nodes"] = pruned_nodes
        except Exception:
            # Non-fatal: leave as-is if anything unexpected occurs
            pass
        
        return trapi_query

    def identify_nodes(self, query: str) -> Dict[str, str]:
        """
        LLM-driven, context-sensitive determination of subject/object node types based on query (LangGraph Prototype logic)
        """
        node_types = [
            "biolink:Disease", "biolink:Gene", "biolink:SmallMolecule", "biolink:PathologicalProcess",
            "biolink:PhysiologicalProcess", "biolink:Polypeptide", "biolink:BiologicalEntity", "biolink:PhenotypicFeature"
        ]
        prompt = f"""
Help construct a TRAPI query.
- Query: {query}
Available node types:
{node_types}
For the given query, determine which node type should be subject and which should be object, using the vocabulary above. Output only a dictionary: {{'subject': ..., 'object': ...}}. Example output: {{'subject': 'biolink:Disease', 'object': 'biolink:SmallMolecule'}}
"""
        llm_result = self.llm.invoke(prompt).content.strip()
        node_map = self.extract_dict(llm_result)
        if not isinstance(node_map, dict) or 'subject' not in node_map or 'object' not in node_map:
            logger.warning(f"LLM failed to parse node types for query '{query}'. Falling back to Disease-SmallMolecule.")
            return {"subject": "biolink:Disease", "object": "biolink:SmallMolecule"}
        return node_map

# Simple TRAPI validation function used by MCP TRAPI tool
from typing import Tuple

def validate_trapi(trapi_query: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate TRAPI query structure and enforce minimal checks.
    Returns (is_valid, message).
    """
    try:
        if not isinstance(trapi_query, dict):
            return False, "TRAPI query must be a dict"
        message = trapi_query.get("message")
        if not isinstance(message, dict):
            return False, "Missing 'message' key"
        qg = message.get("query_graph")
        if not isinstance(qg, dict):
            return False, "Missing 'query_graph' key in message"
        nodes = qg.get("nodes", {})
        edges = qg.get("edges", {})
        if not isinstance(nodes, dict) or not isinstance(edges, dict):
            return False, "'nodes' or 'edges' is not a dict"
        # Basic single-hop sanity: at most 1 edge
        if len(edges) > 1:
            return False, f"Multi-hop query detected: {len(edges)} edges found"
        if len(edges) == 0:
            return False, "No edges specified"
        # Ensure referenced nodes exist and have categories
        (edge_id, edge_data) = next(iter(edges.items()))
        subj = edge_data.get("subject")
        obj = edge_data.get("object")
        if subj not in nodes or obj not in nodes:
            return False, f"Edge {edge_id} references missing nodes"
        for nid, ndata in nodes.items():
            cats = ndata.get("categories")
            if not isinstance(cats, list) or not cats or not all(isinstance(c, str) for c in cats):
                return False, f"Node {nid} has invalid categories"
        return True, "Valid single-hop TRAPI query"
    except Exception as e:
        return False, f"Validation error: {e}"

    
    # ID injection helper
    def _inject_ids_from_entity_data(self, trapi_query: Dict[str, Any], entity_data: Dict[str, str], query_text: str) -> None:
        """Inject category-consistent IDs from entity_data onto missing node(s).
        Behavior:
        - If neither node has ids: inject onto one node using category-consistent prefixes (legacy behavior).
        - If exactly one node already has ids: inject onto the other node (unblocks autonomous multi-hop chaining).
        """
        if not isinstance(trapi_query, dict):
            return
        qg = trapi_query.get("message", {}).get("query_graph", {})
        nodes = qg.get("nodes", {})
        edges = qg.get("edges", {})
        if not nodes or not edges:
            return
        node_keys = list(nodes.keys())
        if len(node_keys) < 2:
            return

        prefix_map = {
            'biolink:Disease': ['MONDO:', 'DOID:', 'UMLS:'],
            'biolink:SmallMolecule': ['CHEBI:', 'ChEMBL:', 'DRUGBANK:'],
            'biolink:Gene': ['HGNC:', 'NCBIGene:', 'ENSEMBL:'],
            'biolink:Protein': ['PR:', 'UniProtKB:'],
            'biolink:BiologicalProcess': ['GO:']
        }
        # Heuristic: prefer IDs whose source names occur in the query text (use entity_data keys)
        qt = (query_text or "").lower()
        matched_ids = []
        try:
            for name, cid in (entity_data or {}).items():
                if isinstance(name, str) and name and name.lower() in qt:
                    matched_ids.append(str(cid))
        except Exception:
            matched_ids = []

        def collect_for(node_key: str, limit: int = 500) -> list:
            cat = (nodes.get(node_key, {}).get('categories') or [None])[0]
            if not cat:
                return []
            # First, use ids whose names appear in the query text
            out = []
            seen = set()
            for scid in matched_ids:
                if scid not in seen:
                    seen.add(scid)
                    out.append(scid)
                    if len(out) >= limit:
                        return out
            # Fallback to category-consistent pool from entity_data (keeps autonomy but prevents mismatched namespaces)
            prefixes = prefix_map.get(cat, [])
            for _, cid in (entity_data or {}).items():
                scid = str(cid)
                if any(scid.startswith(p) for p in prefixes):
                    if scid not in seen:
                        seen.add(scid)
                        out.append(scid)
                        if len(out) >= limit:
                            break
            return out

        # Identify subject/object from the single edge
        e_id, e_data = next(iter(edges.items()))
        subj = e_data.get('subject', node_keys[0])
        obj = e_data.get('object', node_keys[1] if len(node_keys) > 1 else node_keys[0])

        # Determine pinned state
        def has_ids(nk: str) -> bool:
            v = nodes.get(nk, {}).get('ids')
            return isinstance(v, list) and len(v) > 0
        subj_has = has_ids(subj)
        obj_has = has_ids(obj)

        # Case 1: neither has ids → legacy behavior (inject on one node)
        if not subj_has and not obj_has:
            subj_ids = collect_for(subj)
            if subj_ids:
                nodes.setdefault(subj, {})['ids'] = subj_ids
                return
            obj_ids = collect_for(obj)
            if obj_ids:
                nodes.setdefault(obj, {})['ids'] = obj_ids
            return

        # Case 2: exactly one side has ids → inject on the other to enable chaining
        if subj_has != obj_has:
            target = obj if subj_has else subj
            t_ids = collect_for(target)
            if t_ids:
                nodes.setdefault(target, {})['ids'] = t_ids
            return

        # Case 3: both have ids → do nothing (avoid exploding combinations here)
        return
