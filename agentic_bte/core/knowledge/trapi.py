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
from .trapi_legacy_builder import LegacyTRAPIBuilder
import requests

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
        # Legacy builder (original LangGraph logic)
        self.legacy_builder = LegacyTRAPIBuilder(openai_api_key=self.openai_api_key)

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
        Legacy-first TRAPI builder: delegate to original LangGraph implementation to avoid malformed queries.
        Returns a single TRAPI query; batching is handled downstream.
        """
        failed_trapis = failed_trapis or []
        # Delegate to legacy builder
        trapi_query = self.legacy_builder.build(query, entity_data or {}, failed_trapis)

        # Optional post-processing (off by default)
        try:
            if getattr(self.settings, 'bp_prefilter_mode', 'off') == 'enforce':
                trapi_query = self._apply_bp_gene_prefilter(trapi_query, entity_data, query)
        except Exception as e:
            logger.debug(f"BP prefilter skipped: {e}")
        try:
            if getattr(self.settings, 'enforce_two_node', False):
                trapi_query = self._enforce_two_node_single_edge(trapi_query, entity_data, query)
        except Exception as e:
            logger.debug(f"Two-node enforcement (builder) skipped: {e}")
        return trapi_query

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
        batch_size = self.settings.trapi_batch_limit
        batches = []
        for i in range(0, len(all_ids), batch_size):
            t = deepcopy(trapi)
            t["message"]["query_graph"]["nodes"][key]["ids"] = all_ids[i:i+batch_size]
            batches.append(t)
        logger.info(f"TRAPI batching (all batches): node={key}, original_count={len(all_ids)}, batch_size={batch_size}, batches={len(batches)}")
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

        # Normalize current predicate list for inspection
        current_preds = edge.get("predicates") or []
        if not isinstance(current_preds, list):
            current_preds = [current_preds] if current_preds else []

        # If there are predicates for the chosen category pair, ensure they are supported; otherwise attempt repairs
        preds = has_coverage(subj_cat, obj_cat)
        if preds:
            # Map common mis-directions to supported forms
            try:
                # Gene -> BiologicalProcess should use participates_in (not has_participant)
                if subj_cat == 'biolink:Gene' and obj_cat == 'biolink:BiologicalProcess':
                    # Prefer participates_in when supported
                    preferred = [p for p in preds if p.endswith(':participates_in')]
                    if preferred:
                        edge["predicates"] = [preferred[0]]
                    else:
                        edge["predicates"] = [preds[0]]
                # SmallMolecule -> Disease should use treats (highest specificity)
                elif subj_cat == 'biolink:SmallMolecule' and obj_cat == 'biolink:Disease':
                    preferred = [p for p in preds if p.endswith(':treats')]
                    edge["predicates"] = [preferred[0] if preferred else preds[0]]
                # SmallMolecule -> Gene should prefer targets if available
                elif subj_cat == 'biolink:SmallMolecule' and obj_cat == 'biolink:Gene':
                    preferred = [p for p in preds if p.endswith(':targets')]
                    edge["predicates"] = [preferred[0] if preferred else preds[0]]
                else:
                    # If current preds are unsupported, set first supported
                    if not any(p in preds for p in current_preds):
                        edge["predicates"] = [preds[0]]
                    elif not current_preds:
                        edge["predicates"] = [preds[0]]
            except Exception:
                # Fallback to first supported predicate
                if not current_preds:
                    edge["predicates"] = [preds[0]]
            return trapi_query

        # If both nodes share the same category, repair to a meaningful pair
        repaired = False
        try:
            def entity_data_has_prefix(prefixes: List[str]) -> bool:
                for _, cid in (entity_data or {}).items():
                    sc = str(cid)
                    if any(sc.startswith(p) for p in prefixes):
                        return True
                return False
            # Normalize identical category pairs
            if subj_cat == obj_cat and isinstance(subj_cat, str):
                # SmallMolecule→SmallMolecule is nonsensical; prefer SmallMolecule→Disease if disease IDs present
                if subj_cat == 'biolink:SmallMolecule':
                    # Clear object ids if they are chemical-like to avoid SM→SM
                    try:
                        if isinstance(obj_node.get('ids'), list) and obj_node.get('ids'):
                            obj_node.pop('ids', None)
                    except Exception:
                        pass
                    if entity_data_has_prefix(['MONDO:', 'DOID:', 'UMLS:']):
                        obj_node['categories'] = ['biolink:Disease']
                        cand = has_coverage('biolink:SmallMolecule', 'biolink:Disease')
                        pref = [p for p in cand if p.endswith(':treats')]
                        edge['predicates'] = [pref[0] if pref else (cand[0] if cand else 'biolink:treats')]
                        repaired = True
                    else:
                        # Fallback: target genes of small molecules
                        obj_node['categories'] = ['biolink:Gene']
                        cand = has_coverage('biolink:SmallMolecule', 'biolink:Gene')
                        pref = [p for p in cand if p.endswith(':targets')]
                        edge['predicates'] = [pref[0] if pref else (cand[0] if cand else 'biolink:affects')]
                        repaired = True
                elif subj_cat == 'biolink:BiologicalProcess':
                    # Standardize to Gene participates_in BiologicalProcess
                    subj_node['categories'] = ['biolink:Gene']
                    obj_node['categories'] = ['biolink:BiologicalProcess']
                    cand = has_coverage('biolink:Gene', 'biolink:BiologicalProcess')
                    pref = [p for p in cand if p.endswith(':participates_in')]
                    edge['predicates'] = [pref[0] if pref else (cand[0] if cand else 'biolink:participates_in')]
                    # Ensure direction Gene→Process
                    edge['subject'], edge['object'] = subj_key, obj_key
                    repaired = True
        except Exception:
            pass
        
        # Repair strategies

        # Strategy A: If the query involves a GO term (BiologicalProcess) on either side,
        # prefer Gene ↔ BiologicalProcess with participates_in (gene→process) or has_participant (process→gene)
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
                # We standardize to Gene (subject) -> BiologicalProcess (object) for participates_in
                if preferred.endswith(':participates_in'):
                    edge["subject"], edge["object"] = (obj_key, subj_key) if proc_on_subject else (subj_key, obj_key)
                repaired = True

        # Strategy B: If the query involves a Disease, prefer SmallMolecule ↔ Disease with treats (small_molecule→disease)
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
                # Fix common misuses: gene→disease with therapeutic predicates → use genetic association
                if subj_cat == 'biolink:Gene' and obj_cat == 'biolink:Disease':
                    assoc_preds = self.find_predicates('biolink:Gene', 'biolink:Disease')
                    preferred = [p for p in assoc_preds if p.endswith(':gene_associated_with_condition') or p.endswith(':genetically_associated_with')]
                    if assoc_preds:
                        edge["predicates"] = [preferred[0] if preferred else assoc_preds[0]]
                        repaired = True
                # Otherwise try flipping
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
            if eid.startswith('CHEBI:') or eid.startswith('ChEMBL:') or eid.startswith('DRUGBANK:') or eid.startswith('UNII:'):
                return 'biolink:ChemicalSubstance'
            if eid.startswith('NCBIGene:') or eid.startswith('HGNC:') or eid.startswith('ENSEMBL:'):
                return 'biolink:Gene'
            if eid.startswith('GO:'):
                return 'biolink:BiologicalProcess'
            if eid.startswith('MONDO:') or eid.startswith('DOID:'):
                return 'biolink:Disease'
            # Do not force-map UMLS to a specific category; leave as-is for LLM/repairs to decide
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

    def _normalize_trapi_ids(self, trapi_query: Dict[str, Any]) -> None:
        """Normalize node IDs using SRI Node Normalization to preferred prefixes per category.
        - Disease: prefer MONDO, DOID, then UMLS
        - BiologicalProcess: GO
        - ChemicalSubstance/Drug: CHEBI, DRUGBANK, ChEMBL, UNII
        - Gene: NCBIGene, HGNC, ENSEMBL
        - Protein: UniProtKB, PR
        """
        qg = trapi_query.get("message", {}).get("query_graph", {})
        nodes = qg.get("nodes", {}) if isinstance(qg.get("nodes"), dict) else {}
        if not nodes:
            return
        # Gather all ids to normalize
        all_ids = []
        node_ids_map = {}
        for nid, ndata in nodes.items():
            ids = ndata.get("ids")
            if isinstance(ids, list) and ids:
                node_ids_map[nid] = [str(x) for x in ids]
                all_ids.extend([str(x) for x in ids])
        if not all_ids:
            return
        # Call Node Normalization in one request (best-effort)
        try:
            url = "https://nodenormalization-sri.renci.org/1.5/get_normalized_nodes"
            payload = {"curies": list(dict.fromkeys(all_ids)), "conflate": True, "description": True}
            r = requests.post(url, json=payload, timeout=15)
            r.raise_for_status()
            nn = r.json() or {}
        except Exception:
            nn = {}
        # Preference order by category
        pref = {
            'biolink:Disease': ["MONDO:", "DOID:", "UMLS:"],
            'biolink:BiologicalProcess': ["GO:"],
            'biolink:ChemicalSubstance': ["CHEBI:", "DRUGBANK:", "ChEMBL", "UNII:"],
            'biolink:Drug': ["CHEBI:", "DRUGBANK:", "ChEMBL", "UNII:"],
            'biolink:Gene': ["NCBIGene:", "HGNC:", "ENSEMBL:"],
            'biolink:Protein': ["UniProtKB:", "PR:"]
        }
        def choose(curie: str, category: str) -> str:
            entry = nn.get(curie)
            if not isinstance(entry, dict):
                return curie
            # Build candidate list: primary id + equivalents
            cands = []
            try:
                main = (entry.get("id") or {}).get("identifier")
                if main:
                    cands.append(main)
            except Exception:
                pass
            for eq in (entry.get("equivalent_identifiers", []) or []):
                eid = eq.get("identifier")
                if eid:
                    cands.append(eid)
            # Pick first matching preferred prefix
            for pfx in pref.get(category, []):
                for cid in cands:
                    if str(cid).startswith(pfx):
                        return cid
            return cands[0] if cands else curie
        # Rewrite node ids in place to normalized, de-duplicated lists
        for nid, ids in node_ids_map.items():
            category = (nodes.get(nid, {}).get("categories") or ["biolink:NamedThing"])[0]
            new_ids = []
            seen = set()
            for cid in ids:
                nc = choose(cid, category)
                if nc not in seen:
                    seen.add(nc)
                    new_ids.append(nc)
            nodes[nid]["ids"] = new_ids

    def _edge_priority(self, subj_cat: Optional[str], obj_cat: Optional[str]) -> int:
        """Heuristic priority for selecting the best edge to keep in a two-node graph."""
        pair = (subj_cat or "", obj_cat or "")
        if pair in (("biolink:ChemicalSubstance", "biolink:Disease"), ("biolink:SmallMolecule", "biolink:Disease")):
            return 0
        if pair in (("biolink:ChemicalSubstance", "biolink:Gene"), ("biolink:SmallMolecule", "biolink:Gene")):
            return 1
        if pair == ("biolink:Gene", "biolink:BiologicalProcess"):
            return 2
        return 10

    def _enforce_two_node_single_edge(self, trapi_query: Dict[str, Any], entity_data: Dict[str, str], query_text: str) -> Dict[str, Any]:
        """Reduce a TRAPI query to exactly one edge and its two referenced nodes using sensible edge selection.
        If the remaining edge is SmallMolecule→SmallMolecule, try to coerce the object to Disease (if any disease ID in
        entity_data) or to Gene as a fallback, and keep the predicates if possible."""
        qg = trapi_query.get("message", {}).get("query_graph", {})
        edges = qg.get("edges", {}) if isinstance(qg.get("edges"), dict) else {}
        nodes = qg.get("nodes", {}) if isinstance(qg.get("nodes"), dict) else {}
        if not edges or not nodes:
            return trapi_query
        # Choose best edge
        best = None
        best_score = 999
        for eid, e in edges.items():
            sj = e.get("subject"); ob = e.get("object")
            sc = (nodes.get(sj, {}).get("categories") or [None])[0]
            oc = (nodes.get(ob, {}).get("categories") or [None])[0]
            score = self._edge_priority(sc, oc)
            if score < best_score:
                best = (eid, e, sj, ob, sc, oc)
                best_score = score
        if not best:
            return trapi_query
        _, edge, sj, ob, sc, oc = best
        # Keep only the chosen edge (normalize edge id)
        qg["edges"] = {"e01": {"subject": sj, "object": ob, "predicates": edge.get("predicates", [])}}
        # Keep referenced nodes only
        kept = {}
        if sj in nodes: kept[sj] = nodes[sj]
        if ob in nodes: kept[ob] = nodes[ob]
        qg["nodes"] = kept
        # Coerce SM→SM to a more useful pair
        if (sc in ("biolink:SmallMolecule", "biolink:ChemicalSubstance")) and (oc in ("biolink:SmallMolecule", "biolink:ChemicalSubstance")):
            # If entity_data contains a disease-like ID, coerce object to Disease
            has_dis = any(str(v).startswith(("MONDO:", "DOID:", "UMLS:")) for v in (entity_data or {}).values())
            if has_dis:
                if ob in qg["nodes"]:
                    qg["nodes"][ob]["categories"] = ["biolink:Disease"]
            else:
                if ob in qg["nodes"]:
                    qg["nodes"][ob]["categories"] = ["biolink:Gene"]
        return trapi_query

    def _prefilter_genes_by_biological_process(self, go_ids: List[str], max_results: int = 1000) -> List[str]:
        """Fetch Gene IDs that participate_in any of the provided GO BiologicalProcess IDs via BTE."""
        if not go_ids:
            return []
        # Build a TRAPI: Gene --participates_in--> BiologicalProcess (ids = go_ids)
        q = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"categories": ["biolink:Gene"]},
                        "n1": {"categories": ["biolink:BiologicalProcess"], "ids": list(dict.fromkeys(go_ids))},
                    },
                    "edges": {
                        "e01": {"subject": "n0", "object": "n1", "predicates": ["biolink:participates_in"]}
                    }
                }
            }
        }
        try:
            resp = self.bte_client.execute_trapi_with_batching(q, max_results=max_results, k=0)
            results = resp[0] if isinstance(resp, tuple) else []
            genes = []
            for r in results:
                if (r.get("subject_type") or "").endswith(":Gene"):
                    gid = r.get("subject_id")
                    if gid and gid not in genes:
                        genes.append(gid)
                elif (r.get("object_type") or "").endswith(":Gene"):
                    gid = r.get("object_id")
                    if gid and gid not in genes:
                        genes.append(gid)
            return genes
        except Exception:
            return []

    def _apply_bp_gene_prefilter(self, trapi_query: Dict[str, Any], entity_data: Dict[str, str], query_text: str) -> Dict[str, Any]:
        """If a GO BiologicalProcess is present in entity_data, prefilter genes participating in it and
        rebuild the TRAPI as ChemicalSubstance→Gene with those gene IDs (predicate: targets if available)."""
        try:
            go_ids = [str(v) for v in (entity_data or {}).values() if isinstance(v, str) and v.startswith("GO:")]
            if not go_ids:
                return trapi_query
            gene_ids = self._prefilter_genes_by_biological_process(go_ids)
            if not gene_ids:
                return trapi_query
            # Rebuild graph to ChemicalSubstance -> Gene, ids on the Gene node
            qg = trapi_query.get("message", {}).get("query_graph", {})
            qg["nodes"] = {
                "n0": {"categories": ["biolink:ChemicalSubstance"]},
                "n1": {"categories": ["biolink:Gene"], "ids": gene_ids[:500]},
            }
            # Choose predicate with coverage
            preds = self.find_predicates("biolink:ChemicalSubstance", "biolink:Gene")
            preferred = next((p for p in preds if p.endswith(":targets")), None)
            pred = preferred or (preds[0] if preds else "biolink:affects")
            qg["edges"] = {"e01": {"subject": "n0", "object": "n1", "predicates": [pred]}}
            return trapi_query
        except Exception:
            return trapi_query

    def identify_nodes(self, query: str) -> Dict[str, str]:
        """
        LLM-driven, context-sensitive determination of subject/object node types based on query (LangGraph Prototype logic)
        """
        node_types = [
            "biolink:Disease", "biolink:Gene", "biolink:ChemicalSubstance", "biolink:Drug", "biolink:PathologicalProcess",
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
        # Basic sanity: require at least one edge (multi-edge/multi-hop allowed)
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
            'biolink:ChemicalSubstance': ['CHEBI:', 'ChEMBL:', 'DRUGBANK:', 'UNII:'],
            'biolink:Drug': ['CHEBI:', 'ChEMBL:', 'DRUGBANK:', 'UNII:'],
            'biolink:SmallMolecule': ['CHEBI:', 'ChEMBL:', 'DRUGBANK:', 'UNII:'],
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
            # Collect IDs without prefix filtering; prioritize those whose names appear in the query text
            out = []
            seen = set()
            for scid in matched_ids:
                if scid not in seen:
                    seen.add(scid)
                    out.append(scid)
                    if len(out) >= limit:
                        return out
            # Then include all IDs from the global entity registry
            for _, cid in (entity_data or {}).items():
                scid = str(cid)
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
            source = subj if subj_has else obj
            target = obj if subj_has else subj
            # Avoid creating SmallMolecule→SmallMolecule by skipping injection when both categories are SmallMolecule
            try:
                src_cat = (nodes.get(source, {}).get('categories') or [None])[0]
                tgt_cat = (nodes.get(target, {}).get('categories') or [None])[0]
                if src_cat == tgt_cat == 'biolink:SmallMolecule':
                    return
            except Exception:
                pass
            t_ids = collect_for(target)
            if t_ids:
                nodes.setdefault(target, {})['ids'] = t_ids
            return

        # Case 3: both have ids → do nothing (avoid exploding combinations here)
        return
