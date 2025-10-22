"""
TRAPI Query Builder for stand-alone Prototype.
Builds a TRAPI query via LLM with meta-KG validation and light repair.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
import os

from langchain_openai import ChatOpenAI

from .settings import get_settings
from .bte_client import BTEClient

logger = logging.getLogger(__name__)


class TRAPIQueryBuilder:
    def __init__(self, openai_api_key: Optional[str] = None):
        s = get_settings()
        # Offline fallback if no API key or explicit env switch
        offline = (not (openai_api_key or s.openai_api_key)) or (os.getenv("PROTOTYPE_OFFLINE", "0") in ("1", "true", "True"))
        if not offline:
            self.llm = ChatOpenAI(
                temperature=s.openai_temperature,
                model=s.openai_model,
                api_key=openai_api_key or s.openai_api_key,
                max_tokens=s.openai_max_tokens,
            )
        else:
            self.llm = None
        self.bte = BTEClient()
        self._meta_kg: Optional[Dict[str, Any]] = None

    @property
    def meta_kg(self) -> Dict[str, Any]:
        if self._meta_kg is None:
            self._meta_kg = self.bte.get_meta_knowledge_graph()
        return self._meta_kg

    def _extract_dict(self, raw: str) -> Dict[str, Any]:
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[len("```json"):].strip()
            if raw.endswith("```"):
                raw = raw[:-3].strip()
        elif raw.startswith("```") and raw.endswith("```"):
            raw = raw[3:-3].strip()
        # try to find first {...}
        start = raw.find("{")
        if start != -1:
            depth = 0
            end = start
            for i, ch in enumerate(raw[start:], start):
                if ch == '{': depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            try:
                return json.loads(raw[start:end+1])
            except Exception:
                pass
        try:
            return json.loads(raw)
        except Exception:
            return {"error": "could not parse dict"}

    def _find_predicates(self, subject_cat: str, object_cat: str) -> List[str]:
        preds: List[str] = []
        for e in self.meta_kg.get("edges", []):
            if e.get("subject") == subject_cat and e.get("object") == object_cat:
                p = e.get("predicate")
                if p:
                    preds.append(p)
        return list(dict.fromkeys(preds))  # dedupe preserve order

    def build_trapi_query(self, query: str, entity_data: Dict[str, str] | None, failed_trapis: List[Dict] | None = None) -> Dict[str, Any]:
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
                    },
                }
            }
        }

        category_guidance = """
IMPORTANT BIOLINK CATEGORY MAPPINGS:
- Drugs, medications, compounds, chemicals → biolink:SmallMolecule (NOT biolink:Drug)
- Diseases, disorders, conditions → biolink:Disease
- Genes → biolink:Gene
- Proteins → biolink:Protein
- Biological processes, pathways → biolink:BiologicalProcess
- Phenotypes, symptoms → biolink:PhenotypicFeature
"""

        prompt = f"""
You are a biomedical knowledge graph assistant and must construct a TRAPI query.

{category_guidance}

- Query: {query}
- Entities and IDs: {json.dumps(entity_data or {}, indent=2)}
- Example TRAPI: {json.dumps(trapi_example, indent=2)}

Rules:
1. Use biolink:SmallMolecule for drugs/chemicals; do NOT use biolink:Drug
2. Create only one 'ids' field on exactly one node
3. Put IDs on the node that is most informative for this subquery
4. Use correct subject-predicate-object direction
5. Avoid repeating failed TRAPI attempts: {failed_trapis}

Output only a valid JSON TRAPI object, with no commentary.
"""
        # Offline: synthesize a minimal but valid TRAPI if no LLM
        if self.llm is None:
            n0_cat = "biolink:Disease"
            n1_cat = "biolink:SmallMolecule" if ("drug" in (query or "").lower() or "molecule" in (query or "").lower()) else "biolink:Gene"
            trapi = {
                "message": {
                    "query_graph": {
                        "nodes": {
                            "n0": {"categories": [n0_cat]},
                            "n1": {"categories": [n1_cat]},
                        },
                        "edges": {
                            "e01": {"subject": "n1", "object": "n0", "predicates": ["biolink:related_to"]}
                        }
                    }
                }
            }
            # Use entity_data if provided to set ids for one node
            ed = entity_data or {}
            if ed:
                # pick first id
                try:
                    first_key = next(iter(ed.keys()))
                    trapi["message"]["query_graph"]["nodes"]["n0"]["ids"] = [ed[first_key]] if isinstance(ed[first_key], str) else ed[first_key]
                except Exception:
                    pass
            return trapi
        out = self.llm.invoke(prompt)
        trapi = self._extract_dict(out.content or "{}")
        trapi = self._validate_and_repair(trapi, query, entity_data or {})
        return trapi

    def _validate_and_repair(self, trapi: Dict[str, Any], user_query: str, entity_data: Dict[str, str]) -> Dict[str, Any]:
        qg = trapi.get("message", {}).get("query_graph", {})
        nodes = qg.get("nodes", {})
        edges = qg.get("edges", {})
        if not nodes or not edges:
            return trapi
        edge_key = next(iter(edges.keys()))
        edge = edges[edge_key]
        s_key = edge.get("subject")
        o_key = edge.get("object")
        if s_key not in nodes or o_key not in nodes:
            return trapi
        s_node = nodes[s_key]
        o_node = nodes[o_key]
        s_cat = (s_node.get("categories") or [None])[0]
        o_cat = (o_node.get("categories") or [None])[0]

        preds = self._find_predicates(s_cat, o_cat) if s_cat and o_cat else []
        if preds and not edge.get("predicates"):
            edge["predicates"] = [preds[0]]
            return trapi

        # Repair: if BP on either side, favor Gene↔BiologicalProcess with participates_in or related_to
        def is_bp(node: Dict[str, Any]) -> bool:
            return any(isinstance(c, str) and c.endswith(":BiologicalProcess") for c in node.get("categories", []))

        if is_bp(s_node) or is_bp(o_node):
            # force the non-BP side to Gene for better coverage
            if is_bp(s_node):
                o_node["categories"] = ["biolink:Gene"]
            else:
                s_node["categories"] = ["biolink:Gene"]
            # try to assign predicates again
            s_cat = (s_node.get("categories") or [None])[0]
            o_cat = (o_node.get("categories") or [None])[0]
            preds = self._find_predicates(s_cat, o_cat)
            if preds:
                edge["predicates"] = [preds[0]]
        return trapi


__all__ = ["TRAPIQueryBuilder"]
