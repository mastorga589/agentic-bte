"""
Legacy TRAPI Builder (ported from BTE-LLM Prototype/tools/BTECall.py)

Clean integration wrapper that reproduces the original LangGraph TRAPI construction
logic while using Agentic BTE settings and BTEClient for meta-KG access.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI

from ...config.settings import get_settings
from .bte_client import BTEClient

logger = logging.getLogger(__name__)


class LegacyTRAPIBuilder:
    def __init__(self, openai_api_key: Optional[str] = None):
        s = get_settings()
        api_key = openai_api_key or s.openai_api_key
        self.llm = ChatOpenAI(temperature=0, model=s.openai_model, api_key=api_key) if api_key else ChatOpenAI(temperature=0, model=s.openai_model)
        self.bte = BTEClient()
        self._meta_kg: Optional[Dict[str, Any]] = None
        # mapping for category normalization
        self._id_prefix_to_cat = {
            'CHEBI:': 'biolink:SmallMolecule',
            'ChEMBL': 'biolink:SmallMolecule',
            'DRUGBANK:': 'biolink:SmallMolecule',
            'UNII:': 'biolink:SmallMolecule',
            'NCBIGene:': 'biolink:Gene',
            'HGNC:': 'biolink:Gene',
            'ENSEMBL:': 'biolink:Gene',
            'GO:': 'biolink:BiologicalProcess',
            'MONDO:': 'biolink:Disease',
            'DOID:': 'biolink:Disease',
            'PR:': 'biolink:Protein',
            'UniProtKB:': 'biolink:Protein',
        }

    @property
    def meta_kg(self) -> Dict[str, Any]:
        if self._meta_kg is None:
            try:
                self._meta_kg = self.bte.get_meta_knowledge_graph()
            except Exception:
                logger.warning("Meta-KG unavailable; proceeding with empty edges set")
                self._meta_kg = {"edges": []}
        return self._meta_kg

    # ---- Helpers ported from original ----
    def _extract_dict(self, raw: str) -> Dict[str, Any]:
        rawstring = (raw or "").strip()
        # Try to find first well-formed {...}
        start_index = rawstring.find("{")
        end_index = rawstring.find("}\n`")
        extracted_dict = None
        if start_index != -1 and end_index != -1 and end_index >= start_index:
            extracted_dict = rawstring[start_index:end_index + 1].strip()
        if extracted_dict:
            try:
                return json.loads(extracted_dict)
            except Exception:
                pass
        # Fallback: parse whole
        try:
            return json.loads(rawstring)
        except Exception:
            return {"error": "could not parse dict"}

    def _invoke_with_retries(self, prompt: str, parser_func, max_retries: int = 3, delay: float = 3.0):
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self.llm.invoke(prompt)
                return parser_func(getattr(resp, "content", ""))
            except Exception as e:
                last_err = e
                logger.debug(f"LLM retry {attempt+1}/{max_retries} failed: {e}")
                time.sleep(delay)
        if last_err:
            logger.error(f"LLM invocation failed after {max_retries} retries: {last_err}")
        return None

    def _find_predicates(self, subject_cat: str, object_cat: str) -> List[str]:
        preds: List[str] = []
        try:
            for edge in self.meta_kg.get("edges", []) or []:
                if (str(edge.get("subject", "")).lower() == str(subject_cat).lower()) and (
                    str(edge.get("object", "")).lower() == str(object_cat).lower()
                ):
                    p = edge.get("predicate")
                    if p:
                        preds.append(p)
        except Exception:
            pass
        # de-dupe preserving order
        out = []
        seen = set()
        for p in preds:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _choose_predicate(self, predicate_list: List[str], query: str) -> str:
        if not predicate_list:
            return "biolink:related_to"

        # Structured output with allowed literals, as in original implementation
        class PredicateChoice(TypedDict):
            predicate: str

        try:
            # Use a short prompt; we'll post-validate membership instead of Literal[*list]
            pred_prompt = f"""
Choose the most specific predicate for the TRAPI query.
Query: {query}
Choices: {predicate_list}
Return only the predicate string.
"""
            resp = self.llm.invoke(pred_prompt)
            chosen = (getattr(resp, "content", "") or "").strip()
            for p in predicate_list:
                if p == chosen or p in chosen:
                    return p
        except Exception as e:
            logger.debug(f"Predicate choice failed, defaulting to first: {e}")
        return predicate_list[0]

    def _identify_nodes(self, query: str) -> Optional[Dict[str, str]]:
        nodeprompt = f"""
Your task is to help build a TRAPI query by identifying the correct subject and object nodes from the list below.
Subject: The entity that initiates or is the focus of the relationship.
Object: The entity affected or related to the subject.
Each must include the prefix 'biolink:'. Use the query context to choose correctly.
Nodes: Disease, PhysiologicalProcess, BiologicalEntity, Gene, PathologicalProcess, Polypeptide, SmallMolecule, PhenotypicFeature
Here is the user query: {query}
Return ONLY a JSON dict with keys 'subject' and 'object'.
"""
        return self._invoke_with_retries(nodeprompt, self._extract_dict)

    # ---- Post-sanitization helpers ----
    def _normalize_categories_and_ids(self, trapi: Dict[str, Any]) -> None:
        qg = (trapi.get('message') or {}).get('query_graph') or {}
        nodes = qg.get('nodes') or {}
        # Normalize legacy category names
        def canonical_cat(cat: Optional[str]) -> Optional[str]:
            if not isinstance(cat, str):
                return cat
            if cat.endswith(':PhysiologicalProcess'):
                return 'biolink:BiologicalProcess'
            return cat
        for nid, ndata in list(nodes.items()):
            cats = ndata.get('categories') or []
            if isinstance(cats, list) and cats:
                ndata['categories'] = [canonical_cat(cats[0])]
            # If ids are present, align category to ID namespace
            ids = ndata.get('ids') or []
            if isinstance(ids, list) and ids:
                cid = str(ids[0])
                for pfx, tgt in self._id_prefix_to_cat.items():
                    if cid.startswith(pfx):
                        ndata['categories'] = [tgt]
                        break

    def _repair_predicates(self, trapi: Dict[str, Any]) -> None:
        qg = (trapi.get('message') or {}).get('query_graph') or {}
        nodes = qg.get('nodes') or {}
        edges = qg.get('edges') or {}
        if not isinstance(nodes, dict) or not isinstance(edges, dict):
            return
        for eid, edata in list(edges.items()):
            subj = edata.get('subject'); obj = edata.get('object')
            if subj not in nodes or obj not in nodes:
                continue
            sc = (nodes.get(subj, {}).get('categories') or [None])[0]
            oc = (nodes.get(obj, {}).get('categories') or [None])[0]
            # Determine preferred predicate given pair
            preferred = None
            pair = (sc, oc)
            if pair in (("biolink:SmallMolecule", "biolink:Disease"), ("biolink:ChemicalSubstance", "biolink:Disease")):
                # prefer treats
                preds = self._find_predicates(sc, oc)
                preferred = next((p for p in preds if p.endswith(':treats')), preds[0] if preds else 'biolink:treats')
            elif pair in (("biolink:SmallMolecule", "biolink:BiologicalProcess"), ("biolink:ChemicalSubstance", "biolink:BiologicalProcess")):
                preds = self._find_predicates(sc, oc)
                # Force 'affects' for mechanism-style SM→BP even if MKG lists only generic predicates
                preferred = next((p for p in preds if p.endswith(':affects')), 'biolink:affects')
            elif pair == ("biolink:Gene", "biolink:BiologicalProcess"):
                preds = self._find_predicates(sc, oc)
                preferred = next((p for p in preds if p.endswith(':participates_in')), preds[0] if preds else 'biolink:participates_in')
            elif pair == ("biolink:Disease", "biolink:BiologicalProcess"):
                preds = self._find_predicates(sc, oc)
                preferred = preds[0] if preds else 'biolink:related_to'
            else:
                # fallback to first supported predicate if available
                preds = self._find_predicates(sc, oc)
                preferred = preds[0] if preds else (edata.get('predicates', ['biolink:related_to'])[0])
            # assign predicate
            edata['predicates'] = [preferred]

    def build(self, query: str, entity_data: Dict[str, str], failed_trapis: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        failed_trapis = failed_trapis or []
        # 1) Identify subject/object categories
        subject_object = self._identify_nodes(query)
        if not isinstance(subject_object, dict) or "subject" not in subject_object or "object" not in subject_object:
            logger.warning("Legacy builder could not determine subject/object; returning empty TRAPI")
            return {"message": {"query_graph": {"nodes": {}, "edges": {}}}}

        # 2) Predicate selection from meta-KG and pruning of failed predicates
        predicate_list = self._find_predicates(subject_object.get("subject"), subject_object.get("object"))
        # remove predicates that previously failed
        try:
            failed_predicates = set()
            for tq in failed_trapis:
                preds = (tq.get("message", {}).get("query_graph", {}).get("edges", {}).get("e01", {}) or {}).get("predicates", [])
                for p in preds or []:
                    failed_predicates.add(p)
            predicate_list = [p for p in predicate_list if p not in failed_predicates]
        except Exception:
            pass
        predicate = self._choose_predicate(predicate_list, query)

        # 3) Build TRAPI via LLM prompt (verbatim from original, minimal edits)
        trapi_ex = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"categories": ["biolink:Disease"], "ids": ["MONDO:0016575"]},
                        "n1": {"categories": ["biolink:PhenotypicFeature"]},
                    },
                    "edges": {
                        "e01": {"subject": "n0", "object": "n1", "predicates": ["biolink:has_phenotype"]}
                    },
                }
            }
        }
        trapi_prompt = f"""
You are a smart assistant that can parse the user prompt into a TRAPI query.
Example TRAPI:
{json.dumps(trapi_ex, indent=2)}

User query: "{query}"

Entities and IDs extracted from the query:
{json.dumps(entity_data or {}, indent=2)}

Nodes you MUST use:
{json.dumps(subject_object, indent=2)}

Chosen predicate:
{predicate}

Do NOT use the following TRAPI queries as they have failed previously:
{json.dumps(failed_trapis or [], indent=2)}

Some predicates have directionality ("treated_by" is NOT "treats"). Use correct direction.
Subject is source (domain), object is target (range).
Only one node may include an "ids" field.
Output ONLY a JSON TRAPI object with no commentary.
"""
        trapi = self._invoke_with_retries(trapi_prompt, self._extract_dict)
        if isinstance(trapi, dict):
            # Post-sanitize: align categories to ID prefixes and repair predicates
            try:
                self._normalize_categories_and_ids(trapi)
                self._repair_predicates(trapi)
            except Exception as e:
                logger.debug(f"Legacy TRAPI post-sanitize skipped: {e}")
            return trapi
        return {"message": {"query_graph": {"nodes": {}, "edges": {}}}}
