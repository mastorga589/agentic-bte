"""
Simplified predicate selection strategy for stand-alone Prototype.
Selects top-3 predicates per (intent, subject_category, object_category),
using meta-KG support filtering where available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    THERAPEUTIC = "therapeutic"
    GENETIC = "genetic"
    MECHANISM = "mechanism"
    GENERAL = "general"


@dataclass
class PredicateConfig:
    max_predicates_per_subquery: int = 3
    min_provider_support: int = 1
    enable_meta_kg_filtering: bool = True


class PredicateSelector:
    PREDICATE_TIERS: Dict[QueryIntent, Dict[str, List[str]]] = {
        QueryIntent.THERAPEUTIC: {
            "primary": [
                "biolink:treats",
                "biolink:treated_by",
                "biolink:associated_with",
                "biolink:related_to",
            ],
            "secondary": [
                "biolink:affects",
                "biolink:applied_to_treat",
                "biolink:in_clinical_trials_for",
            ],
        },
        QueryIntent.GENETIC: {
            "primary": [
                "biolink:gene_associated_with_condition",
                "biolink:condition_associated_with_gene",
                "biolink:associated_with",
                "biolink:related_to",
            ],
            "secondary": [
                "biolink:causes",
                "biolink:contributes_to",
                "biolink:genetically_associated_with",
            ],
        },
        QueryIntent.MECHANISM: {
            "primary": [
                "biolink:directly_physically_interacts_with",
                "biolink:interacts_with",
                "biolink:affects",
                "biolink:associated_with",
                "biolink:related_to",
            ],
            "secondary": [
                "biolink:regulates",
                "biolink:physically_interacts_with",
                "biolink:modulates",
            ],
        },
        QueryIntent.GENERAL: {
            "primary": [
                "biolink:related_to",
                "biolink:associated_with",
                "biolink:affects",
            ],
            "secondary": [
                "biolink:interacts_with",
                "biolink:correlated_with",
            ],
        },
    }

    FALLBACK_PREDICATE = "biolink:related_to"

    def __init__(self, meta_kg: Optional[Dict] = None, config: Optional[PredicateConfig] = None):
        self.meta_kg = meta_kg or {}
        self.config = config or PredicateConfig()
        self._support_map: Dict[Tuple[str, str, str], int] = {}
        self._build_support_map()

    def _build_support_map(self) -> None:
        edges = self.meta_kg.get("edges", [])
        for e in edges:
            s = e.get("subject")
            p = e.get("predicate")
            o = e.get("object")
            if s and p and o:
                key = (s, p, o)
                self._support_map[key] = self._support_map.get(key, 0) + 1

    def get_predicate_support(self, subject_category: str, predicate: str, object_category: str) -> int:
        return self._support_map.get((subject_category, predicate, object_category), 0)

    def detect_query_intent(self, query: str, entities: List[Dict]) -> QueryIntent:
        q = (query or "").lower()
        if any(k in q for k in ["treat", "drug", "therapy", "medication", "compound"]):
            return QueryIntent.THERAPEUTIC
        if any(k in q for k in ["gene", "mutation", "genetic", "hereditary", "chromosome"]):
            return QueryIntent.GENETIC
        if any(k in q for k in ["mechanism", "pathway", "process", "interact", "affect", "regulate", "target"]):
            return QueryIntent.MECHANISM
        return QueryIntent.GENERAL

    def _filter_supported(self, cands: List[str], s_cat: str, o_cat: str) -> List[Tuple[str, int]]:
        if not self.config.enable_meta_kg_filtering:
            return [(p, 1) for p in cands]
        out: List[Tuple[str, int]] = []
        for p in cands:
            sup = self.get_predicate_support(s_cat, p, o_cat)
            if sup >= self.config.min_provider_support:
                out.append((p, sup))
        return out

    def select_predicates(self, intent: QueryIntent, subject_category: str, object_category: str) -> List[Tuple[str, float]]:
        tiers = self.PREDICATE_TIERS.get(intent, self.PREDICATE_TIERS[QueryIntent.GENERAL])
        primary = self._filter_supported(tiers.get("primary", []), subject_category, object_category)
        secondary = self._filter_supported(tiers.get("secondary", []), subject_category, object_category)

        ranked: List[Tuple[str, float]] = []
        # Score primary higher than secondary; use support as a tie-breaker boost.
        for i, (p, sup) in enumerate(primary):
            ranked.append((p, 1.0 - 0.05 * i + min(0.2, sup * 0.01)))
        for i, (p, sup) in enumerate(secondary):
            ranked.append((p, 0.6 - 0.05 * i + min(0.2, sup * 0.01)))

        # Always consider fallback if not enough
        if len(ranked) < self.config.max_predicates_per_subquery:
            ranked.append((self.FALLBACK_PREDICATE, 0.3))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[: self.config.max_predicates_per_subquery]


__all__ = [
    "QueryIntent",
    "PredicateConfig",
    "PredicateSelector",
]
