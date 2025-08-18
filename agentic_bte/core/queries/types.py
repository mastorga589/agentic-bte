"""
Query Types - Biomedical Query Classification

This module provides query type definitions and enums for categorizing
different types of biomedical research questions for optimal processing.
"""

from enum import Enum
from typing import Dict, List, Set
from dataclasses import dataclass


class QueryType(Enum):
    """
    Types of biomedical queries for strategic planning and optimization
    
    Each query type corresponds to a different approach for query decomposition,
    execution planning, and result processing.
    """
    DRUG_MECHANISM = "drug_mechanism"           # How does drug X work/affect Y?
    DISEASE_TREATMENT = "disease_treatment"     # What treats disease X?
    GENE_FUNCTION = "gene_function"             # What does gene X do?
    PATHWAY_ANALYSIS = "pathway_analysis"       # What pathways are involved in X?
    DRUG_TARGET = "drug_target"                 # What does drug X target?
    DISEASE_GENE = "disease_gene"               # What genes cause disease X?
    PHENOTYPE_GENE = "phenotype_gene"           # What genes affect phenotype X?
    PROTEIN_FUNCTION = "protein_function"       # What does protein X do?
    GENE_DISEASE = "gene_disease"               # What diseases does gene X cause?
    DRUG_DISEASE = "drug_disease"               # What diseases does drug X treat?
    UNKNOWN = "unknown"                         # Query type could not be determined


@dataclass
class QueryTypeInfo:
    """Information about a specific query type including processing strategies"""
    query_type: QueryType
    description: str
    typical_patterns: List[str]
    entity_types: Set[str]
    decomposition_strategy: str
    complexity_score: int  # 1-5, where 5 is most complex


# Query type information registry
QUERY_TYPE_INFO: Dict[QueryType, QueryTypeInfo] = {
    QueryType.DRUG_MECHANISM: QueryTypeInfo(
        query_type=QueryType.DRUG_MECHANISM,
        description="Questions asking HOW a drug works, its mechanism of action, or mode of action",
        typical_patterns=[
            "How does [drug] treat [disease]?",
            "What is the mechanism of action of [drug]?",
            "How does [drug] work?",
            "By what mechanism does [drug] affect [process]?"
        ],
        entity_types={"Drug", "SmallMolecule", "Disease", "BiologicalProcess", "Gene"},
        decomposition_strategy="mechanistic_pathway",
        complexity_score=4
    ),
    
    QueryType.DISEASE_TREATMENT: QueryTypeInfo(
        query_type=QueryType.DISEASE_TREATMENT,
        description="Questions asking WHAT treats a disease or therapeutic options",
        typical_patterns=[
            "What drugs treat [disease]?",
            "What are the treatment options for [disease]?",
            "Which medications are used for [disease]?",
            "What therapies exist for [disease]?"
        ],
        entity_types={"Disease", "Drug", "SmallMolecule", "BiologicalProcess"},
        decomposition_strategy="bidirectional_search",
        complexity_score=3
    ),
    
    QueryType.GENE_FUNCTION: QueryTypeInfo(
        query_type=QueryType.GENE_FUNCTION,
        description="Questions about gene roles, functions, or biological activities",
        typical_patterns=[
            "What does the [gene] gene do?",
            "What is the function of [gene]?",
            "What role does [gene] play in [process]?",
            "What biological processes involve [gene]?"
        ],
        entity_types={"Gene", "BiologicalProcess", "Protein", "Pathway"},
        decomposition_strategy="functional_analysis",
        complexity_score=3
    ),
    
    QueryType.PATHWAY_ANALYSIS: QueryTypeInfo(
        query_type=QueryType.PATHWAY_ANALYSIS,
        description="Questions about biological pathways, processes, or networks",
        typical_patterns=[
            "What genes are involved in [process]?",
            "What pathways regulate [process]?",
            "Which proteins participate in [pathway]?",
            "How is [process] controlled?"
        ],
        entity_types={"BiologicalProcess", "Pathway", "Gene", "Protein"},
        decomposition_strategy="network_analysis",
        complexity_score=4
    ),
    
    QueryType.DRUG_TARGET: QueryTypeInfo(
        query_type=QueryType.DRUG_TARGET,
        description="Questions about molecular targets that drugs interact with",
        typical_patterns=[
            "What proteins does [drug] target?",
            "What receptors does [drug] bind to?",
            "What are the targets of [drug]?",
            "Which molecules does [drug] interact with?"
        ],
        entity_types={"Drug", "SmallMolecule", "Protein", "Gene"},
        decomposition_strategy="target_identification",
        complexity_score=2
    ),
    
    QueryType.DISEASE_GENE: QueryTypeInfo(
        query_type=QueryType.DISEASE_GENE,
        description="Questions about genes associated with or causing diseases",
        typical_patterns=[
            "What genes cause [disease]?",
            "Which genes are associated with [disease]?",
            "What genetic factors contribute to [disease]?",
            "Which genes predispose to [disease]?"
        ],
        entity_types={"Disease", "Gene", "Variant"},
        decomposition_strategy="genetic_association",
        complexity_score=3
    ),
    
    QueryType.PHENOTYPE_GENE: QueryTypeInfo(
        query_type=QueryType.PHENOTYPE_GENE,
        description="Questions linking observable traits to genetic factors",
        typical_patterns=[
            "What genes affect [trait]?",
            "Which genes influence [phenotype]?",
            "What genetic variants cause [trait]?",
            "Which genes determine [characteristic]?"
        ],
        entity_types={"Gene", "PhenotypicFeature", "Variant"},
        decomposition_strategy="phenotype_mapping",
        complexity_score=3
    ),
    
    QueryType.PROTEIN_FUNCTION: QueryTypeInfo(
        query_type=QueryType.PROTEIN_FUNCTION,
        description="Questions about protein roles, activities, or interactions",
        typical_patterns=[
            "What does [protein] do?",
            "What is the function of [protein]?",
            "How does [protein] work?",
            "What role does [protein] play?"
        ],
        entity_types={"Protein", "BiologicalProcess", "Gene"},
        decomposition_strategy="functional_analysis",
        complexity_score=3
    ),
    
    QueryType.GENE_DISEASE: QueryTypeInfo(
        query_type=QueryType.GENE_DISEASE,
        description="Questions about diseases caused by specific genes",
        typical_patterns=[
            "What diseases does [gene] cause?",
            "Which conditions are associated with [gene]?",
            "What disorders result from [gene] mutations?",
            "What phenotypes are linked to [gene]?"
        ],
        entity_types={"Gene", "Disease", "PhenotypicFeature"},
        decomposition_strategy="genetic_association",
        complexity_score=3
    ),
    
    QueryType.DRUG_DISEASE: QueryTypeInfo(
        query_type=QueryType.DRUG_DISEASE,
        description="Questions about diseases or conditions treated by specific drugs",
        typical_patterns=[
            "What diseases does [drug] treat?",
            "Which conditions is [drug] used for?",
            "What indications does [drug] have?",
            "For what is [drug] prescribed?"
        ],
        entity_types={"Drug", "SmallMolecule", "Disease"},
        decomposition_strategy="therapeutic_indication",
        complexity_score=2
    ),
    
    QueryType.UNKNOWN: QueryTypeInfo(
        query_type=QueryType.UNKNOWN,
        description="Query type could not be determined or doesn't fit standard categories",
        typical_patterns=[],
        entity_types=set(),
        decomposition_strategy="generic",
        complexity_score=1
    )
}


def get_query_type_info(query_type: QueryType) -> QueryTypeInfo:
    """Get detailed information about a query type"""
    return QUERY_TYPE_INFO.get(query_type, QUERY_TYPE_INFO[QueryType.UNKNOWN])


def get_all_query_types() -> List[QueryType]:
    """Get all available query types"""
    return list(QueryType)


def get_query_types_by_complexity(min_complexity: int = 1, max_complexity: int = 5) -> List[QueryType]:
    """Get query types filtered by complexity score"""
    return [
        qt for qt, info in QUERY_TYPE_INFO.items()
        if min_complexity <= info.complexity_score <= max_complexity
    ]


def get_query_types_with_entity_type(entity_type: str) -> List[QueryType]:
    """Get query types that typically involve a specific entity type"""
    return [
        qt for qt, info in QUERY_TYPE_INFO.items()
        if entity_type in info.entity_types
    ]


def suggest_decomposition_strategy(query_type: QueryType) -> str:
    """Get the suggested decomposition strategy for a query type"""
    info = get_query_type_info(query_type)
    return info.decomposition_strategy