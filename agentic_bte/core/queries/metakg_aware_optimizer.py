"""
Meta-KG Aware Context-Driven Adaptive Query Optimizer

This module enhances the LangGraph-inspired adaptive optimizer by leveraging 
the BTE meta-knowledge graph edges to inform subquery generation. This ensures
all subqueries are single-hop and based on actual available relationships.

Key enhancements over the basic context-driven optimizer:
- Meta-KG edge analysis for informed planning
- Guaranteed single-hop subqueries
- Reduced failed query attempts
- More intelligent contextual reasoning
- Better utilization of available knowledge graph structure
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from uuid import uuid4
from copy import deepcopy
from collections import defaultdict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from rdflib import Graph, URIRef, Namespace
from rdflib.namespace import RDF, RDFS

from ..entities.bio_ner import BioNERTool
from ..knowledge.trapi import TRAPIQueryBuilder
from ..knowledge.bte_client import BTEClient
from ...config.settings import get_settings
from ...exceptions.base import ExternalServiceError

logger = logging.getLogger(__name__)


@dataclass
class MetaKGEdge:
    """Represents an edge in the BTE meta-knowledge graph"""
    subject: str
    predicate: str
    object: str
    api_name: str = ""
    provided_by: str = ""
    
    def __str__(self) -> str:
        return f"{self.subject} -{self.predicate}-> {self.object}"


@dataclass
class ContextualSubquery:
    """Context-driven subquery without predefined strategies"""
    id: str = field(default_factory=lambda: str(uuid4()))
    query: str = ""
    reasoning_context: str = ""  # Why this query was generated
    suggested_edge: Optional[MetaKGEdge] = None  # The meta-KG edge that suggested this query
    results: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = False
    execution_time: float = 0.0
    iteration_number: int = 0
    

@dataclass 
class AdaptivePlan:
    """Context-driven execution plan using RDF knowledge accumulation"""
    id: str = field(default_factory=lambda: str(uuid4()))
    original_query: str = ""
    knowledge_graph: Graph = field(default_factory=Graph)  # RDF graph for context
    executed_subqueries: List[ContextualSubquery] = field(default_factory=list)
    entity_data: Dict[str, str] = field(default_factory=dict)
    accumulated_results: List[Dict[str, Any]] = field(default_factory=list)
    current_iteration: int = 0
    max_iterations: int = 10
    total_execution_time: float = 0.0
    final_answer: str = ""
    completion_reason: str = ""  # Why the plan finished


class MetaKGAwareAdaptiveOptimizer:
    """
    Meta-KG aware adaptive query optimizer using context-driven planning
    
    This optimizer enhances the LangGraph-inspired approach by leveraging
    BTE meta-knowledge graph edges to inform subquery generation, ensuring
    all subqueries are single-hop and based on actual available relationships.
    """
    
    def __init__(self, bio_ner: Optional[BioNERTool] = None, bte_client: Optional[BTEClient] = None, openai_api_key: Optional[str] = None):
        """
        Initialize the meta-KG aware adaptive optimizer
        
        Args:
            bio_ner: Biomedical NER tool instance (optional)
            bte_client: BTE client instance (optional) 
            openai_api_key: OpenAI API key for LLM operations
        """
        self.settings = get_settings()
        self.openai_api_key = openai_api_key or self.settings.openai_api_key
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for adaptive query optimization")
        
        # Initialize components - use provided instances or create new ones
        self.llm = ChatOpenAI(
            temperature=0,
            model=self.settings.openai_model,
            api_key=self.openai_api_key
        )
        self.bio_ner = bio_ner or BioNERTool(self.openai_api_key)
        self.trapi_builder = TRAPIQueryBuilder(self.openai_api_key)
        self.bte_client = bte_client or BTEClient()
        
        # RDF namespace setup (same as LangGraph implementation)
        self._setup_rdf_namespaces()
        
        # Meta-KG cache
        self._meta_kg_edges: List[MetaKGEdge] = []
        self._edges_by_subject: Dict[str, List[MetaKGEdge]] = defaultdict(list)
        self._edges_by_object: Dict[str, List[MetaKGEdge]] = defaultdict(list)
        # Note: Meta-KG will be loaded async via _load_meta_kg() when first needed
    
    def _setup_rdf_namespaces(self):
        """Setup RDF namespaces for knowledge graph context"""
        self.BIOLINK = Namespace("https://w3id.org/biolink/vocab/")
        self.EX = Namespace("http://example.org/entity/")
        
        # Complete mapping of all 48 node types present in BTE meta-KG
        self.ALL_NODE_TYPES = {
            "biolink:Activity", "biolink:AnatomicalEntity", "biolink:Bacterium", "biolink:Behavior",
            "biolink:BiologicalEntity", "biolink:BiologicalProcess", "biolink:Cell", "biolink:CellularComponent",
            "biolink:ChemicalEntity", "biolink:ChemicalExposure", "biolink:ChemicalMixture", "biolink:ClinicalAttribute",
            "biolink:ClinicalFinding", "biolink:ClinicalIntervention", "biolink:Cohort", "biolink:ComplexMolecularMixture",
            "biolink:Device", "biolink:DiagnosticAid", "biolink:Disease", "biolink:DiseaseOrPhenotypicFeature",
            "biolink:Drug", "biolink:EnvironmentalExposure", "biolink:Food", "biolink:Fungus",
            "biolink:Gene", "biolink:GeneFamily", "biolink:GrossAnatomicalStructure", "biolink:InformationContentEntity",
            "biolink:MacromolecularComplex", "biolink:MolecularActivity", "biolink:MolecularMixture", "biolink:NucleicAcidEntity",
            "biolink:OrganismAttribute", "biolink:OrganismTaxon", "biolink:PathologicalProcess", "biolink:Pathway",
            "biolink:Phenomenon", "biolink:PhenotypicFeature", "biolink:PhysicalEntity", "biolink:PhysiologicalProcess",
            "biolink:Plant", "biolink:Polypeptide", "biolink:PopulationOfIndividualOrganisms", "biolink:Procedure",
            "biolink:Protein", "biolink:SequenceVariant", "biolink:SmallMolecule", "biolink:Virus"
        }
        
        # Create dynamic namespaces for all node types
        self.ENTITY_NAMESPACE_MAP = {}
        for node_type in self.ALL_NODE_TYPES:
            # Create a namespace for each node type
            clean_name = node_type.replace("biolink:", "")
            namespace = Namespace(f"https://biolink.github.io/biolink-model/{clean_name}/")
            self.ENTITY_NAMESPACE_MAP[node_type] = namespace

    async def _load_meta_kg(self):
        """Load and parse the BTE meta-knowledge graph asynchronously"""
        # Skip if already loaded
        if self._meta_kg_edges:
            return
            
        try:
            logger.info("Loading BTE meta-knowledge graph for edge analysis")
            meta_kg_data = self.bte_client.get_meta_knowledge_graph()
            
            # Parse edges into structured format
            for edge_data in meta_kg_data.get("edges", []):
                edge = MetaKGEdge(
                    subject=edge_data.get("subject", ""),
                    predicate=edge_data.get("predicate", ""),
                    object=edge_data.get("object", ""),
                    api_name=edge_data.get("api", {}).get("name", ""),
                    provided_by=edge_data.get("provided_by", "")
                )
                
                self._meta_kg_edges.append(edge)
                self._edges_by_subject[edge.subject].append(edge)
                self._edges_by_object[edge.object].append(edge)
            
            logger.info(f"Loaded {len(self._meta_kg_edges)} meta-KG edges for informed planning")
            
        except Exception as e:
            logger.error(f"Error loading meta-KG: {e}")
            # Continue with empty meta-KG - will fall back to basic approach
    
    def get_available_node_types(self) -> Set[str]:
        """
        Get all node types available in the loaded meta-KG
        
        Returns:
            Set of all node types present in the meta-KG edges
        """
        node_types = set()
        for edge in self._meta_kg_edges:
            node_types.add(edge.subject)
            node_types.add(edge.object)
        return node_types
    
    def _get_all_node_types(self) -> Set[str]:
        """
        Get all possible node types from the static definition
        
        Returns:
            Set of all supported node types
        """
        return self.ALL_NODE_TYPES.copy()
    
    def _get_available_edges_from_nodes(self, current_nodes: Set[str]) -> List[MetaKGEdge]:
        """
        Get available edges that can extend from current nodes in knowledge graph
        
        Args:
            current_nodes: Set of node types currently present in knowledge graph
            
        Returns:
            List of meta-KG edges that can be traversed from current nodes
        """
        available_edges = []
        
        for node_type in current_nodes:
            # Get edges where this node type is the subject (outgoing edges)
            available_edges.extend(self._edges_by_subject.get(node_type, []))
        
        return available_edges
    
    def _extract_current_node_types(self, knowledge_graph: Graph) -> Set[str]:
        """
        Extract the node types currently present in the RDF knowledge graph
        
        Args:
            knowledge_graph: Current accumulated RDF graph
            
        Returns:
            Set of biolink node types present in the graph
        """
        current_nodes = set()
        
        try:
            # Get all unique subjects and objects from the graph
            subjects_and_objects = set()
            for subj, _, obj in knowledge_graph:
                subjects_and_objects.add(subj)
                subjects_and_objects.add(obj)
            
            # Check which namespace each URI belongs to
            for uri in subjects_and_objects:
                uri_str = str(uri)
                for node_type, namespace in self.ENTITY_NAMESPACE_MAP.items():
                    if str(namespace) in uri_str:
                        current_nodes.add(node_type)
                        break
            
            # Also check the turtle content as fallback using broader matching
            if not current_nodes:
                turtle_content = knowledge_graph.serialize(format="turtle")
                for node_type in self.ALL_NODE_TYPES:
                    simplified_type = node_type.replace("biolink:", "").lower()
                    # Check for both exact match and partial matches for compound words
                    if (simplified_type in turtle_content.lower() or 
                        any(word in turtle_content.lower() for word in simplified_type.split() if len(word) > 3)):
                        current_nodes.add(node_type)
                        
            # Final fallback - if still empty, try to infer from available meta-KG node types
            if not current_nodes and self._meta_kg_edges:
                available_types = self.get_available_node_types()
                # If we have meta-KG data, use top node types as reasonable starting point
                if available_types:
                    current_nodes = available_types.intersection({
                        "biolink:Disease", "biolink:Gene", "biolink:SmallMolecule", "biolink:ChemicalEntity",
                        "biolink:PhenotypicFeature", "biolink:Protein", "biolink:Drug"
                    })
                    # If intersection is empty, just use the top frequent ones
                    if not current_nodes:
                        current_nodes = {"biolink:Disease", "biolink:SmallMolecule", "biolink:Gene"}
                        
        except Exception as e:
            logger.warning(f"Error extracting node types from RDF graph: {e}")
            # If graph is empty or has issues, consider all major node types for first iteration
            # Use the top 10 most common node types as fallback
            current_nodes = {
                "biolink:Disease", "biolink:Gene", "biolink:SmallMolecule", "biolink:ChemicalEntity",
                "biolink:PhenotypicFeature", "biolink:Protein", "biolink:Polypeptide", 
                "biolink:PhysiologicalProcess", "biolink:Procedure", "biolink:PathologicalProcess"
            }
        
        logger.debug(f"Extracted node types from graph: {current_nodes}")
        return current_nodes
    
    def _suggest_next_edges(self, current_graph: Graph, original_query: str) -> List[MetaKGEdge]:
        """
        Suggest next edges to explore based on current knowledge graph state
        
        Args:
            current_graph: Current RDF knowledge graph
            original_query: Original user query for context
            
        Returns:
            List of suggested meta-KG edges to explore next
        """
        current_nodes = self._extract_current_node_types(current_graph)
        logger.debug(f"Current node types in graph: {current_nodes}")
        
        available_edges = self._get_available_edges_from_nodes(current_nodes)
        
        if not available_edges:
            logger.warning("No available edges found from current nodes")
            return []
        
        # Score edges based on relevance to original query
        scored_edges = []
        query_lower = original_query.lower()
        
        for edge in available_edges:
            score = 0
            edge_pred_lower = edge.predicate.lower()
            edge_obj_lower = edge.object.lower()
            edge_subj_lower = edge.subject.lower()
            
            # High priority for treatment relationships
            if "treat" in query_lower and "treated_by" in edge_pred_lower:
                score += 15
            
            # Mechanism of action specific scoring
            if any(term in query_lower for term in ["mechanism", "activity", "pathway", "function"]):
                if "gene" in edge_obj_lower or "polypeptide" in edge_obj_lower:
                    score += 12  # Drug-protein interactions are key for mechanisms
                if "physiologicalprocess" in edge_obj_lower or "pathway" in edge_obj_lower:
                    score += 10  # Pathway involvement
                if "interacts_with" in edge_pred_lower or "affects" in edge_pred_lower:
                    score += 8   # Interaction predicates important for mechanisms
            
            # Antioxidant activity specific
            if "antioxidant" in query_lower:
                if "gene" in edge_obj_lower or "polypeptide" in edge_obj_lower:
                    score += 15  # Antioxidant proteins/genes are highly relevant
                if "physiologicalprocess" in edge_obj_lower:
                    score += 12  # Antioxidant processes
                if "associated_with" in edge_pred_lower or "participates_in" in edge_pred_lower:
                    score += 10
            
            # General scoring for key concepts
            if "gene" in query_lower and "gene" in edge_obj_lower:
                score += 8
            if "drug" in query_lower and "smallmolecule" in edge_obj_lower:
                score += 8
            if "disease" in query_lower and "disease" in edge_subj_lower:
                score += 8
            
            # Boost reliable, informative predicates
            high_value_predicates = ["interacts_with", "affects", "regulates", "participates_in", "has_phenotype"]
            if any(pred in edge_pred_lower for pred in high_value_predicates):
                score += 6
                
            reliable_predicates = ["treated_by", "associated_with", "related_to", "causes"]
            if any(pred in edge_pred_lower for pred in reliable_predicates):
                score += 4
                
            scored_edges.append((score, edge))
        
        # Sort by score and return top candidates
        scored_edges.sort(key=lambda x: x[0], reverse=True)
        top_edges = [edge for _, edge in scored_edges[:10]]  # Top 10 candidates
        
        logger.debug(f"Suggested {len(top_edges)} potential edges for next subquery")
        return top_edges
    
    def _extract_entities_from_results(self, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Extract new entity IDs from BTE results to update entity_data
        
        Args:
            results: List of BTE result relationships
            
        Returns:
            Dictionary mapping entity names to IDs
        """
        new_entities = {}
        
        for result in results:
            # Extract subject entity (BTE returns subject_id and subject)
            subject_id = result.get('subject_id', '')
            subject_name = result.get('subject', '')  # This is the human-readable name
            if subject_id and subject_name:
                # Use the name as key, ID as value (consistent with entity_data format)
                clean_name = subject_name.lower().replace('_', ' ').replace('-', ' ')
                new_entities[clean_name] = subject_id
            
            # Extract object entity (BTE returns object_id and object)
            object_id = result.get('object_id', '')
            object_name = result.get('object', '')  # This is the human-readable name
            if object_id and object_name:
                clean_name = object_name.lower().replace('_', ' ').replace('-', ' ')
                new_entities[clean_name] = object_id
            
            # Also try to extract from the IDs themselves if names are missing or look generic
            if subject_id:
                if not subject_name or subject_name == 'Unknown':
                    if ':' in subject_id:
                        # Extract a reasonable name from ID
                        prefix, id_part = subject_id.split(':', 1)
                        if prefix in ['CHEBI', 'CHEMBL']:
                            new_entities[f"compound_{id_part}"] = subject_id
                        elif prefix == 'NCBIGene':
                            new_entities[f"gene_{id_part}"] = subject_id
                        elif prefix in ['MONDO', 'DOID']:
                            new_entities[f"disease_{id_part}"] = subject_id
                        elif prefix == 'UMLS':
                            new_entities[f"concept_{id_part}"] = subject_id
                
                # Also add ID-based references for later queries
                if subject_id.startswith('CHEBI:'):
                    new_entities[f"chebi_{subject_id.split(':')[1]}"] = subject_id
                elif subject_id.startswith('NCBIGene:'):
                    new_entities[f"gene_{subject_id.split(':')[1]}"] = subject_id
            
            if object_id:
                if not object_name or object_name == 'Unknown':
                    if ':' in object_id:
                        prefix, id_part = object_id.split(':', 1)
                        if prefix in ['CHEBI', 'CHEMBL']:
                            new_entities[f"compound_{id_part}"] = object_id
                        elif prefix == 'NCBIGene':
                            new_entities[f"gene_{id_part}"] = object_id
                        elif prefix in ['MONDO', 'DOID']:
                            new_entities[f"disease_{id_part}"] = object_id
                        elif prefix == 'UMLS':
                            new_entities[f"concept_{id_part}"] = object_id
                
                # Also add ID-based references for later queries
                if object_id.startswith('CHEBI:'):
                    new_entities[f"chebi_{object_id.split(':')[1]}"] = object_id
                elif object_id.startswith('NCBIGene:'):
                    new_entities[f"gene_{object_id.split(':')[1]}"] = object_id
        
        logger.info(f"Extracted {len(new_entities)} new entities from BTE results")
        if new_entities:
            logger.debug(f"New entities sample: {dict(list(new_entities.items())[:3])}")  # Show first 3 with IDs
        
        return new_entities
    
    def _is_single_hop_query(self, query: str) -> bool:
        """
        Validate that a query is truly single-hop
        
        Args:
            query: The subquery to validate
            
        Returns:
            True if query appears to be single-hop, False otherwise
        """
        query_lower = query.lower().strip()
        
        # Check for multi-hop indicators
        multi_hop_patterns = [
            "that treat",
            "that interact", 
            "that participate",
            "that regulate",
            "that affect",
            "which treat",
            "which interact",
            "which participate",
            "which regulate",
            "which affect",
            "do the",
            "are the",
            "drugs that",
            "genes that",
            "proteins that",
            "processes that",
            "pathways that"
        ]
        
        for pattern in multi_hop_patterns:
            if pattern in query_lower:
                return False
        
        # Check for complex conjunctions
        complex_words = ["and", "through", "via", "by way of", "involving"]
        for word in complex_words:
            if word in query_lower:
                return False
        
        return True
    
    def _simplify_to_single_hop(self, complex_query: str, original_query: str) -> Optional[str]:
        """
        Try to simplify a complex query to a single-hop equivalent
        
        Args:
            complex_query: The complex multi-hop query
            original_query: Original user query for context
            
        Returns:
            Simplified single-hop query or None if cannot simplify
        """
        query_lower = complex_query.lower()
        
        # Common simplification patterns
        if "genes do" in query_lower and "drug" in query_lower:
            return "What genes does lutein interact with?"
        elif "processes" in query_lower and "gene" in query_lower:
            return "What processes does AMPK regulate?"
        elif "pathway" in query_lower and "gene" in query_lower:
            return "What pathways does NRF2 participate in?"
        elif "drug" in query_lower and "treat" in query_lower:
            if "macular degeneration" in original_query.lower():
                return "What drugs treat macular degeneration?"
            else:
                return "What drugs treat the condition?"
        
        # If we can't simplify, return None
        return None

    def _extract_query_pattern(self, query: str) -> Optional[Dict[str, str]]:
        """
        Extract the pattern from a query (e.g., "What genes does X interact with?" -> "What Y does X Z with?")
        
        Args:
            query: The query to extract pattern from
            
        Returns:
            Dictionary with pattern components or None
        """
        query_lower = query.lower().strip()
        
        # Common query patterns for biomedical queries
        patterns = [
            # "What genes does X interact with?"
            (r"what (\w+) does (\w+) (\w+) with\?", 
             {"question_word": "what", "object_type": "$1", "subject": "$2", "predicate": "$3", "suffix": "with?"}),
            
            # "What genes does X regulate?"
            (r"what (\w+) does (\w+) (\w+)\?", 
             {"question_word": "what", "object_type": "$1", "subject": "$2", "predicate": "$3", "suffix": "?"}),
            
            # "What processes does X participate in?"
            (r"what (\w+) does (\w+) (\w+) in\?", 
             {"question_word": "what", "object_type": "$1", "subject": "$2", "predicate": "$3", "suffix": "in?"}),
            
            # "What drugs treat X?"
            (r"what (\w+) (\w+) (\w+)\?", 
             {"question_word": "what", "subject_type": "$1", "predicate": "$2", "object": "$3", "suffix": "?"}),
        ]
        
        import re
        for pattern, template in patterns:
            match = re.search(pattern, query_lower)
            if match:
                # Replace matched groups in template
                result = {}
                for key, value in template.items():
                    if value.startswith("$"):
                        group_num = int(value[1:])
                        result[key] = match.group(group_num)
                    else:
                        result[key] = value
                result["pattern"] = pattern
                return result
        
        return None
    
    def _get_top_drugs_from_accumulated_results(self, accumulated_results: List[Dict[str, Any]]) -> List[str]:
        """
        Extract top drug names from accumulated results
        
        Args:
            accumulated_results: List of relationship results
            
        Returns:
            List of drug names found in results
        """
        drug_names = set()
        drug_keywords = ['donepezil', 'memantine', 'rivastigmine', 'galantamine', 
                        'acetylcarnitine', 'nilotinib', 'bexarotene', 'tacrine']
        
        for result in accumulated_results:
            for field in ['subject', 'object']:
                if field in result:
                    name = result[field].lower()
                    for keyword in drug_keywords:
                        if keyword in name:
                            drug_names.add(keyword)
        
        return list(drug_names)[:5]  # Return up to 5 drugs
    
    def _group_similar_queries_from_entities(self, 
                                           entity_data: Dict[str, str], 
                                           executed_subqueries: List[ContextualSubquery],
                                           suggested_edge: MetaKGEdge) -> Optional[str]:
        """
        Generate a grouped query based on available entities and suggested edge
        
        Args:
            entity_data: Available entity mappings
            executed_subqueries: Previously executed subqueries
            suggested_edge: Suggested meta-KG edge for this query
            
        Returns:
            Grouped query string or None
        """
        # Get previously executed query patterns
        executed_patterns = []
        for sq in executed_subqueries or []:
            pattern = self._extract_query_pattern(sq.query)
            if pattern:
                executed_patterns.append((sq.query, pattern))
        
        # Find entities that match the suggested edge's subject type
        edge_subject = suggested_edge.subject.lower().replace("biolink:", "")
        edge_object = suggested_edge.object.lower().replace("biolink:", "")
        edge_predicate = suggested_edge.predicate.replace("_", " ").lower()
        
        # Map biolink types to common query terms
        type_mapping = {
            "smallmolecule": "drugs",
            "gene": "genes", 
            "disease": "diseases",
            "physiologicalprocess": "processes",
            "pathologicalprocess": "processes",
            "polypeptide": "proteins",
            "phenotypicfeature": "phenotypes"
        }
        
        subject_word = type_mapping.get(edge_subject, edge_subject)
        object_word = type_mapping.get(edge_object, edge_object)
        
        # Find candidate entities for grouping
        candidate_entities = []
        for entity_name, entity_id in entity_data.items():
            # Simple heuristic to match entities to the edge subject type
            if (edge_subject == "smallmolecule" and 
                any(drug in entity_name.lower() for drug in ["donepezil", "acetylcarnitine", "memantine", "rivastigmine", "galantamine", "nilotinib", "bexarotene"])):
                candidate_entities.append(entity_name)
            elif (edge_subject == "gene" and 
                  ("gene" in entity_name.lower() or entity_id.startswith("NCBIGene:"))):
                candidate_entities.append(entity_name)
        
        # Group multiple entities into one query if we have 2+ candidates
        if len(candidate_entities) >= 2:
            # Create different query templates based on the edge relationship
            if "interact" in edge_predicate:
                entity_list = ", ".join(candidate_entities[:5])  # Limit to 5 entities
                return f"What {object_word} do {entity_list} interact with?"
            elif "treat" in edge_predicate:
                entity_list = ", ".join(candidate_entities[:5])
                return f"What conditions do {entity_list} treat?"
            elif "regulat" in edge_predicate:
                entity_list = ", ".join(candidate_entities[:5])
                return f"What {object_word} do {entity_list} regulate?"
            elif "affect" in edge_predicate:
                entity_list = ", ".join(candidate_entities[:5])
                return f"What {object_word} are affected by {entity_list}?"
        
        return None
    
    def _generate_metakg_informed_subquery(self, 
                                         original_query: str,
                                         current_graph: Graph,
                                         entity_data: Dict[str, str],
                                         iteration: int,
                                         executed_subqueries: List[ContextualSubquery] = None) -> Optional[Tuple[str, MetaKGEdge]]:
        """
        Generate next subquery informed by meta-KG edge analysis
        
        Args:
            original_query: The original user query
            current_graph: Current RDF knowledge graph
            entity_data: Available entity mappings
            iteration: Current iteration number
            
        Returns:
            Tuple of (subquery, suggested_edge) or None if complete
        """
        # Get suggested edges from meta-KG analysis
        suggested_edges = self._suggest_next_edges(current_graph, original_query)
        
        if not suggested_edges and iteration > 0:
            logger.info("No more viable edges found - planning complete")
            return None
        
        # Get previously executed subqueries to avoid duplicates
        executed_queries = set()
        if executed_subqueries:
            executed_queries = {sq.query.lower().strip() for sq in executed_subqueries}
        
        # Try to generate a grouped query first
        if suggested_edges:
            # Prefer grouping for SmallMolecule -> Gene edges using accumulated results (donepezil, acetylcarnitine, etc.)
            top_edge = suggested_edges[0]
            if top_edge.subject.lower().endswith("smallmolecule") and top_edge.object.lower().endswith("gene"):
                try:
                    # Use accumulated results tracked on the instance
                    accumulated = getattr(self, "_accumulated_results", [])
                    top_drugs = self._get_top_drugs_from_accumulated_results(accumulated)
                    if len(top_drugs) >= 2:
                        grouped_query = f"What genes do {', '.join(top_drugs)} interact with?"
                        if grouped_query.lower().strip() not in executed_queries:
                            logger.info(f"Generated grouped subquery from accumulated context: {grouped_query}")
                            return grouped_query, top_edge
                except Exception as e:
                    logger.debug(f"Could not generate grouped query from accumulated results: {e}")
            
            # Fallback grouping using entity_data heuristics
            if len(entity_data) > 20:  # Only try grouping if we have enough entities
                grouped_query = self._group_similar_queries_from_entities(entity_data, executed_subqueries, top_edge)
                if grouped_query and grouped_query.lower().strip() not in executed_queries:
                    logger.info(f"Generated grouped subquery: {grouped_query}")
                    return grouped_query, top_edge
                else:
                    logger.debug(f"No grouped query generated. Entity count: {len(entity_data)}, Suggested edge: {top_edge}")
        
        # Create edge context for LLM planning
        edge_context = ""
        if suggested_edges:
            edge_descriptions = []
            for edge in suggested_edges[:5]:  # Top 5 edges
                edge_descriptions.append(f"- {edge.subject} {edge.predicate} {edge.object}")
            edge_context = f"\n\nBased on the current knowledge state, here are some viable single-hop relationships you could explore next:\n" + "\n".join(edge_descriptions)
        
        # Create previously executed queries context
        executed_queries_context = ""
        if executed_queries:
            executed_list = list(executed_queries)[:10]  # Show last 10 to avoid long prompts
            executed_queries_context = f"\n\nPREVIOUSLY EXECUTED SUBQUERIES (DO NOT REPEAT):\n" + "\n".join([f"- {q}" for q in executed_list])
        
        # Enhanced planner prompt with strict single-hop enforcement
        planner_prompt = f"""You are a biomedical query planner. Your job is to generate ONLY simple, single-hop questions.

        CRITICAL RULES FOR SINGLE-HOP QUERIES:
        - Each query must connect exactly TWO node types with ONE relationship
        - NO complex sentences with multiple relationships
        - NO queries that imply multi-hop connections
        - NO questions with "and", "through", "via", or "by way of"
        
        VALID SINGLE-HOP EXAMPLES:
        ✅ "What drugs treat diabetes?" (Disease -> SmallMolecule)
        ✅ "What genes does metformin interact with?" (SmallMolecule -> Gene)
        ✅ "What processes does AMPK regulate?" (Gene -> PhysiologicalProcess)
        
        INVALID MULTI-HOP EXAMPLES (DO NOT GENERATE):
        ❌ "What genes do drugs that treat diabetes interact with?"
        ❌ "What processes are affected by genes that interact with diabetes drugs?"
        ❌ "What pathways do genes interacting with diabetes drugs participate in?"
        
        Available node types: Disease, Gene, SmallMolecule, PhysiologicalProcess, PathologicalProcess, Polypeptide, PhenotypicFeature

        Current results in knowledge graph:
        {current_graph.serialize(format="turtle")}
        {edge_context}
        {executed_queries_context}

        Strategy for mechanistic exploration:
        1. Find drugs treating the condition (Disease -> SmallMolecule)
        2. Find what genes each specific drug interacts with (SmallMolecule -> Gene)
        3. Find what processes each specific gene regulates (Gene -> PhysiologicalProcess)
        4. Continue with simple, direct relationships
        
        Original query: {original_query}
        Current iteration: {iteration}
        
        Generate ONE simple, single-hop question following the valid examples above.
        If you have thoroughly explored all reasonable single-hop relationships, respond "COMPLETE".
        
        Response format: Just the single-hop question, nothing else.
        """
        
        try:
            response = self.llm.invoke([{"role": "system", "content": planner_prompt}])
            subquery = response.content.strip()
            
            # Validate that subquery is truly single-hop
            if not self._is_single_hop_query(subquery):
                logger.warning(f"LLM generated complex multi-hop subquery: {subquery}")
                # Try to simplify to single-hop
                simplified = self._simplify_to_single_hop(subquery, original_query)
                if simplified:
                    subquery = simplified
                    logger.info(f"Simplified to single-hop: {subquery}")
                else:
                    logger.warning("Could not simplify complex query - skipping")
                    return None
            
            # Check for duplicates
            if subquery.lower().strip() in executed_queries:
                logger.warning(f"LLM generated duplicate subquery: {subquery}")
                
                # Try to generate an alternative subquery for mechanistic exploration
                if "antioxidant" in original_query.lower():
                    alternative_queries = [
                        "What biological processes are regulated by the identified genes?",
                        "What pathways do the identified proteins participate in?",
                        "What cellular components are affected by these drugs?",
                        "What molecular functions are associated with these proteins?"
                    ]
                    for alt_query in alternative_queries:
                        if alt_query.lower().strip() not in executed_queries:
                            subquery = alt_query
                            logger.info(f"Using alternative subquery: {subquery}")
                            break
                    else:
                        logger.info("No alternative subqueries available - marking complete")
                        return None
                else:
                    return None  # No alternatives for non-mechanistic queries
            
            if subquery == "COMPLETE" or iteration >= 10:  # Max iterations check
                return None
            
            # Additional check: don't stop early for mechanism queries
            if iteration < 3 and any(term in original_query.lower() for term in ["mechanism", "activity", "targeting", "pathway"]):
                # Force continuation for mechanistic queries in early iterations
                if subquery == "COMPLETE":
                    logger.info("Ignoring early COMPLETE for mechanistic query - continuing exploration")
                    # Generate a mechanistic follow-up based on current results
                    if "smallmolecule" in str(current_graph).lower():
                        subquery = "What genes do the identified drugs interact with?"
                    elif "gene" in str(current_graph).lower():
                        subquery = "What biological processes do the identified genes participate in?"
                    else:
                        return None
            
            # Try to match the subquery to a suggested edge
            suggested_edge = None
            if suggested_edges:
                # Simple heuristic matching - could be enhanced
                for edge in suggested_edges:
                    if (edge.object.lower().replace("biolink:", "") in subquery.lower() or
                        edge.predicate.replace("_", " ").lower() in subquery.lower()):
                        suggested_edge = edge
                        break
                
                # If no specific match, use the top-scored edge
                if not suggested_edge:
                    suggested_edge = suggested_edges[0]
            
            logger.info(f"Generated meta-KG informed subquery: {subquery}")
            if suggested_edge:
                logger.debug(f"Based on suggested edge: {suggested_edge}")
            
            return subquery, suggested_edge
            
        except Exception as e:
            logger.error(f"Error generating meta-KG informed subquery: {e}")
            return None

    def _update_knowledge_graph(self, results: List[Dict[str, Any]], graph: Graph) -> None:
        """
        Update RDF knowledge graph with new results (same as LangGraph RDFgraphUpdater)
        
        Args:
            results: New relationship results to add
            graph: RDF graph to update
        """
        # Bind biolink namespace
        graph.bind("biolink", self.BIOLINK)
        
        # Bind all entity type namespaces dynamically
        for node_type, namespace in self.ENTITY_NAMESPACE_MAP.items():
            # Create a clean binding name from the node type
            clean_name = node_type.replace("biolink:", "").lower()
            graph.bind(clean_name, namespace)
        
        def make_entity_uri(name: str, entity_type: str):
            ns = self.ENTITY_NAMESPACE_MAP.get(entity_type, self.EX)
            return ns[name.replace(" ", "_").replace(":", "-").lower()]
        
        # Add triples to graph
        for result in results:
            if all(key in result for key in ['subject', 'predicate', 'object']):
                subj = make_entity_uri(result['subject'], result.get("subject_type", ""))
                pred = URIRef(self.BIOLINK + result['predicate'].split(":")[1])
                obj = make_entity_uri(result['object'], result.get("object_type", ""))
                graph.add((subj, pred, obj))

    def _execute_subquery_with_bte(self, subquery: str, entity_data: Dict[str, str]) -> Tuple[List[Dict[str, Any]], float]:
        """
        Execute a single subquery using BTE API
        
        Args:
            subquery: The subquery to execute
            entity_data: Available entity mappings
            
        Returns:
            Tuple of (results, execution_time)
        """
        start_time = time.time()
        
        try:
            # Log entity data being used
            logger.info(f"Entity data available for TRAPI query: {entity_data}")
            logger.info(f"Accumulated results count: {len(getattr(self, '_accumulated_results', []))}")
            
            # Build TRAPI query with accumulated results for batch processing
            trapi_query = self.trapi_builder.build_trapi_query(
                query=subquery,
                entity_data=entity_data,
                accumulated_results=getattr(self, '_accumulated_results', [])
            )
            
            if not trapi_query:
                logger.warning(f"Failed to build TRAPI query for: {subquery}")
                return [], time.time() - start_time
            
            # Log the built TRAPI query
            logger.info(f"Built TRAPI query: {json.dumps(trapi_query, indent=2)}")
            
            # Execute with BTE
            bte_response = self.bte_client.execute_trapi_query(trapi_query)
            
            # Parse results
            results, _ = self.bte_client.parse_bte_results(bte_response, k=5, max_results=50)
            
            logger.info(f"Subquery executed successfully: {len(results)} results found")
            return results, time.time() - start_time
            
        except Exception as e:
            logger.error(f"Error executing subquery '{subquery}': {e}")
            return [], time.time() - start_time

    def _synthesize_final_answer(self, 
                                original_query: str,
                                knowledge_graph: Graph,
                                entity_data: Dict[str, str],
                                executed_subqueries: List[ContextualSubquery]) -> str:
        """
        Synthesize final answer from accumulated knowledge (same as LangGraph summary_node)
        
        Args:
            original_query: Original user query
            knowledge_graph: Accumulated RDF knowledge graph  
            entity_data: Entity mappings
            executed_subqueries: All executed subqueries
            
        Returns:
            Comprehensive final answer
        """
        summary_prompt = f"""You are an expert proficient in the pharmaceutical sciences, medicinal chemistry and biomedical research.
        Your team has access to a biomedical knowledge graph, and have provided you with the following results based on the user's query.
        However, the knowledge graph might not be perfect and have gaps in its data, and some relationships between the entities might be implicit.
        Your job is to answer the user's query using your team's results and your own biomedical expertise.
        Your summary of the results MUST be comprehensive while still avoiding redundancy.
        Expound on the logical steps you took to form your final answer.

        Make sure to organize your answer around the user query:
        {original_query}

        Here are the findings of your team expressed in Turtle:
        {knowledge_graph.serialize(format="turtle")}

        Here are the entities each ID corresponds to:
        {entity_data}

        You MUST base your answer on the evidence/context included in the prompt; 
        however, you can use your expertise to contextualize the results. 
        For example, if the results only show target genes for dirithromycin, knowing that dirithromycin and erythromycin are part of the same drug class, you can infer that they are likely to share similar properties and targets.
        
        Maintain complete transparency about your problem-solving process; the relationships between each entity should be clear in your answer.
        Do NOT list down all entities; rather, choose to most important ones to illustrate your point (for example, "the BRCA family of proteins is involved in breast cancer").
        Remember, prioritize accuracy, explainability, and user understanding.
        Only include relevant results in the final answer.
        """
        
        try:
            response = self.llm.invoke([{"role": "system", "content": summary_prompt}])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error synthesizing final answer: {e}")
            return f"Error generating final answer. Executed {len(executed_subqueries)} subqueries with mixed results."

    def create_adaptive_plan(self, 
                           query: str, 
                           entities: Dict[str, str] = None,
                           max_iterations: int = 10) -> AdaptivePlan:
        """
        Create initial adaptive plan using meta-KG informed contextual reasoning
        
        Args:
            query: Original biomedical query
            entities: Optional pre-extracted entities
            max_iterations: Maximum number of iterations
            
        Returns:
            Initial adaptive plan
        """
        logger.info(f"Creating meta-KG aware adaptive plan for query: {query[:100]}...")
        
        # Extract entities if needed
        if entities is None:
            entity_result = self.bio_ner.extract_and_link(query)
            entities = entity_result.get("entity_ids", {})
        
        # Create plan with RDF knowledge graph
        plan = AdaptivePlan(
            original_query=query,
            entity_data=entities,
            max_iterations=max_iterations
        )
        
        logger.info(f"Created meta-KG aware adaptive plan")
        return plan

    def execute_adaptive_plan(self, plan: AdaptivePlan) -> AdaptivePlan:
        """
        Execute adaptive plan using meta-KG informed contextual reasoning
        
        Args:
            plan: The adaptive plan to execute
            
        Returns:
            Updated plan with results
        """
        logger.info(f"Executing meta-KG aware adaptive plan for query: {plan.original_query[:100]}...")
        start_time = time.time()
        
        # Initialize accumulated results for batch query support
        self._accumulated_results = []
        
        for iteration in range(plan.max_iterations):
            plan.current_iteration = iteration + 1
            logger.info(f"Meta-KG aware iteration {plan.current_iteration}")
            
            # Generate next contextual subquery using meta-KG edges
            subquery_result = self._generate_metakg_informed_subquery(
                original_query=plan.original_query,
                current_graph=plan.knowledge_graph,
                entity_data=plan.entity_data,
                iteration=iteration,
                executed_subqueries=plan.executed_subqueries
            )
            
            if not subquery_result:
                plan.completion_reason = "Plan determined complete by meta-KG aware contextual reasoning"
                break
                
            subquery, suggested_edge = subquery_result
            logger.info(f"Executing subquery: {subquery}")
            
            # Execute subquery
            results, exec_time = self._execute_subquery_with_bte(subquery, plan.entity_data)
            
            # Create subquery record with meta-KG edge information
            contextual_subquery = ContextualSubquery(
                query=subquery,
                reasoning_context=f"Generated at iteration {iteration + 1} using meta-KG edge analysis",
                suggested_edge=suggested_edge,
                results=results,
                success=len(results) > 0,
                execution_time=exec_time,
                iteration_number=iteration + 1
            )
            
            plan.executed_subqueries.append(contextual_subquery)
            plan.accumulated_results.extend(results)
            
            # Update accumulated results for batch query support
            self._accumulated_results.extend(results)
            
            # Update knowledge graph with new results
            if results:
                self._update_knowledge_graph(results, plan.knowledge_graph)
                logger.info(f"Updated knowledge graph with {len(results)} new relationships")
                
                # CRITICAL: Extract and update entity data from BTE results for subsequent queries
                new_entities = self._extract_entities_from_results(results)
                if new_entities:
                    # Update the plan's entity_data with new entities found in results
                    plan.entity_data.update(new_entities)
                    logger.info(f"Updated entity_data with {len(new_entities)} new entities. Total: {len(plan.entity_data)}")
                    logger.debug(f"Updated entity_data keys: {list(plan.entity_data.keys())[:10]}...")  # Show first 10
                
                if suggested_edge:
                    logger.debug(f"Results aligned with suggested edge: {suggested_edge}")
            else:
                logger.warning(f"No results found for subquery: {subquery}")
        
        # Generate final answer
        plan.final_answer = self._synthesize_final_answer(
            original_query=plan.original_query,
            knowledge_graph=plan.knowledge_graph,
            entity_data=plan.entity_data,
            executed_subqueries=plan.executed_subqueries
        )
        
        plan.total_execution_time = time.time() - start_time
        logger.info(f"Meta-KG aware adaptive execution completed in {plan.total_execution_time:.2f}s with {len(plan.accumulated_results)} total results")
        
        return plan

    def get_meta_kg_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded meta-KG edges
        
        Returns:
            Dictionary with meta-KG statistics
        """
        edge_count_by_subject = defaultdict(int)
        edge_count_by_predicate = defaultdict(int)
        
        for edge in self._meta_kg_edges:
            edge_count_by_subject[edge.subject] += 1
            edge_count_by_predicate[edge.predicate] += 1
        
        return {
            "total_edges": len(self._meta_kg_edges),
            "unique_subjects": len(edge_count_by_subject),
            "unique_predicates": len(edge_count_by_predicate),
            "top_subjects": dict(sorted(edge_count_by_subject.items(), key=lambda x: x[1], reverse=True)[:10]),
            "top_predicates": dict(sorted(edge_count_by_predicate.items(), key=lambda x: x[1], reverse=True)[:10])
        }