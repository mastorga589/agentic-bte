"""
RDF Graph Manager - Knowledge Graph Accumulation

This module manages the RDF graph used to accumulate biomedical knowledge
across agent iterations, providing structured storage and retrieval of
biomedical relationships discovered through the research process.

Migrated and enhanced from the original BTE-LLM implementation.
"""

from typing import List, Dict, Any, Optional
from rdflib import Graph, URIRef, Namespace
from rdflib.namespace import RDF, RDFS
import logging

logger = logging.getLogger(__name__)


class RDFGraphManager:
    """
    Manages RDF graph for accumulating biomedical knowledge
    
    Handles namespace management, entity URI creation, and structured
    knowledge storage across multi-agent research iterations.
    """
    
    def __init__(self):
        """Initialize RDF graph with biomedical namespaces"""
        self.graph = Graph()
        self._setup_namespaces()
        self._setup_entity_mappings()
    
    def _setup_namespaces(self):
        """Set up biomedical namespaces for RDF graph"""
        # Define biomedical namespaces
        self.BIOLINK = Namespace("https://w3id.org/biolink/vocab/")
        self.EX = Namespace("http://example.org/entity/")
        self.GENE = Namespace("https://biolink.github.io/biolink-model/Gene/")
        self.DISEASE = Namespace("https://biolink.github.io/biolink-model/Disease/")
        self.PHYSPROCESS = Namespace("https://biolink.github.io/biolink-model/PhysiologicalProcess/")
        self.BIOENT = Namespace("https://biolink.github.io/biolink-model/BiologicalEntity/")
        self.PATHPROCESS = Namespace("https://biolink.github.io/biolink-model/PathologicalProcess/")
        self.SMALLMOL = Namespace("https://biolink.github.io/biolink-model/SmallMolecule/")
        self.PHENFEATURE = Namespace("https://biolink.github.io/biolink-model/PhenotypicFeature/")
        self.POLYPEPTIDE = Namespace("https://biolink.github.io/biolink-model/Polypeptide/")
        
        # Bind namespaces for readable output
        self.graph.bind("biolink", self.BIOLINK)
        self.graph.bind("ex", self.EX)
        self.graph.bind("gene", self.GENE)
        self.graph.bind("disease", self.DISEASE)
        self.graph.bind("physprocess", self.PHYSPROCESS)
        self.graph.bind("phenprocess", self.PHENFEATURE)
        self.graph.bind("bioent", self.BIOENT)
        self.graph.bind("pathprocess", self.PATHPROCESS)
        self.graph.bind("smallmol", self.SMALLMOL)
        self.graph.bind("polypeptide", self.POLYPEPTIDE)
    
    def _setup_entity_mappings(self):
        """Map biolink entity types to their namespaces"""
        self.entity_namespace_map = {
            "biolink:Gene": self.GENE,
            "biolink:Disease": self.DISEASE,
            "biolink:PhysiologicalProcess": self.PHYSPROCESS,
            "biolink:BiologicalEntity": self.BIOENT,
            "biolink:PathologicalProcess": self.PATHPROCESS,
            "biolink:SmallMolecule": self.SMALLMOL,
            "biolink:PhenotypicFeature": self.PHENFEATURE,
            "biolink:Polypeptide": self.POLYPEPTIDE,
        }
    
    def make_entity_uri(self, name: str, entity_type: str = "") -> URIRef:
        """
        Create a URI for a biomedical entity
        
        Args:
            name: Entity name
            entity_type: Biolink entity type (e.g., "biolink:Gene")
            
        Returns:
            URIRef for the entity
        """
        namespace = self.entity_namespace_map.get(entity_type, self.EX)
        clean_name = name.replace(" ", "_").replace(":", "-").lower()
        return namespace[clean_name]
    
    def add_triples(self, triples: List[Dict[str, Any]]) -> int:
        """
        Add biomedical relationship triples to the RDF graph
        
        Args:
            triples: List of relationship dictionaries with subject, predicate, object
            
        Returns:
            Number of triples successfully added
        """
        added_count = 0
        
        for triple in triples:
            try:
                # Extract triple components
                subject_name = triple.get('subject', '')
                predicate = triple.get('predicate', '')
                object_name = triple.get('object', '')
                
                subject_type = triple.get('subject_type', '')
                object_type = triple.get('object_type', '')
                
                if not all([subject_name, predicate, object_name]):
                    logger.warning(f"Incomplete triple data: {triple}")
                    continue
                
                # Create URIs
                subject_uri = self.make_entity_uri(subject_name, subject_type)
                object_uri = self.make_entity_uri(object_name, object_type)
                
                # Create predicate URI (remove biolink: prefix)
                predicate_clean = predicate.split(":")[-1] if ":" in predicate else predicate
                predicate_uri = URIRef(self.BIOLINK + predicate_clean)
                
                # Add triple to graph
                self.graph.add((subject_uri, predicate_uri, object_uri))
                added_count += 1
                
            except Exception as e:
                logger.error(f"Error adding triple {triple}: {str(e)}")
                continue
        
        logger.info(f"Added {added_count}/{len(triples)} triples to RDF graph")
        return added_count
    
    def get_turtle_representation(self) -> str:
        """
        Get Turtle serialization of the RDF graph
        
        Returns:
            Turtle-formatted string representation
        """
        try:
            return self.graph.serialize(format="turtle")
        except Exception as e:
            logger.error(f"Error serializing graph to Turtle: {str(e)}")
            return ""
    
    def get_triple_count(self) -> int:
        """
        Get the number of triples in the graph
        
        Returns:
            Number of RDF triples
        """
        return len(self.graph)
    
    def clear_graph(self):
        """Clear all triples from the graph"""
        self.graph.remove((None, None, None))
        logger.info("Cleared RDF graph")
    
    def query_graph(self, sparql_query: str) -> List[Dict[str, Any]]:
        """
        Execute SPARQL query on the RDF graph
        
        Args:
            sparql_query: SPARQL query string
            
        Returns:
            Query results as list of dictionaries
        """
        try:
            results = []
            for row in self.graph.query(sparql_query):
                result_dict = {}
                for var in row.labels:
                    result_dict[str(var)] = str(row[var])
                results.append(result_dict)
            return results
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {str(e)}")
            return []
    
    def get_entity_relationships(self, entity_name: str) -> List[Dict[str, str]]:
        """
        Get all relationships for a specific entity
        
        Args:
            entity_name: Name of the entity to query
            
        Returns:
            List of relationships involving the entity
        """
        # Simple pattern matching for entity relationships
        sparql_query = f"""
        PREFIX biolink: <https://w3id.org/biolink/vocab/>
        SELECT ?subject ?predicate ?object
        WHERE {{
            {{
                ?subject ?predicate ?object .
                FILTER(CONTAINS(LCASE(STR(?subject)), LCASE("{entity_name}")) ||
                       CONTAINS(LCASE(STR(?object)), LCASE("{entity_name}")))
            }}
        }}
        """
        
        return self.query_graph(sparql_query)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about the RDF graph
        
        Returns:
            Dictionary with graph statistics
        """
        return {
            "total_triples": self.get_triple_count(),
            "unique_subjects": len(set(self.graph.subjects())),
            "unique_predicates": len(set(self.graph.predicates())),
            "unique_objects": len(set(self.graph.objects())),
            "namespaces": len(list(self.graph.namespaces())),
        }