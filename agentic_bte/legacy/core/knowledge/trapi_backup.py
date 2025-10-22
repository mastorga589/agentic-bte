"""
TRAPI Query Builder - Build TRAPI queries from natural language

This module provides functionality to convert natural language biomedical
queries into TRAPI (Translator Reasoner API) format using LLMs and knowledge
graph metadata.

Migrated and enhanced from the original BTE-LLM implementation.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from copy import deepcopy

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Literal

from ...config.settings import get_settings
from ...exceptions.base import ExternalServiceError
from .bte_client import BTEClient

logger = logging.getLogger(__name__)


# Legacy convenience functions (kept for backward compatibility)
def build_trapi_query_legacy(query: str, entity_data: Dict[str, str] = None, 
                            failed_trapis: List[Dict] = None, 
                            openai_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Legacy convenience function - use TRAPIQueryBuilder.build_trapi_query instead
    """
    builder = TRAPIQueryBuilder(openai_api_key)
    return builder.build_trapi_query(query, entity_data, failed_trapis)


def validate_trapi(trapi_query: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate TRAPI query structure
    
    Args:
        trapi_query: TRAPI query to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    from ...config.settings import get_settings
    settings = get_settings()
    # Simple validation without requiring full TRAPIQueryBuilder initialization
    try:
        if "message" not in trapi_query:
            return False, "Missing 'message' key"
        message = trapi_query["message"]
        if "query_graph" not in message:
            return False, "Missing 'query_graph' key in message"
        query_graph = message["query_graph"]
        if "nodes" not in query_graph or "edges" not in query_graph:
            return False, "Missing 'nodes' or 'edges' in query_graph"
        return True, "Valid TRAPI query"
    except Exception as e:
        return False, f"Validation error: {str(e)}"
    """
    TRAPI Query Builder for BioThings Explorer
    
    This class builds TRAPI (Translator Reasoner API) queries from natural language
    biomedical questions, with support for entity mapping and predicate selection.
    
    Migrated from original BTE-LLM implementations with enhancements.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the TRAPI query builder
        
        Args:
            openai_api_key: OpenAI API key for LLM operations
        """
        self.settings = get_settings()
        self.openai_api_key = openai_api_key or self.settings.openai_api_key
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for TRAPI query building")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=0,
            model=self.settings.openai_model,
            api_key=self.openai_api_key
        )
        
        # Initialize BTE client for meta knowledge graph
        self.bte_client = BTEClient()
        
        # Cache for meta knowledge graph
        self._meta_kg = None
    
    @property
    def meta_kg(self) -> Dict[str, Any]:
        """Get cached meta knowledge graph"""
        if self._meta_kg is None:
            self._meta_kg = self.bte_client.get_meta_knowledge_graph()
        return self._meta_kg
    
    def extract_dict(self, raw_string: str) -> Dict[str, Any]:
        """
        Extract dictionary from raw LLM response string
        
        Args:
            raw_string: Raw response string from LLM
            
        Returns:
            Parsed dictionary or error dict
        """
        # Strip whitespace
        raw_string = raw_string.strip()
        
        # Handle markdown code blocks
        if raw_string.startswith("```json"):
            # Remove ```json from start and ``` from end
            raw_string = raw_string[7:]  # Remove ```json
            if raw_string.endswith("```"):
                raw_string = raw_string[:-3]  # Remove ```
            raw_string = raw_string.strip()
        elif raw_string.startswith("```"):
            # Handle generic code blocks
            lines = raw_string.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]  # Remove first line
            if lines[-1].strip() == "```":
                lines = lines[:-1]  # Remove last line
            raw_string = '\n'.join(lines).strip()
        
        # Try to find JSON object within the text
        start_index = raw_string.find("{")
        if start_index != -1:
            # Find the matching closing brace
            brace_count = 0
            end_index = start_index
            for i, char in enumerate(raw_string[start_index:], start_index):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_index = i
                        break
            
            if brace_count == 0:  # Found matching brace
                extracted_dict = raw_string[start_index:end_index + 1].strip()
                try:
                    return json.loads(extracted_dict)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}")
                    logger.error(f"Attempted to parse: {extracted_dict}")
        
        # Try parsing the entire string as JSON
        try:
            return json.loads(raw_string)
        except json.JSONDecodeError:
            logger.error(f"Could not parse as JSON: {raw_string}")
            return {"error": "could not parse dict"}
    
    def find_predicates(self, subject: str, obj: str) -> List[str]:
        """
        Find predicates between subject and object types using meta knowledge graph
        
        Args:
            subject: Subject biolink type (e.g., "biolink:Disease")
            obj: Object biolink type (e.g., "biolink:SmallMolecule")
            
        Returns:
            List of available predicates
        """
        predicates = []
        meta_edges = self.meta_kg.get("edges", [])
        
        # Get excluded predicates from settings
        excluded_predicates = self.settings.excluded_predicates
        
        for edge in meta_edges:
            if (edge.get("subject", "").lower() == subject.lower() and 
                edge.get("object", "").lower() == obj.lower()):
                predicate = edge.get("predicate")
                # Only add predicate if it's not in the excluded list
                if predicate and predicate not in excluded_predicates:
                    predicates.append(predicate)
        
        logger.debug(f"Found {len(predicates)} predicates for {subject} -> {obj}")
        return predicates
    
    def choose_predicate(self, predicate_list: List[str], query: str) -> str:
        """
        Choose the most appropriate predicate for the query using LLM
        
        Args:
            predicate_list: List of available predicates
            query: Original natural language query
            
        Returns:
            Selected predicate
        """
        if not predicate_list:
            logger.warning("No predicates available for selection")
            return ""
        
        if len(predicate_list) == 1:
            return predicate_list[0]
        
        try:
            predicate_prompt = f"""Choose the most specific predicate by examining the query closely and choosing the closest answer.  
                
                Here is the query: {query}
                Here are the available predicates: {predicate_list}
                
                Return only the chosen predicate string.
                """
            
            response = self.llm.invoke(predicate_prompt)
            chosen = response.content.strip()
            
            # Find the predicate that matches the response
            for predicate in predicate_list:
                if predicate in chosen:
                    logger.debug(f"Selected predicate '{predicate}' for query: {query}")
                    return predicate
            
            # If no exact match, return the first one
            logger.warning(f"No exact predicate match found, using first: {predicate_list[0]}")
            return predicate_list[0]
            
        except Exception as e:
            logger.error(f"Error choosing predicate: {e}")
            return predicate_list[0] if predicate_list else ""
    
    def invoke_with_retries(self, prompt: str, parser_func, max_retries: int = 3, delay: int = 5):
        """
        Invoke LLM with retry logic
        
        Args:
            prompt: LLM prompt
            parser_func: Function to parse the response
            max_retries: Maximum number of retries
            delay: Delay between retries in seconds
            
        Returns:
            Parsed response or None on failure
        """
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                return parser_func(response.content)
            except Exception as e:
                logger.warning(f"Retry {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:  # Don't sleep on last attempt
                    time.sleep(delay)
        
        logger.error(f"All {max_retries} attempts failed")
        return None
    
    def identify_nodes(self, query: str) -> Dict[str, str]:
        """
        Identify subject and object nodes for TRAPI query using LLM
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary with subject and object biolink types
        """
        node_prompt = f"""
        Your task is to help build a TRAPI query by identifying the correct subject and object nodes from the list of nodes below. 

        In biomedical knowledge graphs:
        - Subject: The starting entity (what we know or are querying FROM)
        - Object: The target entity (what we want to find or are querying TO)
        
        For queries like "Which drugs treat diabetes?", diabetes (Disease) is the subject and drugs (SmallMolecule) is the object.
        For queries like "What genes are associated with cancer?", cancer (Disease) is the subject and genes (Gene) is the object.
        
        Each of these must have the prefix "biolink:". You must use the context and intent behind the user query to make the correct choice.

        Available node types: Disease, PhysiologicalProcess, BiologicalEntity, Gene, PathologicalProcess, Polypeptide, SmallMolecule, PhenotypicFeature

        Here is the user query: {query}

        Be as specific as possible as downstream results will be affected by your node choices.
        Your response must be a JSON object with "subject" and "object" as keys and their biolink node types as values.
        """
        
        result = self.invoke_with_retries(node_prompt, self.extract_dict)
        if result and "error" not in result:
            logger.info(f"Identified nodes for query '{query}': {result}")
            return result
        else:
            logger.error(f"Failed to identify nodes for query: {query}")
            return {"error": "Could not determine subject/object nodes"}
    
    def build_trapi_query_structure(self, query: str, subject_object: Dict[str, str], predicate: str, 
                                  entity_data: Dict[str, str], failed_trapis: List[Dict]) -> Dict[str, Any]:
        """
        Build the actual TRAPI query structure using LLM
        
        Args:
            query: Original natural language query
            subject_object: Dictionary with subject and object types
            predicate: Selected predicate
            entity_data: Entity names to IDs mapping
            failed_trapis: List of previously failed TRAPI queries
            
        Returns:
            TRAPI query dictionary
        """
        print("\\n\\n Building trapi....")
        
        # TRAPI example for reference
        trapi_example = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "categories": [
                                "biolink:Disease"
                            ],
                            "ids": [
                                "MONDO:0016575"
                            ]
                        },
                        "n1": {
                            "categories": [
                                "biolink:PhenotypicFeature"
                            ]
                        }
                    },
                    "edges": {
                        "e01": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": [
                                "biolink:has_phenotype"
                            ]
                        }
                    }
                }
            }
        }
        
        trapi_prompt = f"""
            You are a smart assistant that has access to a knowledge graph. 
            You are tasked with correctly parsing the user prompt into a TRAPI query. 
            Here's an example TRAPI query:
            {json.dumps(trapi_example, indent=2)}

            Here is the actual user query: "{query}"

            Here are the biological entities and their IDs extracted from the query:
            {json.dumps(entity_data, indent=2)}

            Here are the nodes you MUST use:
            {json.dumps(subject_object, indent=2)}

            Here is the chosen predicate:
            {predicate}

            Do NOT use the following TRAPI queries as they have failed to get results: 
            {failed_trapis}

            Some predicates have directionality ("treated_by" is NOT the same as "treats"). When defining the edges, you MUST use the correct predicate with the correct directionality.
            (e.g., biolink:Disease is "treated_by" (NOT "treats") biolink:SmallMolecule; 
            biolink:Disease as subject and biolink:Gene as object should result in "condition_associated_with_gene", NOT "gene_associated_with_condition")

            In this KG, the subject of the predicate is always the source (domain), and the object is the target (range). For example, if 'X treats Y', then X is the subject and Y is the object.
            
            Please make sure that only one node in the TRAPI query has the "ids" field (either "n0" or "n1" but NOT both); 
            decide which one would result in more useful results (for "Which of these genes are involved in a physiological process?", including the IDs for Gene would result in more useful results. In this case DO NOT include the ID for Physiological Process)

            Using these, output a JSON object containing the completed TRAPI query. 
        """
        
        result = self.invoke_with_retries(trapi_prompt, self.extract_dict)
        
        if result and "error" not in result:
            logger.info(f"Successfully built TRAPI query for: {query}")
            return result
        else:
            logger.error(f"Failed to build TRAPI query for: {query}")
            return {"error": "Could not build TRAPI query"}
    
    def build_trapi_query(self, query: str, entity_data: Optional[Dict[str, str]] = None, 
                         failed_trapis: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Main method to build a complete TRAPI query from natural language
        
        Args:
            query: Natural language biomedical query
            entity_data: Optional mapping of entity names to IDs
            failed_trapis: Optional list of previously failed TRAPI queries
            
        Returns:
            Complete TRAPI query dictionary
        """
        if entity_data is None:
            entity_data = {}
        if failed_trapis is None:
            failed_trapis = []
        
        max_retries = 3
        
        try:
            # Step 1: Identify subject and object nodes
            subject_object = self.identify_nodes(query)
            if not subject_object or "error" in subject_object:
                logger.error("Could not determine subject/object nodes")
                return {"error": "Could not determine subject/object nodes"}
            
            print(f"\\n\\nIdentified subject/object: {json.dumps(subject_object)}")
            
            # Step 2: Find available predicates
            predicate_list = self.find_predicates(
                subject_object.get("subject", ""), 
                subject_object.get("object", "")
            )
            
            # Step 3: Remove failed predicates from previous attempts
            failed_predicates = set()
            for trapi_query in failed_trapis:
                try:
                    predicates = (trapi_query.get("message", {})
                                .get("query_graph", {})
                                .get("edges", {})
                                .get("e01", {})
                                .get("predicates", []))
                    failed_predicates.update(predicates)
                except Exception as e:
                    logger.warning(f"Error extracting predicates from failed_trapis: {e}")
            
            # Remove failed predicates from predicate list
            predicate_list = [p for p in predicate_list if p not in failed_predicates]
            
            if not predicate_list:
                logger.error("No valid predicates found")
                return {"error": "No valid predicates found"}
            
            # Step 4: Choose the best predicate
            predicate = self.choose_predicate(predicate_list, query)
            
            # Step 5: Build the TRAPI query structure
            trapi = self.build_trapi_query_structure(query, subject_object, predicate, entity_data, failed_trapis)
            
            if trapi and "error" not in trapi:
                logger.info("Successfully built TRAPI query")
                return trapi
            
            # Step 6: Retry if initial attempt fails
            for attempt in range(max_retries):
                logger.warning(f"TRAPI build failed, retry {attempt + 1}/{max_retries}")
                trapi = self.build_trapi_query_structure(query, subject_object, predicate, entity_data, failed_trapis)
                if trapi and "error" not in trapi:
                    logger.info(f"TRAPI query built successfully on retry {attempt + 1}")
                    return trapi
            
            logger.error("All TRAPI build attempts failed")
            return {"error": "Invalid TRAPI"}
            
        except Exception as e:
            logger.error(f"Error building TRAPI query: {e}")
            raise ExternalServiceError(
                f"TRAPI query building failed: {str(e)}",
                service_name="trapi_builder"
            ) from e


# Convenience functions
def build_trapi_query(query: str, entity_data: Optional[Dict[str, str]] = None, 
                     failed_trapis: Optional[List[Dict]] = None,
                     openai_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to build a TRAPI query
    
    Args:
        query: Natural language biomedical query
        entity_data: Optional entity name to ID mapping
        failed_trapis: Optional list of failed TRAPI queries
        openai_api_key: Optional OpenAI API key
        
    Returns:
        TRAPI query dictionary
    """
    builder = TRAPIQueryBuilder(openai_api_key)
    return builder.build_trapi_query(query, entity_data, failed_trapis)