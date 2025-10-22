#!/usr/bin/env python3
"""
TRAPI Single-Hop Query Validator

This module ensures that all TRAPI queries conform to the single-hop constraint
required by BioThings Explorer. Multi-hop queries cause 400 Bad Request errors.

Key features:
1. Validate TRAPI queries to ensure they have exactly one edge
2. Fix multi-hop queries by breaking them into single-hop components  
3. Provide enhanced TRAPI builder that enforces single-hop constraint
4. Maintain compatibility with existing BTE query patterns
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy

logger = logging.getLogger(__name__)


class TRAPISingleHopValidator:
    """
    Validator to ensure TRAPI queries conform to single-hop constraint
    """
    
    def __init__(self):
        """Initialize the validator"""
        self.validation_errors = []
    
    def validate_trapi_query(self, trapi_query: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that a TRAPI query conforms to single-hop constraint
        
        Args:
            trapi_query: TRAPI query to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Check basic structure
            if 'message' not in trapi_query:
                errors.append("Missing 'message' key in TRAPI query")
                return False, errors
            
            message = trapi_query['message']
            
            if 'query_graph' not in message:
                errors.append("Missing 'query_graph' key in message")
                return False, errors
            
            query_graph = message['query_graph']
            
            if 'nodes' not in query_graph or 'edges' not in query_graph:
                errors.append("Missing 'nodes' or 'edges' in query_graph")
                return False, errors
            
            nodes = query_graph['nodes']
            edges = query_graph['edges']
            
            # Check single-hop constraint
            if len(edges) == 0:
                errors.append("Query has no edges - at least one edge is required")
                return False, errors
            
            if len(edges) > 1:
                errors.append(f"Multi-hop query detected: {len(edges)} edges found. BTE requires exactly 1 edge.")
                return False, errors
            
            # Validate edge structure
            edge_id, edge_data = next(iter(edges.items()))
            
            if 'subject' not in edge_data or 'object' not in edge_data:
                errors.append(f"Edge {edge_id} missing 'subject' or 'object'")
                return False, errors
            
            subject_node = edge_data['subject']
            object_node = edge_data['object']
            
            # Check that referenced nodes exist
            if subject_node not in nodes:
                errors.append(f"Edge references non-existent subject node: {subject_node}")
                return False, errors
            
            if object_node not in nodes:
                errors.append(f"Edge references non-existent object node: {object_node}")
                return False, errors
            
            # Check that we have exactly 2 nodes (for single-hop)
            if len(nodes) != 2:
                errors.append(f"Single-hop query should have exactly 2 nodes, found {len(nodes)}")
                return False, errors
            
            # Check node structure
            for node_id, node_data in nodes.items():
                if 'categories' not in node_data:
                    errors.append(f"Node {node_id} missing 'categories'")
                    return False, errors
                
                categories = node_data['categories']
                if not isinstance(categories, list) or len(categories) == 0:
                    errors.append(f"Node {node_id} has invalid categories")
                    return False, errors
                
                # Check biolink category format
                for category in categories:
                    if not category.startswith('biolink:'):
                        errors.append(f"Node {node_id} has invalid biolink category: {category}")
                        return False, errors
            
            # Check predicate format
            if 'predicates' in edge_data:
                predicates = edge_data['predicates']
                if not isinstance(predicates, list) or len(predicates) == 0:
                    errors.append(f"Edge {edge_id} has invalid predicates")
                    return False, errors
                
                for predicate in predicates:
                    if not predicate.startswith('biolink:'):
                        errors.append(f"Edge {edge_id} has invalid biolink predicate: {predicate}")
                        return False, errors
            
            # If we reach here, the query is valid
            return True, []
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
    def fix_multi_hop_query(self, trapi_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Break down a multi-hop query into multiple single-hop queries
        
        Args:
            trapi_query: Multi-hop TRAPI query
            
        Returns:
            List of single-hop TRAPI queries
        """
        try:
            query_graph = trapi_query['message']['query_graph']
            nodes = query_graph['nodes']
            edges = query_graph['edges']
            
            if len(edges) <= 1:
                # Already single-hop or no edges
                return [trapi_query]
            
            logger.info(f"Breaking down multi-hop query with {len(edges)} edges into single-hop queries")
            
            single_hop_queries = []
            
            # Create one single-hop query for each edge
            for i, (edge_id, edge_data) in enumerate(edges.items()):
                subject_node = edge_data['subject']
                object_node = edge_data['object']
                
                # Create new query with just this edge and its nodes
                new_query = {
                    'message': {
                        'query_graph': {
                            'nodes': {
                                subject_node: deepcopy(nodes[subject_node]),
                                object_node: deepcopy(nodes[object_node])
                            },
                            'edges': {
                                f'e{i}': deepcopy(edge_data)
                            }
                        }
                    }
                }
                
                # Update edge to use consistent node IDs
                new_query['message']['query_graph']['edges'][f'e{i}']['subject'] = subject_node
                new_query['message']['query_graph']['edges'][f'e{i}']['object'] = object_node
                
                single_hop_queries.append(new_query)
                logger.debug(f"Created single-hop query {i+1}: {subject_node} -> {object_node}")
            
            return single_hop_queries
            
        except Exception as e:
            logger.error(f"Error fixing multi-hop query: {e}")
            return [trapi_query]  # Return original on error
    
    def create_single_hop_query(self, subject_category: str, object_category: str, 
                               predicate: str, subject_id: Optional[str] = None,
                               object_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a properly formatted single-hop TRAPI query
        
        Args:
            subject_category: Subject biolink category (e.g., "biolink:Disease")
            object_category: Object biolink category (e.g., "biolink:SmallMolecule") 
            predicate: Biolink predicate (e.g., "biolink:treated_by")
            subject_id: Optional subject entity ID
            object_id: Optional object entity ID
            
        Returns:
            Valid single-hop TRAPI query
        """
        query = {
            'message': {
                'query_graph': {
                    'nodes': {
                        'n0': {
                            'categories': [subject_category]
                        },
                        'n1': {
                            'categories': [object_category]
                        }
                    },
                    'edges': {
                        'e01': {
                            'subject': 'n0',
                            'object': 'n1',
                            'predicates': [predicate]
                        }
                    }
                }
            }
        }
        
        # Add IDs if provided
        if subject_id:
            query['message']['query_graph']['nodes']['n0']['ids'] = [subject_id]
        
        if object_id:
            query['message']['query_graph']['nodes']['n1']['ids'] = [object_id]
        
        # Validate the created query
        is_valid, errors = self.validate_trapi_query(query)
        if not is_valid:
            logger.warning(f"Created invalid query: {errors}")
        
        return query


def enhance_trapi_builder_with_single_hop_validation(trapi_builder):
    """
    Enhance existing TRAPI builder with single-hop validation
    
    Args:
        trapi_builder: Existing TRAPIQueryBuilder instance
        
    Returns:
        Enhanced TRAPI builder with validation
    """
    validator = TRAPISingleHopValidator()
    
    # Store original build method
    original_build_trapi_query_structure = trapi_builder.build_trapi_query_structure
    
    def enhanced_build_trapi_query_structure(query: str, subject_object: Dict[str, str], 
                                           predicate: str, entity_data: Dict[str, str], 
                                           failed_trapis: List[Dict]) -> Dict[str, Any]:
        """Enhanced TRAPI building with single-hop validation"""
        
        # Call original method
        result = original_build_trapi_query_structure(
            query, subject_object, predicate, entity_data, failed_trapis
        )
        
        if 'error' in result:
            return result
        
        # Validate the generated query
        is_valid, errors = validator.validate_trapi_query(result)
        
        if not is_valid:
            logger.warning(f"Generated invalid TRAPI query: {errors}")
            
            # Try to fix multi-hop queries
            if any('Multi-hop' in error for error in errors):
                logger.info("Attempting to fix multi-hop query by using first edge only")
                
                try:
                    # Extract just the first edge to make it single-hop
                    query_graph = result['message']['query_graph']
                    edges = query_graph['edges']
                    
                    if len(edges) > 1:
                        # Keep only the first edge
                        first_edge_id = list(edges.keys())[0]
                        first_edge = edges[first_edge_id]
                        
                        # Get the nodes used by this edge
                        subject_node = first_edge['subject']
                        object_node = first_edge['object']
                        
                        # Create new single-hop query
                        fixed_query = {
                            'message': {
                                'query_graph': {
                                    'nodes': {
                                        subject_node: query_graph['nodes'][subject_node],
                                        object_node: query_graph['nodes'][object_node]
                                    },
                                    'edges': {
                                        'e01': {
                                            'subject': subject_node,
                                            'object': object_node,
                                            'predicates': first_edge['predicates']
                                        }
                                    }
                                }
                            }
                        }
                        
                        # Validate the fixed query
                        is_fixed_valid, fixed_errors = validator.validate_trapi_query(fixed_query)
                        
                        if is_fixed_valid:
                            logger.info("Successfully fixed multi-hop query to single-hop")
                            return fixed_query
                        else:
                            logger.warning(f"Fixed query is still invalid: {fixed_errors}")
                
                except Exception as e:
                    logger.error(f"Error fixing multi-hop query: {e}")
            
            # If we can't fix it, try to create a simple valid query
            try:
                logger.info("Creating fallback single-hop query")
                fallback_query = validator.create_single_hop_query(
                    subject_object.get('subject', 'biolink:NamedThing'),
                    subject_object.get('object', 'biolink:NamedThing'),
                    predicate
                )
                return fallback_query
                
            except Exception as e:
                logger.error(f"Error creating fallback query: {e}")
                return {"error": f"Could not create valid single-hop TRAPI query: {'; '.join(errors)}"}
        
        logger.debug("Generated valid single-hop TRAPI query")
        return result
    
    # Replace the method with enhanced version
    trapi_builder.build_trapi_query_structure = enhanced_build_trapi_query_structure
    trapi_builder._single_hop_validator = validator
    
    return trapi_builder


def enhance_mcp_trapi_tool_with_validation():
    """
    Enhance the MCP TRAPI tool with single-hop validation
    
    This function modifies the MCP TRAPI tool to ensure all generated queries
    conform to the single-hop constraint.
    """
    try:
        # Import the MCP TRAPI tool handler
        from agentic_bte.servers.mcp.tools.trapi_tool import handle_trapi_query
        from agentic_bte.core.knowledge.trapi import TRAPIQueryBuilder
        
        # Store original handler
        original_handle_trapi_query = handle_trapi_query
        
        async def enhanced_handle_trapi_query(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Enhanced TRAPI query handler with single-hop validation"""
            
            try:
                # Get query parameters
                query = arguments.get("query")
                entity_data = arguments.get("entity_data", {})
                failed_trapis = arguments.get("failed_trapis", [])
                
                if not query:
                    return {
                        "error": "Query parameter is required",
                        "content": [{"type": "text", "text": "Error: Query parameter is required"}]
                    }
                
                # Create enhanced TRAPI builder
                trapi_builder = TRAPIQueryBuilder()
                trapi_builder = enhance_trapi_builder_with_single_hop_validation(trapi_builder)
                
                # Build TRAPI query with validation
                trapi_query = trapi_builder.build_trapi_query(query, entity_data, failed_trapis)
                
                if "error" in trapi_query:
                    return {
                        "error": trapi_query["error"],
                        "content": [{"type": "text", "text": f"TRAPI building error: {trapi_query['error']}"}]
                    }
                
                # Final validation check
                validator = TRAPISingleHopValidator()
                is_valid, validation_errors = validator.validate_trapi_query(trapi_query)
                
                if not is_valid:
                    error_msg = f"Generated invalid TRAPI query: {'; '.join(validation_errors)}"
                    return {
                        "error": error_msg,
                        "content": [{"type": "text", "text": error_msg}]
                    }
                
                # Success - return valid single-hop query
                response_text = f"Generated valid single-hop TRAPI query for: {query}\n\n"
                response_text += f"Query Structure:\n```json\n{json.dumps(trapi_query, indent=2)}\n```\n\n"
                response_text += "Validation: Valid single-hop TRAPI query"
                
                return {
                    "content": [{"type": "text", "text": response_text}],
                    "result_data": {"query": trapi_query}
                }
                
            except Exception as e:
                error_msg = f"Error building validated TRAPI query: {str(e)}"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "content": [{"type": "text", "text": error_msg}]
                }
        
        # Replace the original handler (this would need to be done at import time)
        logger.info("Enhanced MCP TRAPI tool with single-hop validation")
        return enhanced_handle_trapi_query
        
    except ImportError as e:
        logger.warning(f"Could not enhance MCP TRAPI tool: {e}")
        return None


# Test functions
def test_validator():
    """Test the TRAPI validator with example queries"""
    validator = TRAPISingleHopValidator()
    
    print("=== TRAPI Single-Hop Validator Test ===")
    
    # Test 1: Valid single-hop query
    valid_query = {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {"categories": ["biolink:Disease"], "ids": ["MONDO:0005015"]},
                    "n1": {"categories": ["biolink:SmallMolecule"]}
                },
                "edges": {
                    "e01": {
                        "subject": "n0",
                        "object": "n1", 
                        "predicates": ["biolink:treated_by"]
                    }
                }
            }
        }
    }
    
    is_valid, errors = validator.validate_trapi_query(valid_query)
    print(f"Valid query test: {is_valid}, errors: {errors}")
    
    # Test 2: Invalid multi-hop query
    multi_hop_query = {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {"categories": ["biolink:Disease"]},
                    "n1": {"categories": ["biolink:Gene"]},
                    "n2": {"categories": ["biolink:SmallMolecule"]}
                },
                "edges": {
                    "e01": {"subject": "n0", "object": "n1", "predicates": ["biolink:associated_with"]},
                    "e12": {"subject": "n1", "object": "n2", "predicates": ["biolink:targeted_by"]}
                }
            }
        }
    }
    
    is_valid, errors = validator.validate_trapi_query(multi_hop_query)
    print(f"Multi-hop query test: {is_valid}, errors: {errors}")
    
    # Test fixing multi-hop query
    fixed_queries = validator.fix_multi_hop_query(multi_hop_query)
    print(f"Fixed multi-hop into {len(fixed_queries)} single-hop queries")
    
    for i, fixed_query in enumerate(fixed_queries):
        is_fixed_valid, fixed_errors = validator.validate_trapi_query(fixed_query)
        print(f"  Fixed query {i+1} valid: {is_fixed_valid}, errors: {fixed_errors}")


if __name__ == "__main__":
    test_validator()