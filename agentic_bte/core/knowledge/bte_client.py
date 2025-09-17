"""
BTE Client - BioThings Explorer API Client

This module provides client functionality for the BioThings Explorer API,
including TRAPI query execution and meta knowledge graph retrieval.

Migrated and enhanced from original BTE-LLM implementations.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from copy import deepcopy
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ...config.settings import get_settings
from ...exceptions.base import ExternalServiceError

logger = logging.getLogger(__name__)


class BTEClient:
    """
    Client for BioThings Explorer (BTE) API
    
    This client handles communication with BTE services including:
    - TRAPI query execution
    - Meta knowledge graph retrieval
    - Result parsing and processing
    
    Migrated from original BTE-LLM implementations with enhancements.
    """
    
    def __init__(self, base_url: Optional[str] = None, timeout: int = 300):
        """
        Initialize BTE client
        
        Args:
            base_url: Base URL for BTE API (defaults to production URL)
            timeout: Request timeout in seconds
        """
        self.settings = get_settings()
        self.base_url = base_url or self.settings.bte_api_base_url
        self.timeout = timeout
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Cache for meta knowledge graph
        self._meta_kg_cache = None
        self._meta_kg_cache_time = 0
        self._cache_ttl = 3600  # 1 hour cache
    
    def _make_request(self, endpoint: str, method: str = "GET", 
                     data: Optional[Dict] = None, params: Optional[Dict] = None) -> requests.Response:
        """
        Make HTTP request to BTE API with error handling
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            data: Request data for POST requests
            params: Query parameters
            
        Returns:
            Response object
            
        Raises:
            ExternalServiceError: On API request failure
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params, timeout=self.timeout)
            elif method.upper() == "POST":
                headers = {'Content-Type': 'application/json'}
                response = self.session.post(
                    url, 
                    json=data, 
                    params=params, 
                    headers=headers, 
                    timeout=self.timeout
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.Timeout as e:
            logger.error(f"BTE API request timeout for {url}")
            raise ExternalServiceError(
                f"BTE API request timeout: {str(e)}", 
                service_name="bte_api"
            ) from e
        except requests.exceptions.RequestException as e:
            logger.error(f"BTE API request failed for {url}: {e}")
            raise ExternalServiceError(
                f"BTE API request failed: {str(e)}", 
                service_name="bte_api"
            ) from e
    
    def get_meta_knowledge_graph(self) -> Dict[str, Any]:
        """
        Get meta knowledge graph from BTE
        
        Returns:
            Meta knowledge graph data
        """
        # Check cache first
        current_time = time.time()
        if (self._meta_kg_cache and 
            current_time - self._meta_kg_cache_time < self._cache_ttl):
            logger.debug("Using cached meta knowledge graph")
            return self._meta_kg_cache
        
        try:
            logger.info("Fetching meta knowledge graph from BTE")
            response = self._make_request("metakg")
            meta_kg = response.json()
            
            # Update cache
            self._meta_kg_cache = meta_kg
            self._meta_kg_cache_time = current_time
            
            logger.info(f"Retrieved meta knowledge graph with {len(meta_kg.get('edges', []))} edges")
            return meta_kg
            
        except Exception as e:
            logger.error(f"Failed to get meta knowledge graph: {e}")
            raise ExternalServiceError(
                f"Failed to retrieve meta knowledge graph: {str(e)}",
                service_name="bte_api"
            ) from e
    
    def execute_trapi_query(self, trapi_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute TRAPI query against BTE API
        
        Args:
            trapi_query: TRAPI query dictionary
            
        Returns:
            BTE API response
        """
        try:
            logger.info("Executing TRAPI query against BTE")
            logger.debug(f"TRAPI query: {json.dumps(trapi_query, indent=2)}")
            
            # Use v1/query endpoint for TRAPI queries
            response = self._make_request("v1/query", method="POST", data=trapi_query)
            result = response.json()
            
            # Check for BTE-specific errors in response
            if "message" in result and "status" in result["message"]:
                status = result["message"]["status"]
                if status and any(log.get("level") == "ERROR" for log in status.get("logs", [])):
                    error_logs = [log for log in status["logs"] if log.get("level") == "ERROR"]
                    error_msg = "; ".join([log.get("message", "Unknown error") for log in error_logs])
                    raise ExternalServiceError(
                        f"BTE API returned error: {error_msg}",
                        service_name="bte_api"
                    )
            
            logger.info("TRAPI query executed successfully")
            return result
            
        except ExternalServiceError:
            raise  # Re-raise BTE-specific errors
        except Exception as e:
            logger.error(f"Error executing TRAPI query: {e}")
            raise ExternalServiceError(
                f"TRAPI query execution failed: {str(e)}",
                service_name="bte_api"
            ) from e
    
    def parse_bte_results(self, bte_response: Dict[str, Any], k: int = 5, max_results: int = 50) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Parse BTE API response into structured format
        
        Args:
            bte_response: Raw BTE API response
            k: Maximum results per entity
            max_results: Maximum total results
            
        Returns:
            Tuple of (parsed_results, entity_name_to_id_mapping)
        """
        try:
            # Extract all nodes, edges, and results
            nodes = bte_response.get("message", {}).get("knowledge_graph", {}).get("nodes", {})
            edges = bte_response.get("message", {}).get("knowledge_graph", {}).get("edges", {})
            results = bte_response.get("message", {}).get("results", [])
            
            # Parse node data into a structured format
            node_data = {}
            for node_id, node_info in nodes.items():
                # BTE API returns names as lists - extract first (primary) name
                name_list = node_info.get("name", ["Unknown"])
                primary_name = name_list[0] if isinstance(name_list, list) and name_list else str(name_list) if name_list else "Unknown"
                
                node_data[node_id] = {
                    "ID": node_id,
                    "name": primary_name,  # Now always a string
                    "category": node_info.get("categories", ["Unknown"])[0]
                }
            
            # Extract edge data
            edge_data = {}
            for edge_id, edge_info in edges.items():
                edge_data[edge_id] = {
                    "subject": edge_info.get("subject"),
                    "predicate": edge_info.get("predicate"),
                    "object": edge_info.get("object")
                }
            
            # Parse through results to match IDs and create structured output
            parsed_results = []
            entity_name_mapping = {}
            
            # Track how many results have been collected per input ID
            results_per_id = {}
            
            for result in results:
                if len(parsed_results) >= max_results:
                    break
                    
                bindings = result.get("node_bindings", {})
                subject_id = bindings.get("n0", [{}])[0].get("id", "Unknown")
                object_id = bindings.get("n1", [{}])[0].get("id", "Unknown")
                
                subject_name = node_data.get(subject_id, {}).get("name", "Unknown")
                object_name = node_data.get(object_id, {}).get("name", "Unknown")
                
                # Check if we've already collected enough results for this entity
                if k > 0:
                    subject_count = results_per_id.get(subject_id, {}).get("result_count", 0)
                    object_count = results_per_id.get(object_id, {}).get("result_count", 0)
                    
                    if subject_count >= k and object_count >= k:
                        continue
                
                # Find the relationship predicate
                relationship = None
                for edge_id, edge_info in edge_data.items():
                    if (edge_data[edge_id].get("subject") == subject_id and 
                        edge_data[edge_id].get("object") == object_id):
                        relationship = edge_data[edge_id].get("predicate")
                        break
                
                if relationship:
                    parsed_results.append({
                        "subject": subject_name,
                        "subject_id": subject_id,
                        "subject_type": node_data.get(subject_id, {}).get("category", "Unknown"),
                        "predicate": relationship,
                        "object": object_name,
                        "object_id": object_id,
                        "object_type": node_data.get(object_id, {}).get("category", "Unknown")
                    })
                    
                    # Update result counts
                    if subject_id not in results_per_id:
                        results_per_id[subject_id] = {"result_count": 0, "name": subject_name}
                    if object_id not in results_per_id:
                        results_per_id[object_id] = {"result_count": 0, "name": object_name}
                    
                    results_per_id[subject_id]["result_count"] += 1
                    results_per_id[object_id]["result_count"] += 1
                    
                    # Update entity name mapping
                    if subject_name not in entity_name_mapping:
                        entity_name_mapping[subject_name] = subject_id
                    if object_name not in entity_name_mapping:
                        entity_name_mapping[object_name] = object_id
            
            logger.info(f"Parsed {len(parsed_results)} relationships from BTE API")
            return parsed_results, entity_name_mapping
            
        except Exception as e:
            logger.error(f"Error parsing BTE results: {e}")
            return [], {}
    
    def split_trapi_query(self, trapi_query: Dict[str, Any], batch_limit: int = 50) -> List[Dict[str, Any]]:
        """
        Split TRAPI query with large entity lists into smaller batches
        
        Args:
            trapi_query: Original TRAPI query
            batch_limit: Maximum entities per batch
            
        Returns:
            List of TRAPI query batches
        """
        try:
            query_graph = trapi_query.get("message", {}).get("query_graph", {})
            nodes = query_graph.get("nodes", {})
            
            # Identify nodes that need to be split
            split_nodes = {k: v for k, v in nodes.items() if "ids" in v and len(v["ids"]) > batch_limit}
            
            if not split_nodes:
                return [trapi_query]  # No need to split
            
            # For simplicity, assume only ONE node needs splitting (common case)
            node_id, node_data = next(iter(split_nodes.items()))
            id_chunks = [
                node_data["ids"][i:i + batch_limit]
                for i in range(0, len(node_data["ids"]), batch_limit)
            ]
            
            # Create a new query for each chunk
            queries = []
            for chunk in id_chunks:
                new_query = deepcopy(trapi_query)
                new_query["message"]["query_graph"]["nodes"][node_id]["ids"] = chunk
                queries.append(new_query)
            
            logger.info(f"Split TRAPI query into {len(queries)} batches")
            return queries
            
        except Exception as e:
            logger.error(f"Error splitting TRAPI query: {e}")
            return [trapi_query]
    
    def execute_trapi_with_batching(self, trapi_query: Dict[str, Any], 
                                  max_results: int = 50, k: int = 5, 
                                  batch_limit: int = 50) -> Tuple[List[Dict], Dict[str, str], Dict[str, Any]]:
        """
        Execute TRAPI query with automatic batching for large entity sets
        
        Args:
            trapi_query: TRAPI query to execute
            max_results: Maximum results to return
            k: Maximum results per entity
            batch_limit: Maximum entities per batch
            
        Returns:
            Tuple of (results, entity_mappings, metadata)
        """
        try:
            # Split query into batches if needed
            query_batches = self.split_trapi_query(trapi_query, batch_limit)
            
            all_results = []
            all_mappings = {}
            messages = []
            
            for i, batch_query in enumerate(query_batches):
                logger.info(f"Executing TRAPI batch {i+1}/{len(query_batches)}")
                
                try:
                    # Execute single batch
                    response = self.execute_trapi_query(batch_query)
                    
                    # Parse results
                    batch_results, batch_mappings = self.parse_bte_results(response, k, max_results)
                    
                    # Accumulate results
                    all_results.extend(batch_results)
                    all_mappings.update(batch_mappings)
                    
                    # Track batch success
                    batch_message = response.get("description", f"Batch {i+1} completed")
                    messages.append(batch_message)
                    
                    # Stop if we have enough results
                    if len(all_results) >= max_results:
                        all_results = all_results[:max_results]
                        break
                        
                except Exception as e:
                    logger.warning(f"Batch {i+1} failed: {e}")
                    messages.append(f"Batch {i+1} failed: {str(e)}")
                    continue
            
            metadata = {
                "message": "; ".join(messages),
                "total_batches": len(query_batches),
                "successful_batches": len([m for m in messages if "failed" not in m]),
                "total_results": len(all_results)
            }
            
            return all_results, all_mappings, metadata
            
        except Exception as e:
            logger.error(f"Error in batched TRAPI execution: {e}")
            return [], {}, {"error": str(e)}
    
    def get_bte_response_summary(self, bte_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of BTE API response
        
        Args:
            bte_response: Raw BTE API response
            
        Returns:
            Summary of response with key metrics
        """
        try:
            parsed_results = {
                "results": [],
                "entities": {},
                "total_results": 0,
                "query_metadata": {},
                "entity_mappings": {}  # ID to name mappings
            }
            
            # Extract message from response
            message = bte_response.get("message", {})
            if not message:
                logger.warning("No message found in BTE response")
                return parsed_results
            
            # Extract query metadata
            query_graph = message.get("query_graph", {})
            parsed_results["query_metadata"] = {
                "nodes": len(query_graph.get("nodes", {})),
                "edges": len(query_graph.get("edges", {}))
            }
            
            # Extract knowledge graph entities
            knowledge_graph = message.get("knowledge_graph", {})
            kg_nodes = knowledge_graph.get("nodes", {})
            kg_edges = knowledge_graph.get("edges", {})
            
            # Build entity mappings (ID -> name/label)
            entity_mappings = {}
            for node_id, node_data in kg_nodes.items():
                # Try to get name from various fields
                name = (node_data.get("name") or 
                       node_data.get("label") or 
                       node_data.get("attributes", {}).get("name") or
                       node_id)
                entity_mappings[node_id] = name
            
            parsed_results["entity_mappings"] = entity_mappings
            parsed_results["entities"] = kg_nodes
            
            # Extract and parse results
            results = message.get("results", [])
            parsed_results["total_results"] = len(results)
            
            for result in results:
                # Extract node bindings
                node_bindings = result.get("node_bindings", {})
                
                parsed_result = {
                    "score": result.get("score", 0),
                    "entities": {},
                    "bindings": node_bindings,
                    "edge_bindings": result.get("edge_bindings", {})
                }
                
                # Process node bindings to get entity information
                for query_node, bindings in node_bindings.items():
                    if isinstance(bindings, list) and bindings:
                        # Take the first binding
                        binding = bindings[0]
                        entity_id = binding.get("id")
                        if entity_id and entity_id in kg_nodes:
                            entity_info = kg_nodes[entity_id].copy()
                            entity_info["query_node"] = query_node
                            entity_info["binding_id"] = entity_id
                            parsed_result["entities"][query_node] = entity_info
                
                parsed_results["results"].append(parsed_result)
            
            logger.info(f"Parsed {len(parsed_results['results'])} results from BTE response")
            return parsed_results
            
        except Exception as e:
            logger.error(f"Error parsing BTE results: {e}")
            # Return partial results if parsing fails
            return {
                "results": [],
                "entities": {},
                "total_results": 0,
                "query_metadata": {},
                "entity_mappings": {},
                "parse_error": str(e)
            }
    
    def query_and_parse(self, trapi_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute TRAPI query and parse results in one step
        
        Args:
            trapi_query: TRAPI query dictionary
            
        Returns:
            Parsed BTE results
        """
        raw_response = self.execute_trapi_query(trapi_query)
        return self.parse_bte_results(raw_response)
    
    def get_entity_info(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific entity
        
        Args:
            entity_id: Entity identifier (e.g., MONDO:0005148)
            
        Returns:
            Entity information or None if not found
        """
        try:
            # Use the node info endpoint if available
            response = self._make_request(f"v1/node/{entity_id}")
            return response.json()
        except Exception as e:
            logger.warning(f"Could not retrieve entity info for {entity_id}: {e}")
            return None
    
    def health_check(self) -> bool:
        """
        Check if BTE API is healthy and accessible
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            response = self._make_request("", timeout=10)  # Root endpoint
            return response.status_code == 200
        except Exception as e:
            logger.error(f"BTE API health check failed: {e}")
            return False


# Convenience functions
def execute_trapi_query(trapi_query: Dict[str, Any], 
                       base_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to execute a TRAPI query
    
    Args:
        trapi_query: TRAPI query dictionary
        base_url: Optional BTE API base URL
        
    Returns:
        Parsed BTE results
    """
    client = BTEClient(base_url)
    return client.query_and_parse(trapi_query)


def get_meta_knowledge_graph(base_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to get meta knowledge graph
    
    Args:
        base_url: Optional BTE API base URL
        
    Returns:
        Meta knowledge graph data
    """
    client = BTEClient(base_url)
    return client.get_meta_knowledge_graph()