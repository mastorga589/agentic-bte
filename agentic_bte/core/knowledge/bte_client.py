"""
BTE Client - BioThings Explorer API Client

This module provides client functionality for the BioThings Explorer API,
including TRAPI query execution and meta knowledge graph retrieval.

Migrated and enhanced from original BTE-LLM implementations.
"""

import json
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from copy import deepcopy
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ...config.settings import get_settings
from ...exceptions.base import ExternalServiceError

# Setup comprehensive logging for BTE client
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create formatter for detailed logging
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)

# Add handler if not already present
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class BTEClient:
    """
    Client for BioThings Explorer (BTE) API
    
    This client handles communication with BTE services including:
    - TRAPI query execution
    - Meta knowledge graph retrieval
    - Result parsing and processing
    
    Migrated from original BTE-LLM implementations with enhancements.
    """
    
    def __init__(self, base_url: Optional[str] = None, timeout: int = 300, poll_seconds: int = 120, poll_interval: float = 1.5):
        """
        Initialize BTE client
        
        Args:
            base_url: Base URL for BTE API (defaults to production URL)
            timeout: Request timeout in seconds
        """
        start_time = datetime.now()
        logger.info(f"=== Initializing BTEClient ===")
        logger.debug(f"Parameters - base_url: {base_url}, timeout: {timeout}")
        
        try:
            self.settings = get_settings()
            logger.debug("Successfully loaded settings")
            
            self.base_url = base_url or self.settings.bte_api_base_url
            self.timeout = timeout
            # Async preferences from settings (with sensible fallbacks)
            try:
                self.prefer_async = bool(getattr(self.settings, 'bte_prefer_async', False))
            except Exception:
                self.prefer_async = False
            try:
                self.poll_seconds = int(getattr(self.settings, 'bte_async_poll_seconds', poll_seconds))
            except Exception:
                self.poll_seconds = poll_seconds
            try:
                self.poll_interval = float(getattr(self.settings, 'bte_async_poll_interval', poll_interval))
            except Exception:
                self.poll_interval = poll_interval
            # Enforce reasonable lower bounds to avoid premature timeouts
            if self.poll_seconds < 180:
                logger.info(f"Increasing async poll_seconds from {self.poll_seconds} to 180s (minimum)")
                self.poll_seconds = 180
            if self.poll_interval < 1.0:
                logger.info(f"Increasing async poll_interval from {self.poll_interval} to 1.0s (minimum)")
                self.poll_interval = 1.0
            # Default TRAPI batch size from settings
            try:
                self.batch_limit = getattr(self.settings, 'trapi_batch_limit', 50)
            except Exception:
                self.batch_limit = 50
            
            logger.info(f"BTE client configured with base_url: {self.base_url}, timeout: {timeout}s, prefer_async={self.prefer_async}, poll={self.poll_seconds}s@{self.poll_interval}s")
        
            # Setup session with retry strategy
            logger.debug("Setting up HTTP session with retry strategy...")
            self.session = requests.Session()
            
            # Handle urllib3 version compatibility for Retry parameters
            retry_kwargs = {
                "total": 3,
                "status_forcelist": [429, 500, 502, 503, 504],
                "backoff_factor": 1
            }
            logger.debug(f"Retry configuration: {retry_kwargs}")
            
            # Use allowed_methods for urllib3 >= 1.26, method_whitelist for older versions
            try:
                # Try with the new parameter name first
                retry_strategy = Retry(
                    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
                    **retry_kwargs
                )
                logger.debug("Using 'allowed_methods' for urllib3 retry strategy")
            except TypeError:
                # Fallback to old parameter name for backward compatibility
                retry_strategy = Retry(
                    method_whitelist=["HEAD", "GET", "OPTIONS", "POST"],
                    **retry_kwargs
                )
                logger.debug("Fallback to 'method_whitelist' for urllib3 retry strategy")
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
            logger.info("HTTP session configured with retry strategy")
            
            # Cache for meta knowledge graph
            self._meta_kg_cache = None
            self._meta_kg_cache_time = 0
            self._cache_ttl = 3600  # 1 hour cache
            logger.debug(f"Meta knowledge graph cache initialized (TTL: {self._cache_ttl}s)")
            
            # Store meta-KG reference for propagation to evidence scoring
            self.meta_kg = None
            
            # Simple in-memory cache for ID->resolved name lookups
            self._name_cache: Dict[str, str] = {}
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"BTEClient initialization completed in {duration:.3f}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize BTEClient: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def _make_request(self, endpoint: str, method: str = "GET", 
                     data: Optional[Dict] = None, params: Optional[Dict] = None,
                     timeout: Optional[int] = None) -> requests.Response:
        """
        Make HTTP request to BTE API with error handling
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            data: Request data for POST requests
            params: Query parameters
            timeout: Optional per-request timeout override
            
        Returns:
            Response object
            
        Raises:
            ExternalServiceError: On API request failure
        """
        start_time = datetime.now()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        logger.info(f"=== Making {method.upper()} request to BTE API ===")
        logger.debug(f"URL: {url}")
        logger.debug(f"Parameters: {params}")
        eff_timeout = timeout or self.timeout
        logger.debug(f"Timeout: {eff_timeout}s")
        
        if data:
            logger.debug(f"Request data size: {len(json.dumps(data))} bytes")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Request data preview: {json.dumps(data, indent=2)[:500]}...")
        
        try:
            if method.upper() == "GET":
                logger.debug("Executing GET request...")
                response = self.session.get(url, params=params, timeout=eff_timeout)
            elif method.upper() == "POST":
                headers = {'Content-Type': 'application/json'}
                logger.debug(f"Executing POST request with headers: {headers}")
                response = self.session.post(
                    url, 
                    json=data, 
                    params=params, 
                    headers=headers, 
                    timeout=eff_timeout
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Log response details
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"BTE API request completed in {duration:.3f}s")
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            
            if response.content:
                content_size = len(response.content)
                logger.debug(f"Response content size: {content_size} bytes")
                
                # Log preview of response content for debugging
                if logger.isEnabledFor(logging.DEBUG) and content_size > 0:
                    try:
                        content_preview = response.content.decode('utf-8')[:300]
                        logger.debug(f"Response content preview: {content_preview}...")
                    except UnicodeDecodeError:
                        logger.debug("Response content is binary data")
            
            response.raise_for_status()
            logger.info(f"BTE API request successful: {response.status_code}")
            return response
            
        except requests.exceptions.Timeout as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"BTE API request timeout after {duration:.3f}s for {url}")
            logger.error(f"Timeout details: {str(e)}")
            raise ExternalServiceError(
                f"BTE API request timeout: {str(e)}", 
                service_name="bte_api"
            ) from e
        except requests.exceptions.RequestException as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"BTE API request failed after {duration:.3f}s for {url}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
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
        start_time = datetime.now()
        logger.info(f"=== Getting meta knowledge graph from BTE ===")
        
        # Check cache first
        current_time = time.time()
        cache_age = current_time - self._meta_kg_cache_time
        
        if (self._meta_kg_cache and cache_age < self._cache_ttl):
            logger.info(f"Using cached meta knowledge graph (age: {cache_age:.1f}s)")
            return self._meta_kg_cache
        
        logger.debug(f"Cache status - exists: {bool(self._meta_kg_cache)}, age: {cache_age:.1f}s, TTL: {self._cache_ttl}s")
        
        try:
            logger.info("Fetching fresh meta knowledge graph from BTE API...")
            response = self._make_request("meta_knowledge_graph")
            
            logger.debug("Parsing meta knowledge graph response...")
            meta_kg = response.json()
            
            # Log detailed structure information
            edges_count = len(meta_kg.get('edges', []))
            nodes_count = len(meta_kg.get('nodes', []))
            logger.info(f"Meta KG structure: {nodes_count} nodes, {edges_count} edges")
            
            if logger.isEnabledFor(logging.DEBUG):
                # Log sample of edge predicates
                edges = meta_kg.get('edges', [])
                if edges:
                    predicates = set(edge.get('predicate', 'unknown') for edge in edges[:10])
                    logger.debug(f"Sample predicates: {list(predicates)[:5]}")
            
            # Update cache and store reference
            self._meta_kg_cache = meta_kg
            self._meta_kg_cache_time = current_time
            self.meta_kg = meta_kg  # Store for propagation to evidence scoring
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Meta knowledge graph retrieved successfully in {duration:.3f}s")
            return meta_kg
            
        except Exception as e:
            logger.error(f"Failed to get meta knowledge graph: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise ExternalServiceError(
                f"Failed to retrieve meta knowledge graph: {str(e)}",
                service_name="bte_api"
            ) from e
    
    def _extract_job_id(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract a job identifier from various async response shapes."""
        for k in ("id", "job_id", "request_id", "task_id"):
            v = data.get(k)
            if isinstance(v, str) and v:
                return v
        # sometimes nested under 'data'
        nested = data.get("data")
        if isinstance(nested, dict):
            return self._extract_job_id(nested)
        return None

    def _fetch_async_result(self, job_id: str) -> Dict[str, Any]:
        """Poll BTE async endpoints until the TRAPI response is ready or timeout.
        Uses BTE's TRAPI 1.5 endpoints: /asyncquery_status/{job_id} and /asyncquery_response/{job_id}.
        """
        # Extend polling window to improve completion rate on slow jobs
        deadline = time.time() + (self.poll_seconds * 3)
        status_url = f"{self.base_url}/asyncquery_status/{job_id}"
        response_url = f"{self.base_url}/asyncquery_response/{job_id}"
        last_status: Optional[str] = None
        while time.time() < deadline:
            # Prefer polling the response endpoint which returns status until finished
            try:
                rr = self.session.get(response_url, timeout=self.timeout)
                if rr.status_code in (200, 201):
                    data = rr.json()
                    if isinstance(data, dict):
                        # If finished, this will be a full TRAPI Response with a 'message'
                        if "message" in data:
                            return data
                        # If still running, it mirrors status payload
                        status = str(data.get("status") or data.get("state") or "").lower()
                        if status:
                            last_status = status
                            if status in ("failed", "error"):
                                raise ExternalServiceError(f"BTE async job failed: {data}", service_name="bte_api")
                # fallthrough to check status endpoint
            except Exception:
                pass

            # Query status endpoint for authoritative status and response_url
            try:
                rs = self.session.get(status_url, timeout=self.timeout)
                if rs.status_code in (200, 201):
                    sdata = rs.json() or {}
                    status = str(sdata.get("status") or sdata.get("state") or "").lower()
                    if status:
                        last_status = status
                        # Capture canonical response_url if provided
                        try:
                            ru = sdata.get("response_url")
                            if isinstance(ru, str) and ru.strip():
                                response_url = ru
                        except Exception:
                            pass
                        if status in ("succeeded", "success", "completed", "complete", "done"):
                            # Loop will fetch response_url again on next iteration immediately
                            pass
                        elif status in ("failed", "error"):
                            raise ExternalServiceError(f"BTE async job failed: {sdata}", service_name="bte_api")
                elif rs.status_code == 404:
                    # brief grace period if status not yet materialized
                    pass
            except Exception:
                pass

            time.sleep(self.poll_interval)
        raise ExternalServiceError(f"Timed out waiting for async result (last_status={last_status})", service_name="bte_api")

    def _fetch_async_result_from_url(self, job_url: str) -> Dict[str, Any]:
        """Poll an async job URL until TRAPI message is returned or timeout.
        Supports BTE job URLs that point to /asyncquery_status/{job_id} and derives the
        corresponding /asyncquery_response/{job_id} endpoint for result retrieval.
        """
        if not job_url:
            raise ExternalServiceError("Missing async job URL", service_name="bte_api")
        # Normalize absolute URL
        if job_url.startswith("/"):
            job_url = f"{self.base_url}{job_url}"
        base = job_url.rstrip("/")
        # Extract job_id and derive canonical endpoints
        job_id = base.rsplit("/", 1)[-1]
        # Prefer explicit patterns
        if "/asyncquery_status/" in base:
            status_url = base
            response_url = base.replace("/asyncquery_status/", "/asyncquery_response/")
        elif "/asyncquery_response/" in base:
            response_url = base
            status_url = base.replace("/asyncquery_response/", "/asyncquery_status/")
        else:
            # Fallback to BTE TRAPI 1.5 endpoints
            status_url = f"{self.base_url}/asyncquery_status/{job_id}"
            response_url = f"{self.base_url}/asyncquery_response/{job_id}"

        # Extend polling window to improve completion rate on slow jobs
        deadline = time.time() + (self.poll_seconds * 3)
        last_status: Optional[str] = None

        while time.time() < deadline:
            # Try the response endpoint first; it returns status until finished, then TRAPI Response
            try:
                rr = self.session.get(response_url, timeout=self.timeout)
                if rr.status_code in (200, 201):
                    data = rr.json()
                    if isinstance(data, dict):
                        if "message" in data:
                            return data
                        status = str(data.get("status") or data.get("state") or "").lower()
                        if status:
                            last_status = status
                            if status in ("failed", "error"):
                                raise ExternalServiceError(f"BTE async job failed: {data}", service_name="bte_api")
            except Exception:
                pass

            # Consult status endpoint for authoritative state and response_url
            try:
                rs = self.session.get(status_url, timeout=self.timeout)
                if rs.status_code in (200, 201):
                    sdata = rs.json() or {}
                    status = str(sdata.get("status") or sdata.get("state") or "").lower()
                    if status:
                        last_status = status
                        try:
                            ru = sdata.get("response_url")
                            if isinstance(ru, str) and ru.strip():
                                response_url = ru
                        except Exception:
                            pass
                        if status in ("failed", "error"):
                            raise ExternalServiceError(f"BTE async job failed: {sdata}", service_name="bte_api")
            except Exception:
                pass

            time.sleep(self.poll_interval)
        raise ExternalServiceError(f"Timed out waiting for async result (last_status={last_status})", service_name="bte_api")

    def execute_trapi_query(self, trapi_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute TRAPI query against BTE API (prefers asyncquery per spec; falls back to query)
        
        Args:
            trapi_query: TRAPI query dictionary
            
        Returns:
            BTE API response
        """
        start_time = datetime.now()
        logger.info("=== Executing TRAPI query against BTE API ===")
        
        try:
            # Log query structure and content
            query_message = trapi_query.get('message', {})
            query_graph = query_message.get('query_graph', {})
            
            nodes_info = query_graph.get('nodes', {})
            edges_info = query_graph.get('edges', {})
            
            logger.info(f"TRAPI query structure: {len(nodes_info)} nodes, {len(edges_info)} edges")
            
            # Log node details
            for node_id, node_data in nodes_info.items():
                categories = node_data.get('categories', [])
                ids = node_data.get('ids', [])
                logger.debug(f"Node {node_id}: categories={categories}, entity_count={len(ids)}")
                if ids and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"  Sample IDs: {ids[:3]}...")
            
            # Log edge details
            for edge_id, edge_data in edges_info.items():
                subject = edge_data.get('subject')
                predicate = edge_data.get('predicates', [])
                object_node = edge_data.get('object')
                logger.debug(f"Edge {edge_id}: {subject} --{predicate}--> {object_node}")
            
            # Log full query if debug level
            if logger.isEnabledFor(logging.DEBUG):
                query_str = json.dumps(trapi_query, indent=2)
                logger.debug(f"Full TRAPI query:\n{query_str}")
            
            if self.prefer_async:
                # Prefer asyncquery endpoint per spec
                logger.debug("Attempting asyncquery initiation...")
                try:
                    async_resp = self._make_request("asyncquery", method="POST", data=trapi_query)
                    location_hdr = None
                    try:
                        location_hdr = async_resp.headers.get("Location")
                    except Exception:
                        location_hdr = None
                    if async_resp.status_code in (200, 201, 202):
                        # Some deployments return immediate TRAPI message in body
                        try:
                            data = async_resp.json()
                        except Exception:
                            data = {}
                        job_id = None
                        if isinstance(data, dict) and "message" in data:
                            result = data
                        else:
                            # Prefer Location header URL when provided
                            if location_hdr:
                                logger.info(f"Async job started (Location header): {location_hdr}")
                                result = self._fetch_async_result_from_url(location_hdr)
                            else:
                                # Support BTE's body fields: job_url and job_id
                                job_url = None
                                try:
                                    job_url = data.get("job_url") if isinstance(data, dict) else None
                                except Exception:
                                    job_url = None
                                if isinstance(job_url, str) and job_url.strip():
                                    logger.info(f"Async job started (job_url): {job_url}")
                                    result = self._fetch_async_result_from_url(job_url)
                                else:
                                    job_id = self._extract_job_id(data) if isinstance(data, dict) else None
                                    if job_id:
                                        logger.info(f"Async job started with id={job_id}")
                                        result = self._fetch_async_result(job_id)
                                    else:
                                        # If structure unexpected, try to treat as result (may already be the final message)
                                        result = data if isinstance(data, dict) else {"message": data}
                        logger.debug("Parsing BTE API async response...")
                    else:
                        raise ExternalServiceError(f"asyncquery returned {async_resp.status_code}", service_name="bte_api")
                except Exception as e:
                    # Async failed or timed out — fall back to synchronous /query for robustness
                    logger.error(f"/asyncquery failed: {e}")
                    logger.info("Falling back to synchronous /query endpoint")
                    response = self._make_request("query", method="POST", data=trapi_query)
                    try:
                        if response.status_code == 202:
                            data = response.json()
                            job_id = self._extract_job_id(data) if isinstance(data, dict) else None
                            if job_id:
                                logger.info(f"Async job started via /query with id={job_id}")
                                result = self._fetch_async_result(job_id)
                            else:
                                response.raise_for_status()
                                result = data
                        else:
                            result = response.json()
                    except Exception:
                        response.raise_for_status()
                        result = response.json()
            else:
                # Use synchronous /query directly (faster for deployments with problematic async behavior)
                response = self._make_request("query", method="POST", data=trapi_query)
                try:
                    if response.status_code == 202:
                        data = response.json()
                        job_id = self._extract_job_id(data) if isinstance(data, dict) else None
                        if job_id:
                            logger.info(f"Async job started via /query with id={job_id}")
                            result = self._fetch_async_result(job_id)
                        else:
                            response.raise_for_status()
                            result = data
                    else:
                        result = response.json()
                except Exception:
                    response.raise_for_status()
                    result = response.json()
            
            # Check for BTE-specific errors in response
            if isinstance(result, dict) and "message" in result and "status" in result["message"]:
                status = result["message"]["status"]
                logger.debug(f"Response status information: {status}")
                
                if status and any(log.get("level") == "ERROR" for log in status.get("logs", [])):
                    error_logs = [log for log in status["logs"] if log.get("level") == "ERROR"]
                    error_msg = "; ".join([log.get("message", "Unknown error") for log in error_logs])
                    logger.error(f"BTE API returned errors: {error_msg}")
                    raise ExternalServiceError(
                        f"BTE API returned error: {error_msg}",
                        service_name="bte_api"
                    )
                    
                # Log any warnings
                warning_logs = [log for log in status.get("logs", []) if log.get("level") == "WARNING"]
                for warning in warning_logs:
                    logger.warning(f"BTE API warning: {warning.get('message', 'Unknown warning')}")
            
            # Log result structure
            result_message = result.get('message', {})
            knowledge_graph = result_message.get('knowledge_graph', {})
            results = result_message.get('results', [])
            
            kg_nodes_count = len(knowledge_graph.get('nodes', {}))
            kg_edges_count = len(knowledge_graph.get('edges', {}))
            results_count = len(results)
            
            logger.info(f"TRAPI response: {kg_nodes_count} KG nodes, {kg_edges_count} KG edges, {results_count} results")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"TRAPI query executed successfully in {duration:.3f}s")
            
            return result
            
        except ExternalServiceError:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"TRAPI query failed with BTE error after {duration:.3f}s")
            raise  # Re-raise BTE-specific errors
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"Error executing TRAPI query after {duration:.3f}s: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise ExternalServiceError(
                f"TRAPI query execution failed: {str(e)}",
                service_name="bte_api"
            ) from e
    
    def parse_bte_results(self, bte_response: Dict[str, Any], k: int = 5, max_results: int = 50, 
                         predicate: str = None, query_intent: str = None) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Parse BTE API response into structured format with evidence-weighted scoring
        
        Args:
            bte_response: Raw BTE API response
            k: Maximum results per entity
            max_results: Maximum total results
            predicate: The predicate used in the query (for scoring)
            query_intent: Query intent for predicate relevance scoring
            
        Returns:
            Tuple of (parsed_results, entity_name_to_id_mapping)
        """
        def _looks_generic(name: str) -> bool:
            if not isinstance(name, str):
                return True
            n = name.strip().lower()
            return n in {"unknown", "namedthing", "molecule", "small molecule", "chemical", "biological process", "gene", "protein"}

        def _resolve_name_via_node_norm(eid: str) -> Optional[str]:
            try:
                if not isinstance(eid, str) or not eid:
                    return None
                if eid in self._name_cache:
                    return self._name_cache[eid]
                url = "https://nodenormalization-sri.renci.org/1.5/get_normalized_nodes"
                payload = {"curies": [eid], "conflate": True, "description": True}
                r = self.session.post(url, json=payload, timeout=10)
                r.raise_for_status()
                data = r.json() or {}
                entry = data.get(eid)
                if not isinstance(entry, dict):
                    return None
                # Prefer primary id label
                id_block = entry.get("id") or {}
                label = id_block.get("label")
                if isinstance(label, str) and label:
                    self._name_cache[eid] = label
                    return label
                # Fallback to equivalent identifiers' label
                for eq in entry.get("equivalent_identifiers", []) or []:
                    lbl = eq.get("label")
                    if isinstance(lbl, str) and lbl:
                        self._name_cache[eid] = lbl
                        return lbl
                return None
            except Exception:
                return None

        try:
            # Extract query_graph for directionality
            qg = bte_response.get("message", {}).get("query_graph", {})
            qg_nodes = qg.get("nodes", {}) or {}
            qg_edges = qg.get("edges", {}) or {}
            # Determine query edge orientation (use first edge)
            q_sub_key, q_obj_key = None, None
            if isinstance(qg_edges, dict) and qg_edges:
                e_id, e_data = next(iter(qg_edges.items()))
                q_sub_key = e_data.get("subject")
                q_obj_key = e_data.get("object")
            
            # Extract all nodes, edges, and results from knowledge_graph
            nodes = bte_response.get("message", {}).get("knowledge_graph", {}).get("nodes", {})
            edges = bte_response.get("message", {}).get("knowledge_graph", {}).get("edges", {})
            results = bte_response.get("message", {}).get("results", [])
            
            # Initialize evidence scorer if we have predicate and intent info
            evidence_scorer = None
            if predicate and query_intent:
                from .evidence_scoring import create_evidence_scorer
                from .predicate_strategy import QueryIntent
                try:
                    query_intent_enum = QueryIntent(query_intent.lower())
                    # Pass meta-KG to evidence scorer to avoid warnings
                    evidence_scorer = create_evidence_scorer(self.meta_kg)
                except (ValueError, ImportError) as e:
                    logger.warning(f"Could not initialize evidence scorer: {e}")
                    evidence_scorer = None
            
            # Parse node data into a structured format
            node_data = {}
            for node_id, node_info in nodes.items():
                # BTE API returns names as lists - extract first (primary) name
                name_list = node_info.get("name", ["Unknown"])
                primary_name = name_list[0] if isinstance(name_list, list) and name_list else (name_list if isinstance(name_list, str) else "Unknown")
                # Fallbacks: check common alternate fields
                if _looks_generic(primary_name):
                    alt = None
                    try:
                        alt = node_info.get("label") or node_info.get("labels")
                        if isinstance(alt, list):
                            alt = alt[0] if alt else None
                    except Exception:
                        alt = None
                    if isinstance(alt, str) and alt.strip():
                        primary_name = alt
                # Last-resort: resolve via Node Normalization when generic
                if _looks_generic(primary_name):
                    resolved = _resolve_name_via_node_norm(node_id)
                    if isinstance(resolved, str) and resolved.strip():
                        primary_name = resolved
                
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
            
            # Optionally filter non-mechanistic predicates when an intent is specified
            filter_non_mech = bool(query_intent) and str(query_intent).lower() in ("mechanism", "target")
            non_mechanistic_preds = {"biolink:occurs_together_in_literature_with"}
            
            # Debug: log the structure of the first few results
            if results and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"First result structure: {json.dumps(results[0], indent=2) if results else 'No results'}")
                
            for i, result in enumerate(results):
                if len(parsed_results) >= max_results:
                    break
                
                if i == 0:  # Debug first result in detail
                    logger.debug(f"Processing result {i}: {json.dumps(result, indent=2)[:500]}...")
                    
                bindings = result.get("node_bindings", {})
                logger.debug(f"Node bindings keys: {list(bindings.keys())}")
                
                # Map subject/object by the actual TRAPI query edge orientation
                try:
                    subj_binding_key = q_sub_key if q_sub_key in bindings else 'n0'
                    obj_binding_key = q_obj_key if q_obj_key in bindings else 'n1'
                    subject_id = (bindings.get(subj_binding_key) or [{}])[0].get("id", "Unknown")
                    object_id = (bindings.get(obj_binding_key) or [{}])[0].get("id", "Unknown")
                except Exception:
                    subject_id = bindings.get("n0", [{}])[0].get("id", "Unknown")
                    object_id = bindings.get("n1", [{}])[0].get("id", "Unknown")
                
                logger.debug(f"Extracted IDs: subject={subject_id}, object={object_id}")
                
                subject_name = node_data.get(subject_id, {}).get("name", "Unknown")
                object_name = node_data.get(object_id, {}).get("name", "Unknown")
                
                logger.debug(f"Resolved names: subject={subject_name}, object={object_name}")
                
                # Check if we've already collected enough results for this entity
                if k > 0:
                    subject_count = results_per_id.get(subject_id, {}).get("result_count", 0)
                    object_count = results_per_id.get(object_id, {}).get("result_count", 0)
                    
                    if subject_count >= k and object_count >= k:
                        continue
                
                # Find the relationship predicate
                relationship = None
                if i == 0:  # Debug first result's edge matching
                    logger.debug(f"Looking for edges connecting {subject_id} -> {object_id}")
                    logger.debug(f"Available edges: {len(edge_data)}")
                    if len(edge_data) > 0:
                        sample_edges = list(edge_data.items())[:3]
                        for edge_id, edge_info in sample_edges:
                            logger.debug(f"Sample edge {edge_id}: {edge_info.get('subject')} -[{edge_info.get('predicate')}]-> {edge_info.get('object')}")
                            
                # Prefer directed match subject_id -> object_id
                for edge_id, edge_info in edge_data.items():
                    edge_subject = edge_info.get("subject")
                    edge_object = edge_info.get("object")
                    if edge_subject == subject_id and edge_object == object_id:
                        relationship = edge_info.get("predicate")
                        if i == 0:
                            logger.debug(f"Directed match: {edge_subject} -[{relationship}]-> {edge_object}")
                        break
                # Fallback: reversed match (keep query direction)
                if not relationship:
                    for edge_id, edge_info in edge_data.items():
                        edge_subject = edge_info.get("subject")
                        edge_object = edge_info.get("object")
                        if edge_subject == object_id and edge_object == subject_id:
                            relationship = edge_info.get("predicate")
                            if i == 0:
                                logger.debug(f"Reversed match used for predicate: {edge_subject} -[{relationship}]-> {edge_object}")
                            break
                
                if relationship:
                    # Filter non-mechanistic predicates if requested
                    if filter_non_mech and str(relationship) in non_mechanistic_preds:
                        continue
                    enriched = {
                        "subject": subject_name,
                        "subject_id": subject_id,
                        "subject_type": node_data.get(subject_id, {}).get("category", "Unknown"),
                        "predicate": relationship,
                        "object": object_name,
                        "object_id": object_id,
                        "object_type": node_data.get(object_id, {}).get("category", "Unknown")
                    }
                    
                    # Attach evidence-weighted score if possible
                    if evidence_scorer is not None:
                        try:
                            score = evidence_scorer.score_result(result, edges, relationship, query_intent_enum)
                            enriched["score"] = score
                        except Exception as e:
                            logger.debug(f"Scoring failed, using default: {e}")
                    
                    parsed_results.append(enriched)
                    
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
    
    def split_trapi_query(self, trapi_query: Dict[str, Any], batch_limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Split TRAPI query with large entity lists into smaller batches.
        Ensures at most ONE node carries many IDs by trimming others to a single ID.
        
        Args:
            trapi_query: Original TRAPI query
            batch_limit: Maximum entities per batch (defaults to settings.trapi_batch_limit)
            
        Returns:
            List of TRAPI query batches
        """
        try:
            limit = batch_limit or getattr(self, 'batch_limit', 50)
            query_graph = trapi_query.get("message", {}).get("query_graph", {})
            nodes = query_graph.get("nodes", {}) or {}
            if not isinstance(nodes, dict):
                return [trapi_query]
            
            # Find all nodes with IDs
            nodes_with_ids = [(k, v) for k, v in nodes.items() if isinstance(v.get("ids"), list) and v.get("ids")]
            
            # If multiple nodes have IDs, keep the largest list as primary and trim others to 1 ID
            if len(nodes_with_ids) > 1:
                # Choose primary by length
                primary_id, primary_node = max(nodes_with_ids, key=lambda kv: len(kv[1].get("ids", [])))
                for nid, ndata in nodes_with_ids:
                    if nid != primary_id and isinstance(ndata.get("ids"), list) and ndata["ids"]:
                        # Trim to the first ID to avoid cartesian blow-up / batch limit errors
                        ndata["ids"] = [ndata["ids"][0]]
                # Update local reference
                nodes = query_graph.get("nodes", {})
            
            # Identify node to split (the one with IDs > limit)
            split_nodes = {k: v for k, v in nodes.items() if isinstance(v.get("ids"), list) and len(v.get("ids", [])) > limit}
            if not split_nodes:
                return [trapi_query]
            
            node_id, node_data = next(iter(split_nodes.items()))
            id_list = node_data.get("ids", [])
            id_chunks = [id_list[i:i + limit] for i in range(0, len(id_list), limit)]
            
            # Create a new query for each chunk (other nodes preserved as trimmed above)
            queries = []
            for chunk in id_chunks:
                new_query = deepcopy(trapi_query)
                new_query["message"]["query_graph"]["nodes"][node_id]["ids"] = chunk
                queries.append(new_query)
            
            logger.info(f"Split TRAPI query: node={node_id}, original_count={len(id_list)}, batch_size={limit}, batches={len(queries)}")
            return queries
            
        except Exception as e:
            logger.error(f"Error splitting TRAPI query: {e}")
            return [trapi_query]
    
    def execute_trapi_with_batching(self, trapi_query: Dict[str, Any], 
                                  max_results: int = 50, k: int = 5, 
                                  batch_limit: Optional[int] = None, predicate: str = None, 
                                  query_intent: str = None) -> Tuple[List[Dict], Dict[str, str], Dict[str, Any]]:
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
            # Optionally enforce single-hop, two-node TRAPI (disabled by default)
            try:
                if getattr(self.settings, 'enforce_two_node', False):
                    qg = trapi_query.get("message", {}).get("query_graph", {})
                    edges = qg.get("edges", {}) if isinstance(qg.get("edges"), dict) else {}
                    nodes = qg.get("nodes", {}) if isinstance(qg.get("nodes"), dict) else {}
                    if not edges or not nodes:
                        pass
                    else:
                        # Choose best single edge by heuristic priority
                        def edge_priority(eid_e):
                            eid, e = eid_e
                            sj = e.get("subject"); ob = e.get("object")
                            sc = (nodes.get(sj, {}).get("categories") or [None])[0]
                            oc = (nodes.get(ob, {}).get("categories") or [None])[0]
                            pair = (sc or "", oc or "")
                            if pair in (("biolink:ChemicalSubstance", "biolink:Disease"), ("biolink:SmallMolecule", "biolink:Disease")):
                                return 0
                            if pair in (("biolink:ChemicalSubstance", "biolink:Gene"), ("biolink:SmallMolecule", "biolink:Gene")):
                                return 1
                            if pair == ("biolink:Gene", "biolink:BiologicalProcess"):
                                return 2
                            return 10
                        e_items = sorted(list(edges.items()), key=edge_priority)
                        chosen_eid, chosen_edge = e_items[0]
                        # Determine referenced node keys
                        subj_key = chosen_edge.get("subject")
                        obj_key = chosen_edge.get("object")
                        # Rebuild edges dict with only the chosen edge id normalized to 'e01'
                        qg["edges"] = {"e01": {"subject": subj_key, "object": obj_key, "predicates": chosen_edge.get("predicates", [])}}
                        # Keep only referenced nodes
                        kept_nodes = {}
                        if subj_key in nodes:
                            kept_nodes[subj_key] = nodes[subj_key]
                        if obj_key in nodes:
                            kept_nodes[obj_key] = nodes[obj_key]
                        qg["nodes"] = kept_nodes
                        logger.info(f"Sanitized TRAPI to single-hop two-node graph (nodes={list(kept_nodes.keys())})")
            except Exception as e:
                logger.debug("Two-node enforcement skipped: %s", e)

            # Split query into batches if needed
            def run_with_limit(limit: int):
                # Inspect original ID-carrying node and count
                qg = trapi_query.get("message", {}).get("query_graph", {})
                nodes = qg.get("nodes", {}) or {}
                split_candidates = {k: v for k, v in nodes.items() if isinstance(v.get("ids"), list) and v.get("ids")}
                # Choose the node with the largest ID list, if any
                orig_count = 0
                split_node = None
                if split_candidates:
                    split_node, node_data = max(split_candidates.items(), key=lambda kv: len(kv[1].get("ids", [])))
                    try:
                        orig_count = len(node_data.get("ids", []))
                    except Exception:
                        orig_count = 0
                batches = self.split_trapi_query(trapi_query, limit)
                batch_count = len(batches)
                all_results, all_mappings, messages = [], {}, []
                for i, batch_query in enumerate(batches):
                    logger.info(f"Executing TRAPI batch {i+1}/{len(batches)} (limit={limit})")
                    try:
                        response = self.execute_trapi_query(batch_query)
                        batch_results, batch_mappings = self.parse_bte_results(response, k, max_results, predicate, query_intent)
                        all_results.extend(batch_results)
                        all_mappings.update(batch_mappings)
                        messages.append(response.get("description", f"Batch {i+1} completed"))
                        if len(all_results) >= max_results:
                            all_results = all_results[:max_results]
                            break
                    except Exception as e:
                        messages.append(f"Batch {i+1} failed: {str(e)}")
                        # If BTE complains about batch size, propagate an indicator
                        if "exceeds batch size limit" in str(e):
                            raise RuntimeError("BATCH_LIMIT_EXCEEDED")
                        continue
                return all_results, all_mappings, messages, batch_count, orig_count, split_node
            
            # Determine initial batch limit from settings if not provided
            initial_limit = batch_limit or getattr(self, 'batch_limit', 50)
            # Build fallback ladder only if starting above 10
            limits = [initial_limit] if initial_limit <= 10 else [initial_limit, 25, 10]
            all_results, all_mappings, messages = [], {}, []
            planned_batches, original_count, split_node = 1, 0, None
            used_limit = None
            for lim in limits:
                try:
                    used_limit = lim
                    all_results, all_mappings, messages, planned_batches, original_count, split_node = run_with_limit(lim)
                    break
                except RuntimeError as re:
                    if str(re) == "BATCH_LIMIT_EXCEEDED" and lim != limits[-1]:
                        logger.warning(f"Batch size {lim} too large; retrying with smaller limit")
                        continue
                    else:
                        raise
            
            metadata = {
                "message": "; ".join(messages),
                "total_batches": len(messages),  # executed batches
                "successful_batches": len([m for m in messages if "failed" not in m.lower()]),
                "total_results": len(all_results),
                "planned_batch_count": planned_batches,
                "original_entity_count": original_count,
                "batch_limit": used_limit,
                "id_node": split_node,
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