"""
Stand-alone BTE client (BioThings Explorer) for TRAPI execution and meta-KG retrieval.
Ported and simplified for the Prototype package.
"""

from __future__ import annotations

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .settings import get_settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BTEClient:
    def __init__(self, base_url: Optional[str] = None, timeout: int = 300,
                 query_endpoint: Optional[str] = None, meta_kg_endpoint: Optional[str] = None):
        self.settings = get_settings()
        self.base_url = (base_url or self.settings.bte_api_base_url).rstrip('/')
        self.query_endpoint = (query_endpoint or self.settings.bte_query_endpoint).lstrip('/')
        self.meta_kg_endpoint = (meta_kg_endpoint or self.settings.bte_meta_kg_endpoint).lstrip('/')
        self.timeout = timeout

        self.session = requests.Session()
        retry_kwargs = {
            "total": 3,
            "status_forcelist": [429, 500, 502, 503, 504],
            "backoff_factor": 1,
        }
        try:
            retry_strategy = Retry(allowed_methods=["HEAD", "GET", "OPTIONS", "POST"], **retry_kwargs)
        except TypeError:
            retry_strategy = Retry(method_whitelist=["HEAD", "GET", "OPTIONS", "POST"], **retry_kwargs)
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self._meta_kg_cache: Optional[Dict[str, Any]] = None
        self._meta_kg_cache_time: float = 0.0
        self._cache_ttl: int = 3600

    def _request(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None, params: Optional[Dict] = None) -> requests.Response:
        # If endpoint is absolute, use it directly
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            url = endpoint
        else:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if method.upper() == "GET":
            return self.session.get(url, params=params, timeout=self.timeout)
        if method.upper() == "POST":
            return self.session.post(url, json=data, params=params, timeout=self.timeout, headers={"Content-Type": "application/json"})
        raise ValueError(f"Unsupported method: {method}")

    def get_meta_knowledge_graph(self) -> Dict[str, Any]:
        now = time.time()
        if self._meta_kg_cache and (now - self._meta_kg_cache_time) < self._cache_ttl:
            return self._meta_kg_cache
        try:
            resp = self._request(self.meta_kg_endpoint)
            resp.raise_for_status()
            meta = resp.json()
        except Exception:
            # Fallback to public meta-KG if local instance doesn't expose it
            fallback = "https://bte.transltr.io/v1/meta_knowledge_graph"
            r = self._request(fallback)
            r.raise_for_status()
            meta = r.json()
        self._meta_kg_cache = meta
        self._meta_kg_cache_time = now
        return meta

    def _extract_job_id(self, data: Dict[str, Any]) -> Optional[str]:
        for k in ("id", "job_id", "request_id", "task_id"):
            v = data.get(k)
            if isinstance(v, str) and v:
                return v
        # sometimes nested
        if isinstance(data.get("data"), dict):
            return self._extract_job_id(data["data"])  # type: ignore
        return None

    def _fetch_async_result(self, job_id: str, poll_seconds: int = 60, interval: float = 1.5) -> Dict[str, Any]:
        deadline = time.time() + poll_seconds
        result_urls = [
            f"{self.base_url}/result/{job_id}",
            f"{self.base_url}/results/{job_id}",
            f"{self.base_url}/asyncquery/{job_id}",
            f"{self.base_url}/response/{job_id}",
        ]
        status_urls = [
            f"{self.base_url}/status/{job_id}",
            f"{self.base_url}/status?id={job_id}",
        ]
        last_status: Optional[str] = None
        while time.time() < deadline:
            # Try result endpoints first
            for url in result_urls:
                try:
                    r = self._request(url, method="GET")
                    if r.status_code == 404:
                        continue
                    if r.status_code in (200, 201):
                        data = r.json()
                        # Heuristic: TRAPI responses have 'message'
                        if isinstance(data, dict) and "message" in data:
                            return data
                        # Some servers wrap result under 'data'
                        if isinstance(data.get("data"), dict) and "message" in data["data"]:
                            return data["data"]
                except Exception:
                    continue
            # Check status endpoints
            for url in status_urls:
                try:
                    r = self._request(url, method="GET")
                    if r.status_code in (200, 201):
                        data = r.json()
                        status = str(data.get("status") or data.get("state") or "").lower()
                        if status:
                            last_status = status
                            if status in ("succeeded", "success", "completed", "complete", "done"):
                                # One more attempt to fetch result next loop
                                break
                            if status in ("failed", "error"):
                                raise RuntimeError(f"BTE async job failed: {data}")
                except Exception:
                    continue
            time.sleep(interval)
        raise TimeoutError(f"Timed out waiting for async result (last_status={last_status})")

    def _extract_job_id(self, data: Dict[str, Any]) -> Optional[str]:
        for k in ("id", "job_id", "request_id", "task_id"):
            v = data.get(k)
            if isinstance(v, str) and v:
                return v
        # sometimes nested under 'data'
        if isinstance(data.get("data"), dict):
            nested = data.get("data")
            if isinstance(nested, dict):
                return self._extract_job_id(nested)  # type: ignore
        return None

    def _fetch_async_result(self, job_id: str, poll_seconds: int = 90, interval: float = 1.5) -> Dict[str, Any]:
        deadline = time.time() + poll_seconds
        # Common result endpoints observed across BTE deployments
        result_urls = [
            f"{self.base_url}/asyncquery/{job_id}",
            f"{self.base_url}/result/{job_id}",
            f"{self.base_url}/results/{job_id}",
            f"{self.base_url}/response/{job_id}",
        ]
        status_urls = [
            f"{self.base_url}/status/{job_id}",
            f"{self.base_url}/status?id={job_id}",
        ]
        last_status: Optional[str] = None
        while time.time() < deadline:
            # Try result endpoints first
            for url in result_urls:
                try:
                    r = self._request(url, method="GET")
                    if r.status_code == 404:
                        continue
                    if r.status_code in (200, 201):
                        data = r.json()
                        # TRAPI responses either appear directly or under 'response'
                        if isinstance(data, dict):
                            if "message" in data:
                                return data
                            if isinstance(data.get("response"), dict) and "message" in data["response"]:
                                return data["response"]
                            if isinstance(data.get("data"), dict) and "message" in data["data"]:
                                return data["data"]
                except Exception:
                    continue
            # Check status endpoints for progress
            for url in status_urls:
                try:
                    r = self._request(url, method="GET")
                    if r.status_code in (200, 201):
                        data = r.json()
                        status = str(data.get("status") or data.get("state") or "").lower()
                        if status:
                            last_status = status
                            if status in ("succeeded", "success", "completed", "complete", "done"):
                                # Loop will try result endpoints again
                                break
                            if status in ("failed", "error"):
                                raise RuntimeError(f"BTE async job failed: {data}")
                except Exception:
                    continue
            time.sleep(interval)
        raise TimeoutError(f"Timed out waiting for async result (last_status={last_status})")

    def execute_trapi_query(self, trapi_query: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._request(self.query_endpoint, method="POST", data=trapi_query)
        # Async endpoints may return 202 or 200 with job id
        if resp.status_code in (200, 201, 202):
            try:
                data = resp.json()
            except Exception:
                resp.raise_for_status()
                return {}
            # If direct TRAPI message returned
            if isinstance(data, dict) and "message" in data:
                return data
            # If job id present, poll
            job_id = self._extract_job_id(data) if isinstance(data, dict) else None
            if job_id:
                return self._fetch_async_result(job_id)
            # Otherwise, raise for unexpected format
            resp.raise_for_status()
            return data
        # Other status codes
        resp.raise_for_status()
        return resp.json()

    def parse_bte_results(self, bte_response: Dict[str, Any], k: int = 5, max_results: int = 50,
                          predicate: Optional[str] = None, query_intent: Optional[str] = None) -> Tuple[List[Dict], Dict[str, str]]:
        nodes = bte_response.get("message", {}).get("knowledge_graph", {}).get("nodes", {})
        edges = bte_response.get("message", {}).get("knowledge_graph", {}).get("edges", {})
        results = bte_response.get("message", {}).get("results", [])

        # Nodes: normalize name
        node_data: Dict[str, Dict[str, Any]] = {}
        for node_id, node_info in nodes.items():
            name_raw = node_info.get("name", ["Unknown"])  # can be list in some BTE instances
            name = name_raw[0] if isinstance(name_raw, list) and name_raw else (name_raw if isinstance(name_raw, str) else "Unknown")
            node_data[node_id] = {
                "ID": node_id,
                "name": name,
                "category": node_info.get("categories", ["Unknown"])[0]
            }

        edge_data: Dict[str, Dict[str, Any]] = {}
        for edge_id, edge_info in edges.items():
            edge_data[edge_id] = {
                "subject": edge_info.get("subject"),
                "predicate": edge_info.get("predicate"),
                "object": edge_info.get("object")
            }

        parsed: List[Dict[str, Any]] = []
        entity_map: Dict[str, str] = {}
        results_per_id: Dict[str, Dict[str, Any]] = {}

        for res in results:
            bindings = res.get("node_bindings", {})
            subject_id = bindings.get("n0", [{}])[0].get("id", "Unknown")
            object_id = bindings.get("n1", [{}])[0].get("id", "Unknown")

            subject_name = node_data.get(subject_id, {}).get("name", subject_id)
            object_name = node_data.get(object_id, {}).get("name", object_id)

            relationship = None
            for eid, einfo in edge_data.items():
                if einfo.get("subject") == subject_id and einfo.get("object") == object_id:
                    relationship = einfo.get("predicate")
                    break
                if einfo.get("subject") == object_id and einfo.get("object") == subject_id:
                    # Flip
                    relationship = einfo.get("predicate")
                    subject_id, object_id = object_id, subject_id
                    subject_name, object_name = object_name, subject_name
                    break

            if not relationship:
                continue

            enriched = {
                "subject": subject_name,
                "subject_id": subject_id,
                "subject_type": node_data.get(subject_id, {}).get("category", "Unknown"),
                "predicate": relationship,
                "object": object_name,
                "object_id": object_id,
                "object_type": node_data.get(object_id, {}).get("category", "Unknown"),
            }
            parsed.append(enriched)

            # Maintain per-ID caps k if desired (soft cap by early continue)
            if k > 0:
                results_per_id.setdefault(subject_id, {"count": 0})
                results_per_id.setdefault(object_id, {"count": 0})
                results_per_id[subject_id]["count"] += 1
                results_per_id[object_id]["count"] += 1

            # Update entity map
            entity_map.setdefault(subject_name, subject_id)
            entity_map.setdefault(object_name, object_id)

            if len(parsed) >= max_results:
                break

        return parsed, entity_map

    def split_trapi_query(self, trapi_query: Dict[str, Any], batch_limit: int = 50) -> List[Dict[str, Any]]:
        qg = trapi_query.get("message", {}).get("query_graph", {})
        nodes = qg.get("nodes", {})
        split_nodes = {k: v for k, v in nodes.items() if isinstance(v.get("ids"), list) and len(v["ids"]) > batch_limit}
        if not split_nodes:
            return [trapi_query]
        node_id, node_data = next(iter(split_nodes.items()))
        id_chunks = [node_data["ids"][i:i + batch_limit] for i in range(0, len(node_data["ids"]), batch_limit)]
        queries: List[Dict[str, Any]] = []
        for chunk in id_chunks:
            q = json.loads(json.dumps(trapi_query))
            q["message"]["query_graph"]["nodes"][node_id]["ids"] = chunk
            queries.append(q)
        return queries

    def execute_trapi_with_batching(self, trapi_query: Dict[str, Any], max_results: int = 50, k: int = 5,
                                    batch_limit: int = 50, predicate: Optional[str] = None,
                                    query_intent: Optional[str] = None) -> Tuple[List[Dict], Dict[str, str], Dict[str, Any]]:
        queries = self.split_trapi_query(trapi_query, batch_limit)
        all_results: List[Dict] = []
        all_maps: Dict[str, str] = {}
        messages: List[str] = []
        for i, q in enumerate(queries):
            try:
                resp = self.execute_trapi_query(q)
                results, mapping = self.parse_bte_results(resp, k=k, max_results=max_results, predicate=predicate, query_intent=query_intent)
                all_results.extend(results)
                all_maps.update(mapping)
                messages.append(resp.get("description", f"Batch {i+1} completed"))
                if len(all_results) >= max_results:
                    all_results = all_results[:max_results]
                    break
            except Exception as e:
                messages.append(f"Batch {i+1} failed: {e}")
                continue
        meta = {
            "message": "; ".join(messages),
            "total_batches": len(queries),
            "successful_batches": len([m for m in messages if "failed" not in m.lower()]),
            "total_results": len(all_results),
        }
        return all_results, all_maps, meta


__all__ = ["BTEClient"]
