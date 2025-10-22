"""
Result Caching System for Query Optimizers

This module provides a unified caching system that can be shared across
all query optimizers to improve performance and reduce API calls.
"""

import hashlib
import json
import time
import pickle
import os
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from threading import RLock
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached query result"""
    key: str
    query: str
    strategy: str
    result: Dict[str, Any]
    timestamp: float
    ttl: int  # Time to live in seconds
    hit_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired"""
        return time.time() - self.timestamp > self.ttl
    
    def touch(self):
        """Update hit count and access time"""
        self.hit_count += 1
        # Note: We don't update timestamp on access to maintain true TTL


class QueryCache:
    """
    Thread-safe caching system for query results
    
    Supports both in-memory and persistent disk caching with configurable TTL,
    size limits, and cache eviction policies.
    """
    
    def __init__(self, 
                 max_memory_entries: int = 1000,
                 default_ttl: int = 3600,
                 cache_dir: Optional[str] = None,
                 enable_disk_cache: bool = True,
                 max_disk_cache_mb: int = 100):
        """
        Initialize the query cache
        
        Args:
            max_memory_entries: Maximum entries to keep in memory
            default_ttl: Default TTL in seconds
            cache_dir: Directory for disk cache (None for temp)
            enable_disk_cache: Whether to enable persistent disk cache
            max_disk_cache_mb: Maximum disk cache size in MB
        """
        self.max_memory_entries = max_memory_entries
        self.default_ttl = default_ttl
        self.enable_disk_cache = enable_disk_cache
        self.max_disk_cache_bytes = max_disk_cache_mb * 1024 * 1024
        
        # In-memory cache
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # For LRU eviction
        self._lock = RLock()
        
        # Disk cache setup
        if enable_disk_cache:
            if cache_dir:
                self.cache_dir = Path(cache_dir)
            else:
                self.cache_dir = Path.home() / ".agentic_bte" / "cache"
            
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized disk cache at {self.cache_dir}")
        else:
            self.cache_dir = None
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "disk_reads": 0,
            "disk_writes": 0,
            "errors": 0
        }
    
    def _generate_cache_key(self, query: str, strategy: str, params: Optional[Dict] = None) -> str:
        """Generate a unique cache key for the query and parameters"""
        # Normalize query (lowercase, strip whitespace)
        normalized_query = query.lower().strip()
        
        # Create key components
        key_data = {
            "query": normalized_query,
            "strategy": strategy,
            "params": params or {}
        }
        
        # Create hash
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    def get(self, query: str, strategy: str, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached result for a query
        
        Args:
            query: The biomedical query
            strategy: Optimization strategy used
            params: Additional parameters that affect the result
            
        Returns:
            Cached result or None if not found/expired
        """
        cache_key = self._generate_cache_key(query, strategy, params)
        
        with self._lock:
            # Try memory cache first
            if cache_key in self._memory_cache:
                entry = self._memory_cache[cache_key]
                if not entry.is_expired():
                    entry.touch()
                    self._update_access_order(cache_key)
                    self.stats["hits"] += 1
                    logger.debug(f"Memory cache hit for key: {cache_key}")
                    return entry.result
                else:
                    # Remove expired entry
                    del self._memory_cache[cache_key]
                    if cache_key in self._access_order:
                        self._access_order.remove(cache_key)
            
            # Try disk cache if enabled
            if self.enable_disk_cache:
                disk_result = self._get_from_disk(cache_key)
                if disk_result:
                    # Add back to memory cache
                    entry = CacheEntry(
                        key=cache_key,
                        query=query,
                        strategy=strategy,
                        result=disk_result["result"],
                        timestamp=disk_result["timestamp"],
                        ttl=disk_result["ttl"]
                    )
                    
                    if not entry.is_expired():
                        self._add_to_memory(entry)
                        self.stats["hits"] += 1
                        self.stats["disk_reads"] += 1
                        logger.debug(f"Disk cache hit for key: {cache_key}")
                        return entry.result
            
            self.stats["misses"] += 1
            return None
    
    def put(self, query: str, strategy: str, result: Dict[str, Any], 
            params: Optional[Dict] = None, ttl: Optional[int] = None):
        """
        Store a result in the cache
        
        Args:
            query: The biomedical query
            strategy: Optimization strategy used
            result: The result to cache
            params: Additional parameters that affect the result
            ttl: Time to live (uses default if None)
        """
        cache_key = self._generate_cache_key(query, strategy, params)
        ttl = ttl or self.default_ttl
        
        entry = CacheEntry(
            key=cache_key,
            query=query,
            strategy=strategy,
            result=result,
            timestamp=time.time(),
            ttl=ttl
        )
        
        with self._lock:
            # Add to memory cache
            self._add_to_memory(entry)
            
            # Add to disk cache if enabled
            if self.enable_disk_cache:
                try:
                    self._save_to_disk(entry)
                    self.stats["disk_writes"] += 1
                except Exception as e:
                    logger.warning(f"Failed to save to disk cache: {e}")
                    self.stats["errors"] += 1
        
        logger.debug(f"Cached result for key: {cache_key}")
    
    def _add_to_memory(self, entry: CacheEntry):
        """Add entry to memory cache with LRU eviction"""
        # Remove if already exists
        if entry.key in self._memory_cache:
            if entry.key in self._access_order:
                self._access_order.remove(entry.key)
        
        # Check if we need to evict
        while len(self._memory_cache) >= self.max_memory_entries:
            self._evict_lru()
        
        # Add new entry
        self._memory_cache[entry.key] = entry
        self._access_order.append(entry.key)
    
    def _update_access_order(self, key: str):
        """Update access order for LRU tracking"""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self._access_order:
            return
        
        lru_key = self._access_order.pop(0)
        if lru_key in self._memory_cache:
            del self._memory_cache[lru_key]
            self.stats["evictions"] += 1
            logger.debug(f"Evicted LRU entry: {lru_key}")
    
    def _get_from_disk(self, cache_key: str) -> Optional[Dict]:
        """Load entry from disk cache"""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.cache"
        
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    
                # Check if expired
                if time.time() - data["timestamp"] > data["ttl"]:
                    cache_file.unlink()  # Remove expired file
                    return None
                
                return data
        except Exception as e:
            logger.warning(f"Error reading disk cache {cache_file}: {e}")
            # Try to remove corrupted file
            try:
                cache_file.unlink()
            except:
                pass
        
        return None
    
    def _save_to_disk(self, entry: CacheEntry):
        """Save entry to disk cache"""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{entry.key}.cache"
        
        data = {
            "query": entry.query,
            "strategy": entry.strategy,
            "result": entry.result,
            "timestamp": entry.timestamp,
            "ttl": entry.ttl
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    
    def invalidate(self, query: str, strategy: str, params: Optional[Dict] = None):
        """Remove specific entry from cache"""
        cache_key = self._generate_cache_key(query, strategy, params)
        
        with self._lock:
            # Remove from memory
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            
            # Remove from disk
            if self.cache_dir:
                cache_file = self.cache_dir / f"{cache_key}.cache"
                try:
                    cache_file.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Error removing disk cache file: {e}")
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._memory_cache.clear()
            self._access_order.clear()
            
            # Clear disk cache
            if self.cache_dir and self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.cache"):
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.warning(f"Error removing cache file {cache_file}: {e}")
        
        logger.info("Cache cleared")
    
    def cleanup_expired(self):
        """Remove expired entries from both memory and disk caches"""
        with self._lock:
            # Clean memory cache
            expired_keys = []
            for key, entry in self._memory_cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._memory_cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
            
            # Clean disk cache
            if self.cache_dir and self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.cache"):
                    try:
                        # Check if file is expired by looking at modification time
                        mtime = cache_file.stat().st_mtime
                        if time.time() - mtime > self.default_ttl:
                            cache_file.unlink()
                    except Exception as e:
                        logger.warning(f"Error cleaning cache file {cache_file}: {e}")
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache performance statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            **self.stats,
            "memory_entries": len(self._memory_cache),
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "disk_cache_enabled": self.enable_disk_cache
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information"""
        with self._lock:
            memory_entries = []
            for key, entry in self._memory_cache.items():
                memory_entries.append({
                    "key": key,
                    "query_preview": entry.query[:50] + "..." if len(entry.query) > 50 else entry.query,
                    "strategy": entry.strategy,
                    "age_seconds": time.time() - entry.timestamp,
                    "ttl_remaining": max(0, entry.ttl - (time.time() - entry.timestamp)),
                    "hit_count": entry.hit_count,
                    "expired": entry.is_expired()
                })
            
            disk_size = 0
            disk_files = 0
            if self.cache_dir and self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.cache"):
                    try:
                        disk_size += cache_file.stat().st_size
                        disk_files += 1
                    except:
                        pass
            
            return {
                "memory_entries": memory_entries,
                "disk_cache_size_mb": disk_size / (1024 * 1024),
                "disk_files": disk_files,
                "max_memory_entries": self.max_memory_entries,
                "max_disk_cache_mb": self.max_disk_cache_bytes / (1024 * 1024),
                "cache_dir": str(self.cache_dir) if self.cache_dir else None,
                "stats": self.get_stats()
            }


# Global cache instance
_global_cache: Optional[QueryCache] = None


def get_global_cache() -> QueryCache:
    """Get the global cache instance, creating it if necessary"""
    global _global_cache
    if _global_cache is None:
        _global_cache = QueryCache()
    return _global_cache


def configure_global_cache(max_memory_entries: int = 1000,
                          default_ttl: int = 3600,
                          cache_dir: Optional[str] = None,
                          enable_disk_cache: bool = True,
                          max_disk_cache_mb: int = 100):
    """Configure the global cache instance"""
    global _global_cache
    _global_cache = QueryCache(
        max_memory_entries=max_memory_entries,
        default_ttl=default_ttl,
        cache_dir=cache_dir,
        enable_disk_cache=enable_disk_cache,
        max_disk_cache_mb=max_disk_cache_mb
    )
    return _global_cache


# Convenience functions for direct cache operations
def cache_get(query: str, strategy: str, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """Get cached result using global cache"""
    return get_global_cache().get(query, strategy, params)


def cache_put(query: str, strategy: str, result: Dict[str, Any], 
              params: Optional[Dict] = None, ttl: Optional[int] = None):
    """Store result using global cache"""
    return get_global_cache().put(query, strategy, result, params, ttl)


def cache_invalidate(query: str, strategy: str, params: Optional[Dict] = None):
    """Invalidate cached result using global cache"""
    return get_global_cache().invalidate(query, strategy, params)


def cache_clear():
    """Clear global cache"""
    return get_global_cache().clear()


def cache_stats() -> Dict[str, Union[int, float]]:
    """Get global cache statistics"""
    return get_global_cache().get_stats()


# Aliases for test framework compatibility
def clear_cache():
    """Alias for cache_clear"""
    return cache_clear()


def get_cache_stats() -> Dict[str, Union[int, float]]:
    """Alias for cache_stats"""
    return cache_stats()
