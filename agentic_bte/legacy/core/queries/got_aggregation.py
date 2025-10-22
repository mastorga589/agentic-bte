"""
Enhanced Biomedical Thought Aggregation for GoT Framework

This module implements sophisticated aggregation strategies specifically designed
for biomedical data, with confidence scoring, iterative refinement, and
quality assessment based on domain-specific knowledge.
"""

import logging
import time
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
import json
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .got_framework import GoTThought, ThoughtType, GoTMetrics
from .interfaces import OptimizationResult
# from call_mcp_tool import call_mcp_tool  # Removed - not needed in aggregation

logger = logging.getLogger(__name__)


@dataclass
class AggregationScore:
    """Comprehensive scoring for aggregated thoughts"""
    confidence_score: float = 0.0      # Weighted confidence from source thoughts
    diversity_score: float = 0.0       # Diversity of information sources
    consistency_score: float = 0.0     # Consistency across sources
    relevance_score: float = 0.0       # Relevance to original query
    biomedical_score: float = 0.0      # Domain-specific quality
    overall_score: float = 0.0         # Combined weighted score


class BiomedicalAggregator:
    """
    Advanced aggregator for biomedical thoughts with domain-specific logic
    """
    
    def __init__(self, enable_refinement: bool = True, max_refinement_iterations: int = 3):
        """
        Initialize the biomedical aggregator
        
        Args:
            enable_refinement: Enable iterative refinement of aggregated results
            max_refinement_iterations: Maximum refinement iterations
        """
        self.enable_refinement = enable_refinement
        self.max_refinement_iterations = max_refinement_iterations
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Biomedical entity type hierarchies for smarter aggregation
        self.entity_hierarchies = {
            'Gene': ['NCBIGene', 'HGNC', 'ENSEMBL'],
            'Disease': ['MONDO', 'DOID', 'HP', 'MESH'],
            'Chemical': ['CHEBI', 'PUBCHEM.COMPOUND', 'DRUGBANK'],
            'Protein': ['PR', 'UniProtKB'],
            'Pathway': ['REACTOME', 'KEGG', 'GO'],
            'Phenotype': ['HP', 'MP', 'ZP']
        }
        
        # Quality weights for different aspects
        self.quality_weights = {
            'confidence': 0.3,
            'diversity': 0.2,
            'consistency': 0.2,
            'relevance': 0.15,
            'biomedical': 0.15
        }
    
    async def aggregate_entity_thoughts(self, thoughts: List[GoTThought], 
                                      context: Dict[str, Any]) -> GoTThought:
        """
        Aggregate entity extraction thoughts with sophisticated deduplication
        and confidence scoring
        """
        logger.info(f"Aggregating {len(thoughts)} entity thoughts")
        
        # Collect all entities with their sources
        entity_sources = defaultdict(list)
        total_confidence = 0.0
        
        for thought in thoughts:
            entities = thought.content.get("entities", {})
            for entity_name, entity_id in entities.items():
                entity_sources[entity_name].append({
                    'id': entity_id,
                    'source_thought': thought.id,
                    'confidence': thought.confidence,
                    'timestamp': thought.created_at
                })
            total_confidence += thought.confidence
        
        # Resolve entity conflicts and create final entity mapping
        resolved_entities = await self._resolve_entity_conflicts(entity_sources)
        
        # Calculate aggregation score
        score = self._calculate_entity_aggregation_score(
            resolved_entities, thoughts, context
        )
        
        aggregated_thought = GoTThought(
            id="",  # Will be auto-generated
            thought_type=ThoughtType.ENTITY_EXTRACTION,
            content={
                "entities": resolved_entities,
                "aggregated_from": [t.id for t in thoughts],
                "aggregation_score": score.__dict__,
                "entity_conflicts_resolved": len([k for k, v in entity_sources.items() if len(v) > 1])
            },
            confidence=score.overall_score,
            dependencies=set().union(*(t.dependencies for t in thoughts)),
            metadata={
                "aggregation_method": "biomedical_entity_resolution",
                "source_count": len(thoughts),
                "entity_count": len(resolved_entities)
            }
        )
        
        logger.info(f"Entity aggregation completed: {len(resolved_entities)} entities, "
                   f"score={score.overall_score:.3f}")
        
        return aggregated_thought
    
    async def aggregate_result_thoughts(self, thoughts: List[GoTThought], 
                                      context: Dict[str, Any]) -> GoTThought:
        """
        Aggregate API execution results with advanced ranking and deduplication
        """
        logger.info(f"Aggregating {len(thoughts)} result thoughts")
        
        all_results = []
        source_metadata = {}
        
        # Collect all results with metadata
        for thought in thoughts:
            results = thought.content.get("results", [])
            for i, result in enumerate(results):
                enhanced_result = result.copy()
                enhanced_result['source_thought'] = thought.id
                enhanced_result['source_confidence'] = thought.confidence
                enhanced_result['original_index'] = i
                all_results.append(enhanced_result)
                
            source_metadata[thought.id] = {
                'confidence': thought.confidence,
                'result_count': len(results),
                'dependencies': thought.dependencies
            }
        
        # Apply sophisticated deduplication and ranking
        deduplicated_results = await self._deduplicate_biomedical_results(all_results)
        ranked_results = await self._rank_biomedical_results(deduplicated_results, context)
        
        # Apply iterative refinement if enabled
        if self.enable_refinement and len(ranked_results) > 0:
            refined_results = await self._iteratively_refine_results(
                ranked_results, context, source_metadata
            )
        else:
            refined_results = ranked_results
        
        # Calculate aggregation score
        score = self._calculate_result_aggregation_score(
            refined_results, thoughts, context
        )
        
        # Generate comprehensive final answer
        final_answer = self._generate_comprehensive_answer(refined_results, context)
        
        aggregated_thought = GoTThought(
            id="",
            thought_type=ThoughtType.RESULT_AGGREGATION,
            content={
                "results": refined_results,
                "final_answer": final_answer,
                "aggregated_from": [t.id for t in thoughts],
                "aggregation_score": score.__dict__,
                "deduplication_stats": {
                    "original_count": len(all_results),
                    "deduplicated_count": len(deduplicated_results),
                    "final_count": len(refined_results)
                }
            },
            confidence=score.overall_score,
            dependencies=set().union(*(t.dependencies for t in thoughts)),
            metadata={
                "aggregation_method": "biomedical_result_ranking",
                "refinement_applied": self.enable_refinement,
                "source_thoughts": list(source_metadata.keys())
            }
        )
        
        logger.info(f"Result aggregation completed: {len(refined_results)} final results, "
                   f"score={score.overall_score:.3f}")
        
        return aggregated_thought
    
    async def _resolve_entity_conflicts(self, entity_sources: Dict[str, List[Dict]]) -> Dict[str, str]:
        """
        Resolve conflicts when the same entity appears with different IDs
        """
        resolved_entities = {}
        
        for entity_name, sources in entity_sources.items():
            if len(sources) == 1:
                # No conflict, use the single source
                resolved_entities[entity_name] = sources[0]['id']
            else:
                # Multiple sources, resolve conflict
                resolved_id = await self._resolve_single_entity_conflict(entity_name, sources)
                resolved_entities[entity_name] = resolved_id
        
        return resolved_entities
    
    async def aggregate_results(self, results: List[Dict[str, Any]], 
                              entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Simple method to aggregate and rank results for production use
        
        Args:
            results: List of API results to aggregate
            entities: List of entities for context
            
        Returns:
            Aggregated and ranked results
        """
        if not results:
            return results
        
        try:
            context = {
                'entities': entities,
                'query': 'aggregation context'
            }
            
            # Apply deduplication and ranking
            deduplicated_results = await self._deduplicate_biomedical_results(results)
            ranked_results = await self._rank_biomedical_results(deduplicated_results, context)
            
            logger.info(f"Aggregated {len(results)} results to {len(ranked_results)} final results")
            
            return ranked_results
            
        except Exception as e:
            logger.warning(f"Result aggregation failed: {e}")
            return results
    
    async def _resolve_single_entity_conflict(self, entity_name: str, 
                                           sources: List[Dict]) -> str:
        """
        Resolve a single entity conflict using domain knowledge and confidence
        """
        # Sort sources by confidence and recency
        sources.sort(key=lambda x: (x['confidence'], x['timestamp']), reverse=True)
        
        # Check if IDs are actually different or just differently formatted
        unique_ids = set(s['id'] for s in sources)
        if len(unique_ids) == 1:
            return sources[0]['id']
        
        # Apply biomedical entity resolution logic
        for entity_type, prefixes in self.entity_hierarchies.items():
            matching_sources = []
            for source in sources:
                entity_id = source['id']
                if any(entity_id.startswith(prefix) for prefix in prefixes):
                    matching_sources.append(source)
            
            if matching_sources:
                # Return the highest confidence ID from the appropriate namespace
                best_match = max(matching_sources, key=lambda x: x['confidence'])
                return best_match['id']
        
        # Fallback: return highest confidence source
        return sources[0]['id']
    
    async def _deduplicate_biomedical_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Advanced deduplication for biomedical results using content similarity
        """
        if len(results) <= 1:
            return results
        
        # Extract text content for similarity comparison
        result_texts = []
        for result in results:
            text_parts = []
            
            # Extract relevant text from node bindings
            node_bindings = result.get('node_bindings', {})
            for node_id, bindings in node_bindings.items():
                if isinstance(bindings, list):
                    for binding in bindings:
                        if 'id' in binding:
                            text_parts.append(binding['id'])
                        if 'name' in binding:
                            text_parts.append(binding['name'])
            
            # Extract text from edge bindings
            edge_bindings = result.get('edge_bindings', {})
            for edge_id, bindings in edge_bindings.items():
                if isinstance(bindings, list):
                    for binding in bindings:
                        if 'id' in binding:
                            text_parts.append(binding['id'])
            
            result_texts.append(' '.join(text_parts))
        
        # Calculate similarity matrix
        if len(result_texts) > 1 and any(text.strip() for text in result_texts):
            try:
                tfidf_matrix = self.vectorizer.fit_transform(result_texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)
            except:
                # Fallback to simple deduplication
                return self._simple_deduplicate_results(results)
        else:
            return self._simple_deduplicate_results(results)
        
        # Group similar results
        similarity_threshold = 0.8
        groups = []
        used_indices = set()
        
        for i in range(len(results)):
            if i in used_indices:
                continue
                
            group = [i]
            used_indices.add(i)
            
            for j in range(i + 1, len(results)):
                if j in used_indices:
                    continue
                    
                if similarity_matrix[i][j] > similarity_threshold:
                    group.append(j)
                    used_indices.add(j)
            
            groups.append(group)
        
        # Select best representative from each group
        deduplicated = []
        for group in groups:
            # Select result with highest combined score (original score + source confidence)
            best_result = None
            best_score = -1
            
            for idx in group:
                result = results[idx]
                original_score = result.get('score', 0)
                source_confidence = result.get('source_confidence', 0)
                combined_score = original_score * 0.7 + source_confidence * 0.3
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_result = result
            
            if best_result:
                best_result['deduplication_group_size'] = len(group)
                deduplicated.append(best_result)
        
        logger.info(f"Deduplication: {len(results)} -> {len(deduplicated)} results")
        return deduplicated
    
    def _simple_deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple deduplication based on exact matching"""
        seen_hashes = set()
        deduplicated = []
        
        for result in results:
            # Create hash from key result properties
            result_key = json.dumps({
                k: v for k, v in result.items() 
                if k not in ['source_thought', 'source_confidence', 'original_index']
            }, sort_keys=True)
            result_hash = hashlib.md5(result_key.encode()).hexdigest()
            
            if result_hash not in seen_hashes:
                seen_hashes.add(result_hash)
                deduplicated.append(result)
        
        return deduplicated
    
    async def _rank_biomedical_results(self, results: List[Dict[str, Any]], 
                                     context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rank biomedical results using multiple quality factors
        """
        if not results:
            return results
        
        query = context.get('query', '').lower()
        query_tokens = set(query.split())
        
        for result in results:
            scores = {}
            
            # Original score from BTE API
            original_score = result.get('score', 0)
            scores['api_score'] = original_score
            
            # Source confidence score
            source_confidence = result.get('source_confidence', 0)
            scores['source_confidence'] = source_confidence
            
            # Query relevance score
            relevance_score = self._calculate_query_relevance(result, query_tokens)
            scores['relevance'] = relevance_score
            
            # Biomedical quality score
            biomedical_score = self._calculate_biomedical_quality(result)
            scores['biomedical_quality'] = biomedical_score
            
            # Edge diversity score (more edges = more comprehensive)
            edge_count = len(result.get('edge_bindings', {}))
            scores['edge_diversity'] = min(1.0, edge_count / 5.0)
            
            # Calculate weighted final score
            final_score = (
                scores['api_score'] * 0.3 +
                scores['source_confidence'] * 0.2 +
                scores['relevance'] * 0.2 +
                scores['biomedical_quality'] * 0.2 +
                scores['edge_diversity'] * 0.1
            )
            
            result['ranking_scores'] = scores
            result['final_ranking_score'] = final_score
        
        # Sort by final ranking score
        ranked_results = sorted(results, key=lambda x: x['final_ranking_score'], reverse=True)
        
        logger.info(f"Ranked {len(ranked_results)} results, top score: "
                   f"{ranked_results[0]['final_ranking_score']:.3f}")
        
        return ranked_results
    
    def _calculate_query_relevance(self, result: Dict[str, Any], query_tokens: Set[str]) -> float:
        """Calculate how relevant a result is to the original query"""
        relevance_score = 0.0
        
        # Check node bindings for query term matches
        node_bindings = result.get('node_bindings', {})
        for node_id, bindings in node_bindings.items():
            if isinstance(bindings, list):
                for binding in bindings:
                    name = binding.get('name', '').lower()
                    if name:
                        name_tokens = set(name.split())
                        overlap = len(name_tokens.intersection(query_tokens))
                        if overlap > 0:
                            relevance_score += overlap / len(query_tokens)
        
        return min(1.0, relevance_score)
    
    def _calculate_biomedical_quality(self, result: Dict[str, Any]) -> float:
        """Calculate biomedical-specific quality score"""
        quality_score = 0.0
        
        # Check for high-quality biomedical namespaces
        high_quality_namespaces = {
            'NCBIGene', 'HGNC', 'UniProtKB', 'CHEBI', 'MONDO', 'HP', 'GO', 'REACTOME'
        }
        
        node_bindings = result.get('node_bindings', {})
        namespace_count = 0
        quality_namespace_count = 0
        
        for node_id, bindings in node_bindings.items():
            if isinstance(bindings, list):
                for binding in bindings:
                    entity_id = binding.get('id', '')
                    if ':' in entity_id:
                        namespace = entity_id.split(':')[0]
                        namespace_count += 1
                        if namespace in high_quality_namespaces:
                            quality_namespace_count += 1
        
        if namespace_count > 0:
            quality_score = quality_namespace_count / namespace_count
        
        # Bonus for having multiple evidence edges
        edge_count = len(result.get('edge_bindings', {}))
        if edge_count > 2:
            quality_score += 0.2
        
        return min(1.0, quality_score)
    
    async def _iteratively_refine_results(self, results: List[Dict[str, Any]], 
                                        context: Dict[str, Any],
                                        source_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Iteratively refine results using feedback loops
        """
        current_results = results.copy()
        
        for iteration in range(self.max_refinement_iterations):
            logger.info(f"Refinement iteration {iteration + 1}/{self.max_refinement_iterations}")
            
            # Analyze current results for potential improvements
            refinement_needed = self._analyze_refinement_needs(current_results, context)
            
            if not refinement_needed:
                logger.info("No refinement needed, stopping iterations")
                break
            
            # Apply refinements
            refined_results = await self._apply_refinements(current_results, refinement_needed, context)
            
            # Check if refinement improved the results
            if self._evaluate_refinement_improvement(current_results, refined_results):
                current_results = refined_results
                logger.info(f"Refinement iteration {iteration + 1} improved results")
            else:
                logger.info(f"Refinement iteration {iteration + 1} did not improve results, stopping")
                break
        
        return current_results
    
    def _analyze_refinement_needs(self, results: List[Dict[str, Any]], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze what refinements might be needed"""
        needs = {
            'low_confidence_results': [],
            'missing_entity_types': set(),
            'sparse_evidence': [],
            'query_mismatch': []
        }
        
        query = context.get('query', '').lower()
        query_tokens = set(query.split())
        
        for i, result in enumerate(results):
            # Check for low confidence
            if result.get('final_ranking_score', 0) < 0.5:
                needs['low_confidence_results'].append(i)
            
            # Check for sparse evidence
            edge_count = len(result.get('edge_bindings', {}))
            if edge_count < 2:
                needs['sparse_evidence'].append(i)
            
            # Check for query mismatch
            relevance = result.get('ranking_scores', {}).get('relevance', 0)
            if relevance < 0.3:
                needs['query_mismatch'].append(i)
        
        return needs
    
    async def _apply_refinements(self, results: List[Dict[str, Any]], 
                               refinement_needs: Dict[str, Any],
                               context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply specific refinements based on identified needs"""
        refined_results = results.copy()
        
        # For now, implement basic refinement by re-ranking with adjusted weights
        # In a full implementation, this could trigger additional API calls or
        # use different aggregation strategies
        
        for i in refinement_needs.get('low_confidence_results', []):
            if i < len(refined_results):
                # Boost results that might have been undervalued
                result = refined_results[i]
                biomedical_quality = result.get('ranking_scores', {}).get('biomedical_quality', 0)
                if biomedical_quality > 0.7:
                    # This is actually a high-quality result, boost its score
                    result['final_ranking_score'] *= 1.2
        
        # Re-sort after refinements
        refined_results.sort(key=lambda x: x['final_ranking_score'], reverse=True)
        
        return refined_results
    
    def _evaluate_refinement_improvement(self, original: List[Dict[str, Any]], 
                                       refined: List[Dict[str, Any]]) -> bool:
        """Evaluate if refinement actually improved the results"""
        if not original or not refined:
            return len(refined) > len(original)
        
        original_avg_score = np.mean([r.get('final_ranking_score', 0) for r in original[:5]])
        refined_avg_score = np.mean([r.get('final_ranking_score', 0) for r in refined[:5]])
        
        return refined_avg_score > original_avg_score * 1.05  # 5% improvement threshold
    
    def _calculate_entity_aggregation_score(self, entities: Dict[str, str], 
                                          thoughts: List[GoTThought],
                                          context: Dict[str, Any]) -> AggregationScore:
        """Calculate comprehensive score for entity aggregation"""
        score = AggregationScore()
        
        # Confidence score: weighted average of source confidences
        if thoughts:
            total_weight = sum(len(t.content.get("entities", {})) for t in thoughts)
            if total_weight > 0:
                weighted_confidence = sum(
                    t.confidence * len(t.content.get("entities", {})) for t in thoughts
                ) / total_weight
                score.confidence_score = weighted_confidence
        
        # Diversity score: based on entity type diversity
        entity_types = set()
        for entity_id in entities.values():
            if ':' in entity_id:
                namespace = entity_id.split(':')[0]
                entity_types.add(namespace)
        score.diversity_score = min(1.0, len(entity_types) / 10.0)
        
        # Consistency score: how often entities appear across sources
        entity_counts = Counter()
        for thought in thoughts:
            for entity in thought.content.get("entities", {}):
                entity_counts[entity] += 1
        
        if entity_counts:
            max_count = max(entity_counts.values())
            consistency_scores = [count / max_count for count in entity_counts.values()]
            score.consistency_score = np.mean(consistency_scores)
        
        # Overall score
        score.overall_score = (
            score.confidence_score * self.quality_weights['confidence'] +
            score.diversity_score * self.quality_weights['diversity'] +
            score.consistency_score * self.quality_weights['consistency']
        )
        
        return score
    
    def _calculate_result_aggregation_score(self, results: List[Dict[str, Any]], 
                                          thoughts: List[GoTThought],
                                          context: Dict[str, Any]) -> AggregationScore:
        """Calculate comprehensive score for result aggregation"""
        score = AggregationScore()
        
        if not results:
            return score
        
        # Confidence score: average of top results
        top_results = results[:min(5, len(results))]
        score.confidence_score = np.mean([r.get('final_ranking_score', 0) for r in top_results])
        
        # Diversity score: based on result diversity
        unique_node_types = set()
        for result in top_results:
            node_bindings = result.get('node_bindings', {})
            unique_node_types.update(node_bindings.keys())
        score.diversity_score = min(1.0, len(unique_node_types) / 5.0)
        
        # Consistency score: how consistent are the scores across results
        scores = [r.get('final_ranking_score', 0) for r in top_results]
        if scores:
            score_std = np.std(scores)
            score.consistency_score = max(0, 1.0 - score_std)
        
        # Relevance score: average relevance to query
        relevance_scores = [
            r.get('ranking_scores', {}).get('relevance', 0) for r in top_results
        ]
        score.relevance_score = np.mean(relevance_scores) if relevance_scores else 0
        
        # Biomedical score: average biomedical quality
        biomedical_scores = [
            r.get('ranking_scores', {}).get('biomedical_quality', 0) for r in top_results
        ]
        score.biomedical_score = np.mean(biomedical_scores) if biomedical_scores else 0
        
        # Overall score
        score.overall_score = (
            score.confidence_score * self.quality_weights['confidence'] +
            score.diversity_score * self.quality_weights['diversity'] +
            score.consistency_score * self.quality_weights['consistency'] +
            score.relevance_score * self.quality_weights['relevance'] +
            score.biomedical_score * self.quality_weights['biomedical']
        )
        
        return score
    
    def _generate_comprehensive_answer(self, results: List[Dict[str, Any]], 
                                     context: Dict[str, Any]) -> str:
        """Generate a comprehensive final answer from aggregated results"""
        if not results:
            return "No results found after aggregation and refinement."
        
        query = context.get('query', '')
        answer_parts = []
        
        # Summary statistics
        answer_parts.append(f"Found {len(results)} high-quality results for: '{query}'")
        
        # Top result details
        if results:
            top_result = results[0]
            top_score = top_result.get('final_ranking_score', 0)
            answer_parts.append(f"Top result scored {top_score:.3f} out of 1.0.")
            
            # Extract key entities from top result
            node_bindings = top_result.get('node_bindings', {})
            if node_bindings:
                entities = []
                for node_id, bindings in node_bindings.items():
                    if isinstance(bindings, list) and bindings:
                        binding = bindings[0]
                        name = binding.get('name', binding.get('id', ''))
                        entities.append(name)
                
                if entities:
                    answer_parts.append(f"Key entities involved: {', '.join(entities[:3])}")
        
        # Quality assessment
        avg_score = np.mean([r.get('final_ranking_score', 0) for r in results[:5]])
        if avg_score > 0.7:
            quality = "high"
        elif avg_score > 0.5:
            quality = "moderate"
        else:
            quality = "low"
        
        answer_parts.append(f"Overall result quality: {quality} (avg score: {avg_score:.3f})")
        
        # Aggregation summary
        dedup_stats = context.get('deduplication_stats', {})
        if dedup_stats:
            original_count = dedup_stats.get('original_count', 0)
            final_count = dedup_stats.get('final_count', len(results))
            if original_count > final_count:
                answer_parts.append(f"Results refined from {original_count} to {final_count} entries.")
        
        return " ".join(answer_parts)


class IterativeRefinementEngine:
    """
    Engine for iterative query refinement with feedback loops
    """
    
    def __init__(self, max_iterations: int = 3, improvement_threshold: float = 0.1):
        """
        Initialize the refinement engine
        
        Args:
            max_iterations: Maximum refinement iterations
            improvement_threshold: Minimum improvement required to continue
        """
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.aggregator = BiomedicalAggregator()
    
    async def refine_query_execution(self, initial_result: OptimizationResult, 
                                   context: Dict[str, Any]) -> OptimizationResult:
        """
        Iteratively refine a query execution result
        
        Args:
            initial_result: Initial optimization result
            context: Execution context with query and parameters
            
        Returns:
            Refined optimization result
        """
        current_result = initial_result
        iteration = 0
        
        while iteration < self.max_iterations:
            logger.info(f"Refinement iteration {iteration + 1}/{self.max_iterations}")
            
            # Analyze current result for improvement opportunities
            refinement_plan = await self._analyze_result_quality(current_result, context)
            
            if not refinement_plan.get('needs_refinement', False):
                logger.info("Result quality is satisfactory, no refinement needed")
                break
            
            # Apply refinements
            try:
                refined_result = await self._apply_result_refinements(
                    current_result, refinement_plan, context
                )
                
                # Evaluate improvement
                improvement = self._calculate_improvement(current_result, refined_result)
                
                if improvement >= self.improvement_threshold:
                    current_result = refined_result
                    current_result.reasoning_chain.append(
                        f"Refinement iteration {iteration + 1} improved quality by {improvement:.3f}"
                    )
                    logger.info(f"Iteration {iteration + 1} improved result by {improvement:.3f}")
                else:
                    logger.info(f"Iteration {iteration + 1} improvement {improvement:.3f} "
                              f"below threshold {self.improvement_threshold}")
                    break
                    
            except Exception as e:
                logger.warning(f"Refinement iteration {iteration + 1} failed: {e}")
                break
            
            iteration += 1
        
        # Add final refinement summary
        current_result.reasoning_chain.append(
            f"Iterative refinement completed after {iteration} iterations"
        )
        
        return current_result
    
    async def _analyze_result_quality(self, result: OptimizationResult, 
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze result quality to determine refinement needs"""
        analysis = {
            'needs_refinement': False,
            'issues': [],
            'refinement_strategies': []
        }
        
        # Check result count
        if len(result.results) < 5:
            analysis['issues'].append('low_result_count')
            analysis['refinement_strategies'].append('expand_search')
        
        # Check quality score
        if result.metrics.quality_score < 0.6:
            analysis['issues'].append('low_quality_score')
            analysis['refinement_strategies'].append('improve_ranking')
        
        # Check entity coverage
        if len(result.entities) < 2:
            analysis['issues'].append('low_entity_coverage')
            analysis['refinement_strategies'].append('enhance_entities')
        
        # Set refinement flag
        analysis['needs_refinement'] = len(analysis['issues']) > 0
        
        return analysis
    
    async def _apply_result_refinements(self, result: OptimizationResult, 
                                      refinement_plan: Dict[str, Any],
                                      context: Dict[str, Any]) -> OptimizationResult:
        """Apply specific refinements to improve result quality"""
        refined_result = result  # Start with current result
        
        for strategy in refinement_plan.get('refinement_strategies', []):
            if strategy == 'improve_ranking':
                refined_result = await self._improve_result_ranking(refined_result, context)
            elif strategy == 'expand_search':
                refined_result = await self._expand_search_results(refined_result, context)
            elif strategy == 'enhance_entities':
                refined_result = await self._enhance_entity_extraction(refined_result, context)
        
        return refined_result
    
    async def _improve_result_ranking(self, result: OptimizationResult, 
                                    context: Dict[str, Any]) -> OptimizationResult:
        """Improve result ranking using advanced scoring"""
        if not result.results:
            return result
        
        # Re-rank results using biomedical aggregator
        ranked_results = await self.aggregator._rank_biomedical_results(
            result.results, context
        )
        
        # Update result
        new_result = result
        new_result.results = ranked_results
        
        # Recalculate quality score
        if ranked_results:
            avg_score = np.mean([r.get('final_ranking_score', 0) for r in ranked_results[:5]])
            new_result.metrics.quality_score = min(1.0, avg_score)
        
        return new_result
    
    async def _expand_search_results(self, result: OptimizationResult, 
                                   context: Dict[str, Any]) -> OptimizationResult:
        """Expand search results by adjusting parameters"""
        # For now, this is a placeholder - in a full implementation,
        # this would trigger additional API calls with modified parameters
        return result
    
    async def _enhance_entity_extraction(self, result: OptimizationResult, 
                                       context: Dict[str, Any]) -> OptimizationResult:
        """Enhance entity extraction with additional techniques"""
        query = context.get('query', '')
        
        try:
            # Re-extract entities with enhanced focus
            ner_response = call_mcp_tool("bio_ner", query=query)
            enhanced_entities = ner_response.get("entities", {})
            
            # Merge with existing entities
            combined_entities = {**result.entities, **enhanced_entities}
            
            new_result = result
            new_result.entities = combined_entities
            new_result.metrics.entities_found = len(combined_entities)
            
            return new_result
            
        except Exception as e:
            logger.warning(f"Failed to enhance entity extraction: {e}")
            return result
    
    async def refine_results(self, results: List[Dict[str, Any]], 
                           entities: List[Dict[str, Any]], 
                           max_iterations: int = 3) -> List[Dict[str, Any]]:
        """
        Simple result refinement for production use
        
        Args:
            results: List of results to refine
            entities: List of entities for context
            max_iterations: Maximum iterations (not used in this simple version)
            
        Returns:
            Refined results list
        """
        if not results:
            return results
        
        try:
            # Apply biomedical aggregation improvements
            context = {
                'entities': entities,
                'query': 'refinement context'
            }
            
            # Use the aggregator to re-rank and improve results
            improved_results = await self.aggregator._rank_biomedical_results(results, context)
            
            logger.info(f"Refined {len(results)} results, top score: {improved_results[0].get('final_ranking_score', 0):.3f}" if improved_results else "No results to refine")
            
            return improved_results
            
        except Exception as e:
            logger.warning(f"Result refinement failed: {e}")
            return results
    
    def _calculate_improvement(self, original: OptimizationResult,
                             refined: OptimizationResult) -> float:
        """Calculate improvement between original and refined results"""
        original_score = original.metrics.quality_score
        refined_score = refined.metrics.quality_score
        
        if original_score == 0:
            return 1.0 if refined_score > 0 else 0.0
        
        return (refined_score - original_score) / original_score