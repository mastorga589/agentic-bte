"""
Tests for UnifiedBiomedicalAgent

This module contains comprehensive tests for the main unified interface
for biomedical query processing.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from agentic_bte.unified.agent import (
    UnifiedBiomedicalAgent, QueryRequest, QueryResponse, BatchQueryRequest,
    BatchQueryResponse, QueryMode, ProcessingStage
)
from agentic_bte.unified.config import UnifiedConfig
from agentic_bte.unified.types import (
    BiomedicalResult, Entity, EntityType, QueryComplexity
)
from agentic_bte.unified.strategy_router import StrategyType


@pytest.fixture
async def agent():
    """Create a UnifiedBiomedicalAgent for testing"""
    config = UnifiedConfig()
    agent = UnifiedBiomedicalAgent(
        config=config,
        enable_caching=True,
        enable_parallel=True
    )
    
    # Mock the component initializations to avoid actual API calls
    agent.entity_processor.initialize = AsyncMock()
    agent.strategy_router.initialize = AsyncMock()
    agent.execution_engine.initialize = AsyncMock()
    agent.knowledge_manager.initialize = AsyncMock()
    
    yield agent
    
    # Cleanup
    await agent.shutdown()


@pytest.fixture
def sample_entities():
    """Sample entities for testing"""
    return [
        Entity(
            entity_id="MONDO:0005015",
            name="diabetes",
            entity_type=EntityType.DISEASE,
            synonyms=["diabetes mellitus"],
            confidence=0.95
        ),
        Entity(
            entity_id="CHEBI:6801",
            name="metformin",
            entity_type=EntityType.SMALL_MOLECULE,
            synonyms=["glucophage"],
            confidence=0.88
        )
    ]


@pytest.fixture
def sample_results():
    """Sample biomedical results for testing"""
    return [
        BiomedicalResult(
            result_id="result1",
            subject_entity="diabetes",
            predicate="biolink:treats",
            object_entity="metformin",
            confidence=0.9,
            evidence_count=15,
            source_databases=["ChEMBL", "DrugBank"]
        ),
        BiomedicalResult(
            result_id="result2", 
            subject_entity="metformin",
            predicate="biolink:affects",
            object_entity="blood glucose",
            confidence=0.85,
            evidence_count=20,
            source_databases=["PubMed", "CTD"]
        )
    ]


class TestUnifiedBiomedicalAgent:
    """Test cases for UnifiedBiomedicalAgent"""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test agent initialization"""
        agent = UnifiedBiomedicalAgent()
        
        # Should not be initialized yet
        assert not agent._initialized
        assert agent.enable_caching
        assert agent.enable_parallel
        
        # Check components exist
        assert agent.entity_processor is not None
        assert agent.strategy_router is not None
        assert agent.execution_engine is not None
        assert agent.knowledge_manager is not None
        assert agent.parallel_executor is not None
    
    @pytest.mark.asyncio
    async def test_initialization_without_parallel(self):
        """Test agent initialization without parallel execution"""
        agent = UnifiedBiomedicalAgent(enable_parallel=False)
        
        assert agent.parallel_executor is None
        assert not agent.enable_parallel
    
    @pytest.mark.asyncio
    async def test_component_initialization(self, agent):
        """Test component initialization"""
        await agent.initialize()
        
        assert agent._initialized
        agent.entity_processor.initialize.assert_called_once()
        agent.strategy_router.initialize.assert_called_once()
        agent.execution_engine.initialize.assert_called_once()
        agent.knowledge_manager.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_simple_query(self, agent, sample_entities, sample_results):
        """Test processing a simple query"""
        # Mock component methods
        agent.entity_processor.extract_entities = AsyncMock(return_value=sample_entities)
        agent.strategy_router.select_strategy = AsyncMock(return_value=StrategyType.COMPREHENSIVE)
        agent.knowledge_manager.build_trapi_queries = AsyncMock(return_value=[])
        agent.knowledge_manager.score_results = AsyncMock(return_value=sample_results)
        agent.execution_engine.execute = AsyncMock(return_value=sample_results)
        
        response = await agent.process_query(
            text="What drugs treat diabetes?",
            query_mode=QueryMode.FAST,
            max_results=10
        )
        
        assert isinstance(response, QueryResponse)
        assert response.total_results == 2
        assert len(response.results) == 2
        assert response.strategy_used == StrategyType.COMPREHENSIVE
        assert ProcessingStage.COMPLETED in response.stages_completed
        assert response.error is None
    
    @pytest.mark.asyncio
    async def test_query_caching(self, agent, sample_entities, sample_results):
        """Test query result caching"""
        # Mock component methods
        agent.entity_processor.extract_entities = AsyncMock(return_value=sample_entities)
        agent.strategy_router.select_strategy = AsyncMock(return_value=StrategyType.COMPREHENSIVE)
        agent.knowledge_manager.build_trapi_queries = AsyncMock(return_value=[])
        agent.knowledge_manager.score_results = AsyncMock(return_value=sample_results)
        agent.execution_engine.execute = AsyncMock(return_value=sample_results)
        
        # First query
        response1 = await agent.process_query("What drugs treat diabetes?")
        assert not response1.cached
        
        # Second identical query should be cached
        response2 = await agent.process_query("What drugs treat diabetes?")
        assert response2.cached
        
        # Verify entity processor was only called once
        agent.entity_processor.extract_entities.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_with_error(self, agent):
        """Test query processing with error"""
        # Mock entity processor to raise an error
        agent.entity_processor.extract_entities = AsyncMock(
            side_effect=Exception("Entity extraction failed")
        )
        
        response = await agent.process_query("What drugs treat diabetes?")
        
        assert response.error is not None
        assert "Entity extraction failed" in response.error
        assert ProcessingStage.FAILED in response.stages_completed
        assert response.total_results == 0
    
    @pytest.mark.asyncio
    async def test_confidence_threshold_filtering(self, agent, sample_entities):
        """Test filtering results by confidence threshold"""
        # Create results with different confidence scores
        results = [
            BiomedicalResult(
                result_id="high_conf",
                subject_entity="diabetes",
                predicate="biolink:treats",
                object_entity="metformin",
                confidence=0.9
            ),
            BiomedicalResult(
                result_id="low_conf",
                subject_entity="diabetes", 
                predicate="biolink:treats",
                object_entity="aspirin",
                confidence=0.3
            )
        ]
        
        # Mock component methods
        agent.entity_processor.extract_entities = AsyncMock(return_value=sample_entities)
        agent.strategy_router.select_strategy = AsyncMock(return_value=StrategyType.COMPREHENSIVE)
        agent.knowledge_manager.build_trapi_queries = AsyncMock(return_value=[])
        agent.knowledge_manager.score_results = AsyncMock(return_value=results)
        agent.execution_engine.execute = AsyncMock(return_value=results)
        
        response = await agent.process_query(
            "What drugs treat diabetes?",
            confidence_threshold=0.5
        )
        
        # Should only return high confidence result
        assert response.total_results == 1
        assert response.results[0].result_id == "high_conf"
    
    @pytest.mark.asyncio
    async def test_max_results_limiting(self, agent, sample_entities):
        """Test limiting number of results"""
        # Create many results
        results = []
        for i in range(10):
            results.append(BiomedicalResult(
                result_id=f"result_{i}",
                subject_entity="diabetes",
                predicate="biolink:treats",
                object_entity=f"drug_{i}",
                confidence=0.8
            ))
        
        # Mock component methods
        agent.entity_processor.extract_entities = AsyncMock(return_value=sample_entities)
        agent.strategy_router.select_strategy = AsyncMock(return_value=StrategyType.COMPREHENSIVE)
        agent.knowledge_manager.build_trapi_queries = AsyncMock(return_value=[])
        agent.knowledge_manager.score_results = AsyncMock(return_value=results)
        agent.execution_engine.execute = AsyncMock(return_value=results)
        
        response = await agent.process_query(
            "What drugs treat diabetes?",
            max_results=5
        )
        
        # Should only return 5 results
        assert response.total_results == 5
        assert len(response.results) == 5
    
    @pytest.mark.asyncio
    async def test_query_complexity_assessment(self, agent):
        """Test query complexity assessment"""
        # Simple query
        simple_entities = [Entity(
            entity_id="test", name="test", entity_type=EntityType.DISEASE, confidence=0.8
        )]
        simple_request = QueryRequest(
            query_id="test", text="test", query_mode=QueryMode.FAST, max_results=10
        )
        
        complexity = agent._assess_query_complexity(simple_request, simple_entities)
        assert complexity == QueryComplexity.LOW
        
        # Complex query
        complex_entities = []
        for i in range(6):
            complex_entities.append(Entity(
                entity_id=f"entity_{i}",
                name=f"entity_{i}",
                entity_type=EntityType.DISEASE if i % 2 == 0 else EntityType.SMALL_MOLECULE,
                confidence=0.8
            ))
        
        complex_request = QueryRequest(
            query_id="test",
            text="This is a very long and complex biomedical query that involves multiple entities and relationships across different biological domains and should be classified as high complexity",
            query_mode=QueryMode.COMPREHENSIVE,
            max_results=1000
        )
        
        complexity = agent._assess_query_complexity(complex_request, complex_entities)
        assert complexity == QueryComplexity.HIGH
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, agent, sample_entities, sample_results):
        """Test batch query processing"""
        # Mock component methods
        agent.entity_processor.extract_entities = AsyncMock(return_value=sample_entities)
        agent.strategy_router.select_strategy = AsyncMock(return_value=StrategyType.COMPREHENSIVE)
        agent.knowledge_manager.build_trapi_queries = AsyncMock(return_value=[])
        agent.knowledge_manager.score_results = AsyncMock(return_value=sample_results)
        agent.execution_engine.execute = AsyncMock(return_value=sample_results)
        
        queries = [
            "What drugs treat diabetes?",
            "What are the side effects of metformin?",
            "How does insulin work?"
        ]
        
        batch_response = await agent.process_batch(
            queries=queries,
            max_concurrency=2
        )
        
        assert isinstance(batch_response, BatchQueryResponse)
        assert len(batch_response.responses) == 3
        assert batch_response.successful_queries == 3
        assert batch_response.failed_queries == 0
        assert batch_response.average_response_time > 0
    
    @pytest.mark.asyncio
    async def test_batch_with_query_objects(self, agent, sample_entities, sample_results):
        """Test batch processing with QueryRequest objects"""
        # Mock component methods
        agent.entity_processor.extract_entities = AsyncMock(return_value=sample_entities)
        agent.strategy_router.select_strategy = AsyncMock(return_value=StrategyType.COMPREHENSIVE)
        agent.knowledge_manager.build_trapi_queries = AsyncMock(return_value=[])
        agent.knowledge_manager.score_results = AsyncMock(return_value=sample_results)
        agent.execution_engine.execute = AsyncMock(return_value=sample_results)
        
        queries = [
            QueryRequest(query_id="q1", text="What drugs treat diabetes?", query_mode=QueryMode.FAST),
            QueryRequest(query_id="q2", text="Side effects of metformin", query_mode=QueryMode.BALANCED)
        ]
        
        batch_response = await agent.process_batch(queries=queries)
        
        assert len(batch_response.responses) == 2
        assert batch_response.responses[0].request.query_mode == QueryMode.FAST
        assert batch_response.responses[1].request.query_mode == QueryMode.BALANCED
    
    @pytest.mark.asyncio
    async def test_batch_fail_fast(self, agent):
        """Test batch processing with fail_fast option"""
        # Mock first query to succeed, second to fail
        call_count = 0
        
        async def mock_process_single_query(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First query succeeds
                return QueryResponse(
                    query_id=request.query_id,
                    request=request,
                    results=[],
                    total_results=0,
                    processing_time=0.1,
                    strategy_used=StrategyType.COMPREHENSIVE,
                    stages_completed=[ProcessingStage.COMPLETED],
                    performance_metrics={}
                )
            else:
                # Second query fails
                return QueryResponse(
                    query_id=request.query_id,
                    request=request,
                    results=[],
                    total_results=0,
                    processing_time=0.1,
                    strategy_used=StrategyType.COMPREHENSIVE,
                    stages_completed=[ProcessingStage.FAILED],
                    performance_metrics={},
                    error="Simulated failure"
                )
        
        agent._process_single_query = mock_process_single_query
        
        queries = ["Query 1", "Query 2", "Query 3"]
        
        batch_response = await agent.process_batch(
            queries=queries,
            fail_fast=True
        )
        
        # Should stop after second query fails
        assert len(batch_response.responses) == 2
        assert batch_response.successful_queries == 1
        assert batch_response.failed_queries == 1
    
    def test_cache_key_generation(self, agent):
        """Test cache key generation"""
        request1 = QueryRequest(query_id="1", text="diabetes treatment")
        request2 = QueryRequest(query_id="2", text="diabetes treatment")  # Same text
        request3 = QueryRequest(query_id="3", text="diabetes symptoms")  # Different text
        
        key1 = agent._get_cache_key(request1)
        key2 = agent._get_cache_key(request2)
        key3 = agent._get_cache_key(request3)
        
        assert key1 == key2  # Same query text should have same key
        assert key1 != key3  # Different query text should have different key
    
    def test_get_active_queries(self, agent):
        """Test getting active queries"""
        # Initially no active queries
        active = agent.get_active_queries()
        assert len(active) == 0
        
        # Add a mock active query
        mock_response = QueryResponse(
            query_id="test",
            request=QueryRequest(query_id="test", text="test"),
            results=[],
            total_results=0,
            processing_time=0.0,
            strategy_used=StrategyType.COMPREHENSIVE,
            stages_completed=[],
            performance_metrics={}
        )
        agent._active_queries["test"] = mock_response
        
        active = agent.get_active_queries()
        assert len(active) == 1
        assert "test" in active
    
    def test_get_query_history(self, agent):
        """Test getting query history"""
        # Add some mock history
        for i in range(5):
            response = QueryResponse(
                query_id=f"query_{i}",
                request=QueryRequest(query_id=f"query_{i}", text=f"test {i}"),
                results=[],
                total_results=0,
                processing_time=0.1,
                strategy_used=StrategyType.COMPREHENSIVE,
                stages_completed=[ProcessingStage.COMPLETED],
                performance_metrics={},
                error="error" if i == 2 else None
            )
            agent._query_history.append(response)
        
        # Get all history
        history = agent.get_query_history()
        assert len(history) == 5
        
        # Get limited history
        history = agent.get_query_history(limit=3)
        assert len(history) == 3
        
        # Get history without errors
        history = agent.get_query_history(include_errors=False)
        assert len(history) == 4  # 5 total - 1 error
    
    def test_performance_summary(self, agent):
        """Test performance summary generation"""
        # No queries processed yet
        summary = agent.get_performance_summary()
        assert "message" in summary
        
        # Add some mock history
        for i in range(10):
            response = QueryResponse(
                query_id=f"query_{i}",
                request=QueryRequest(query_id=f"query_{i}", text=f"test {i}"),
                results=[],
                total_results=0,
                processing_time=0.1 * (i + 1),
                strategy_used=StrategyType.COMPREHENSIVE if i % 2 == 0 else StrategyType.OPTIMIZED,
                stages_completed=[ProcessingStage.COMPLETED],
                performance_metrics={},
                error="error" if i == 7 else None,
                cached=i < 3
            )
            agent._query_history.append(response)
        
        summary = agent.get_performance_summary()
        
        assert summary["total_queries"] == 10
        assert summary["successful_queries"] == 9
        assert summary["failed_queries"] == 1
        assert summary["success_rate"] == 0.9
        assert summary["cache_hit_rate"] == 0.3
        assert "comprehensive" in summary["strategy_usage"]
        assert "optimized" in summary["strategy_usage"]
    
    def test_clear_cache(self, agent):
        """Test clearing cache"""
        # Add some mock cache entries
        for i in range(5):
            agent._cache[f"key_{i}"] = f"value_{i}"
        
        assert len(agent._cache) == 5
        
        cleared = agent.clear_cache()
        assert cleared == 5
        assert len(agent._cache) == 0
    
    def test_clear_history(self, agent):
        """Test clearing history"""
        # Add some mock history
        for i in range(5):
            response = QueryResponse(
                query_id=f"query_{i}",
                request=QueryRequest(query_id=f"query_{i}", text=f"test {i}"),
                results=[],
                total_results=0,
                processing_time=0.1,
                strategy_used=StrategyType.COMPREHENSIVE,
                stages_completed=[],
                performance_metrics={}
            )
            agent._query_history.append(response)
        
        assert len(agent._query_history) == 5
        
        cleared = agent.clear_history()
        assert cleared == 5
        assert len(agent._query_history) == 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, agent):
        """Test health check"""
        await agent.initialize()
        
        health = await agent.health_check()
        
        assert health["initialized"]
        assert health["active_queries"] == 0
        assert health["cache_size"] == 0
        assert health["history_size"] == 0
        assert "components" in health
        assert health["components"]["entity_processor"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, agent, sample_entities, sample_results):
        """Test parallel execution mode"""
        from agentic_bte.unified.knowledge_manager import TRAPIQuery
        
        # Mock TRAPI queries
        mock_queries = [
            TRAPIQuery(
                query_graph={}, 
                query_id="q1", 
                predicate="biolink:treats", 
                entities=["diabetes", "drug"]
            ),
            TRAPIQuery(
                query_graph={}, 
                query_id="q2", 
                predicate="biolink:affects", 
                entities=["drug", "protein"]
            )
        ]
        
        # Mock component methods
        agent.entity_processor.extract_entities = AsyncMock(return_value=sample_entities)
        agent.strategy_router.select_strategy = AsyncMock(return_value=StrategyType.COMPREHENSIVE)
        agent.knowledge_manager.build_trapi_queries = AsyncMock(return_value=mock_queries)
        agent.knowledge_manager.score_results = AsyncMock(return_value=sample_results)
        
        # Mock parallel executor
        agent.parallel_executor.execute_parallel_predicates = AsyncMock(
            return_value=[(mock_queries[0], sample_results), (mock_queries[1], sample_results)]
        )
        
        response = await agent.process_query(
            text="What drugs treat diabetes and how do they work?",
            enable_parallel=True
        )
        
        assert response.parallel_execution
        agent.parallel_executor.execute_parallel_predicates.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_knowledge_integration(self, agent, sample_entities, sample_results):
        """Test knowledge integration in comprehensive mode"""
        # Mock component methods
        agent.entity_processor.extract_entities = AsyncMock(return_value=sample_entities)
        agent.strategy_router.select_strategy = AsyncMock(return_value=StrategyType.COMPREHENSIVE)
        agent.knowledge_manager.build_trapi_queries = AsyncMock(return_value=[])
        agent.knowledge_manager.score_results = AsyncMock(return_value=sample_results)
        agent.execution_engine.execute = AsyncMock(return_value=sample_results)
        agent.knowledge_manager.integrate_results = AsyncMock(return_value=[])
        agent.knowledge_manager.build_knowledge_graph = AsyncMock(return_value={"nodes": [], "edges": []})
        
        response = await agent.process_query(
            text="What drugs treat diabetes?",
            query_mode=QueryMode.COMPREHENSIVE
        )
        
        assert response.knowledge_graph is not None
        agent.knowledge_manager.integrate_results.assert_called_once()
        agent.knowledge_manager.build_knowledge_graph.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown(self, agent):
        """Test agent shutdown"""
        await agent.initialize()
        
        # Add some mock active queries
        agent._active_queries["test"] = "mock_response"
        
        await agent.shutdown()
        
        assert not agent._initialized
        assert len(agent._active_queries) == 0


class TestQueryRequestResponse:
    """Test query request and response data structures"""
    
    def test_query_request_creation(self):
        """Test QueryRequest creation"""
        request = QueryRequest(
            query_id="test_id",
            text="What drugs treat diabetes?",
            query_mode=QueryMode.FAST,
            max_results=50,
            timeout_seconds=60.0
        )
        
        assert request.query_id == "test_id"
        assert request.text == "What drugs treat diabetes?"
        assert request.query_mode == QueryMode.FAST
        assert request.max_results == 50
        assert request.timeout_seconds == 60.0
        assert request.enable_parallel
        assert request.enable_caching
    
    def test_query_response_creation(self):
        """Test QueryResponse creation"""
        request = QueryRequest(query_id="test", text="test query")
        
        response = QueryResponse(
            query_id="test",
            request=request,
            results=[],
            total_results=0,
            processing_time=1.5,
            strategy_used=StrategyType.COMPREHENSIVE,
            stages_completed=[ProcessingStage.COMPLETED],
            performance_metrics={}
        )
        
        assert response.query_id == "test"
        assert response.total_results == 0
        assert response.processing_time == 1.5
        assert response.strategy_used == StrategyType.COMPREHENSIVE
        assert not response.cached
        assert not response.parallel_execution


class TestQueryModes:
    """Test different query modes"""
    
    def test_query_mode_enum(self):
        """Test QueryMode enum values"""
        assert QueryMode.STANDARD.value == "standard"
        assert QueryMode.FAST.value == "fast"
        assert QueryMode.COMPREHENSIVE.value == "comprehensive"
        assert QueryMode.BALANCED.value == "balanced"
        assert QueryMode.EXPERIMENTAL.value == "experimental"


class TestProcessingStages:
    """Test processing stage tracking"""
    
    def test_processing_stage_enum(self):
        """Test ProcessingStage enum values"""
        assert ProcessingStage.INITIALIZED.value == "initialized"
        assert ProcessingStage.ENTITY_EXTRACTION.value == "entity_extraction"
        assert ProcessingStage.STRATEGY_SELECTION.value == "strategy_selection"
        assert ProcessingStage.QUERY_BUILDING.value == "query_building"
        assert ProcessingStage.EXECUTION.value == "execution"
        assert ProcessingStage.RESULT_PROCESSING.value == "result_processing"
        assert ProcessingStage.KNOWLEDGE_INTEGRATION.value == "knowledge_integration"
        assert ProcessingStage.COMPLETED.value == "completed"
        assert ProcessingStage.FAILED.value == "failed"