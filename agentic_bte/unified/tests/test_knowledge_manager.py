"""
Tests for the unified knowledge manager
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from agentic_bte.unified.config import UnifiedConfig
from agentic_bte.unified.types import (
    BiomedicalResult, EntityContext, Entity, EntityType, 
    BiomedicalRelationship, ExecutionContext
)
from agentic_bte.unified.knowledge_manager import (
    UnifiedKnowledgeManager, TRAPIQuery, KnowledgeAssertion,
    KnowledgeEvidence, KnowledgeSource, PredicateRanking
)
from agentic_bte.core.knowledge.predicate_strategy import QueryIntent


@pytest.fixture
def mock_config():
    """Create a mock unified config"""
    config = Mock(spec=UnifiedConfig)
    
    # Quality config
    config.quality = Mock()
    config.quality.evidence_weight = 0.35
    config.quality.predicate_weight = 0.25
    config.quality.source_weight = 0.20
    config.quality.multiplicity_weight = 0.10
    config.quality.study_weight = 0.10
    config.quality.min_results_threshold = 10
    config.quality.fallback_threshold = 5
    
    # Domain config
    config.domain = Mock()
    config.domain.max_predicates_per_query = 4
    config.domain.max_queries_per_strategy = 10
    
    return config


@pytest.fixture
def sample_entity_context():
    """Create sample entity context"""
    entities = [
        Entity(
            name="diabetes",
            entity_id="MONDO:0005015",
            entity_type=EntityType.DISEASE,
            confidence=0.9
        ),
        Entity(
            name="metformin",
            entity_id="CHEBI:6801", 
            entity_type=EntityType.SMALL_MOLECULE,
            confidence=0.8
        )
    ]
    
    return EntityContext(
        query="What drugs treat diabetes?",
        entities=entities,
        confidence=0.85
    )


class TestUnifiedKnowledgeManager:
    """Test the unified knowledge manager"""
    
    @patch('agentic_bte.unified.knowledge_manager.RDFGraphManager')
    @patch('agentic_bte.unified.knowledge_manager.EvidenceScorer')
    @patch('agentic_bte.unified.knowledge_manager.PredicateSelector')
    def test_knowledge_manager_initialization(self, mock_predicate_selector, 
                                            mock_evidence_scorer, mock_rdf_manager, mock_config):
        """Test knowledge manager initialization"""
        manager = UnifiedKnowledgeManager(mock_config)
        
        assert manager.config == mock_config
        assert manager.knowledge_assertions == {}
        assert manager.cached_trapi_queries == {}
        assert manager.query_build_count == 0
        assert manager.evidence_score_count == 0
        assert manager.knowledge_assertion_count == 0
        
        # Verify components were initialized
        mock_rdf_manager.assert_called_once()
        mock_evidence_scorer.assert_called_once()
        mock_predicate_selector.assert_called_once()


class TestKnowledgeDataStructures:
    """Test knowledge data structures"""
    
    def test_trapi_query_creation(self):
        """Test TRAPI query creation"""
        query_graph = {
            "nodes": {"n0": {"categories": ["biolink:Disease"]}},
            "edges": {"e01": {"subject": "n0", "object": "n1"}}
        }
        
        trapi_query = TRAPIQuery(
            query_graph=query_graph,
            query_id="test_query",
            predicate="biolink:treats",
            entities=["diabetes", "metformin"],
            confidence=0.8,
            source_intent=QueryIntent.THERAPEUTIC,
            estimated_results=50
        )
        
        assert trapi_query.query_id == "test_query"
        assert trapi_query.predicate == "biolink:treats"
        assert len(trapi_query.entities) == 2
        assert trapi_query.confidence == 0.8
        assert trapi_query.source_intent == QueryIntent.THERAPEUTIC
        assert trapi_query.estimated_results == 50
    
    def test_knowledge_assertion_creation(self):
        """Test knowledge assertion creation"""
        evidence = KnowledgeEvidence(
            source=KnowledgeSource.BTE_API,
            confidence=0.9,
            provenance=["infores:drugbank"],
            attributes={"phase": 4},
            study_count=5,
            clinical_phase=4
        )
        
        assertion = KnowledgeAssertion(
            subject="diabetes",
            predicate="biolink:treated_by",
            object="metformin",
            subject_type="biolink:Disease", 
            object_type="biolink:SmallMolecule",
            evidence=[evidence],
            aggregated_confidence=0.85,
            provenance_count=1
        )
        
        assert assertion.subject == "diabetes"
        assert assertion.predicate == "biolink:treated_by"
        assert assertion.object == "metformin"
        assert len(assertion.evidence) == 1
        assert assertion.evidence[0].source == KnowledgeSource.BTE_API
        assert assertion.aggregated_confidence == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])