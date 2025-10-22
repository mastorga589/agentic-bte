"""
Tests for the unified knowledge manager
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from ..config import UnifiedConfig
from ..types import (
    BiomedicalResult, EntityContext, Entity, EntityType, 
    BiomedicalRelationship, ExecutionContext
)
from ..knowledge_manager import (
    UnifiedKnowledgeManager, TRAPIQuery, KnowledgeAssertion,
    KnowledgeEvidence, KnowledgeSource, PredicateRanking
)
from ...core.knowledge.predicate_strategy import QueryIntent


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


@pytest.fixture
def sample_execution_context():
    """Create sample execution context"""
    return ExecutionContext(
        query="What drugs treat diabetes?",
        strategy=Mock(),
        entity_context=Mock(),
        knowledge_graph=None,
        config=Mock()
    )


@pytest.fixture 
def sample_bte_results():
    """Create sample BTE API results"""
    return [
        {
            "analyses": [
                {
                    "edge_bindings": {
                        "e01": [{"id": "edge_1"}]
                    },
                    "node_bindings": {
                        "n0": [{"id": "MONDO:0005015"}],
                        "n1": [{"id": "CHEBI:6801"}]
                    }
                }
            ]
        },
        {
            "analyses": [
                {
                    "edge_bindings": {
                        "e01": [{"id": "edge_2"}]
                    },
                    "node_bindings": {
                        "n0": [{"id": "MONDO:0005015"}],
                        "n1": [{"id": "CHEBI:71193"}]
                    }
                }
            ]
        }
    ]


@pytest.fixture
def sample_edges_data():
    """Create sample edges data"""
    return {
        "edge_1": {
            "subject": "MONDO:0005015",
            "object": "CHEBI:6801",
            "predicate": "biolink:treated_by",
            "attributes": [
                {
                    "attribute_type_id": "biolink:max_research_phase",
                    "value": 4
                }
            ],
            "sources": [
                {"resource_id": "infores:drugbank"}
            ]
        },
        "edge_2": {
            "subject": "MONDO:0005015", 
            "object": "CHEBI:71193",
            "predicate": "biolink:treated_by",
            "attributes": [
                {
                    "attribute_type_id": "biolink:clinical_approval_status",
                    "value": "approved_for_condition"
                }
            ],
            "sources": [
                {"resource_id": "infores:clinicaltrials"}
            ]
        }
    }


@pytest.fixture
def sample_nodes_data():
    """Create sample nodes data"""
    return {
        "MONDO:0005015": {
            "name": "diabetes mellitus",
            "categories": ["biolink:Disease"]
        },
        "CHEBI:6801": {
            "name": "Metformin", 
            "categories": ["biolink:SmallMolecule"]
        },
        "CHEBI:71193": {
            "name": "Liraglutide",
            "categories": ["biolink:SmallMolecule"]
        }
    }


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
    
    @patch('agentic_bte.unified.knowledge_manager.RDFGraphManager')
    @patch('agentic_bte.unified.knowledge_manager.EvidenceScorer')
    @patch('agentic_bte.unified.knowledge_manager.PredicateSelector')
    @pytest.mark.asyncio
    async def test_process_biomedical_query(self, mock_predicate_selector_cls, mock_evidence_scorer_cls,
                                          mock_rdf_manager_cls, mock_config, sample_entity_context,
                                          sample_execution_context):
        """Test processing biomedical query into TRAPI queries"""
        # Setup mocks
        mock_predicate_selector = Mock()
        mock_predicate_selector.detect_query_intent.return_value = QueryIntent.THERAPEUTIC
        mock_predicate_selector.select_predicates.return_value = [
            ("biolink:treated_by", 0.9),
            ("biolink:affects", 0.7)
        ]
        mock_predicate_selector_cls.return_value = mock_predicate_selector
        
        manager = UnifiedKnowledgeManager(mock_config)
        
        # Execute
        trapi_queries = await manager.process_biomedical_query(
            "What drugs treat diabetes?", 
            sample_entity_context,
            sample_execution_context
        )
        
        # Verify
        assert len(trapi_queries) > 0
        assert all(isinstance(q, TRAPIQuery) for q in trapi_queries)
        
        # Check that queries were built for entity pairs
        mock_predicate_selector.detect_query_intent.assert_called_once()
        mock_predicate_selector.select_predicates.assert_called()
        
        # Verify query structure
        first_query = trapi_queries[0]
        assert "nodes" in first_query.query_graph
        assert "edges" in first_query.query_graph
        assert first_query.source_intent == QueryIntent.THERAPEUTIC
    
    @patch('agentic_bte.unified.knowledge_manager.RDFGraphManager')
    @patch('agentic_bte.unified.knowledge_manager.EvidenceScorer')
    @patch('agentic_bte.unified.knowledge_manager.PredicateSelector')
    def test_score_knowledge_results(self, mock_predicate_selector_cls, mock_evidence_scorer_cls,
                                   mock_rdf_manager_cls, mock_config, sample_bte_results, 
                                   sample_edges_data):
        """Test scoring of knowledge results"""
        # Setup evidence scorer mock
        mock_evidence_scorer = Mock()
        mock_evidence_scorer.score_result.side_effect = [0.9, 0.7]  # Mock scores
        mock_evidence_scorer_cls.return_value = mock_evidence_scorer
        
        manager = UnifiedKnowledgeManager(mock_config)
        
        # Execute
        scored_results = manager.score_knowledge_results(
            sample_bte_results,
            sample_edges_data, 
            "biolink:treated_by",
            QueryIntent.THERAPEUTIC
        )
        
        # Verify
        assert len(scored_results) == 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in scored_results)
        assert scored_results[0][1] == 0.9  # Higher score first
        assert scored_results[1][1] == 0.7
        
        # Verify scoring was called
        assert mock_evidence_scorer.score_result.call_count == 2
    
    @patch('agentic_bte.unified.knowledge_manager.RDFGraphManager')
    @patch('agentic_bte.unified.knowledge_manager.EvidenceScorer')
    @patch('agentic_bte.unified.knowledge_manager.PredicateSelector')
    def test_build_knowledge_graph(self, mock_predicate_selector_cls, mock_evidence_scorer_cls,
                                 mock_rdf_manager_cls, mock_config, sample_bte_results,
                                 sample_edges_data, sample_nodes_data):
        """Test building unified knowledge graph"""
        # Setup RDF manager mock
        mock_rdf_manager = Mock()
        mock_rdf_manager.add_triples.return_value = 2
        mock_rdf_manager_cls.return_value = mock_rdf_manager
        
        manager = UnifiedKnowledgeManager(mock_config)
        
        # Create scored results
        scored_results = [(sample_bte_results[0], 0.9), (sample_bte_results[1], 0.7)]
        
        # Execute
        knowledge_graph = manager.build_knowledge_graph(
            scored_results,
            sample_edges_data,
            sample_nodes_data
        )
        
        # Verify knowledge graph structure
        assert knowledge_graph.entity_count >= 0
        assert knowledge_graph.relationship_count >= 0
        assert isinstance(knowledge_graph.confidence_distribution, dict)
        assert isinstance(knowledge_graph.provenance_summary, dict)
        
        # Verify RDF triples were added
        mock_rdf_manager.add_triples.assert_called_once()
        
        # Verify knowledge assertions were created
        assert len(manager.knowledge_assertions) > 0
        
        # Check assertion structure
        first_assertion = list(manager.knowledge_assertions.values())[0]
        assert isinstance(first_assertion, KnowledgeAssertion)
        assert first_assertion.subject
        assert first_assertion.predicate
        assert first_assertion.object
        assert len(first_assertion.evidence) > 0
    
    @patch('agentic_bte.unified.knowledge_manager.RDFGraphManager')
    @patch('agentic_bte.unified.knowledge_manager.EvidenceScorer')
    @patch('agentic_bte.unified.knowledge_manager.PredicateSelector')
    def test_get_entity_knowledge(self, mock_predicate_selector_cls, mock_evidence_scorer_cls,
                                mock_rdf_manager_cls, mock_config):
        """Test getting entity knowledge"""
        # Setup RDF manager mock
        mock_rdf_manager = Mock()
        mock_rdf_manager.get_entity_relationships.return_value = [
            {"subject": "diabetes", "predicate": "treated_by", "object": "metformin"}
        ]
        mock_rdf_manager_cls.return_value = mock_rdf_manager
        
        manager = UnifiedKnowledgeManager(mock_config)
        
        # Add sample knowledge assertion
        assertion = KnowledgeAssertion(
            subject="diabetes",
            predicate="biolink:treated_by",
            object="metformin",
            subject_type="biolink:Disease",
            object_type="biolink:SmallMolecule",
            evidence=[],
            aggregated_confidence=0.8
        )
        manager.knowledge_assertions["test_key"] = assertion
        
        # Execute
        entity_knowledge = manager.get_entity_knowledge("diabetes")
        
        # Verify
        assert entity_knowledge['entity_name'] == "diabetes"
        assert len(entity_knowledge['rdf_relationships']) == 1
        assert len(entity_knowledge['knowledge_assertions']) == 1
        assert entity_knowledge['assertion_count'] == 1
        assert entity_knowledge['relationship_count'] == 1
        
        # Verify RDF manager was called
        mock_rdf_manager.get_entity_relationships.assert_called_once_with("diabetes")
    
    @patch('agentic_bte.unified.knowledge_manager.RDFGraphManager')
    @patch('agentic_bte.unified.knowledge_manager.EvidenceScorer') 
    @patch('agentic_bte.unified.knowledge_manager.PredicateSelector')
    def test_get_knowledge_statistics(self, mock_predicate_selector_cls, mock_evidence_scorer_cls,
                                    mock_rdf_manager_cls, mock_config):
        """Test getting knowledge statistics"""
        # Setup RDF manager mock
        mock_rdf_manager = Mock()
        mock_rdf_manager.get_summary_stats.return_value = {
            'total_triples': 10,
            'unique_subjects': 5,
            'unique_predicates': 3,
            'unique_objects': 5
        }
        mock_rdf_manager_cls.return_value = mock_rdf_manager
        
        manager = UnifiedKnowledgeManager(mock_config)
        
        # Add some test data
        manager.query_build_count = 5
        manager.evidence_score_count = 15
        
        # Add sample TRAPI query
        trapi_query = TRAPIQuery(
            query_graph={},
            query_id="test_query",
            predicate="biolink:treats",
            entities=["diabetes", "metformin"],
            confidence=0.8,
            source_intent=QueryIntent.THERAPEUTIC
        )
        manager.cached_trapi_queries["test_key"] = trapi_query
        
        # Add sample knowledge assertion
        assertion = KnowledgeAssertion(
            subject="diabetes",
            predicate="biolink:treated_by", 
            object="metformin",
            subject_type="biolink:Disease",
            object_type="biolink:SmallMolecule",
            evidence=[],
            aggregated_confidence=0.8
        )
        manager.knowledge_assertions["test_assertion"] = assertion
        
        # Execute
        stats = manager.get_knowledge_statistics()
        
        # Verify
        assert 'rdf_graph' in stats
        assert stats['knowledge_assertions'] == 1
        assert stats['trapi_queries_built'] == 5
        assert stats['cached_queries'] == 1
        assert stats['results_scored'] == 15
        assert 'intent_distribution' in stats
        assert 'predicate_usage' in stats
        assert 'avg_confidence' in stats
        
        # Verify intent distribution
        assert stats['intent_distribution']['therapeutic'] == 1
        
        # Verify predicate usage
        assert stats['predicate_usage']['biolink:treated_by'] == 1
        
        # Verify average confidence
        assert stats['avg_confidence'] == 0.8
    
    @patch('agentic_bte.unified.knowledge_manager.RDFGraphManager')
    @patch('agentic_bte.unified.knowledge_manager.EvidenceScorer')
    @patch('agentic_bte.unified.knowledge_manager.PredicateSelector')
    def test_export_knowledge_graph(self, mock_predicate_selector_cls, mock_evidence_scorer_cls,
                                   mock_rdf_manager_cls, mock_config):
        """Test exporting knowledge graph in different formats"""
        # Setup RDF manager mock
        mock_rdf_manager = Mock()
        mock_rdf_manager.get_turtle_representation.return_value = "@prefix bio: <http://biolink/> ."
        mock_rdf_manager_cls.return_value = mock_rdf_manager
        
        manager = UnifiedKnowledgeManager(mock_config)
        
        # Add sample assertion for JSON export
        assertion = KnowledgeAssertion(
            subject="diabetes",
            predicate="biolink:treated_by",
            object="metformin", 
            subject_type="biolink:Disease",
            object_type="biolink:SmallMolecule",
            evidence=[],
            aggregated_confidence=0.8
        )
        manager.knowledge_assertions["test_key"] = assertion
        
        # Test turtle export
        turtle_export = manager.export_knowledge_graph("turtle")
        assert turtle_export == "@prefix bio: <http://biolink/> ."
        mock_rdf_manager.get_turtle_representation.assert_called_once()
        
        # Test JSON export
        json_export = manager.export_knowledge_graph("json")
        json_data = json.loads(json_export)
        assert len(json_data) == 1
        assert json_data[0]['subject'] == "diabetes"
        assert json_data[0]['predicate'] == "biolink:treated_by"
        assert json_data[0]['object'] == "metformin"
        
        # Test unsupported format
        with pytest.raises(ValueError):
            manager.export_knowledge_graph("xml")
    
    @patch('agentic_bte.unified.knowledge_manager.RDFGraphManager')
    @patch('agentic_bte.unified.knowledge_manager.EvidenceScorer')
    @patch('agentic_bte.unified.knowledge_manager.PredicateSelector')
    @pytest.mark.asyncio
    async def test_validate_knowledge_consistency(self, mock_predicate_selector_cls, 
                                                mock_evidence_scorer_cls, mock_rdf_manager_cls, mock_config):
        """Test knowledge consistency validation"""
        manager = UnifiedKnowledgeManager(mock_config)
        
        # Add conflicting assertions
        assertion1 = KnowledgeAssertion(
            subject="drug_a",
            predicate="biolink:treats",
            object="disease_b",
            subject_type="biolink:SmallMolecule",
            object_type="biolink:Disease", 
            evidence=[],
            aggregated_confidence=0.8
        )
        
        assertion2 = KnowledgeAssertion(
            subject="drug_a",
            predicate="biolink:causes", 
            object="disease_b",
            subject_type="biolink:SmallMolecule",
            object_type="biolink:Disease",
            evidence=[],
            aggregated_confidence=0.6
        )
        
        # Add low confidence assertion
        assertion3 = KnowledgeAssertion(
            subject="drug_c",
            predicate="biolink:affects",
            object="disease_d",
            subject_type="biolink:SmallMolecule", 
            object_type="biolink:Disease",
            evidence=[],
            aggregated_confidence=0.2  # Low confidence
        )
        
        manager.knowledge_assertions["assertion1"] = assertion1
        manager.knowledge_assertions["assertion2"] = assertion2  
        manager.knowledge_assertions["assertion3"] = assertion3
        
        # Execute validation
        report = await manager.validate_knowledge_consistency()
        
        # Verify report structure
        assert 'total_assertions' in report
        assert 'conflicts_detected' in report
        assert 'low_confidence_assertions' in report
        assert 'missing_evidence_assertions' in report
        assert 'conflicts' in report
        assert 'recommendations' in report
        
        # Verify counts
        assert report['total_assertions'] == 3
        assert report['conflicts_detected'] == 1  # treats vs causes conflict
        assert report['low_confidence_assertions'] == 1  # assertion3
        assert report['missing_evidence_assertions'] == 3  # All have empty evidence
        
        # Verify conflict details
        assert len(report['conflicts']) == 1
        conflict = report['conflicts'][0]
        assert conflict['entity_pair'] == "drug_a:disease_b"
        assert 'biolink:treats' in conflict['conflicting_predicates']
        assert 'biolink:causes' in conflict['conflicting_predicates']
        
        # Verify recommendations
        assert len(report['recommendations']) >= 1
    
    @patch('agentic_bte.unified.knowledge_manager.RDFGraphManager')
    @patch('agentic_bte.unified.knowledge_manager.EvidenceScorer')
    @patch('agentic_bte.unified.knowledge_manager.PredicateSelector')
    def test_clear_knowledge_cache(self, mock_predicate_selector_cls, mock_evidence_scorer_cls,
                                 mock_rdf_manager_cls, mock_config):
        """Test clearing knowledge cache"""
        # Setup RDF manager mock
        mock_rdf_manager = Mock()
        mock_rdf_manager_cls.return_value = mock_rdf_manager
        
        manager = UnifiedKnowledgeManager(mock_config)
        
        # Add some test data
        manager.knowledge_assertions["test"] = Mock()
        manager.cached_trapi_queries["test"] = Mock()
        manager.query_build_count = 10
        manager.evidence_score_count = 20
        manager.knowledge_assertion_count = 5
        
        # Execute clear
        manager.clear_knowledge_cache()
        
        # Verify everything was cleared
        assert len(manager.knowledge_assertions) == 0
        assert len(manager.cached_trapi_queries) == 0
        assert manager.query_build_count == 0
        assert manager.evidence_score_count == 0
        assert manager.knowledge_assertion_count == 0
        
        # Verify RDF graph was cleared
        mock_rdf_manager.clear_graph.assert_called_once()


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
    
    def test_predicate_ranking_creation(self):
        """Test predicate ranking creation"""
        ranking = PredicateRanking(
            predicate="biolink:treats",
            relevance_score=0.9,
            support_count=15,
            tier="primary",
            confidence=0.85
        )
        
        assert ranking.predicate == "biolink:treats"
        assert ranking.relevance_score == 0.9
        assert ranking.support_count == 15
        assert ranking.tier == "primary"
        assert ranking.confidence == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])