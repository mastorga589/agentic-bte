"""
Unit tests for biomedical entity processing components

Tests for entity recognition, linking, and BioNER functionality without
requiring external API calls or spaCy model loading.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from agentic_bte.core.entities.bio_ner import BioNERTool
from agentic_bte.core.entities.linking import EntityLinker
from agentic_bte.core.entities.recognition import EntityRecognizer


@pytest.mark.unit
class TestBioNERTool:
    """Test BioNER tool functionality"""
    
    def test_bioNER_tool_initialization(self):
        """Test BioNER tool can be initialized"""
        with patch('agentic_bte.core.entities.bio_ner.EntityRecognizer'):
            with patch('agentic_bte.core.entities.bio_ner.EntityLinker'):
                bio_ner = BioNERTool()
                assert bio_ner is not None
    
    @patch('agentic_bte.core.entities.bio_ner.EntityRecognizer')
    @patch('agentic_bte.core.entities.bio_ner.EntityLinker')
    def test_extract_and_link_success(self, mock_linker_class, mock_recognizer_class):
        """Test successful entity extraction and linking"""
        # Mock recognizer
        mock_recognizer = Mock()
        mock_recognizer.extract_entities.return_value = [
            {
                "text": "diabetes",
                "label": "DISEASE",
                "start": 0,
                "end": 8
            }
        ]
        mock_recognizer_class.return_value = mock_recognizer
        
        # Mock linker
        mock_linker = Mock()
        mock_linker.link_entities.return_value = {
            "diabetes": {
                "id": "MONDO:0005015",
                "name": "diabetes mellitus",
                "type": "disease"
            }
        }
        mock_linker_class.return_value = mock_linker
        
        bio_ner = BioNERTool()
        result = bio_ner.extract_and_link("What drugs treat diabetes?")
        
        assert "entities" in result
        assert "entity_ids" in result
        assert "diabetes" in result["entities"]
        assert result["entities"]["diabetes"]["id"] == "MONDO:0005015"
    
    @patch('agentic_bte.core.entities.bio_ner.EntityRecognizer')
    @patch('agentic_bte.core.entities.bio_ner.EntityLinker')
    def test_extract_and_link_no_entities(self, mock_linker_class, mock_recognizer_class):
        """Test extraction with no entities found"""
        # Mock recognizer returning no entities
        mock_recognizer = Mock()
        mock_recognizer.extract_entities.return_value = []
        mock_recognizer_class.return_value = mock_recognizer
        
        mock_linker = Mock()
        mock_linker_class.return_value = mock_linker
        
        bio_ner = BioNERTool()
        result = bio_ner.extract_and_link("Hello world")
        
        assert result["entities"] == {}
        assert result["entity_ids"] == {}
    
    @patch('agentic_bte.core.entities.bio_ner.EntityRecognizer')
    @patch('agentic_bte.core.entities.bio_ner.EntityLinker')
    def test_extract_and_link_error_handling(self, mock_linker_class, mock_recognizer_class):
        """Test error handling in extraction process"""
        # Mock recognizer that raises exception
        mock_recognizer = Mock()
        mock_recognizer.extract_entities.side_effect = Exception("Recognition failed")
        mock_recognizer_class.return_value = mock_recognizer
        
        mock_linker = Mock()
        mock_linker_class.return_value = mock_linker
        
        bio_ner = BioNERTool()
        result = bio_ner.extract_and_link("test query")
        
        assert "error" in result
        assert "Recognition failed" in result["error"]


@pytest.mark.unit  
class TestEntityLinker:
    """Test entity linking functionality"""
    
    def test_entity_linker_initialization(self):
        """Test EntityLinker can be initialized"""
        with patch('agentic_bte.core.entities.linking.requests'):
            linker = EntityLinker()
            assert linker is not None
            assert hasattr(linker, 'sri_resolver_url')
    
    @patch('agentic_bte.core.entities.linking.requests.post')
    def test_link_entities_success(self, mock_post):
        """Test successful entity linking"""
        # Mock successful SRI resolver response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "diabetes": [
                {
                    "curie": "MONDO:0005015",
                    "label": "diabetes mellitus"
                }
            ]
        }
        mock_post.return_value = mock_response
        
        linker = EntityLinker()
        entities = [
            {
                "text": "diabetes",
                "label": "DISEASE",
                "start": 0,
                "end": 8
            }
        ]
        
        result = linker.link_entities(entities)
        
        assert "diabetes" in result
        assert result["diabetes"]["id"] == "MONDO:0005015"
        assert result["diabetes"]["name"] == "diabetes mellitus"
    
    @patch('agentic_bte.core.entities.linking.requests.post')
    def test_link_entities_api_failure(self, mock_post):
        """Test handling of API failure"""
        # Mock failed API response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_post.return_value = mock_response
        
        linker = EntityLinker()
        entities = [
            {
                "text": "diabetes",
                "label": "DISEASE", 
                "start": 0,
                "end": 8
            }
        ]
        
        result = linker.link_entities(entities)
        
        # Should return empty result or fallback IDs
        assert isinstance(result, dict)
    
    def test_normalize_entity_text(self):
        """Test entity text normalization"""
        linker = EntityLinker()
        
        # Test various normalization cases
        assert linker._normalize_entity_text("Type 2 Diabetes") == "type 2 diabetes"
        assert linker._normalize_entity_text("COVID-19") == "covid-19"
        assert linker._normalize_entity_text("  spaced  ") == "spaced"


@pytest.mark.unit
class TestEntityRecognizer:
    """Test entity recognition functionality"""
    
    def test_entity_recognizer_initialization(self):
        """Test EntityRecognizer can be initialized"""
        with patch('spacy.load'):
            recognizer = EntityRecognizer()
            assert recognizer is not None
    
    @patch('spacy.load')
    def test_extract_entities_with_mock_spacy(self, mock_spacy_load):
        """Test entity extraction with mocked spaCy model"""
        # Mock spaCy model and document
        mock_nlp = Mock()
        mock_doc = Mock()
        
        # Mock entity
        mock_ent = Mock()
        mock_ent.text = "diabetes"
        mock_ent.label_ = "DISEASE"
        mock_ent.start_char = 0
        mock_ent.end_char = 8
        mock_doc.ents = [mock_ent]
        
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp
        
        recognizer = EntityRecognizer()
        result = recognizer.extract_entities("diabetes treatment")
        
        assert len(result) == 1
        assert result[0]["text"] == "diabetes"
        assert result[0]["label"] == "DISEASE"
        assert result[0]["start"] == 0
        assert result[0]["end"] == 8
    
    @patch('spacy.load')
    def test_extract_entities_no_entities(self, mock_spacy_load):
        """Test extraction with no entities found"""
        # Mock spaCy model returning no entities
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.ents = []
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp
        
        recognizer = EntityRecognizer()
        result = recognizer.extract_entities("hello world")
        
        assert result == []
    
    def test_entity_type_mapping(self):
        """Test entity type mapping functionality"""
        recognizer = EntityRecognizer()
        
        # Test standard biomedical entity type mappings
        assert recognizer._map_entity_type("DISEASE") == "disease"
        assert recognizer._map_entity_type("CHEMICAL") == "chemical"
        assert recognizer._map_entity_type("GENE_OR_GENE_PRODUCT") == "gene"
        assert recognizer._map_entity_type("UNKNOWN_TYPE") == "general"