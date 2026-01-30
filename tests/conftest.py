"""
Pytest configuration and shared fixtures for Agentic BTE tests

This module provides common fixtures, test configuration, and mock data
used across the test suite for consistent and reliable testing.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List
import tempfile
import os
from pathlib import Path

# Test data and fixtures
@pytest.fixture
def sample_biomedical_query():
    """Sample biomedical query for testing"""
    return "What drugs can treat type 2 diabetes?"

@pytest.fixture
def sample_entity_data():
    """Sample entity extraction results"""
    return {
        "entities": {
            "diabetes": {
                "id": "MONDO:0005015",
                "type": "disease"
            },
            "drugs": {
                "id": "biolink:ChemicalEntity",
                "type": "general"
            }
        },
        "entity_ids": {
            "diabetes": "MONDO:0005015",
            "drugs": "biolink:ChemicalEntity"
        }
    }

@pytest.fixture
def sample_trapi_query():
    """Sample TRAPI query structure"""
    return {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {
                        "categories": ["biolink:Disease"],
                        "ids": ["MONDO:0005015"]
                    },
                    "n1": {
                        "categories": ["biolink:SmallMolecule"]
                    }
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

@pytest.fixture
def sample_bte_results():
    """Sample BTE API results"""
    return [
        {
            "subject": "MONDO:0005015",
            "subject_type": "biolink:Disease",
            "predicate": "biolink:tested_by_clinical_trials_of",
            "object": "CHEBI:6801",
            "object_type": "biolink:SmallMolecule"
        },
        {
            "subject": "MONDO:0005015", 
            "subject_type": "biolink:Disease",
            "predicate": "biolink:tested_by_clinical_trials_of",
            "object": "CHEBI:71193",
            "object_type": "biolink:SmallMolecule"
        }
    ]

@pytest.fixture
def sample_entity_mappings():
    """Sample entity ID to name mappings"""
    return {
        "diabetes mellitus": "MONDO:0005015",
        "Metformin": "CHEBI:6801",
        "Liraglutide": "CHEBI:71193"
    }

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    mock_response = Mock()
    mock_response.content = "What genes are associated with diabetes?"
    return mock_response

@pytest.fixture
def mock_bte_api_response():
    """Mock BTE API response"""
    return {
        "message": {
            "results": [
                {
                    "node_bindings": {
                        "n0": [{"id": "MONDO:0005015"}],
                        "n1": [{"id": "CHEBI:6801"}]
                    },
                    "edge_bindings": {
                        "e01": [{"id": "test_edge_1"}]
                    }
                }
            ],
            "knowledge_graph": {
                "nodes": {
                    "MONDO:0005015": {
                        "categories": ["biolink:Disease"],
                        "name": "diabetes mellitus"
                    },
                    "CHEBI:6801": {
                        "categories": ["biolink:SmallMolecule"], 
                        "name": "Metformin"
                    }
                },
                "edges": {
                    "test_edge_1": {
                        "subject": "MONDO:0005015",
                        "predicate": "biolink:tested_by_clinical_trials_of",
                        "object": "CHEBI:6801"
                    }
                }
            }
        }
    }

@pytest.fixture
def temp_env_file():
    """Create temporary .env file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("AGENTIC_BTE_OPENAI_API_KEY=test_key\n")
        f.write("AGENTIC_BTE_OPENAI_MODEL=gpt-4\n")
        f.write("AGENTIC_BTE_DEBUG_MODE=true\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)

# Mock classes for external dependencies
@pytest.fixture
def mock_spacy_nlp():
    """Mock spaCy NLP model"""
    mock_nlp = Mock()
    
    # Mock document with entities
    mock_doc = Mock()
    mock_ent = Mock()
    mock_ent.text = "diabetes"
    mock_ent.label_ = "DISEASE"
    mock_ent.start_char = 0
    mock_ent.end_char = 8
    mock_doc.ents = [mock_ent]
    
    mock_nlp.return_value = mock_doc
    return mock_nlp

@pytest.fixture
def mock_entity_linker():
    """Mock entity linker"""
    mock_linker = Mock()
    mock_linker.resolve.return_value = {
        "diabetes": {
            "curie": "MONDO:0005015",
            "name": "diabetes mellitus"
        }
    }
    return mock_linker

# Async test support
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# Test markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests that don't require external services"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interaction" 
    )
    config.addinivalue_line(
        "markers", "external: Tests requiring external APIs (BTE, OpenAI, SRI)"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer to run"
    )
    config.addinivalue_line(
        "markers", "mcp: Tests specific to MCP server functionality"
    )
    config.addinivalue_line(
        "markers", "agents: Tests for LangGraph multi-agent system"
    )
    config.addinivalue_line(
        "markers", "benchmark: Benchmark tests for system accuracy and performance"
    )

# Test environment setup
@pytest.fixture(autouse=True)
def test_env_setup(monkeypatch):
    """Automatically set up test environment variables"""
    monkeypatch.setenv("AGENTIC_BTE_OPENAI_API_KEY", "test_key")
    monkeypatch.setenv("AGENTIC_BTE_DEBUG_MODE", "true")
    monkeypatch.setenv("AGENTIC_BTE_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("AGENTIC_BTE_ENABLE_CACHING", "false")

@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

# Benchmark test fixtures
@pytest.fixture
def mock_llm_for_benchmarks():
    """Mock LLM for benchmark testing"""
    mock_llm = Mock()
    
    # Mock LLM response for baseline queries
    def mock_invoke(query):
        mock_response = Mock()
        # Return a realistic but deterministic response
        mock_response.content = (
            "Based on current medical knowledge, several drugs may be relevant:\n"
            "- Metformin\n"
            "- Insulin\n"
            "- Sulfonylureas\n"
            "However, specific targeting information requires deeper analysis."
        )
        return mock_response
    
    mock_llm.invoke = mock_invoke
    return mock_llm

@pytest.fixture
def mock_unified_agent():
    """Mock UnifiedBiomedicalAgent for benchmark testing"""
    from unittest.mock import AsyncMock
    import random
    
    mock_agent = AsyncMock()
    
    # Common drug names that might appear in DMDB dataset
    common_drugs = [
        "Metformin", "Aspirin", "Doxorubicin", "Ibuprofen", "Acetaminophen",
        "Warfarin", "Insulin", "Prednisone", "Simvastatin", "Lisinopril",
        "Amoxicillin", "Ciprofloxacin", "Omeprazole", "Atorvastatin", "Furosemide"
    ]
    
    # Mock process_query to return realistic response with variety
    async def mock_process_query(text, **kwargs):
        # Randomly select drugs to simulate realistic retrieval
        # With 50% chance, include common drugs that might be in ground truth
        num_drugs = random.randint(3, 8)
        selected_drugs = random.sample(common_drugs, min(num_drugs, len(common_drugs)))
        
        mock_response = Mock()
        drug_list = "\n".join([f"{i+1}. {drug}" for i, drug in enumerate(selected_drugs)])
        mock_response.final_answer = (
            f"Based on the knowledge graph, the following drugs target the specified biological process:\n"
            f"{drug_list}\n"
            f"\nThese drugs have been shown to affect the biological process in the context of the disease."
        )
        mock_response.results = [
            {"subject": "Disease", "predicate": "treated_by", "object": drug}
            for drug in selected_drugs
        ]
        mock_response.processing_time = 2.5
        mock_response.error = None
        return mock_response
    
    mock_agent.process_query = mock_process_query
    return mock_agent
