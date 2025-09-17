"""
Integration tests for MCP server functionality

Tests the MCP server tools integration without requiring external APIs.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from agentic_bte.servers.mcp.tools.bio_ner_tool import handle_bio_ner
from agentic_bte.servers.mcp.tools.trapi_tool import handle_trapi_query
from agentic_bte.servers.mcp.tools.bte_tool import handle_bte_call


@pytest.mark.integration
@pytest.mark.mcp
class TestMCPServerIntegration:
    """Test MCP server tool integration"""
    
    @patch('agentic_bte.servers.mcp.tools.bio_ner_tool.BioNERTool')
    @pytest.mark.asyncio
    async def test_bio_ner_tool_integration(self, mock_bio_ner_class):
        """Test BioNER tool integration in MCP server"""
        # Mock BioNER tool
        mock_bio_ner = Mock()
        mock_bio_ner.extract_and_link.return_value = {
            "entities": {
                "diabetes": {
                    "id": "MONDO:0005015",
                    "type": "disease"
                }
            },
            "entity_ids": {
                "diabetes": "MONDO:0005015"
            }
        }
        mock_bio_ner_class.return_value = mock_bio_ner
        
        # Test the tool handler
        arguments = {"query": "What drugs treat diabetes?"}
        result = await handle_bio_ner(arguments)
        
        assert "content" in result
        assert "result_data" in result
        assert "diabetes" in str(result["content"][0]["text"])
    
    @patch('agentic_bte.servers.mcp.tools.trapi_tool.TRAPIQueryBuilder')
    @pytest.mark.asyncio
    async def test_trapi_tool_integration(self, mock_trapi_builder_class):
        """Test TRAPI query building tool integration"""
        # Mock TRAPI builder
        mock_builder = Mock()
        mock_trapi_query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"categories": ["biolink:Disease"], "ids": ["MONDO:0005015"]},
                        "n1": {"categories": ["biolink:SmallMolecule"]}
                    },
                    "edges": {
                        "e01": {"subject": "n0", "object": "n1", "predicates": ["biolink:treated_by"]}
                    }
                }
            }
        }
        mock_builder.build_query.return_value = mock_trapi_query
        mock_builder.validate_trapi_query.return_value = (True, "Valid TRAPI query")
        mock_trapi_builder_class.return_value = mock_builder
        
        # Test the tool handler
        arguments = {
            "query": "What drugs treat diabetes?",
            "entity_data": {"diabetes": "MONDO:0005015"}
        }
        result = await handle_trapi_query(arguments)
        
        assert "content" in result
        assert "trapi_query" in result
        assert result["trapi_query"] == mock_trapi_query
    
    @patch('agentic_bte.servers.mcp.tools.bte_tool.BTEClient')
    @pytest.mark.asyncio
    async def test_bte_call_integration(self, mock_bte_client_class):
        """Test BTE API call tool integration"""
        # Mock BTE client
        mock_client = Mock()
        mock_results = [
            {
                "subject": "MONDO:0005015",
                "predicate": "biolink:treated_by",
                "object": "CHEBI:6801"
            }
        ]
        mock_entity_mappings = {"Metformin": "CHEBI:6801"}
        mock_metadata = {"message": "Success", "total_batches": 1}
        
        mock_client.execute_trapi_with_batching.return_value = (
            mock_results, mock_entity_mappings, mock_metadata
        )
        mock_bte_client_class.return_value = mock_client
        
        # Test the tool handler
        trapi_query = {
            "message": {
                "query_graph": {
                    "nodes": {"n0": {"ids": ["MONDO:0005015"]}},
                    "edges": {"e01": {"subject": "n0", "object": "n1"}}
                }
            }
        }
        arguments = {"json_query": trapi_query, "maxresults": 10, "k": 3}
        result = await handle_bte_call(arguments)
        
        assert "content" in result
        assert "results" in result
        assert "entity_mappings" in result
        assert len(result["results"]) == 1


@pytest.mark.integration
class TestMCPToolChaining:
    """Test chaining MCP tools together"""
    
    @patch('agentic_bte.servers.mcp.tools.bio_ner_tool.BioNERTool')
    @patch('agentic_bte.servers.mcp.tools.trapi_tool.TRAPIQueryBuilder')
    @patch('agentic_bte.servers.mcp.tools.bte_tool.BTEClient')
    @pytest.mark.asyncio
    async def test_full_mcp_workflow(self, mock_bte_client_class, mock_trapi_builder_class, mock_bio_ner_class):
        """Test complete workflow: BioNER -> TRAPI -> BTE"""
        
        # Step 1: Mock BioNER
        mock_bio_ner = Mock()
        mock_bio_ner.extract_and_link.return_value = {
            "entities": {"diabetes": {"id": "MONDO:0005015", "type": "disease"}},
            "entity_ids": {"diabetes": "MONDO:0005015"}
        }
        mock_bio_ner_class.return_value = mock_bio_ner
        
        # Step 2: Mock TRAPI Builder
        mock_builder = Mock()
        mock_trapi_query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"categories": ["biolink:Disease"], "ids": ["MONDO:0005015"]},
                        "n1": {"categories": ["biolink:SmallMolecule"]}
                    },
                    "edges": {
                        "e01": {"subject": "n0", "object": "n1", "predicates": ["biolink:treated_by"]}
                    }
                }
            }
        }
        mock_builder.build_query.return_value = mock_trapi_query
        mock_builder.validate_trapi_query.return_value = (True, "Valid")
        mock_trapi_builder_class.return_value = mock_builder
        
        # Step 3: Mock BTE Client
        mock_client = Mock()
        mock_results = [
            {
                "subject": "MONDO:0005015",
                "predicate": "biolink:treated_by", 
                "object": "CHEBI:6801"
            }
        ]
        mock_client.execute_trapi_with_batching.return_value = (
            mock_results, {"Metformin": "CHEBI:6801"}, {"message": "Success"}
        )
        mock_bte_client_class.return_value = mock_client
        
        # Execute workflow
        query = "What drugs treat diabetes?"
        
        # Step 1: BioNER
        ner_result = await handle_bio_ner({"query": query})
        assert "result_data" in ner_result
        entity_data = ner_result["result_data"]["entity_ids"]
        
        # Step 2: TRAPI
        trapi_result = await handle_trapi_query({
            "query": query,
            "entity_data": entity_data
        })
        assert "trapi_query" in trapi_result
        trapi_query = trapi_result["trapi_query"]
        
        # Step 3: BTE
        bte_result = await handle_bte_call({
            "json_query": trapi_query,
            "maxresults": 10,
            "k": 3
        })
        assert "results" in bte_result
        assert len(bte_result["results"]) > 0
        
        # Verify the workflow produces meaningful results
        assert "MONDO:0005015" in str(bte_result["results"])
        assert "CHEBI:6801" in str(bte_result["results"])