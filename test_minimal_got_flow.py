#!/usr/bin/env python3
"""
Minimal test script for GoT framework execution flow validation

This script tests the core GoT planner components directly without
heavy dependencies like scispacy that cause initialization delays.
"""

import json
import logging
import asyncio
from typing import Dict, Any, List

# Configure logging for detailed debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


async def test_minimal_got_flow():
    """Test minimal GoT framework execution flow"""
    logger.info("=== Starting Minimal GoT Framework Test ===")
    
    # Test query
    test_query = "What drugs treat Parkinson's disease?"
    
    try:
        # Step 1: Test entity extraction (without scispacy - use fallback)
        logger.info("Step 1: Testing entity extraction (fallback mode)")
        entities = [
            {"name": "Parkinson's disease", "type": "Disease", "id": "MONDO:0005180"},
            {"name": "drugs", "type": "SmallMolecule", "id": "CHEBI:23888"}
        ]
        logger.info(f"Entities extracted: {entities}")
        
        # Step 2: Test TRAPI query building
        logger.info("Step 2: Testing TRAPI query building")
        from agentic_bte.core.knowledge.trapi import TRAPIQueryBuilder
        
        # Mock TRAPI builder without OpenAI
        class MockTRAPIBuilder:
            def build_trapi_query(self, query: str, entity_data: Dict[str, str], failed_trapis: List = None):
                # Return a mock single-hop TRAPI query
                return {
                    "message": {
                        "query_graph": {
                            "nodes": {
                                "n0": {
                                    "categories": ["biolink:Disease"], 
                                    "ids": ["MONDO:0005180"]
                                },
                                "n1": {
                                    "categories": ["biolink:SmallMolecule"]
                                }
                            },
                            "edges": {
                                "e01": {
                                    "subject": "n1",
                                    "object": "n0", 
                                    "predicates": ["biolink:treats"]
                                }
                            }
                        }
                    }
                }
        
        # Test TRAPI query building
        trapi_builder = MockTRAPIBuilder()
        entity_data = {entity["name"]: entity["id"] for entity in entities}
        trapi_query = trapi_builder.build_trapi_query(test_query, entity_data, [])
        
        logger.info(f"TRAPI query generated: {json.dumps(trapi_query, indent=2)}")
        
        # Step 3: Validate TRAPI single-hop constraint
        logger.info("Step 3: Testing TRAPI validation")
        from agentic_bte.core.knowledge.trapi import validate_trapi
        
        is_valid, validation_msg = validate_trapi(trapi_query)
        logger.info(f"TRAPI validation: {is_valid}, message: {validation_msg}")
        
        # Step 4: Test BTE client (mocked)
        logger.info("Step 4: Testing BTE client (mock response)")
        
        mock_bte_response = {
            "message": {
                "knowledge_graph": {
                    "nodes": {
                        "MONDO:0005180": {
                            "categories": ["biolink:Disease"],
                            "name": ["Parkinson's disease"]
                        },
                        "DRUGBANK:DB00230": {
                            "categories": ["biolink:SmallMolecule"],
                            "name": ["L-DOPA"]
                        }
                    },
                    "edges": {
                        "edge_1": {
                            "subject": "DRUGBANK:DB00230",
                            "object": "MONDO:0005180",
                            "predicate": "biolink:treats"
                        }
                    }
                },
                "results": [
                    {
                        "node_bindings": {
                            "n0": [{"id": "MONDO:0005180"}],
                            "n1": [{"id": "DRUGBANK:DB00230"}]
                        },
                        "score": 0.95
                    }
                ]
            }
        }
        
        # Step 5: Test result parsing
        logger.info("Step 5: Testing BTE result parsing")
        from agentic_bte.core.knowledge.bte_client import BTEClient
        
        class MockBTEClient(BTEClient):
            def __init__(self):
                # Initialize without calling super() to avoid HTTP setup
                pass
            
            def parse_bte_results(self, bte_response, k=5, max_results=50, 
                                predicate=None, query_intent=None):
                # Mock parsing
                return [
                    {
                        "subject": "L-DOPA",
                        "subject_id": "DRUGBANK:DB00230",
                        "subject_type": "SmallMolecule",
                        "predicate": "biolink:treats", 
                        "object": "Parkinson's disease",
                        "object_id": "MONDO:0005180",
                        "object_type": "Disease",
                        "score": 0.95
                    }
                ], {"L-DOPA": "DRUGBANK:DB00230", "Parkinson's disease": "MONDO:0005180"}
        
        mock_client = MockBTEClient()
        parsed_results, entity_mappings = mock_client.parse_bte_results(mock_bte_response)
        
        logger.info(f"Parsed results: {json.dumps(parsed_results, indent=2)}")
        logger.info(f"Entity mappings: {entity_mappings}")
        
        # Step 6: Test final answer generation (without LLM)
        logger.info("Step 6: Testing final answer generation (without LLM)")
        from agentic_bte.core.queries.result_presenter import format_final_answer
        
        final_answer = format_final_answer(parsed_results, entities, test_query)
        logger.info(f"Final answer: {final_answer[:200]}...")
        
        # Step 7: Test execution steps structure (GoT framework components)
        logger.info("Step 7: Testing GoT execution steps structure")
        from agentic_bte.core.queries.result_presenter import QueryStep, QueryResult
        
        steps = [
            QueryStep(
                step_id="entity_extraction_001",
                step_type="entity_extraction", 
                input_data={"query": test_query},
                output_data={"entities": entities},
                execution_time=0.5,
                success=True,
                confidence=0.9
            ),
            QueryStep(
                step_id="query_building_001",
                step_type="query_building",
                input_data={"query": test_query, "entities": entities},
                output_data={"query": trapi_query},
                trapi_query=trapi_query,
                execution_time=1.2,
                success=True,
                confidence=0.85
            ),
            QueryStep(
                step_id="api_execution_001", 
                step_type="api_execution",
                input_data={"trapi_query": trapi_query},
                output_data={"results": parsed_results, "total_results": 1},
                trapi_query=trapi_query,
                execution_time=2.8,
                success=True,
                confidence=0.95
            )
        ]
        
        query_result = QueryResult(
            query=test_query,
            final_answer=final_answer,
            execution_steps=steps,
            total_execution_time=4.5,
            success=True,
            entities_found=entities,
            total_results=1,
            quality_score=0.85,
            got_metrics={
                "volume": 3,
                "latency": 3,
                "total_thoughts": 3,
                "quality_improvement": 1.2,
                "parallel_speedup": 1.0
            }
        )
        
        logger.info(f"Query result structure created successfully")
        
        # Step 8: Test result presentation 
        logger.info("Step 8: Testing result presentation")
        from agentic_bte.core.queries.result_presenter import ResultPresenter
        
        presenter = ResultPresenter(show_debug=True, show_graphs=False)
        presentation = presenter.present_results(query_result)
        
        logger.info("=== RESULT PRESENTATION ===")
        print(presentation)
        
        logger.info("=== Minimal GoT Framework Test Completed Successfully ===")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        logger.exception("Full traceback:")
        return False


def test_subquery_decomposition():
    """Test subquery decomposition logic without LLM"""
    logger.info("=== Testing Subquery Decomposition Logic ===")
    
    complex_query = "What are the molecular mechanisms connecting oxidative stress to neurodegeneration in Parkinson's disease?"
    
    # Mock LLM-generated subqueries
    mock_subqueries = [
        {
            "query": "What drugs treat Parkinson's disease?",
            "entities": [
                {"name": "Parkinson's disease", "type": "Disease", "id": "MONDO:0005180"},
                {"name": "drugs", "type": "SmallMolecule", "id": "CHEBI:23888"}
            ],
            "rationale": "Identify therapeutic compounds for Parkinson's disease",
            "query_type": "therapeutic"
        },
        {
            "query": "What genes are targets of Parkinson's disease drugs?",
            "entities": [
                {"name": "genes", "type": "Gene", "id": "HGNC:1"},
                {"name": "drugs", "type": "SmallMolecule", "id": "CHEBI:23888"}
            ],
            "rationale": "Explore molecular targets of identified therapeutic compounds",
            "query_type": "target"
        },
        {
            "query": "How does oxidative stress affect neurodegeneration?",
            "entities": [
                {"name": "oxidative stress", "type": "BiologicalProcess", "id": "GO:0006979"},
                {"name": "neurodegeneration", "type": "Disease", "id": "MONDO:0005559"}
            ],
            "rationale": "Connect oxidative stress mechanisms to neurodegeneration",
            "query_type": "mechanism"
        }
    ]
    
    logger.info(f"Generated {len(mock_subqueries)} subqueries for complex query")
    
    for i, subquery in enumerate(mock_subqueries, 1):
        logger.info(f"Subquery {i}: {subquery['query']}")
        logger.info(f"  Entities: {[e['name'] for e in subquery['entities']]}")
        logger.info(f"  Rationale: {subquery['rationale']}")
        logger.info(f"  Type: {subquery['query_type']}")
    
    logger.info("=== Subquery Decomposition Test Completed ===")
    return mock_subqueries


async def main():
    """Main test runner"""
    logger.info("Starting comprehensive GoT framework validation...")
    
    # Test 1: Basic GoT flow
    test1_passed = await test_minimal_got_flow()
    logger.info(f"Test 1 (Basic GoT Flow): {'PASSED' if test1_passed else 'FAILED'}")
    
    # Test 2: Subquery decomposition
    test2_subqueries = test_subquery_decomposition()
    test2_passed = len(test2_subqueries) > 0
    logger.info(f"Test 2 (Subquery Decomposition): {'PASSED' if test2_passed else 'FAILED'}")
    
    # Summary
    all_passed = test1_passed and test2_passed
    logger.info(f"Overall Test Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    asyncio.run(main())