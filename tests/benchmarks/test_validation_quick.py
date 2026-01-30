"""
Quick Validation Test for Benchmark Pipeline

This is a minimal test (n=3) to validate that the benchmark pipeline works
before running the full 50-question suite.
"""

import pytest
import asyncio
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .dmdb_utils import (
    sample_dmdb_questions,
    get_ground_truth_drugs,
    calculate_retrieval_metrics,
    parse_drug_list_from_response
)
from agentic_bte.unified.agent import UnifiedBiomedicalAgent

logger = logging.getLogger(__name__)

# Load env vars
load_dotenv()


@pytest.mark.external
def test_quick_baseline_validation():
    """
    Quick validation test with just 3 questions to ensure pipeline works.
    
    This test validates:
    - DMDB data loading
    - LLM API calls
    - Drug name parsing
    - Metric calculation
    
    Expected time: <30 seconds
    """
    # Sample just 3 questions
    questions = sample_dmdb_questions(n_samples=3, random_seed=42)
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    
    results = []
    logger.info("Running quick validation test with 3 questions...")
    
    for idx, row in questions.iterrows():
        question = row['question_without_id']
        logger.info(f"  Question {idx+1}/3: {question[:60]}...")
        
        # Get LLM response
        response = llm.invoke(question)
        response_text = response.content
        
        # Parse and calculate metrics
        predicted_drugs = parse_drug_list_from_response(response_text)
        ground_truth = get_ground_truth_drugs(row)
        metrics = calculate_retrieval_metrics(predicted_drugs, ground_truth)
        
        logger.info(f"    Predicted: {predicted_drugs[:3]}")
        logger.info(f"    Ground truth: {ground_truth}")
        logger.info(f"    Found: {metrics['found_ground_truth']}")
        
        results.append(metrics['found_ground_truth'])
    
    # Calculate recall
    recall_rate = sum(results) / len(results)
    logger.info(f"\nQuick validation complete!")
    logger.info(f"  Recall: {recall_rate*100:.1f}% ({sum(results)}/{len(results)} questions)")
    
    # Assert test completed (no specific threshold for validation)
    assert len(results) == 3, "Should process all 3 questions"
    logger.info("✓ Validation test passed!")


@pytest.mark.asyncio
@pytest.mark.external
async def test_quick_system_validation():
    """
    Quick validation test for BTE-RAG system with 3 questions.
    
    This test validates:
    - UnifiedBiomedicalAgent initialization
    - Agent query processing
    - System response parsing
    - Full pipeline with knowledge graph
    
    Expected time: <5 minutes (agent is slower than baseline LLM)
    """
    # Sample just 3 questions
    questions = sample_dmdb_questions(n_samples=3, random_seed=42)
    
    logger.info("Initializing BTE-RAG system...")
    agent = UnifiedBiomedicalAgent()
    await agent.initialize()
    logger.info("System initialized!")
    
    results = []
    logger.info("Running quick system validation test with 3 questions...")
    
    for idx, row in questions.iterrows():
        question = row['question_without_id']
        logger.info(f"  Question {idx+1}/3: {question[:60]}...")
        
        # Query the BTE-RAG system
        response = await agent.process_query(
            text=question,
            max_results=100
        )
        
        # Parse response
        if hasattr(response, 'final_answer'):
            response_text = response.final_answer
        else:
            response_text = str(response)
        
        # Parse and calculate metrics
        predicted_drugs = parse_drug_list_from_response(response_text)
        ground_truth = get_ground_truth_drugs(row)
        metrics = calculate_retrieval_metrics(predicted_drugs, ground_truth)
        
        logger.info(f"    Predicted: {predicted_drugs[:3]}")
        logger.info(f"    Ground truth: {ground_truth}")
        logger.info(f"    Found: {metrics['found_ground_truth']}")
        
        results.append(metrics['found_ground_truth'])
    
    # Calculate recall
    recall_rate = sum(results) / len(results)
    logger.info(f"\nQuick system validation complete!")
    logger.info(f"  System Recall: {recall_rate*100:.1f}% ({sum(results)}/{len(results)} questions)")
    
    # Assert test completed
    assert len(results) == 3, "Should process all 3 questions"
    logger.info("✓ System validation test passed!")


if __name__ == "__main__":
    # Can run directly for quick testing
    import sys
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("Running baseline validation...")
    test_quick_baseline_validation()
    
    print("\n" + "="*50)
    print("Running system validation...")
    asyncio.run(test_quick_system_validation())
