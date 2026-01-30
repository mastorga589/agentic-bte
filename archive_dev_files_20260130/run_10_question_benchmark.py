#!/usr/bin/env python3
"""
10-Question Baseline Benchmark with Detailed Output
"""
import os
import sys
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment
load_dotenv()

sys.path.insert(0, '/Users/mastorga/Documents/agentic-bte')

from tests.benchmarks.dmdb_utils import (
    sample_dmdb_questions,
    get_ground_truth_drugs,
    calculate_retrieval_metrics,
    parse_drug_list_from_response
)

# Setup
api_key = os.getenv('AGENTIC_BTE_OPENAI_API_KEY')
if not api_key:
    print("ERROR: AGENTIC_BTE_OPENAI_API_KEY not found in environment")
    sys.exit(1)

os.environ['OPENAI_API_KEY'] = api_key
logging.basicConfig(level=logging.WARNING)  # Reduce noise

print('=' * 80)
print('10-QUESTION BASELINE BENCHMARK - DETAILED OUTPUT')
print('=' * 80)
print()

# Sample 10 questions
questions_df = sample_dmdb_questions(n_samples=10, random_seed=42)
llm = ChatOpenAI(temperature=0, model='gpt-4o')

results = []

for idx, row in questions_df.iterrows():
    question = row['question_without_id']
    ground_truth = get_ground_truth_drugs(row)
    
    print(f"\n{'=' * 80}")
    print(f'QUESTION {idx + 1}/10')
    print(f"{'=' * 80}")
    print(f'\n📝 Question:')
    print(f'   {question}')
    print(f'\n✅ Ground Truth Drug(s):')
    print(f"   {', '.join(ground_truth)}")
    
    # Get LLM response
    print(f'\n🤖 Querying GPT-4o...')
    response = llm.invoke(question)
    response_text = response.content
    
    print(f'\n💬 LLM Response:')
    # Print full response for better analysis
    lines = response_text.split('\n')
    for line in lines[:15]:  # First 15 lines
        print(f'   {line}')
    if len(lines) > 15:
        print(f'   ... ({len(lines) - 15} more lines)')
    
    # Parse predicted drugs
    predicted_drugs = parse_drug_list_from_response(response_text)
    
    print(f'\n🔍 Extracted Drugs:')
    if predicted_drugs:
        for i, drug in enumerate(predicted_drugs[:10], 1):
            print(f'   {i}. {drug}')
        if len(predicted_drugs) > 10:
            print(f'   ... and {len(predicted_drugs) - 10} more')
    else:
        print(f'   (none extracted)')
    
    # Calculate metrics
    metrics = calculate_retrieval_metrics(predicted_drugs, ground_truth)
    
    print(f'\n📊 Metrics:')
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall: {metrics['recall']:.3f}")
    print(f"   F1 Score: {metrics['f1']:.3f}")
    print(f"   Found Ground Truth: {'✓ YES' if metrics['found_ground_truth'] else '✗ NO'}")
    
    results.append({
        'question': question,
        'ground_truth': ground_truth,
        'predicted': predicted_drugs,
        'found': metrics['found_ground_truth'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1']
    })

# Aggregate results
print(f"\n\n{'=' * 80}")
print('AGGREGATE RESULTS')
print(f"{'=' * 80}")

found_count = sum(r['found'] for r in results)
total = len(results)
avg_precision = sum(r['precision'] for r in results) / total
avg_recall = sum(r['recall'] for r in results) / total
avg_f1 = sum(r['f1'] for r in results) / total

print(f'\nTotal Questions: {total}')
print(f'Found Ground Truth: {found_count}/{total} ({found_count/total*100:.1f}%)')
print(f'Average Precision: {avg_precision:.3f}')
print(f'Average Recall: {avg_recall:.3f}')
print(f'Average F1 Score: {avg_f1:.3f}')

print(f"\n{'=' * 80}")
print('BENCHMARK COMPLETE')
print(f"{'=' * 80}")
