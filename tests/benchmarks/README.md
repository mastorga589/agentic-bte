# Benchmark Tests for Agentic BTE

This directory contains benchmark tests that evaluate the accuracy and performance of the Agentic BTE system using real datasets and external APIs.

## Overview

The benchmark suite replicates experiments from the archived prototype notebooks to validate system performance on biomedical query answering tasks. All tests use **real data** from authoritative datasets and **actual system responses** (not mocks).

## Test Suite: 50-Question Drug-Disease-BP Benchmark

Located in: `test_50questions_drug_disease_bp.py`

This test suite replicates the experiment from:
```
/Users/mastorga/Documents/BTE-LLM/archive/Prototype/50questions_drugfromdiseasebp_8.04.25.ipynb
```

### Experiment Design

1. **Data Source**: DMDB dataset (`DMDB_go_bp_filtered_jjoy_05_08_2025.csv`)
   - Contains validated drug-disease-biological process triplets
   - 50 questions sampled deterministically (seed=42)

2. **Question Formats**:
   - Without entity IDs: "Which drugs can treat [disease] by targeting [biological process]?"
   - With entity IDs: "Which drugs can treat [disease] (ID: MONDO:XXX) by targeting [process] (ID: GO:XXX)?"

3. **Systems Tested**:
   - **Baseline LLM**: GPT-4o without knowledge graph access
   - **BTE-RAG System**: UnifiedBiomedicalAgent with BTE knowledge graph

4. **Parameter Variations**:
   - k values: 5, 10, 15 (number of top results)
   - maxresults: 100 (maximum results from BTE API)
   - With/without entity IDs in queries

### Test Categories

#### 1. Sample Generation (`test_sample_generation`)
Validates DMDB data loading and question generation.

#### 2. Baseline LLM Performance
- `test_baseline_llm_without_ids`: LLM-only, no entity IDs
- `test_baseline_llm_with_ids`: LLM-only, with entity IDs

Measures baseline accuracy without knowledge graph access.

#### 3. BTE-RAG System Performance
Six tests covering all parameter combinations:
- `test_bte_rag_without_ids_k5`, `test_bte_rag_without_ids_k10`, `test_bte_rag_without_ids_k15`
- `test_bte_rag_with_ids_k5`, `test_bte_rag_with_ids_k10`, `test_bte_rag_with_ids_k15`

Each test:
- Queries real UnifiedBiomedicalAgent
- Extracts predicted drugs from responses
- Compares against ground truth from DMDB
- Calculates precision, recall, and F1 metrics

#### 4. Comparative Analysis
- `test_comparison_ids_vs_no_ids`: Impact of providing entity IDs
- `test_k_parameter_impact`: Effect of k parameter on retrieval quality

#### 5. Integration Test
- `test_complete_benchmark_pipeline`: Validates end-to-end pipeline

## Running the Tests

### Prerequisites

1. **Environment Variables**: Set OpenAI API key
   ```bash
   export AGENTIC_BTE_OPENAI_API_KEY="your-key-here"
   ```

2. **DMDB Dataset**: Must be present at:
   ```
   /Users/mastorga/Documents/BTE-LLM/archive/Prototype/data/DMDB_go_bp_filtered_jjoy_05_08_2025.csv
   ```

3. **External Services**:
   - OpenAI API (for LLM)
   - BTE API (for knowledge graph queries)
   - SRI Name Resolver (for entity resolution)

### Run All Benchmark Tests

```bash
# From agentic-bte root directory
pytest tests/benchmarks/ -v -m benchmark

# Run with external service tests
pytest tests/benchmarks/ -v -m external
```

### Run Specific Test Categories

```bash
# Run only baseline LLM tests
pytest tests/benchmarks/test_50questions_drug_disease_bp.py::TestDrugDiseaseBPBenchmark::test_baseline_llm_without_ids -v

# Run only BTE-RAG tests with k=5
pytest tests/benchmarks/test_50questions_drug_disease_bp.py -k "k5" -v

# Run comparison tests
pytest tests/benchmarks/test_50questions_drug_disease_bp.py -k "comparison" -v
```

### Run Without External APIs (Structure Validation Only)

```bash
# Run integration test (no external calls)
pytest tests/benchmarks/test_50questions_drug_disease_bp.py::TestBenchmarkIntegration -v
```

## Metrics Calculated

For each test, the following metrics are calculated and logged:

1. **Recall Rate**: Percentage of queries where ground truth drug was found
2. **Average Recall**: Mean recall across all queries
3. **Average Precision**: Mean precision across all queries  
4. **Average F1**: Harmonic mean of precision and recall

### Example Output

```
Baseline LLM (no IDs) - Found ground truth in 12.0% of queries
Baseline LLM (no IDs) - Average recall: 0.120

BTE-RAG (without IDs, k=5, maxresults=100):
  - Found ground truth in 68.0% of queries
  - Average recall: 0.680
  - Average precision: 0.245
  - Average F1: 0.361
  --> System successfully retrieved ground truth drugs!
```

## Test Markers

All benchmark tests use pytest markers for filtering:

- `@pytest.mark.benchmark`: All benchmark tests
- `@pytest.mark.external`: Tests requiring external APIs
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.asyncio`: Async tests

## Interpreting Results

### Baseline vs. System Performance

The tests document the **actual measured performance** of both baseline LLM and BTE-RAG system. Key comparisons:

1. **Baseline LLM**: Expected to have limited recall (<30%) without knowledge graph
2. **BTE-RAG System**: Should significantly outperform baseline through structured knowledge retrieval

### ID Impact

Tests compare performance with/without entity IDs to measure:
- Whether providing IDs improves accuracy
- System robustness to entity linking challenges

### Parameter Tuning

k-parameter tests reveal:
- Optimal k value for retrieval quality
- Trade-offs between recall and precision

## Data Utilities

Located in: `dmdb_utils.py`

Key functions:
- `load_dmdb_dataset()`: Load full DMDB dataset
- `sample_dmdb_questions(n_samples, random_seed)`: Generate test questions
- `get_ground_truth_drugs(row)`: Extract correct answers
- `calculate_retrieval_metrics(predicted, ground_truth)`: Compute metrics
- `parse_drug_list_from_response(text)`: Extract drugs from LLM response

## Adding New Benchmarks

To add new benchmark tests:

1. Create data utility functions in `dmdb_utils.py` or new utility file
2. Create test class in `test_*.py` following existing patterns
3. Mark tests with `@pytest.mark.benchmark` and `@pytest.mark.external`
4. Use real fixtures (`real_llm`, `real_unified_agent`)
5. Calculate and log actual metrics
6. Document expected vs. actual performance

## Notes

- All tests use **real data and real system responses** (no mocks)
- Tests require external API access (OpenAI, BTE, SRI)
- Tests are deterministic (fixed random seed for sampling)
- Metrics are logged for inspection, not hardcoded expectations
- Tests document actual system performance at time of execution
