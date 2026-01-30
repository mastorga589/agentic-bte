# Production-Ready Graph of Thoughts (GoT) Biomedical Query System

## 🎯 Overview

This is a complete, production-ready implementation of the **Graph of Thoughts (GoT)** framework specifically optimized for biomedical query processing. The system integrates with MCP (Model Context Protocol) tools to provide intelligent query optimization with comprehensive result presentation and debugging capabilities.

### 🔬 Key Features

- **Graph of Thoughts Framework**: Complete implementation based on Besta et al.'s research paper
- **Real MCP Integration**: Connects to actual biomedical tools (bio_ner, build_trapi_query, call_bte_api)
- **Comprehensive Result Presentation**: Final answers, execution plans, and debugging information
- **TRAPI Query Debugging**: Shows actual TRAPI queries used for BTE calls
- **Production-Ready Architecture**: Error handling, fallbacks, and performance monitoring
- **User-Friendly Interface**: Command-line tool with interactive mode
- **Extensive Validation**: Comprehensive test suite for quality assurance

---

## 🚀 Quick Start

### Basic Usage

```bash
# Simple query execution
python got_query_runner.py "What genes are associated with diabetes?"

# Interactive mode
python got_query_runner.py --interactive

# Debug mode with detailed output
python got_query_runner.py --debug "How does TP53 interact with other proteins?"
```

### Advanced Usage

```bash
# Complex research query with full debugging
python got_query_runner.py --debug --graphs --save-results \
  "Construct a comprehensive analysis of the molecular interactions between oxidative stress, inflammation, and neurodegeneration in Parkinson's disease"

# Custom configuration
python got_query_runner.py \
  --timeout 120 \
  --confidence-threshold 0.8 \
  --max-iterations 10 \
  "What are the pathways connecting insulin signaling to glucose metabolism in diabetes?"
```

---

## 📋 Command Line Options

### Basic Options
- `--interactive` / `-i`: Run in interactive mode for multiple queries
- `--debug` / `-d`: Enable detailed debugging output and TRAPI query display
- `--graphs` / `-g`: Generate visual execution graphs (requires matplotlib)
- `--examples`: Show usage examples and common biomedical query patterns

### Performance Options
- `--timeout SECONDS`: MCP tool timeout (default: 60)
- `--retries COUNT`: Maximum MCP tool retries (default: 3)
- `--max-iterations COUNT`: Maximum GoT iterations (default: 5)
- `--confidence-threshold FLOAT`: Minimum confidence threshold (default: 0.7)
- `--quality-threshold FLOAT`: Quality threshold for refinement (default: 0.1)
- `--max-concurrent COUNT`: Maximum concurrent operations (default: 3)
- `--no-parallel`: Disable parallel execution
- `--no-save`: Don't save results to files

### Output Options
- `--verbose` / `-v`: Enable verbose logging
- Results are automatically saved to timestamped files unless `--no-save` is used

---

## 🧬 Biomedical Query Examples

### Simple Queries
```bash
python got_query_runner.py "What genes are associated with diabetes?"
python got_query_runner.py "How does TP53 interact with other proteins?"
python got_query_runner.py "What drugs target EGFR?"
python got_query_runner.py "What pathways involve insulin?"
```

### Moderate Complexity
```bash
python got_query_runner.py "How do mutations in CFTR lead to cystic fibrosis symptoms?"
python got_query_runner.py "What is the relationship between oxidative stress and Alzheimer's disease?"
python got_query_runner.py "How does insulin signaling affect glucose metabolism?"
```

### Complex Research Queries
```bash
python got_query_runner.py --debug \
  "What is the molecular network connecting oxidative stress and neurodegeneration in Parkinson's disease?"

python got_query_runner.py --debug --graphs \
  "How do epigenetic modifications regulate gene expression in diabetes pathogenesis?"

python got_query_runner.py --debug \
  "What are the interconnected pathways linking metabolism, immunity, and aging?"
```

---

## 📊 System Architecture

### Core Components

1. **Production GoT Optimizer** (`production_got_optimizer.py`)
   - Main orchestrator for the GoT framework
   - Handles entity extraction, query building, API execution, and aggregation
   - Provides comprehensive error handling and performance monitoring

2. **MCP Integration** (`mcp_integration.py`)
   - Real integration with MCP tools
   - Fallback mechanisms for robust operation
   - Async execution with proper error handling

3. **Result Presenter** (`result_presenter.py`)
   - Comprehensive result formatting
   - TRAPI query debugging output
   - Graph visualization capabilities
   - Executive summaries and detailed breakdowns

4. **GoT Framework Components**
   - `got_framework.py`: Core GoT planner and execution
   - `got_aggregation.py`: Biomedical result aggregation and refinement
   - `got_metrics.py`: Performance metrics calculation

### Integration Flow

```
User Query → Entity Extraction (bio_ner) → Query Building (build_trapi_query) 
→ BTE API Execution (call_bte_api) → Result Aggregation → Final Answer
```

### Graph of Thoughts Process

1. **Thought Generation**: Create thoughts for entity extraction, query building, API execution
2. **Dependency Management**: Track dependencies between thoughts
3. **Parallel Execution**: Execute independent thoughts concurrently
4. **Result Aggregation**: Combine and rank results from multiple paths
5. **Iterative Refinement**: Improve results based on quality thresholds

---

## 🧪 Testing and Validation

### Run Comprehensive Tests

```bash
# Run all validation tests
python test_production_got.py

# The test suite includes:
# - Simple query validation (5 queries)
# - Moderate complexity queries (5 queries)  
# - Complex research queries (5 queries)
# - Edge case handling (5 test cases)
# - Performance benchmarking
# - MCP integration testing
# - Error handling validation
```

### Test Categories

1. **Simple Queries**: Basic biomedical questions
2. **Moderate Queries**: Multi-step biomedical reasoning
3. **Complex Queries**: Research-level comprehensive analysis
4. **Edge Cases**: Error conditions and boundary cases
5. **Performance Tests**: Execution time and consistency
6. **Integration Tests**: MCP tool connectivity
7. **Error Handling**: Graceful failure management

### Expected Test Results

- **Success Rate**: ≥80% for production readiness
- **Average Execution Time**: <5 seconds per query
- **MCP Integration**: All 3 tools (bio_ner, build_trapi_query, call_bte_api) working
- **Error Handling**: ≥80% of error conditions handled gracefully

---

## 📈 Performance Metrics

### GoT Framework Metrics

The system tracks and reports metrics from the original GoT paper:

- **Volume**: Number of thoughts generated
- **Latency**: Number of reasoning hops
- **Quality Improvement**: Enhancement over baseline methods
- **Cost Reduction**: Efficiency gains from parallel execution
- **Parallel Speedup**: Performance improvement from concurrent processing

### Example Output

```
📈 GOT FRAMEWORK METRICS
--------------------------------------------------
Volume: 8 thoughts
Latency: 4 hops
Total Thoughts: 12
Quality Improvement: 1.25x
Cost Reduction: 0.80x
Parallel Speedup: 1.40x
Volume-Latency Efficiency: 2.00

Thought Type Distribution:
  entity_extraction: 2 (16.7%)
  query_building: 4 (33.3%)
  api_execution: 4 (33.3%)
  aggregation: 2 (16.7%)
```

---

## 🐛 Debugging and Troubleshooting

### Enable Debug Mode

```bash
python got_query_runner.py --debug "your query here"
```

Debug mode provides:

- **Step-by-step execution details**
- **TRAPI query visualization** for each BTE call
- **Entity extraction results** with confidence scores
- **API response analysis** with result counts
- **GoT framework metrics** and thought distribution
- **Error messages** with context

### TRAPI Query Debugging

When debug mode is enabled, you'll see the actual TRAPI queries:

```json
TRAPI Query 1 (query_building):
Step ID: query_building_1234567890
```json
{
  "message": {
    "query_graph": {
      "nodes": {
        "n0": {
          "categories": ["biolink:Gene"],
          "ids": ["GENE:1234"],
          "name": "insulin"
        }
      },
      "edges": {}
    }
  },
  "submitter": "GoT-Framework",
  "query_type": "biomedical_research"
}
```

### Common Issues and Solutions

1. **MCP Tool Connection Issues**
   - Check MCP server is running
   - Verify tool availability with test suite
   - Review timeout settings

2. **Low Quality Results**
   - Adjust confidence thresholds
   - Enable iterative refinement
   - Check entity extraction quality

3. **Slow Performance**
   - Increase max concurrent operations
   - Reduce max iterations
   - Enable parallel execution

4. **Memory Issues**
   - Reduce max results per query
   - Disable graph visualization
   - Limit concurrent operations

---

## 📁 Project Structure

```
agentic-bte/
├── agentic_bte/
│   └── core/
│       └── queries/
│           ├── production_got_optimizer.py    # Main production optimizer
│           ├── mcp_integration.py            # Real MCP tool integration
│           ├── result_presenter.py           # Comprehensive result presentation
│           ├── got_framework.py              # Core GoT components
│           ├── got_aggregation.py            # Biomedical aggregation
│           ├── got_metrics.py                # Performance metrics
│           └── ...                           # Other framework components
├── got_query_runner.py                       # User-friendly CLI interface
├── test_production_got.py                    # Comprehensive test suite
├── demo_got_functionality.py                # Demo with simulated data
└── README_GOT_PRODUCTION.md                 # This file
```

---

## 🔬 Real MCP Integration

### MCP Tools Integration

The system integrates with real MCP tools:

1. **bio_ner**: Biomedical named entity recognition
2. **build_trapi_query**: TRAPI query construction
3. **call_bte_api**: BioThings Explorer API calls

### Fallback Mechanisms

- **Intelligent fallbacks** when MCP tools are unavailable
- **Keyword-based entity extraction** as backup
- **Structured TRAPI query generation** with fallback patterns
- **Simulated API responses** for testing and development

### Production Deployment

For production deployment with real MCP tools:

1. Ensure MCP server is running and accessible
2. Configure tool endpoints and authentication
3. Test all MCP tools with the integration test suite
4. Monitor tool availability and performance
5. Set appropriate timeouts and retry policies

---

## 🏆 Production Readiness Features

### Error Handling
- **Graceful degradation** when tools are unavailable
- **Comprehensive error messages** with context
- **Fallback mechanisms** for critical components
- **Retry logic** with exponential backoff

### Performance Optimization
- **Parallel execution** of independent operations
- **Caching mechanisms** for repeated queries
- **Resource management** and cleanup
- **Configurable performance parameters**

### Monitoring and Logging
- **Detailed execution logging** at multiple levels
- **Performance metrics** tracking and reporting
- **Error tracking** and analysis
- **Result quality** assessment

### Scalability
- **Configurable concurrency** limits
- **Resource pool management**
- **Horizontal scaling** capabilities
- **Load balancing** ready architecture

---

## 🛠️ Development and Extension

### Adding New MCP Tools

1. Update `mcp_integration.py` with new tool handlers
2. Add integration tests in `test_production_got.py`
3. Update result presentation for new data types
4. Add configuration options as needed

### Customizing Result Presentation

1. Modify `result_presenter.py` for new output formats
2. Add new visualization types
3. Extend debugging information
4. Customize final answer generation

### Performance Tuning

1. Adjust configuration parameters in `ProductionConfig`
2. Monitor execution metrics
3. Profile bottleneck operations
4. Optimize concurrent execution patterns

---

## 📚 References and Citation

This implementation is based on:

**"Graph of Thoughts: Solving Elaborate Problems with Large Language Models"**  
*Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Michal Podstawski, Hubert Niewiadomski, Piotr Nyczyk, Torsten Hoefler*

The implementation adapts the GoT framework specifically for biomedical query optimization with the following enhancements:

- **Domain-specific entity recognition and handling**
- **TRAPI query integration for biomedical APIs**
- **Biomedical result aggregation and confidence scoring**
- **Research-oriented final answer generation**
- **Production-ready error handling and monitoring**

---

## 🤝 Support and Contributing

### Getting Help

1. **Run the test suite** to validate your environment
2. **Check debug output** for detailed execution information
3. **Review log files** for error details and performance metrics
4. **Use interactive mode** to test different query patterns

### Contributing

1. **Follow existing code patterns** and documentation standards
2. **Add comprehensive tests** for new features
3. **Update documentation** for API changes
4. **Maintain backward compatibility** where possible
5. **Follow the established error handling patterns**

---

## 📄 License and Usage

This implementation is designed for research and production use in biomedical query optimization. Please ensure compliance with any applicable licenses for the underlying MCP tools and biomedical databases.

---

**🎉 Ready to explore biomedical knowledge with Graph of Thoughts!**

Start with a simple query to see the system in action:

```bash
python got_query_runner.py "What genes are associated with diabetes?"
```

Or jump into interactive mode for an exploratory session:

```bash
python got_query_runner.py --interactive
```