# Agentic BTE MCP Server - Warp Setup Guide

This guide explains how to set up and configure the Agentic BTE MCP server with Warp terminal.

## Quick Setup

### 1. Installation

The server is installed as part of the `agentic-bte` package:

```bash
cd /Users/mastorga/Documents/agentic-bte
pip install -e .
```

This installs the `agentic-bte-mcp` console command.

### 2. Environment Configuration

Create a `.env` file in the project root (already exists):

```bash
# Required: OpenAI API Key
AGENTIC_BTE_OPENAI_API_KEY=your_openai_api_key_here

# Optional: Model Configuration
AGENTIC_BTE_OPENAI_MODEL=gpt-4o

# Optional: Feature Flags
AGENTIC_BTE_ENABLE_SEMANTIC_CLASSIFICATION=true
AGENTIC_BTE_ENABLE_ENTITY_NAME_RESOLUTION=true
AGENTIC_BTE_ENABLE_QUERY_OPTIMIZATION=true

# Optional: Processing Parameters
AGENTIC_BTE_MAX_SUBQUERIES=10
AGENTIC_BTE_CONFIDENCE_THRESHOLD=0.7
AGENTIC_BTE_MAX_RESULTS_PER_QUERY=50
```

### 3. Warp Configuration

The Warp MCP configuration is in `warp-mcp-config.json`:

```json
{
  "mcpServers": {
    "agentic-bte": {
      "command": "agentic-bte-mcp",
      "args": [],
      "cwd": "/Users/mastorga/Documents/agentic-bte",
      "env": {
        "PYTHONPATH": "/Users/mastorga/Documents/agentic-bte",
        "AGENTIC_BTE_OPENAI_API_KEY": "${AGENTIC_BTE_OPENAI_API_KEY}",
        "AGENTIC_BTE_OPENAI_MODEL": "gpt-4o",
        "AGENTIC_BTE_DEBUG_MODE": "false",
        "AGENTIC_BTE_LOG_LEVEL": "INFO",
        "AGENTIC_BTE_ENABLE_SEMANTIC_CLASSIFICATION": "true",
        "AGENTIC_BTE_ENABLE_ENTITY_NAME_RESOLUTION": "true",
        "AGENTIC_BTE_ENABLE_QUERY_OPTIMIZATION": "true",
        "AGENTIC_BTE_MAX_SUBQUERIES": "10",
        "AGENTIC_BTE_CONFIDENCE_THRESHOLD": "0.7",
        "AGENTIC_BTE_MAX_RESULTS_PER_QUERY": "50"
      }
    }
  }
}
```

## Available MCP Tools

The server provides 5 powerful MCP tools for biomedical research:

1. **`bio_ner`** - Biomedical entity extraction and linking
   - Extracts entities like diseases, drugs, genes, proteins
   - Links to standardized biomedical identifiers (UMLS, etc.)

2. **`build_trapi_query`** - TRAPI query construction
   - Converts natural language to TRAPI format
   - Uses LLM for intelligent query building

3. **`call_bte_api`** - BTE API execution
   - Executes TRAPI queries against BioThings Explorer
   - Handles batching and result aggregation

4. **`plan_and_execute_query`** - Complete biomedical query processing
   - **End-to-end biomedical question answering**
   - **Built-in query optimization:** Intelligent batching, retry logic, result aggregation
   - **Semantic query classification:** Automatic query type detection and optimization
   - **Advanced entity extraction:** spaCy + SciSpaCy + LLM-based biological process extraction
   - **Smart TRAPI construction:** LLM-powered query building with validation and fallbacks
   - **Optimized BTE execution:** Batched API calls with parallel processing
   - **Comprehensive result synthesis:** Entity name resolution and structured output
   - **Perfect for complex biomedical research questions**

5. **`got_biomedical_query`** - **🧠 ADVANCED: Graph of Thoughts (GoT) Framework**
   - **🎯 Research-grade biomedical query optimization**
   - **📊 Graph-based reasoning with parallel thought execution**
   - **🔍 TRAPI query visualization and debugging**
   - **📈 Performance metrics tracking (volume, latency, quality)**
   - **🔄 Comprehensive result aggregation and refinement**
   - **📋 Executive summaries and detailed breakdowns**
   - **⚡ Iterative quality improvement with feedback loops**
   - **🧬 Ideal for: Complex research questions, drug discovery, pathway analysis**
   - **📚 Based on latest research in AI reasoning systems**

## Usage Examples

### Simple Entity Extraction
```
Use the bio_ner tool to extract entities from: "What drugs can treat diabetes?"
```

### Complex Research Query
```
Use the plan_and_execute_query tool to analyze: "What are the molecular mechanisms by which metformin reduces blood glucose levels in Type 2 diabetes patients?"
```

### End-to-End Processing
```
Use the plan_and_execute_query tool for: "Which genes are targeted by drugs used to treat Alzheimer's disease and how do they relate to neuroinflammation?"
```

### 🧠 Advanced GoT Research Queries
```
Use the got_biomedical_query tool for: "Construct a comprehensive analysis of the molecular interactions between oxidative stress, inflammation, and neurodegeneration in Parkinson's disease"
```

### GoT with Custom Configuration
```
Use the got_biomedical_query tool with parameters:
{
  "query": "What is the molecular network connecting insulin signaling to glucose metabolism in diabetes?",
  "output_format": "comprehensive",
  "confidence_threshold": 0.8,
  "show_debug": true
}
```

### GoT Summary Mode
```
Use the got_biomedical_query tool with:
{
  "query": "How do BRCA1 mutations contribute to breast cancer risk?",
  "output_format": "summary"
}
```

## System Requirements

- Python 3.10+
- OpenAI API key
- Internet connection for BTE API
- Optional: spaCy biomedical models for enhanced NER

### SpaCy Models (Optional)

For enhanced biomedical entity recognition:

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_lg-0.5.0.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_ner_bc5cdr_md-0.5.0.tar.gz
```

## Troubleshooting

### Server Won't Start
1. Check that the package is installed: `which agentic-bte-mcp`
2. Verify environment variables are set
3. Check Python path and dependencies

### API Errors
1. Verify OpenAI API key is valid
2. Check internet connection for BTE API
3. Review log output for specific error messages

### spaCy Warnings
The warnings about spaCy model versions are non-fatal and don't affect functionality.

## Development

For development and testing:

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run server directly for debugging
python -m agentic_bte.servers.mcp.server
```

## Security Notes

- Never commit `.env` files with API keys
- Use environment variable references in shared configs
- The `.gitignore` file excludes sensitive files by default

## Architecture

The new server architecture provides:

- **Modular Design**: Clean separation between NER, query building, and execution
- **Advanced Optimization**: Query decomposition and parallel processing
- **Robust Error Handling**: Comprehensive fallbacks and error recovery  
- **Extensible Framework**: Easy to add new biomedical data sources
- **LangGraph Integration**: Multi-agent workflows for complex research

This replaces the previous prototype with a production-ready, scalable system for biomedical knowledge graph queries.