# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository Overview

**Agentic BTE** is a biomedical research platform combining Large Language Models with BioThings Explorer knowledge graphs. It provides both **MCP Server** implementation and **LangGraph multi-agent** workflows for biomedical question answering and drug discovery.

This is a **cleaned and restructured version** migrated from the original BTE-LLM repository, focusing on production-ready, modular architecture.

## Development Setup

### Environment Setup
```bash
# Copy environment template and configure
cp .env.example .env
# Edit .env with your OpenAI API key and preferences

# Install in development mode with all dependencies
pip install -e ".[dev,test-external,notebooks]"

# Install required spaCy biomedical models
python -m spacy download en_core_sci_lg
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz
```

### Development Commands

#### Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests
pytest -m external      # Tests requiring external services (BTE API, OpenAI)

# Run with coverage reporting
pytest --cov=agentic_bte --cov-report=html

# Test a single module
pytest tests/unit/test_entities/ -v
```

#### Code Quality
```bash
# Format code (line length: 100, profiles: black + isort)
black agentic_bte/
isort agentic_bte/

# Type checking
mypy agentic_bte/

# Lint code
flake8 agentic_bte/
```

#### Running Services
```bash
# Launch MCP Server (primary interface)
agentic-bte-mcp

# Alternative server launch
python -m agentic_bte.servers.mcp.server

# Setup/install models script
agentic-bte-setup
```

## Core Architecture

### Module Structure
The codebase follows a **clean, layered architecture** migrated from the original monolithic BTE-LLM implementation:

- **`core/`**: Biomedical processing engine
  - **`entities/`**: NER, linking, classification, resolution (spaCy/SciSpaCy + LLM hybrid)  
  - **`queries/`**: Semantic classification, decomposition, optimization strategies
  - **`knowledge/`**: TRAPI query building, BTE API client, graph processing
- **`agents/`**: Multi-agent implementations (LangGraph orchestration)
- **`servers/`**: MCP server with biomedical tools + handlers
- **`config/`**: Centralized settings with environment variable management

### Key Processing Pipeline
1. **Entity Recognition**: Extract biomedical entities using spaCy/SciSpaCy + LLM classification
2. **Query Classification**: Determine query type (drug mechanism, disease treatment, etc.)
3. **Query Decomposition**: Break complex queries into optimized subqueries with dependency analysis
4. **Knowledge Graph Query**: Execute TRAPI queries against BTE knowledge graph
5. **Result Synthesis**: Generate human-readable answers using LLMs with entity name resolution

### Dual Interface Architecture

#### 1. MCP Server (Primary Interface)
- **Tools**: `bio_ner`, `build_trapi_query`, `call_bte_api`, `plan_and_execute_query`  
- **Optimization**: Query decomposition, parallel execution, dynamic replanning
- **Entity Resolution**: ID-to-name mapping using SRI Name Resolver + BTE metadata
- **Usage**: Direct tool calls via MCP protocol

#### 2. LangGraph Multi-Agent (Advanced Workflows)  
- **Agents**: Annotator → Planner → BTE_Search → Synthesis
- **Orchestration**: Dynamic routing with RDF knowledge graph accumulation
- **Recursive Planning**: Iterative subquery generation based on intermediate results
- **Usage**: Complex research workflows with multi-step reasoning

## Environment Configuration

### Required Settings
```bash
# Core API access
AGENTIC_BTE_OPENAI_API_KEY=your-key-here
AGENTIC_BTE_OPENAI_MODEL=gpt-4o  # or gpt-4

# Query processing tuning
AGENTIC_BTE_MAX_SUBQUERIES=10
AGENTIC_BTE_CONFIDENCE_THRESHOLD=0.7
AGENTIC_BTE_MAX_RESULTS_PER_QUERY=50

# Feature toggles
AGENTIC_BTE_ENABLE_SEMANTIC_CLASSIFICATION=true
AGENTIC_BTE_ENABLE_ENTITY_NAME_RESOLUTION=true
AGENTIC_BTE_ENABLE_QUERY_OPTIMIZATION=true

# Development
AGENTIC_BTE_DEBUG_MODE=false
AGENTIC_BTE_LOG_LEVEL=INFO
```

### External Services
- **BTE API**: `https://bte.transltr.io/v1` (knowledge graph queries)
- **SRI Name Resolver**: `https://name-lookup.ci.transltr.io/lookup` (entity names)
- **OpenAI API**: LLM operations for classification and synthesis

## Common Development Workflows

### Testing New Query Types
```bash
# Test entity recognition for new biomedical queries
python -c "
from agentic_bte.core.entities import extract_and_link_entities
result = extract_and_link_entities('What drugs treat diabetes?')
print(result)
"

# Test query optimization 
pytest tests/integration/test_query_optimization.py::test_diabetes_query -v -s
```

### MCP Tool Development
- **Tool locations**: `agentic_bte/servers/mcp/tools/`
- **Handler registration**: `agentic_bte/servers/mcp/server.py`
- **Input schemas**: Must follow MCP protocol with proper JSON schema validation

### Adding Biomedical Entity Types
- **Recognition**: Update `agentic_bte/core/entities/recognition.py` spaCy model configuration
- **Linking**: Add new linking strategies in `agentic_bte/core/entities/linking.py`  
- **Classification**: Update entity type mappings in classification logic

## Testing Strategy

### Test Categories (use pytest markers)
- **`unit`**: Fast tests for individual components
- **`integration`**: Tests requiring multiple components  
- **`external`**: Tests requiring external APIs (BTE, OpenAI, SRI)
- **`slow`**: Performance/benchmarking tests

### Key Test Scenarios
- **Entity Recognition**: Biomedical NER accuracy across different domains
- **Query Optimization**: Decomposition strategies for complex queries
- **BTE Integration**: TRAPI query building and result parsing
- **Error Handling**: Graceful fallbacks when external services fail

## Migration Notes from BTE-LLM

This repository represents a **cleaned production architecture** extracted from the original BTE-LLM research repository:

### What Was Migrated
- **MCP Server**: Core tools from `/MCP-Server/src/bte_mcp_server/`
- **LangGraph Agents**: Multi-agent orchestration from `/Prototype/Agent.py`  
- **Query Optimization**: Advanced decomposition strategies
- **Entity Processing**: Hybrid spaCy/LLM biomedical NER pipeline

### Architecture Improvements
- **Modular Design**: Separated concerns (entities, queries, knowledge, servers)
- **Configuration Management**: Centralized settings with environment variables
- **Error Handling**: Comprehensive exception hierarchy with graceful fallbacks
- **Testing Framework**: Complete test coverage with fixtures and integration tests
- **Documentation**: API documentation and developer guides

### Performance Optimizations
- **Result Caching**: Configurable TTL for BTE API responses
- **Batch Processing**: TRAPI query splitting for large entity sets
- **Retry Logic**: Exponential backoff for external API failures
- **Memory Management**: Efficient entity name resolution with deduplication

## Common Issues & Solutions

### spaCy Model Installation
If biomedical models are missing:
```bash
python -m spacy download en_core_sci_lg
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz
```

### BTE API Timeout Issues  
- Increase timeout in `agentic_bte/config/settings.py`
- Enable query decomposition to reduce result set size
- Use confidence thresholding to filter low-quality results

### Entity Linking Failures
- Check SRI Name Resolver availability 
- Verify entity text preprocessing (special characters, formatting)
- Enable fallback linking strategies in entity linking configuration

## Integration with External Tools

### MCP Client Usage (Warp, Claude, etc.)
The MCP server provides these tools for biomedical research:
- **`bio_ner`**: Extract and link biomedical entities
- **`build_trapi_query`**: Convert natural language to TRAPI
- **`call_bte_api`**: Execute knowledge graph queries  
- **`plan_and_execute_query`**: End-to-end optimized query execution

### LangGraph Integration
For complex multi-step biomedical research workflows, use the LangGraph multi-agent system with iterative planning and knowledge graph accumulation.