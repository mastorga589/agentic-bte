# Agentic BTE MCP Server Setup

This guide will help you set up and configure the Agentic BTE MCP (Model Context Protocol) server for use with MCP-compatible clients like Claude Desktop.

## Prerequisites

1. **Python Environment**: Ensure you have Python 3.8+ installed
2. **OpenAI API Key**: Required for LLM operations
3. **Package Installation**: Install the Agentic BTE package

## Installation

1. **Install the package in development mode**:
   ```bash
   cd /Users/mastorga/Documents/agentic-bte
   pip install -e .
   ```

2. **Install required spaCy models** (if not already installed):
   ```bash
   python -m spacy download en_core_sci_lg
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root or set environment variables:
   ```bash
   export AGENTIC_BTE_OPENAI_API_KEY="your-openai-api-key-here"
   export AGENTIC_BTE_OPENAI_MODEL="gpt-4o"
   ```

## MCP Server Configuration

### For Claude Desktop

1. **Locate Claude Desktop config file**:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%/Claude/claude_desktop_config.json`

2. **Add the server configuration**:
   ```json
   {
     "mcpServers": {
       "agentic-bte": {
         "command": "agentic-bte-mcp",
         "args": [],
         "cwd": "/Users/mastorga/Documents/agentic-bte",
         "env": {
           "AGENTIC_BTE_OPENAI_API_KEY": "your-openai-api-key-here",
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

3. **Restart Claude Desktop** to load the new configuration

### For Other MCP Clients

Use the provided `mcp-config.json` file as a template and adjust paths/settings as needed for your specific MCP client.

## Available MCP Tools

Once configured, your MCP client will have access to these powerful biomedical research tools:

### 1. `bio_ner`
Extract and link biomedical entities from natural language text.

**Parameters**:
- `query` (string): Text to extract biomedical entities from

**Example**:
```json
{
  "name": "bio_ner",
  "arguments": {
    "query": "Metformin is used to treat type 2 diabetes mellitus"
  }
}
```

### 2. `build_trapi_query`
Build TRAPI (Translator Reasoner API) queries from natural language biomedical questions.

**Parameters**:
- `query` (string): Natural language biomedical query
- `entity_data` (object, optional): Pre-extracted entity ID mappings
- `failed_trapis` (array, optional): Previously failed TRAPI queries to avoid

**Example**:
```json
{
  "name": "build_trapi_query",
  "arguments": {
    "query": "What drugs treat diabetes?",
    "entity_data": {
      "diabetes": "MONDO:0005015"
    }
  }
}
```

### 3. `call_bte_api`
Execute TRAPI queries against the BioThings Explorer knowledge graph.

**Parameters**:
- `json_query` (object): TRAPI query object
- `maxresults` (integer, optional): Maximum number of results (default: 50)
- `k` (integer, optional): Maximum results per entity (default: 5)

### 4. `plan_and_execute_query`
**Recommended**: End-to-end biomedical query processing with optimization.

**Parameters**:
- `query` (string): Complex biomedical query to process
- `entities` (object, optional): Pre-extracted entities
- `max_results` (integer, optional): Maximum results per API call (default: 50)
- `k` (integer, optional): Maximum results per entity (default: 5)
- `confidence_threshold` (number, optional): Minimum confidence threshold (default: 0.7)

**Example**:
```json
{
  "name": "plan_and_execute_query",
  "arguments": {
    "query": "What drugs can treat diabetes?",
    "max_results": 20,
    "k": 3
  }
}
```

### 5. `got_biomedical_query`
**🧠 ADVANCED**: Graph of Thoughts (GoT) biomedical query optimization with research-grade analysis.

**Parameters**:
- `query` (string): Complex biomedical research query
- `output_format` (string, optional): "comprehensive", "summary", or "debug" (default: "comprehensive")
- `show_debug` (boolean, optional): Enable TRAPI query debugging (default: true)
- `confidence_threshold` (number, optional): Minimum confidence threshold (default: 0.7)
- `max_iterations` (integer, optional): Maximum GoT iterations (default: 5)
- `enable_refinement` (boolean, optional): Enable iterative refinement (default: true)
- `save_results` (boolean, optional): Save results to files (default: false)

**Features**:
- Graph-based reasoning with parallel execution
- TRAPI query visualization and debugging
- Performance metrics (volume, latency, quality)
- Executive summaries and detailed breakdowns
- Iterative result refinement

**Example**:
```json
{
  "name": "got_biomedical_query",
  "arguments": {
    "query": "What is the molecular network connecting oxidative stress and neurodegeneration in Parkinson's disease?",
    "output_format": "comprehensive",
    "confidence_threshold": 0.8,
    "show_debug": true
  }
}
```

## Testing the Setup

1. **Test the server directly**:
   ```bash
   agentic-bte-mcp
   ```
   (The server will start and listen on stdio)

2. **Test with your MCP client**: Ask biomedical questions like:
   - "What drugs can treat diabetes?"
   - "What genes are associated with Alzheimer's disease?"
   - "How does aspirin work as a drug?"

## Troubleshooting

### Common Issues

1. **Command not found**: Make sure the package is installed with `pip install -e .`
2. **spaCy models missing**: Run the model installation commands from step 2
3. **OpenAI API errors**: Verify your API key is correctly set in environment variables
4. **Import errors**: Check that all dependencies are installed

### Debug Mode

Enable debug mode for more detailed logging:
```bash
export AGENTIC_BTE_DEBUG_MODE=true
export AGENTIC_BTE_LOG_LEVEL=DEBUG
```

### Logs

Server logs will appear in your MCP client's console or log files, showing:
- Entity extraction results
- TRAPI query building
- BTE API calls and responses
- Error messages and debugging information

## Configuration Options

All environment variables and their defaults:

```bash
# Core API settings
AGENTIC_BTE_OPENAI_API_KEY=""           # Required: OpenAI API key
AGENTIC_BTE_OPENAI_MODEL="gpt-4o"      # OpenAI model to use

# Processing settings
AGENTIC_BTE_MAX_SUBQUERIES="10"        # Max subqueries for complex queries
AGENTIC_BTE_CONFIDENCE_THRESHOLD="0.7" # Minimum confidence for results
AGENTIC_BTE_MAX_RESULTS_PER_QUERY="50" # Max results per BTE API call

# Feature toggles
AGENTIC_BTE_ENABLE_SEMANTIC_CLASSIFICATION="true"  # Enable query classification
AGENTIC_BTE_ENABLE_ENTITY_NAME_RESOLUTION="true"   # Enable entity name resolution
AGENTIC_BTE_ENABLE_QUERY_OPTIMIZATION="true"       # Enable query optimization

# Debug settings
AGENTIC_BTE_DEBUG_MODE="false"          # Enable debug mode
AGENTIC_BTE_LOG_LEVEL="INFO"           # Log level (DEBUG, INFO, WARNING, ERROR)
```

## Support

For issues or questions:
1. Check the logs for error messages
2. Verify all dependencies are installed
3. Ensure environment variables are correctly set
4. Test individual components using the Python API