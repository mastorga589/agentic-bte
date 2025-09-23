# Warp MCP Setup for Agentic BTE

This guide will help you set up the Agentic BTE MCP server in Warp, your AI terminal.

## Prerequisites

1. **Ensure the package is installed**:
   ```bash
   cd /Users/mastorga/Documents/agentic-bte
   pip install -e .
   ```

2. **Set your OpenAI API key**:
   ```bash
   export AGENTIC_BTE_OPENAI_API_KEY="your-openai-api-key-here"
   ```

3. **Verify spaCy models are installed**:
   ```bash
   python -c "import spacy; spacy.load('en_core_sci_lg'); print('✅ Models ready')"
   ```

## Warp MCP Configuration

### Option 1: Using the Pre-configured File

1. **Use the provided configuration**:
   The file `warp-mcp-config.json` is already configured for your system.

2. **Set your API key in your shell profile**:
   Add this to your `~/.zshrc` or `~/.bash_profile`:
   ```bash
   export AGENTIC_BTE_OPENAI_API_KEY="your-openai-api-key-here"
   ```

3. **Apply the configuration to Warp**:
   Copy the contents of `warp-mcp-config.json` to your Warp MCP settings.

### Option 2: Manual Configuration

If you need to customize the configuration, here's the template:

```json
{
  "mcpServers": {
    "agentic-bte": {
      "command": "python",
      "args": ["-m", "agentic_bte.servers.mcp.server"],
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

Once configured, you'll have access to these biomedical research tools in Warp:

### 1. `bio_ner` - Biomedical Entity Recognition
Extract and link biomedical entities from natural language text.

**Example**: "Extract entities from: Metformin is used to treat type 2 diabetes"

### 2. `build_trapi_query` - TRAPI Query Building  
Build TRAPI (Translator Reasoner API) queries from natural language questions.

**Example**: "Build a TRAPI query for: What drugs treat diabetes?"

### 3. `call_bte_api` - BTE Knowledge Graph Query
Execute TRAPI queries against the BioThings Explorer knowledge graph.

**Example**: "Query BTE for drug-disease relationships"

### 4. `plan_and_execute_query` - **Recommended**
End-to-end biomedical query processing with optimization.

**Example**: "Research what genes are associated with Alzheimer's disease"

## Testing Your Setup

1. **Test the server manually**:
   ```bash
   cd /Users/mastorga/Documents/agentic-bte
   python -m agentic_bte.servers.mcp.server
   ```
   
   This should start the MCP server. Press Ctrl+C to stop it.

2. **Test with sample queries**:
   Once configured in Warp, try these biomedical research queries:
   
   - "What drugs can treat diabetes?"
   - "What genes are associated with breast cancer?"
   - "How does aspirin work as an anti-inflammatory drug?"
   - "Which compounds target the EGFR pathway?"

## Configuration Options

### Environment Variables

You can customize the behavior by setting these environment variables:

```bash
# Core settings
export AGENTIC_BTE_OPENAI_API_KEY="your-key"
export AGENTIC_BTE_OPENAI_MODEL="gpt-4o"  # or gpt-4

# Performance tuning
export AGENTIC_BTE_MAX_RESULTS_PER_QUERY="30"  # Reduce for faster responses
export AGENTIC_BTE_CONFIDENCE_THRESHOLD="0.8"  # Increase for higher quality
export AGENTIC_BTE_MAX_SUBQUERIES="5"          # Reduce for simpler queries

# Debug settings
export AGENTIC_BTE_DEBUG_MODE="true"           # Enable for troubleshooting
export AGENTIC_BTE_LOG_LEVEL="DEBUG"           # Detailed logging
```

### Performance Tuning

For different use cases, adjust these settings in the Warp configuration:

**Fast Responses** (for quick queries):
```json
"AGENTIC_BTE_MAX_RESULTS_PER_QUERY": "20",
"AGENTIC_BTE_CONFIDENCE_THRESHOLD": "0.8",
"AGENTIC_BTE_MAX_SUBQUERIES": "5"
```

**Comprehensive Research** (for detailed analysis):
```json
"AGENTIC_BTE_MAX_RESULTS_PER_QUERY": "50",
"AGENTIC_BTE_CONFIDENCE_THRESHOLD": "0.6", 
"AGENTIC_BTE_MAX_SUBQUERIES": "10"
```

## Troubleshooting

### Common Issues

1. **"Module not found" errors**:
   ```bash
   # Reinstall the package
   cd /Users/mastorga/Documents/agentic-bte
   pip install -e .
   ```

2. **spaCy model errors**:
   ```bash
   # Install biomedical models
   python -m spacy download en_core_sci_lg
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz
   ```

3. **OpenAI API errors**:
   - Verify your API key is set correctly
   - Check that you have sufficient OpenAI credits
   - Try switching to "gpt-4" if "gpt-4o" isn't available

4. **Server startup issues**:
   ```bash
   # Check Python path and imports
   cd /Users/mastorga/Documents/agentic-bte
   python -c "from agentic_bte.servers.mcp.server import AgenticBTEMCPServer; print('✅ Import successful')"
   ```

### Debug Mode

Enable detailed logging for troubleshooting:

```json
"AGENTIC_BTE_DEBUG_MODE": "true",
"AGENTIC_BTE_LOG_LEVEL": "DEBUG"
```

### Logs

Server logs will appear in Warp's console, showing:
- Entity extraction results
- TRAPI query building process  
- BTE API calls and responses
- Error messages and debugging information

## Advanced Usage

### Custom Queries

The `plan_and_execute_query` tool can handle complex biomedical research questions:

- **Drug Discovery**: "What drugs can treat Crohn's disease by targeting inflammatory pathways?"
- **Mechanism of Action**: "How do BRCA mutations affect DNA repair and what drugs target these pathways?"
- **Gene-Disease**: "What genes regulate insulin sensitivity and could be drug targets for diabetes?"

### Batch Processing

For multiple queries, you can use the Python API directly:

```python
from agentic_bte.agents import execute_biomedical_research

queries = [
    "What drugs treat diabetes?",
    "What genes cause Alzheimer's?", 
    "How does metformin work?"
]

for query in queries:
    result = execute_biomedical_research(query)
    print(f"Query: {query}")
    print(f"Answer: {result['final_answer']}")
```

## Support

If you encounter issues:

1. Check the main documentation: `README.md` and `MCP_SETUP.md`
2. Review the example scripts in `examples/`
3. Test individual components with the unit tests: `pytest tests/`
4. Check your environment variables and API keys

---

**Happy biomedical research with Warp! 🧬🤖**