# Agentic BTE Examples

This directory contains working examples that demonstrate the capabilities of the Agentic BTE biomedical research platform. These examples show how to use both the MCP server interface and the LangGraph multi-agent system for various biomedical research workflows.

## 🚀 Quick Start

### Prerequisites

1. **Install Agentic BTE**:
   ```bash
   pip install -e .
   ```

2. **Set up environment variables**:
   ```bash
   export AGENTIC_BTE_OPENAI_API_KEY="your-openai-api-key"
   export AGENTIC_BTE_OPENAI_MODEL="gpt-4o"  # or gpt-4
   ```

3. **Install required spaCy models**:
   ```bash
   python -m spacy download en_core_sci_lg
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz
   ```

## 📋 Available Examples

### 1. MCP Basic Usage (`mcp_basic_usage.py`)

**What it demonstrates**:
- Biomedical entity recognition and linking
- TRAPI query building from natural language  
- BTE API execution with knowledge graph queries
- End-to-end biomedical query processing

**Usage**:
```bash
python examples/mcp_basic_usage.py
```

**Key features shown**:
- ✅ Entity extraction from biomedical text
- ✅ Automatic TRAPI query generation
- ✅ BTE knowledge graph integration
- ✅ Entity name resolution with real drug/gene names
- ✅ Error handling and graceful fallbacks

**Expected output**:
```
🧬 BioNER Example
==================================================
Query: What drugs can treat type 2 diabetes mellitus?

✅ Entities extracted successfully!
  • diabetes: MONDO:0005015 (disease)
  • drugs: biolink:ChemicalEntity (general)
```

### 2. LangGraph Multi-Agent System (`langgraph_agents.py`)

**What it demonstrates**:
- Multi-agent research orchestration
- Query decomposition and iterative planning
- RDF knowledge graph accumulation
- Complex biomedical reasoning workflows

**Usage**:
```bash
python examples/langgraph_agents.py
```

**Key features shown**:
- 🤖 Automatic agent orchestration (Annotator → Planner → BTE Search → Summary)
- 🧠 Intelligent query decomposition into single-hop subqueries
- 🕸️ RDF knowledge graph accumulation across iterations
- ⚖️ Comparison of different research strategies
- 📊 Detailed execution analytics and performance metrics

**Expected output**:
```
🔬 Simple Research Example
==================================================
Research Query: What drugs can treat diabetes?

✅ Research completed successfully!
  • Total subqueries: 3
  • Successful subqueries: 3
  • Knowledge triples: 25
  • Execution time: 45.2s
  • Average confidence: 0.73
```

## 🔧 Configuration Options

### Environment Variables

```bash
# Core API settings (required)
export AGENTIC_BTE_OPENAI_API_KEY="your-openai-api-key"
export AGENTIC_BTE_OPENAI_MODEL="gpt-4o"

# Processing parameters (optional)
export AGENTIC_BTE_MAX_SUBQUERIES="10"
export AGENTIC_BTE_CONFIDENCE_THRESHOLD="0.7" 
export AGENTIC_BTE_MAX_RESULTS_PER_QUERY="50"

# Debug settings (optional)
export AGENTIC_BTE_DEBUG_MODE="true"
export AGENTIC_BTE_LOG_LEVEL="INFO"
```

### Customizing Parameters

Both examples accept various parameters you can modify in the code:

```python
# MCP example parameters
result = await handle_plan_and_execute({
    "query": "Your research question",
    "max_results": 30,        # Max results per BTE call
    "k": 5,                  # Max results per entity
    "confidence_threshold": 0.6  # Minimum confidence for results
})

# LangGraph example parameters  
result = orchestrator.execute_research(
    query="Your research question",
    maxresults=25,           # Max results per BTE call
    k=4,                    # Max results per entity  
    confidence_threshold=0.5,  # Minimum confidence
    recursion_limit=50      # Max LangGraph recursion depth
)
```

## 🧪 Example Research Queries

### Drug Discovery
- "What drugs can treat type 2 diabetes?"
- "Which compounds target the EGFR pathway?"
- "What are the side effects of metformin?"

### Gene-Disease Associations  
- "What genes are associated with Alzheimer's disease?"
- "Which genes are involved in breast cancer?"
- "What genetic variants affect drug metabolism?"

### Mechanistic Research
- "How does aspirin work as an anti-inflammatory?"
- "What are the mechanisms of action of immunotherapy?"
- "Which pathways are disrupted in Parkinson's disease?"

### Complex Multi-Step Queries
- "Which drugs can treat Crohn's disease by targeting inflammatory pathways?"
- "What genes regulate insulin sensitivity and could be drug targets for diabetes?"
- "How do BRCA mutations affect DNA repair and what drugs target these pathways?"

## 📊 Understanding the Output

### MCP Tool Results

```python
{
    "content": [...],           # Human-readable response text
    "results": [...],          # Raw relationship data  
    "entities": {...},         # Extracted biomedical entities
    "entity_mappings": {...},  # ID-to-name mappings
    "metadata": {...}          # Execution statistics
}
```

### LangGraph Research Results

```python
{
    "success": True,
    "final_answer": "...",              # Synthesized research answer
    "execution_summary": {
        "total_subqueries": 3,
        "successful_subqueries": 3, 
        "total_execution_time": 45.2,
        "average_confidence": 0.73,
        "rdf_triples_count": 25
    },
    "rdf_graph": "...",                 # Turtle-formatted knowledge graph
    "graph_statistics": {...},         # RDF graph analytics
    "subquery_results": [...]          # Individual subquery performance
}
```

## 🔍 Troubleshooting

### Common Issues

1. **"Command not found" errors**:
   ```bash
   # Ensure package is installed
   pip install -e .
   ```

2. **spaCy model errors**:
   ```bash
   # Install required biomedical models
   python -m spacy download en_core_sci_lg
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz
   ```

3. **OpenAI API errors**:
   ```bash
   # Set your API key
   export AGENTIC_BTE_OPENAI_API_KEY="your-key-here"
   ```

4. **BTE API timeout issues**:
   - Try reducing `max_results` and `k` parameters
   - Increase `confidence_threshold` to get more focused results
   - Use simpler, more specific queries

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
export AGENTIC_BTE_DEBUG_MODE="true"
export AGENTIC_BTE_LOG_LEVEL="DEBUG"
python examples/mcp_basic_usage.py
```

### Performance Tips

1. **For faster results**:
   - Use higher `confidence_threshold` (0.8+)
   - Reduce `max_results` (10-20)
   - Use more specific queries

2. **For comprehensive results**:
   - Use lower `confidence_threshold` (0.5-0.6)
   - Increase `max_results` (30-50)
   - Allow more subqueries in LangGraph

## 🚀 Next Steps

After running these examples:

1. **Explore the Source Code**: 
   - MCP tools: `agentic_bte/servers/mcp/tools/`
   - LangGraph agents: `agentic_bte/agents/`

2. **Customize for Your Use Case**:
   - Modify agent prompts and behavior
   - Add new biomedical query types
   - Integrate with your existing workflows

3. **Scale Up**:
   - Process multiple queries in batch
   - Set up automated research pipelines
   - Deploy as a service for your team

4. **Contribute**:
   - Add new example use cases
   - Improve agent reasoning capabilities
   - Extend to new biomedical domains

## 📚 Additional Resources

- **Main Documentation**: See `README.md` in the project root
- **MCP Setup**: See `MCP_SETUP.md` for Claude Desktop integration
- **Architecture**: See `docs/architecture.md` for technical details  
- **API Reference**: See `agentic_bte/` module documentation

---

**Happy researching! 🧬🤖**