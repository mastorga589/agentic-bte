# Enhanced GoT System - Debugging & Scientific Validation Guide

## 🔬 Overview

This guide demonstrates how to use the comprehensive debugging system for the Enhanced GoT framework. The debugging system provides detailed scientific validation to ensure the system produces scientifically sound results like your Brucellosis example.

## 🚀 Quick Start

### Method 1: Interactive Menu System
```bash
python run_debug_demo.py
```

This launches an interactive menu where you can choose:
1. **Full System Debugging** - Complete analysis with all validation steps
2. **Individual Component Testing** - Test each MCP tool separately  
3. **Quick System Test** - Basic system validation
4. **Entity Extraction Test** - Test only bio_ner tool
5. **TRAPI Query Building Test** - Test only build_trapi_query tool
6. **BTE API Test** - Test only call_bte_api tool

### Method 2: Direct Execution
```bash
python debug_enhanced_got_demo.py
```

Runs the full comprehensive debugging demonstration directly.

## 🔍 What the Debugging System Validates

### 1. **Entity Extraction Validation**
- ✅ **Biomedical Entity Recognition**: Verifies entities are properly classified as diseases, drugs, genes, etc.
- ✅ **Entity ID Assignment**: Confirms entities have valid biomedical IDs (MONDO:, CHEBI:, NCBIGene:, etc.)
- ✅ **Entity Type Accuracy**: Validates entity types match biomedical ontologies

**Example Output:**
```
🧬 STEP 1: ENTITY EXTRACTION & VALIDATION
-----------------------------------------
Raw Entity Response: 1247 characters
Entities Found: 3

📊 ENTITY VALIDATION RESULTS:
  • Total entities: 3
  • Valid biomedical entities: 3
  • Entity types found: ['disease', 'biologicalprocess', 'drug']

🏷️  DETAILED ENTITY BREAKDOWN:
  1. Brucellosis
     Type: disease
     ID: MONDO:0005683
     Valid biomedical: ✅
     Has valid ID: ✅

  2. translation
     Type: biologicalprocess
     ID: GO:0006412
     Valid biomedical: ✅
     Has valid ID: ✅
```

### 2. **TRAPI Query Structure Validation**
- ✅ **Valid TRAPI Format**: Confirms query follows TRAPI specification
- ✅ **Biomedical Categories**: Verifies nodes use proper biolink categories
- ✅ **Scientific Predicates**: Validates edges use meaningful biomedical relationships
- ✅ **Entity ID Integration**: Ensures extracted entities are properly used in queries

**Example Output:**
```
📋 TRAPI QUERY ANALYSIS:
      Valid TRAPI: ✅
      Nodes: 2
      Edges: 1
      📊 TRAPI Structure:
        Node n0:
          Categories: ['biolink:Disease']
          IDs: ['MONDO:0005683']
        Node n1:
          Categories: ['biolink:SmallMolecule']
          IDs: []
        Edge e0:
          Predicates: ['biolink:treats']
          From: n1 To: n0
```

### 3. **Knowledge Graph Results Validation**
- ✅ **Relationship Quality**: Analyzes confidence scores and relationship validity
- ✅ **Entity Name Resolution**: Verifies raw IDs are resolved to readable names
- ✅ **Predicate Distribution**: Shows types of biomedical relationships found
- ✅ **Scientific Soundness**: Validates relationships make biological sense

**Example Output:**
```
🔗 API RESULTS ANALYSIS:
      Total results: 12
      Valid relationships: 12
      Confidence distribution:
        High (>0.7): 4
        Medium (0.4-0.7): 6
        Low (<0.4): 2
      Entity name resolution: 92.3% (24/26)
      📋 Sample relationships:
        1. doxycycline → treats → Brucellosis
           Confidence: 0.856
        2. streptomycin → treats → Brucellosis
           Confidence: 0.734
```

### 4. **Domain Expertise Integration Validation**
- ✅ **Pharmaceutical Context**: Checks for disease/pathophysiology explanation
- ✅ **Mechanistic Reasoning**: Verifies mechanism of action explanations
- ✅ **Drug Classification**: Confirms proper antibiotic class categorization
- ✅ **Expert Inference**: Validates use of domain knowledge to fill gaps
- ✅ **Specific Examples**: Ensures concrete drug examples with mechanisms

**Example Output:**
```
🧠 STEP 4: DOMAIN EXPERTISE ANALYSIS
------------------------------------
Final answer length: 1456 characters

🔬 Domain expertise integration analysis:
  ✅ Pharmaceutical Context: ['brucellosis', 'infectious disease', 'bacteria']
  ✅ Mechanistic Explanation: ['translation', 'protein synthesis', 'ribosome']
  ✅ Drug Classification: ['antibiotic', 'tetracycline', 'aminoglycoside']
  ✅ Expert Inference: ['medicinal chemistry', 'drug class']
  ✅ Specific Examples: ['doxycycline', 'streptomycin']
  ✅ Mechanism Details: ['30S ribosome', 'peptidyl transferase']

📊 Domain Expertise Score: 100.0% (6/6)
🏆 EXCELLENT: High-level pharmaceutical sciences expertise demonstrated!
```

## 🔧 Individual Component Testing

### Entity Extraction Test
Tests the bio_ner MCP tool to ensure proper biomedical entity recognition:

```bash
python run_debug_demo.py
# Select option 4
```

**What it validates:**
- Entity extraction accuracy
- Biomedical type classification
- Entity ID assignment
- Confidence scoring

### TRAPI Query Building Test  
Tests the build_trapi_query MCP tool for proper query construction:

```bash
python run_debug_demo.py
# Select option 5
```

**What it validates:**
- TRAPI specification compliance
- Proper biolink category usage
- Entity ID integration
- Edge/predicate selection

### BTE API Test
Tests the call_bte_api MCP tool for knowledge graph querying:

```bash
python run_debug_demo.py  
# Select option 6
```

**What it validates:**
- API connectivity and response
- Result structure and quality
- Entity name resolution
- Confidence scoring

## 📊 Scientific Validation Criteria

The debugging system uses these criteria to assess scientific soundness:

### ✅ **PASS Criteria**
- **Entity Extraction Success**: ≥1 valid biomedical entity extracted
- **TRAPI Queries Valid**: ≥80% of TRAPI queries follow specification
- **Name Resolution Success**: ≥70% of entity IDs resolved to readable names
- **Domain Expertise Integration**: ≥60% of expertise indicators present
- **Scientific Relationships Found**: ≥1 valid biomedical relationship discovered

### 📈 **Quality Scoring**
- **80-100%**: Production-ready with high scientific rigor
- **60-79%**: Good scientific foundation, minor improvements needed
- **<60%**: Requires significant improvements for scientific accuracy

## 🎯 Expected Results for Brucellosis Query

For the query "What drugs can treat Brucellosis by targeting translation?", a scientifically sound system should demonstrate:

### Domain Context
- Explanation of Brucellosis as bacterial infection
- Importance of translation for bacterial survival

### Mechanistic Understanding
- Translation process and ribosome function
- How translation inhibitors kill bacteria

### Expert Classification
- Tetracyclines (doxycycline): 30S ribosome inhibitor
- Aminoglycosides (streptomycin): translation fidelity inhibitor  
- Chloramphenicol: peptidyl transferase inhibitor
- Rifamycins (rifampicin): indirect via transcription inhibition

### Scientific Evidence
- TRAPI queries targeting disease-drug relationships
- Knowledge graph evidence for therapeutic uses
- Resolved entity names (not raw UMLS/MONDO IDs)

## 🐛 Troubleshooting

### Common Issues

**1. Entity Extraction Fails**
```
❌ Entity extraction failed: OpenAI API key not found
```
**Solution**: Set your OpenAI API key in environment variables:
```bash
export AGENTIC_BTE_OPENAI_API_KEY="your-key-here"
```

**2. MCP Tool Not Found**
```
❌ TRAPI building failed: Unknown tool: build_trapi_query
```
**Solution**: Ensure MCP server is running and tools are available

**3. BTE API Connection Issues**
```
❌ BTE API call failed: Connection timeout
```
**Solution**: Check internet connection and BTE API availability

**4. Low Entity Name Resolution**
```
Entity name resolution: 23.1% (3/13)
```
**Solution**: This indicates the system isn't properly resolving entity IDs to readable names. Check entity resolution components.

### Debug Logging

Enable detailed logging for troubleshooting:
```bash
export AGENTIC_BTE_LOG_LEVEL=DEBUG
python debug_enhanced_got_demo.py
```

## 🔬 System Validation Report

The debugging system generates a comprehensive validation report:

```
📈 STEP 6: SCIENTIFIC VALIDATION SUMMARY
-----------------------------------------
✅ VALIDATION RESULTS:
  Entity Extraction Success: ✅ PASS
  Trapi Queries Valid: ✅ PASS
  Name Resolution Success: ✅ PASS
  Domain Expertise Integration: ✅ PASS
  Scientific Relationships Found: ✅ PASS

🎯 OVERALL SCIENTIFIC VALIDATION: 100.0% (5/5)
🏆 SYSTEM STATUS: Production-ready with high scientific rigor!
```

This validation ensures the system produces responses with the same level of sophistication as your Brucellosis example, combining knowledge graph evidence with domain expertise to provide scientifically accurate and comprehensive answers.

## 📝 Usage Examples

### Quick Validation
```bash
python run_debug_demo.py
# Select option 3 for quick test
```

### Full Scientific Analysis  
```bash
python run_debug_demo.py
# Select option 1 for comprehensive debugging
```

### Component-by-Component Testing
```bash
python run_debug_demo.py
# Select option 2 for individual component tests
```

The debugging system ensures the Enhanced GoT framework maintains the scientific rigor and domain expertise demonstrated in your Brucellosis example while providing full transparency into the system's reasoning process.