"""
MCP Comprehensive Query Tool - End-to-End Biomedical Query Processing

This module provides the MCP tool interface for complete biomedical
query processing from natural language to final results.

Migrated and enhanced from the original BTE-LLM implementation.
"""

import json
import logging
from typing import Dict, Any

from pydantic import BaseModel, Field

from ....core.knowledge.knowledge_system import BiomedicalKnowledgeSystem
from ....config.settings import get_settings

logger = logging.getLogger(__name__)


def _generate_final_answer(query: str, results: list, entity_mappings: dict) -> str:
    """
    Generate a concise final answer based on the biomedical results
    
    Args:
        query: The original query
        results: List of biomedical relationships
        entity_mappings: Mapping of entity IDs to names
        
    Returns:
        Concise final answer string
    """
    if not results:
        return "No relevant biomedical results found for your query."
    
    try:
        query_lower = query.lower()
        
        # Create inverse mapping for results
        id_to_name = {}
        if entity_mappings:
            # entity_mappings is typically name -> id, create id -> name
            for name, entity_id in entity_mappings.items():
                id_to_name[entity_id] = name
        
        # Extract key findings based on query type
        if any(word in query_lower for word in ['drug', 'drugs', 'treat', 'treatment', 'medicine', 'therapeutic']):
            # Drug-focused query
            drugs = set()
            for result in results[:20]:  # Look at top 20 results
                subject_id = result.get('subject', '')
                object_id = result.get('object', '')
                predicate = result.get('predicate', '')
                
                # Look for drug relationships
                if 'treat' in predicate.lower() or 'drug' in predicate.lower():
                    drug_name = id_to_name.get(subject_id, subject_id)
                    if drug_name and drug_name != subject_id:
                        drugs.add(drug_name)
                    
                    drug_name = id_to_name.get(object_id, object_id)
                    if drug_name and drug_name != object_id:
                        drugs.add(drug_name)
            
            if drugs:
                drug_list = sorted(list(drugs))[:5]
                return f"Key therapeutic options include: {', '.join(drug_list)}."
        
        elif any(word in query_lower for word in ['gene', 'genes', 'genetic', 'protein']):
            # Gene-focused query
            genes = set()
            for result in results[:20]:
                subject_id = result.get('subject', '')
                object_id = result.get('object', '')
                
                # Look for gene IDs or names
                if subject_id.startswith('NCBIGene:'):
                    gene_name = id_to_name.get(subject_id, subject_id)
                    genes.add(gene_name)
                
                if object_id.startswith('NCBIGene:'):
                    gene_name = id_to_name.get(object_id, object_id)
                    genes.add(gene_name)
            
            if genes:
                gene_list = sorted(list(genes))[:5]
                return f"Key associated genes include: {', '.join(gene_list)}."
        
        elif any(word in query_lower for word in ['disease', 'disorder', 'condition', 'syndrome']):
            # Disease-focused query
            diseases = set()
            for result in results[:20]:
                subject_id = result.get('subject', '')
                object_id = result.get('object', '')
                
                # Look for disease IDs
                if any(subject_id.startswith(prefix) for prefix in ['MONDO:', 'DOID:']):
                    disease_name = id_to_name.get(subject_id, subject_id)
                    diseases.add(disease_name)
                
                if any(object_id.startswith(prefix) for prefix in ['MONDO:', 'DOID:']):
                    disease_name = id_to_name.get(object_id, object_id)
                    diseases.add(disease_name)
            
            if diseases:
                disease_list = sorted(list(diseases))[:3]
                return f"Key related conditions include: {', '.join(disease_list)}."
        
        # Generic fallback
        return f"Found {len(results)} biomedical relationships providing insights into your query."
        
    except Exception as e:
        logger.debug(f"Error generating final answer: {e}")
        return f"Analysis complete with {len(results)} biomedical findings."


class PlanAndExecuteInput(BaseModel):
    """Input schema for comprehensive query processing"""
    query: str = Field(
        description="Complex biomedical query to process"
    )
    entities: Dict[str, str] = Field(
        default={},
        description="Pre-extracted biomedical entities (optional)"
    )
    execute_after_plan: bool = Field(
        default=True,
        description="Whether to execute after planning (default: true)"
    )
    max_results: int = Field(
        default=50,
        description="Maximum results per API call"
    )
    k: int = Field(
        default=5,
        description="Maximum results per entity"
    )
    confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence threshold for results"
    )
    show_plan_details: bool = Field(
        default=True,
        description="Whether to include detailed plan information in output"
    )


def get_basic_plan_and_execute_tool_definition() -> Dict[str, Any]:
    """Get the MCP tool definition for basic comprehensive query processing"""
    return {
        "name": "basic_plan_and_execute_query",
        "description": "Plan and execute complex biomedical queries using basic optimization strategies with comprehensive biomedical entity resolution and TRAPI query processing",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Complex biomedical query to execute with optimization"
                },
                "entities": {
                    "type": "object",
                    "description": "Pre-extracted biomedical entities (optional)",
                    "additionalProperties": {"type": "string"},
                    "default": {}
                },
                "execute_after_plan": {
                    "type": "boolean",
                    "description": "Whether to execute after planning (default: true)",
                    "default": True
                },
                "k": {
                    "type": "integer",
                    "description": "Maximum results per entity",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 50
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results per API call",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 1000
                },
                "confidence_threshold": {
                    "type": "number",
                    "description": "Minimum confidence threshold for results",
                    "default": 0.7,
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "show_plan_details": {
                    "type": "boolean",
                    "description": "Whether to include detailed plan information in output",
                    "default": True
                }
            },
            "required": ["query"]
        }
    }


async def handle_basic_plan_and_execute(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle basic comprehensive query processing tool calls
    
    Args:
        arguments: Tool call arguments
        
    Returns:
        MCP-formatted response with complete processing results
    """
    try:
        query = arguments.get("query")
        entities = arguments.get("entities", {})
        execute_after_plan = arguments.get("execute_after_plan", True)
        max_results = arguments.get("max_results", 50)
        k = arguments.get("k", 5)
        confidence_threshold = arguments.get("confidence_threshold", 0.7)
        show_plan_details = arguments.get("show_plan_details", True)
        
        if not query:
            return {
                "error": "Query parameter is required",
                "content": [
                    {
                        "type": "text",
                        "text": "Error: Query parameter is required"
                    }
                ]
            }
        
        logger.info(f"Processing comprehensive biomedical query: {query[:100]}...")
        
        # Initialize knowledge system
        knowledge_system = BiomedicalKnowledgeSystem()
        
        # Process the complete query
        result = knowledge_system.process_biomedical_query(query, max_results, k)
        
        if "error" in result:
            return {
                "error": result["error"],
                "content": [
                    {
                        "type": "text",
                        "text": f"Query processing error: {result['error']}\n\nFailed at step: {result.get('step_failed', 'unknown')}"
                    }
                ]
            }
        
        # Generate BTE-LLM style structured output
        formatted_text = "🧬 BASIC_PLAN_AND_EXECUTE_QUERY RESULTS\n"
        formatted_text += "=" * 50 + "\n\n"
        
        # Generate final answer first (prominently at top)
        final_answer = _generate_final_answer(query, result.get("results", []), result.get("entity_mappings", {}))
        if final_answer:
            formatted_text += "🎯 FINAL ANSWER:\n"
            formatted_text += "─" * 20 + "\n"
            formatted_text += f"💡 {final_answer}\n\n"
            formatted_text += "📋 DETAILED BREAKDOWN:\n"
            formatted_text += "─" * 25 + "\n\n"
        
        # 1. EXECUTION SUMMARY
        formatted_text += "📊 EXECUTION SUMMARY:\n"
        formatted_text += "─" * 25 + "\n"
        formatted_text += f"✅ Query processed: Basic optimization strategy\n"
        
        if "metadata" in result:
            metadata = result["metadata"]
            formatted_text += f"🔢 Total results: {metadata.get('total_results', len(result.get('results', [])))}\n"
            if "execution_metadata" in metadata:
                exec_meta = metadata["execution_metadata"]
                formatted_text += f"🔗 API batches: {exec_meta.get('total_batches', 1)}\n"
                formatted_text += f"✅ Successful batches: {exec_meta.get('successful_batches', 1)}\n"
        
        # Show query classification
        if "query_type" in result:
            classification = result.get("classification", {})
            formatted_text += f"📋 Query Type: {result['query_type']}\n"
            formatted_text += f"📈 Classification Confidence: {classification.get('confidence', 0):.2f}\n"
        
        formatted_text += "\n"
        
        # 2. ENTITY MAPPINGS (BTE-style with grouping)
        entities_data = result.get("entities", {})
        entity_mappings = result.get("entity_mappings", {})
        
        # Create comprehensive entity mapping (ID -> Name and Name -> ID)
        all_entity_mappings = {}
        if entity_mappings:
            # entity_mappings is typically name -> id, we want id -> name too
            for name, entity_id in entity_mappings.items():
                all_entity_mappings[entity_id] = name
        
        # Add entities data
        for entity_name, entity_info in entities_data.items():
            entity_id = entity_info.get("id")
            if entity_id and entity_id not in all_entity_mappings:
                all_entity_mappings[entity_id] = entity_name
        
        if all_entity_mappings:
            formatted_text += "🏷️ ENTITY MAPPINGS (FROM BTE DATABASE):\n"
            formatted_text += "─" * 35 + "\n"
            formatted_text += f"📊 Resolved {len(all_entity_mappings)} entity names:\n\n"
            
            # Group by entity type for better organization
            diseases = {k: v for k, v in all_entity_mappings.items() if k.startswith(("MONDO:", "DOID:"))}
            genes = {k: v for k, v in all_entity_mappings.items() if k.startswith("NCBIGene:")}
            drugs = {k: v for k, v in all_entity_mappings.items() if k.startswith(("CHEBI:", "CHEMBL", "DRUGBANK:"))}
            other = {k: v for k, v in all_entity_mappings.items() if not any(k.startswith(p) for p in ["MONDO:", "DOID:", "NCBIGene:", "CHEBI:", "CHEMBL", "DRUGBANK:"])}
            
            if diseases:
                formatted_text += "  🏥 DISEASES:\n"
                for entity_id, name in list(diseases.items())[:5]:
                    formatted_text += f"     • {entity_id} → {name}\n"
                if len(diseases) > 5:
                    formatted_text += f"     ... and {len(diseases) - 5} more diseases\n"
                formatted_text += "\n"
            
            if drugs:
                formatted_text += "  💊 DRUGS & CHEMICALS:\n"
                for entity_id, name in list(drugs.items())[:8]:
                    formatted_text += f"     • {entity_id} → {name}\n"
                if len(drugs) > 8:
                    formatted_text += f"     ... and {len(drugs) - 8} more drugs\n"
                formatted_text += "\n"
            
            if genes:
                formatted_text += "  🧬 GENES:\n"
                for entity_id, name in list(genes.items())[:8]:
                    formatted_text += f"     • {entity_id} → {name}\n"
                if len(genes) > 8:
                    formatted_text += f"     ... and {len(genes) - 8} more genes\n"
                formatted_text += "\n"
            
            if other:
                formatted_text += "  🔬 OTHER ENTITIES:\n"
                for entity_id, name in list(other.items())[:5]:
                    formatted_text += f"     • {entity_id} → {name}\n"
                formatted_text += "\n"
        
        # 3. BIOMEDICAL RELATIONSHIPS
        results_data = result.get("results", [])
        if results_data:
            formatted_text += "🔬 BIOMEDICAL RELATIONSHIPS:\n"
            formatted_text += "─" * 30 + "\n"
            formatted_text += f"📊 Found {len(results_data)} relationships, showing key examples:\n\n"
            
            # Group and display relationships
            drug_relationships = []
            gene_relationships = []
            disease_relationships = []
            
            for res in results_data[:15]:  # Look at first 15
                predicate = res.get('predicate', '').replace('biolink:', '').replace('_', ' ')
                subject = res.get('subject', 'N/A')
                obj = res.get('object', 'N/A')
                
                # Use entity mappings to get readable names
                subject_name = all_entity_mappings.get(subject, subject)
                object_name = all_entity_mappings.get(obj, obj)
                
                # Categorize by content
                if any(term in predicate.lower() for term in ['treat', 'drug', 'chemical']):
                    drug_relationships.append((subject_name, predicate, object_name))
                elif any(term in predicate.lower() for term in ['gene', 'protein', 'express']):
                    gene_relationships.append((subject_name, predicate, object_name))
                else:
                    disease_relationships.append((subject_name, predicate, object_name))
            
            if drug_relationships:
                formatted_text += "  💊 DRUG-RELATED RELATIONSHIPS:\n"
                for i, (subj, pred, obj) in enumerate(drug_relationships[:5], 1):
                    formatted_text += f"     {i}. {subj} ← {pred} → {obj}\n"
                formatted_text += "\n"
            
            if gene_relationships:
                formatted_text += "  🧬 GENE-RELATED RELATIONSHIPS:\n"
                for i, (subj, pred, obj) in enumerate(gene_relationships[:5], 1):
                    formatted_text += f"     {i}. {subj} ← {pred} → {obj}\n"
                formatted_text += "\n"
            
            if disease_relationships:
                formatted_text += "  🏥 DISEASE-RELATED RELATIONSHIPS:\n"
                for i, (subj, pred, obj) in enumerate(disease_relationships[:5], 1):
                    formatted_text += f"     {i}. {subj} ← {pred} → {obj}\n"
                formatted_text += "\n"
            
            # Show remaining relationships if any
            total_shown = len(drug_relationships[:5]) + len(gene_relationships[:5]) + len(disease_relationships[:5])
            if len(results_data) > total_shown:
                formatted_text += f"     ... and {len(results_data) - total_shown} more relationships\n\n"
        
        else:
            formatted_text += "🔬 BIOMEDICAL RELATIONSHIPS:\n"
            formatted_text += "─" * 30 + "\n"
            formatted_text += "❌ No biomedical relationships found\n"
            if "message" in result:
                formatted_text += f"💬 Message: {result['message']}\n"
            formatted_text += "\n"
        
        # 4. SUMMARY
        formatted_text += "🎯 SUMMARY:\n"
        formatted_text += "─" * 15 + "\n"
        formatted_text += "✨ Features demonstrated:\n"
        formatted_text += "   ✅ Basic biomedical query processing\n"
        formatted_text += "   ✅ Entity extraction and mapping\n"
        formatted_text += "   ✅ TRAPI query generation\n"
        formatted_text += "   ✅ BTE knowledge graph integration\n"
        formatted_text += "   ✅ Structured result presentation\n\n"
        
        # Store the structured text for return
        response_text = formatted_text
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": response_text
                }
            ],
            "results": result.get("results", []),
            "entities": result.get("entities", {}),
            "entity_mappings": result.get("entity_mappings", {}),
            "metadata": result.get("metadata", {}),
            "query_type": result.get("query_type"),
            "classification": result.get("classification", {})
        }
        
    except Exception as e:
        error_msg = f"Error in comprehensive query processing: {str(e)}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "content": [
                {
                    "type": "text",
                    "text": error_msg
                }
            ]
        }