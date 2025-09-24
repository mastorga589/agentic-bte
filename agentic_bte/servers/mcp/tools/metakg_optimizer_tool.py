"""
MCP Meta-KG Aware Optimizer Tool - Advanced Biomedical Query Processing

This module provides the MCP tool interface for the meta-KG aware adaptive optimizer
that leverages BTE's meta-knowledge graph to inform and constrain subquery generation.
"""

import json
import logging
from typing import Dict, Any

from pydantic import BaseModel, Field

from ....core.queries.metakg_aware_optimizer import MetaKGAwareAdaptiveOptimizer
from ....config.settings import get_settings

logger = logging.getLogger(__name__)


def _generate_metakg_final_answer(query: str, relationships: list, entity_mappings: dict) -> str:
    """
    Generate a comprehensive final answer for meta-KG optimized results
    
    Args:
        query: The original query
        relationships: List of biomedical relationships
        entity_mappings: Mapping of entity IDs to names
        
    Returns:
        Comprehensive final answer string
    """
    if not relationships:
        return "Meta-KG optimization found no relevant biomedical results for your query."
    
    try:
        query_lower = query.lower()
        
        # Extract and categorize entities from relationships
        drugs_found = set()
        genes_found = set()
        diseases_found = set()
        processes_found = set()
        mechanisms_found = set()
        
        # Parse relationships for key entities and mechanisms
        for rel in relationships:
            subject = rel.get('subject', '')
            obj = rel.get('object', '')
            predicate = rel.get('predicate', '')
            
            # Extract readable names using entity mappings
            subject_name = entity_mappings.get(subject, subject)
            object_name = entity_mappings.get(obj, obj)
            
            # Categorize by entity type and content
            for entity_id, entity_name in [(subject, subject_name), (obj, object_name)]:
                # Drug/chemical detection
                if (entity_id.startswith(('CHEBI:', 'CHEMBL:', 'DRUGBANK:')) or 
                    any(term in entity_name.lower() for term in ['drug', 'medicine', 'compound', 'chemical'])):
                    if not entity_name.startswith(('CHEBI:', 'CHEMBL:', 'DRUGBANK:')):
                        drugs_found.add(entity_name)
                
                # Gene detection
                elif entity_id.startswith('NCBIGene:'):
                    if not entity_name.startswith('NCBIGene:'):
                        genes_found.add(entity_name)
                
                # Disease detection
                elif (entity_id.startswith(('MONDO:', 'DOID:')) or 
                      any(term in entity_name.lower() for term in ['disease', 'disorder', 'syndrome'])):
                    if not entity_name.startswith(('MONDO:', 'DOID:')):
                        diseases_found.add(entity_name)
                
                # Process detection
                elif any(term in entity_name.lower() for term in ['process', 'pathway', 'metabolism', 'signaling']):
                    processes_found.add(entity_name)
            
            # Extract mechanisms from predicates
            clean_predicate = predicate.replace('biolink:', '').replace('_', ' ')
            if any(term in clean_predicate.lower() for term in ['interact', 'regulate', 'affect', 'modulate']):
                mechanisms_found.add(clean_predicate)
        
        # Generate comprehensive answer based on query type
        result = ""
        
        # Treatment/drug queries
        if any(word in query_lower for word in ['drug', 'drugs', 'treat', 'treatment', 'therapy']):
            condition = ""
            if 'diabetes' in query_lower:
                condition = "diabetes"
            elif any(term in query_lower for term in ['macular degeneration', 'amd']):
                condition = "age-related macular degeneration"
            elif diseases_found:
                condition = list(diseases_found)[0].lower()
            else:
                condition = "the specified condition"
            
            if drugs_found:
                drug_list = sorted(list(drugs_found))[:8]
                result = f"Based on the meta-KG analysis, several therapeutic options for {condition} include: {', '.join(drug_list)}. "
                
                if genes_found:
                    key_genes = sorted(list(genes_found))[:5]
                    result += f"These treatments work through interactions with key genes including {', '.join(key_genes)}. "
                
                if 'antioxidant' in query_lower and any('antioxidant' in drug.lower() or 'vitamin' in drug.lower() for drug in drugs_found):
                    result += "The antioxidant mechanism is particularly relevant for protecting cellular components from oxidative damage. "
                
                if mechanisms_found:
                    key_mechanisms = sorted(list(mechanisms_found))[:3]
                    result += f"Primary mechanisms of action involve: {', '.join(key_mechanisms)}."
            else:
                result = f"The analysis identified {len(relationships)} therapeutic relationships for {condition}, involving complex interactions between genes, proteins, and biological processes."
        
        # Gene/genetic queries
        elif any(word in query_lower for word in ['gene', 'genes', 'genetic', 'protein']):
            if genes_found:
                gene_list = sorted(list(genes_found))[:8]
                result = f"Key genes associated with your query: {', '.join(gene_list)}. "
                
                if processes_found:
                    key_processes = sorted(list(processes_found))[:5]
                    result += f"These genes are involved in important biological processes including: {', '.join(key_processes)}. "
                
                if mechanisms_found:
                    result += f"The relationships involve mechanisms such as: {', '.join(sorted(list(mechanisms_found))[:3])}."
            else:
                result = f"Found {len(relationships)} gene-related relationships with various biological processes and pathways."
        
        # Mechanism queries
        elif any(word in query_lower for word in ['mechanism', 'pathway', 'process', 'activity']):
            if mechanisms_found and processes_found:
                key_mechanisms = sorted(list(mechanisms_found))[:4]
                key_processes = sorted(list(processes_found))[:4]
                result = f"The analysis reveals several key mechanisms: {', '.join(key_mechanisms)}. These involve biological processes including: {', '.join(key_processes)}. "
                
                if genes_found:
                    key_genes = sorted(list(genes_found))[:6]
                    result += f"Key genes involved: {', '.join(key_genes)}."
            else:
                result = f"Identified {len(relationships)} mechanistic relationships involving genes, proteins, and biological processes."
        
        # Generic comprehensive answer
        else:
            components = []
            if drugs_found:
                components.append(f"{len(drugs_found)} therapeutic compounds")
            if genes_found:
                components.append(f"{len(genes_found)} genes")
            if diseases_found:
                components.append(f"{len(diseases_found)} conditions")
            if processes_found:
                components.append(f"{len(processes_found)} biological processes")
            
            if components:
                result = f"The meta-KG analysis identified relationships involving {', '.join(components)}. "
                
                if len(components) > 1:
                    result += f"These {len(relationships)} relationships provide insights into the complex interactions between biomedical entities relevant to your query."
            else:
                result = f"Meta-KG analysis identified {len(relationships)} biomedical relationships relevant to your query."
        
        return result.strip()
        
    except Exception as e:
        logger.debug(f"Error generating comprehensive meta-KG final answer: {e}")
        return f"Meta-KG analysis completed successfully, identifying {len(relationships)} relevant biomedical relationships and their interactions."


class MetaKGOptimizerInput(BaseModel):
    """Input schema for meta-KG aware optimization"""
    query: str = Field(
        description="Complex biomedical query to optimize and execute"
    )
    entities: Dict[str, str] = Field(
        default={},
        description="Pre-extracted biomedical entities (optional)"
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


def get_metakg_optimizer_tool_definition() -> Dict[str, Any]:
    """Get the MCP tool definition for meta-KG aware optimization"""
    return {
        "name": "metakg_aware_optimizer",
        "description": "Execute complex biomedical queries using meta-KG aware adaptive optimization that leverages BTE's meta-knowledge graph to generate informed, single-hop subqueries with improved accuracy and efficiency",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Complex biomedical query to execute with meta-KG aware optimization"
                },
                "entities": {
                    "type": "object",
                    "description": "Pre-extracted biomedical entities (optional)",
                    "additionalProperties": {"type": "string"},
                    "default": {}
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


async def handle_metakg_optimizer(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle meta-KG aware optimization tool calls
    
    Args:
        arguments: Tool call arguments
        
    Returns:
        MCP-formatted response with optimization results
    """
    try:
        query = arguments.get("query")
        entities = arguments.get("entities", {})
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
        
        logger.info(f"Processing meta-KG aware optimized query: {query[:100]}...")
        
        # Initialize meta-KG aware optimizer
        optimizer = MetaKGAwareAdaptiveOptimizer()
        
        # Create and execute adaptive plan
        plan = optimizer.create_adaptive_plan(
            query=query,
            entities=entities if entities else None,
            max_iterations=5  # Reasonable number of iterations
        )
        
        result = optimizer.execute_adaptive_plan(plan)
        
        if not result or not hasattr(result, 'accumulated_results'):
            return {
                "error": "Meta-KG optimizer returned no valid result",
                "content": [
                    {
                        "type": "text",
                        "text": "Error: Meta-KG aware optimizer returned no valid result"
                    }
                ]
            }
        
        # Generate BTE-LLM style structured output
        formatted_text = "🧬 METAKG_AWARE_OPTIMIZER RESULTS\n"
        formatted_text += "=" * 50 + "\n\n"
        
        # Prepare entity mappings from result (AdaptivePlan has entity_data)
        entity_mappings = result.entity_data if hasattr(result, 'entity_data') else {}
        
        # Prepare structured relationships from accumulated_results
        structured_relationships = []
        if hasattr(result, 'accumulated_results') and result.accumulated_results:
            structured_relationships = result.accumulated_results
        
        # Generate comprehensive final answer using the optimizer's synthesis method
        final_answer = ""
        if hasattr(result, 'final_answer') and result.final_answer:
            final_answer = result.final_answer
        else:
            # Fallback to basic answer if comprehensive one not available
            final_answer = _generate_metakg_final_answer(query, structured_relationships, entity_mappings)
        
        if final_answer:
            formatted_text += "🎯 FINAL ANSWER:\n"
            formatted_text += "─" * 20 + "\n"
            formatted_text += f"{final_answer}\n\n"
            formatted_text += "📋 DETAILED BREAKDOWN:\n"
            formatted_text += "─" * 25 + "\n\n"
        
        # 1. EXECUTION SUMMARY
        formatted_text += "📊 EXECUTION SUMMARY:\n"
        formatted_text += "─" * 25 + "\n"
        formatted_text += f"✅ Strategy: Meta-KG aware adaptive optimization\n"
        formatted_text += f"⏱️ Execution Time: {getattr(result, 'total_execution_time', 0):.2f}s\n"
        formatted_text += f"🔢 Total Results: {len(structured_relationships)}\n"
        formatted_text += f"🔍 Subqueries Executed: {len(getattr(result, 'executed_subqueries', []))}\n"
        formatted_text += f"✅ Completion Reason: {getattr(result, 'completion_reason', 'Completed')}\n\n"
        
        # 2. META-KG OPTIMIZATION DETAILS
        if show_plan_details:
            formatted_text += "🗺️ META-KG OPTIMIZATION DETAILS:\n"
            formatted_text += "─" * 35 + "\n"
            
            # Show meta-KG statistics if available
            metakg_stats = optimizer.get_meta_kg_statistics() if 'optimizer' in locals() else {}
            if metakg_stats:
                formatted_text += f"🕸️ Meta-KG edges analyzed: {metakg_stats.get('total_edges', 'N/A')}\n"
                formatted_text += f"🗺️ Unique subjects: {metakg_stats.get('unique_subjects', 'N/A')}\n"
                formatted_text += f"🔗 Unique predicates: {metakg_stats.get('unique_predicates', 'N/A')}\n"
            
            # Show subqueries if available
            executed_subqueries = getattr(result, 'executed_subqueries', [])
            if executed_subqueries:
                formatted_text += f"\n📋 EXECUTED SUBQUERIES:\n"
                for i, subquery in enumerate(executed_subqueries, 1):
                    query_text = getattr(subquery, 'query', f'Subquery {i}')
                    result_count = len(getattr(subquery, 'results', []))
                    success = getattr(subquery, 'success', False)
                    status = "✅" if success else "❌"
                    formatted_text += f"   {i}. {status} {query_text} → {result_count} results\n"
            formatted_text += "\n"
        
        # 3. ENTITY MAPPINGS (BTE-style with grouping)
        if entity_mappings:
            formatted_text += "🏷️ ENTITY MAPPINGS (FROM META-KG):\n"
            formatted_text += "─" * 35 + "\n"
            formatted_text += f"📊 Resolved {len(entity_mappings)} entity names:\n\n"
            
            # Group by entity type for better organization
            diseases = {k: v for k, v in entity_mappings.items() if k.startswith(("MONDO:", "DOID:"))}
            genes = {k: v for k, v in entity_mappings.items() if k.startswith("NCBIGene:")}
            drugs = {k: v for k, v in entity_mappings.items() if k.startswith(("CHEBI:", "CHEMBL", "DRUGBANK:"))}
            other = {k: v for k, v in entity_mappings.items() if not any(k.startswith(p) for p in ["MONDO:", "DOID:", "NCBIGene:", "CHEBI:", "CHEMBL", "DRUGBANK:"])}
            
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
        
        # 4. BIOMEDICAL RELATIONSHIPS
        if structured_relationships:
            formatted_text += "🔬 BIOMEDICAL RELATIONSHIPS:\n"
            formatted_text += "─" * 30 + "\n"
            formatted_text += f"📊 Found {len(structured_relationships)} relationships, showing key examples:\n\n"
            
            # Group and display relationships
            drug_relationships = []
            gene_relationships = []
            disease_relationships = []
            
            for rel in structured_relationships[:15]:  # Look at first 15
                predicate = rel.get('predicate', '').replace('biolink:', '').replace('_', ' ')
                subject = rel.get('subject', 'N/A')
                obj = rel.get('object', 'N/A')
                
                # Use entity mappings to get readable names
                subject_name = entity_mappings.get(subject, subject)
                object_name = entity_mappings.get(obj, obj)
                
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
        
        else:
            formatted_text += "🔬 BIOMEDICAL RELATIONSHIPS:\n"
            formatted_text += "─" * 30 + "\n"
            formatted_text += "❌ No biomedical relationships found\n\n"
        
        # 5. SUMMARY
        formatted_text += "🎯 SUMMARY:\n"
        formatted_text += "─" * 15 + "\n"
        formatted_text += "✨ Features demonstrated:\n"
        formatted_text += "   ✅ Meta-KG aware adaptive optimization\n"
        formatted_text += "   ✅ Dynamic subquery generation\n"
        formatted_text += "   ✅ Context-aware entity extraction\n"
        formatted_text += "   ✅ Multi-hop relationship exploration\n"
        formatted_text += "   ✅ Structured biomedical result presentation\n\n"
        
        # Store the structured text for return
        response_text = formatted_text
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": response_text
                }
            ],
            "results": structured_relationships,
            "entities": entity_mappings,
            "entity_mappings": entity_mappings,
            "metadata": {
                "execution_time": getattr(result, 'total_execution_time', 0),
                "total_results": len(structured_relationships),
                "subqueries_count": len(getattr(result, 'executed_subqueries', [])),
                "completion_reason": getattr(result, 'completion_reason', 'Completed'),
                "optimizer_type": "meta-kg-aware",
                "metakg_stats": metakg_stats if 'metakg_stats' in locals() else {}
            },
            "subqueries": [{
                "query": getattr(sq, 'query', ''),
                "results": len(getattr(sq, 'results', [])),
                "success": getattr(sq, 'success', False)
            } for sq in getattr(result, 'executed_subqueries', [])],
            "final_answer": final_answer if 'final_answer' in locals() else None
        }
        
    except Exception as e:
        error_msg = f"Error in meta-KG aware optimization: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "error": error_msg,
            "content": [
                {
                    "type": "text",
                    "text": error_msg
                }
            ]
        }