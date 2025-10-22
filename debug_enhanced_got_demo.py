#!/usr/bin/env python3
"""
Enhanced GoT System - Comprehensive Debugging Demonstration

This script provides detailed debugging information to validate the scientific 
soundness of the enhanced GoT system, including:

1. Subquery decomposition analysis
2. TRAPI query structure examination
3. Entity resolution verification (IDs to readable names)
4. Knowledge graph relationship validation
5. Domain expertise integration verification
6. Step-by-step execution tracing

This addresses your request for transparency in the system's reasoning process.
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List
from pprint import pformat

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from agentic_bte.core.queries.enhanced_got_optimizer import (
    EnhancedGoTOptimizer, EnhancedConfig, DomainExpertAnswerGenerator
)
from agentic_bte.core.queries.production_got_optimizer import ProductionGoTOptimizer
from agentic_bte.core.queries.mcp_integration import call_mcp_tool


class ScientificValidationDebugger:
    """
    Debugging class to validate scientific soundness of the enhanced GoT system
    """
    
    def __init__(self):
        self.execution_log = []
        self.validation_results = {}
        
    def log_step(self, step_type: str, details: Dict[str, Any]):
        """Log execution step with timestamp"""
        self.execution_log.append({
            'timestamp': time.time(),
            'step_type': step_type,
            'details': details
        })
        
    def validate_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate extracted biomedical entities for scientific accuracy"""
        validation = {
            'total_entities': len(entities),
            'entity_types_found': {},
            'valid_biomedical_entities': 0,
            'entity_details': []
        }
        
        biomedical_types = {
            'disease', 'drug', 'gene', 'protein', 'biologicalprocess', 
            'smallmolecule', 'chemical', 'biologicalentity'
        }
        
        for entity in entities:
            entity_type = entity.get('type', '').lower()
            entity_name = entity.get('name', 'Unknown')
            entity_id = entity.get('id', 'No ID')
            
            # Count entity types
            if entity_type in validation['entity_types_found']:
                validation['entity_types_found'][entity_type] += 1
            else:
                validation['entity_types_found'][entity_type] = 1
            
            # Check if it's a valid biomedical entity
            is_biomedical = any(bio_type in entity_type for bio_type in biomedical_types)
            if is_biomedical:
                validation['valid_biomedical_entities'] += 1
            
            validation['entity_details'].append({
                'name': entity_name,
                'type': entity_type,
                'id': entity_id,
                'is_biomedical': is_biomedical,
                'has_valid_id': ':' in entity_id and entity_id != 'No ID'
            })
        
        return validation
    
    def validate_trapi_query(self, trapi_query: Dict[str, Any]) -> Dict[str, Any]:
        """Validate TRAPI query structure for scientific soundness"""
        validation = {
            'is_valid_trapi': False,
            'has_query_graph': False,
            'node_count': 0,
            'edge_count': 0,
            'nodes_analysis': {},
            'edges_analysis': {},
            'scientific_categories': []
        }
        
        try:
            message = trapi_query.get('message', {})
            query_graph = message.get('query_graph', {})
            
            if query_graph:
                validation['has_query_graph'] = True
                
                # Analyze nodes
                nodes = query_graph.get('nodes', {})
                validation['node_count'] = len(nodes)
                
                for node_id, node_data in nodes.items():
                    categories = node_data.get('categories', [])
                    ids = node_data.get('ids', [])
                    
                    validation['nodes_analysis'][node_id] = {
                        'categories': categories,
                        'ids': ids,
                        'id_count': len(ids) if isinstance(ids, list) else 1,
                        'has_biomedical_category': any('biolink:' in cat for cat in categories)
                    }
                    
                    # Collect scientific categories
                    validation['scientific_categories'].extend(categories)
                
                # Analyze edges
                edges = query_graph.get('edges', {})
                validation['edge_count'] = len(edges)
                
                for edge_id, edge_data in edges.items():
                    predicates = edge_data.get('predicates', [])
                    subject = edge_data.get('subject')
                    object_node = edge_data.get('object')
                    
                    validation['edges_analysis'][edge_id] = {
                        'predicates': predicates,
                        'subject': subject,
                        'object': object_node,
                        'has_biomedical_predicate': any('biolink:' in pred for pred in predicates)
                    }
                
                validation['is_valid_trapi'] = (
                    validation['node_count'] > 0 and 
                    validation['edge_count'] >= 0 and
                    any(analysis['has_biomedical_category'] for analysis in validation['nodes_analysis'].values())
                )
        
        except Exception as e:
            validation['error'] = str(e)
        
        return validation
    
    def validate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate knowledge graph results for scientific relationships"""
        validation = {
            'total_results': len(results),
            'valid_relationships': 0,
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'predicate_types': {},
            'entity_name_resolution': {'resolved': 0, 'unresolved': 0},
            'sample_relationships': []
        }
        
        for result in results:
            # Check for valid relationship structure
            has_subject = 'subject' in result and result['subject']
            has_predicate = 'predicate' in result and result['predicate']
            has_object = 'object' in result and result['object']
            
            if has_subject and has_predicate and has_object:
                validation['valid_relationships'] += 1
                
                # Analyze confidence
                confidence = result.get('score', 0.0)
                if confidence > 0.7:
                    validation['confidence_distribution']['high'] += 1
                elif confidence > 0.4:
                    validation['confidence_distribution']['medium'] += 1
                else:
                    validation['confidence_distribution']['low'] += 1
                
                # Analyze predicate types
                predicate = result['predicate']
                if predicate in validation['predicate_types']:
                    validation['predicate_types'][predicate] += 1
                else:
                    validation['predicate_types'][predicate] = 1
                
                # Check entity name resolution (no raw IDs in names)
                subject_name = result['subject']
                object_name = result['object']
                
                if not any(id_prefix in subject_name for id_prefix in ['UMLS:', 'MONDO:', 'CHEBI:', 'NCBIGene:']):
                    validation['entity_name_resolution']['resolved'] += 1
                else:
                    validation['entity_name_resolution']['unresolved'] += 1
                
                if not any(id_prefix in object_name for id_prefix in ['UMLS:', 'MONDO:', 'CHEBI:', 'NCBIGene:']):
                    validation['entity_name_resolution']['resolved'] += 1
                else:
                    validation['entity_name_resolution']['unresolved'] += 1
                
                # Collect sample relationships
                if len(validation['sample_relationships']) < 5:
                    validation['sample_relationships'].append({
                        'subject': subject_name,
                        'predicate': predicate.replace('biolink:', ''),
                        'object': object_name,
                        'confidence': confidence
                    })
        
        return validation


async def demonstrate_enhanced_got_with_debugging():
    """Main demonstration with comprehensive debugging"""
    
    print("🔬 ENHANCED GOT SYSTEM - SCIENTIFIC VALIDATION DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Initialize debugger
    debugger = ScientificValidationDebugger()
    
    # Test query - your Brucellosis example
    query = "What drugs can treat Brucellosis by targeting translation?"
    print(f"📋 RESEARCH QUERY:")
    print(f'"{query}"')
    print()
    
    print("🔍 SCIENTIFIC VALIDATION OBJECTIVES:")
    print("-" * 40)
    print("✓ Validate entity extraction and biomedical typing")
    print("✓ Examine TRAPI query structure for scientific soundness")
    print("✓ Verify entity ID to readable name resolution")
    print("✓ Analyze knowledge graph relationships for validity")
    print("✓ Trace subquery decomposition strategy")
    print("✓ Validate domain expertise integration")
    print()
    
    # Configure enhanced system with maximum debugging
    config = EnhancedConfig(
        # Enable all debugging and expertise features
        enable_expert_inference=True,
        enable_mechanistic_reasoning=True,
        enable_rdf_accumulation=True,
        
        # Domain expertise
        pharmaceutical_expertise=True,
        medicinal_chemistry_expertise=True,
        biomedical_research_expertise=True,
        
        # Debugging settings
        show_debug=True,
        save_results=True,
        
        # Performance settings
        max_iterations=5,
        parallel_execution=True,
        max_concurrent=2,
        mcp_timeout=120
    )
    
    print("⏳ EXECUTING ENHANCED GOT SYSTEM WITH FULL DEBUGGING...")
    print("-" * 55)
    start_time = time.time()
    
    try:
        # Initialize enhanced optimizer
        optimizer = EnhancedGoTOptimizer(config)
        
        # Step 1: Entity Extraction Debugging
        print("\n🧬 STEP 1: ENTITY EXTRACTION & VALIDATION")
        print("-" * 45)
        
        try:
            entity_response = await call_mcp_tool("bio_ner", query=query)
            entities = entity_response.get('entities', [])
            
            print(f"Raw Entity Response: {len(str(entity_response))} characters")
            print(f"Entities Found: {len(entities)}")
            
            # Validate entities scientifically
            entity_validation = debugger.validate_entities(entities)
            debugger.log_step("entity_extraction", entity_validation)
            
            print(f"\n📊 ENTITY VALIDATION RESULTS:")
            print(f"  • Total entities: {entity_validation['total_entities']}")
            print(f"  • Valid biomedical entities: {entity_validation['valid_biomedical_entities']}")
            print(f"  • Entity types found: {list(entity_validation['entity_types_found'].keys())}")
            
            print(f"\n🏷️  DETAILED ENTITY BREAKDOWN:")
            for i, entity_detail in enumerate(entity_validation['entity_details'][:5], 1):
                print(f"  {i}. {entity_detail['name']}")
                print(f"     Type: {entity_detail['type']}")
                print(f"     ID: {entity_detail['id']}")
                print(f"     Valid biomedical: {'✅' if entity_detail['is_biomedical'] else '❌'}")
                print(f"     Has valid ID: {'✅' if entity_detail['has_valid_id'] else '❌'}")
                print()
            
            if len(entities) == 0:
                print("⚠️  WARNING: No entities extracted - this may affect query quality")
        
        except Exception as e:
            print(f"❌ Entity extraction failed: {e}")
            entities = []
        
        # Step 2: Execute full enhanced GoT system
        print("\n🚀 STEP 2: ENHANCED GOT EXECUTION")
        print("-" * 35)
        
        result, presentation = await optimizer.execute_query_with_expertise(query)
        
        execution_time = time.time() - start_time
        print(f"✅ Execution completed in {execution_time:.1f} seconds")
        
        # Step 3: Detailed Analysis of Execution Steps
        print("\n🔍 STEP 3: EXECUTION STEP ANALYSIS")
        print("-" * 35)
        
        if hasattr(result, 'execution_steps'):
            print(f"Total execution steps: {len(result.execution_steps)}")
            
            for i, step in enumerate(result.execution_steps, 1):
                print(f"\n  Step {i}: {step.step_type}")
                print(f"    Success: {'✅' if step.success else '❌'}")
                print(f"    Execution time: {step.execution_time:.3f}s")
                print(f"    Confidence: {step.confidence:.3f}")
                
                # Detailed analysis for TRAPI building steps
                if step.step_type == 'query_building' and hasattr(step, 'output_data'):
                    trapi_query = step.output_data.get('query', {})
                    if trapi_query:
                        print(f"    📋 TRAPI QUERY ANALYSIS:")
                        trapi_validation = debugger.validate_trapi_query(trapi_query)
                        debugger.log_step(f"trapi_validation_step_{i}", trapi_validation)
                        
                        print(f"      Valid TRAPI: {'✅' if trapi_validation['is_valid_trapi'] else '❌'}")
                        print(f"      Nodes: {trapi_validation['node_count']}")
                        print(f"      Edges: {trapi_validation['edge_count']}")
                        
                        # Show readable TRAPI structure
                        print(f"      📊 TRAPI Structure:")
                        if trapi_validation.get('nodes_analysis'):
                            for node_id, node_analysis in trapi_validation['nodes_analysis'].items():
                                categories = node_analysis.get('categories', [])
                                ids = node_analysis.get('ids', [])
                                print(f"        Node {node_id}:")
                                print(f"          Categories: {categories}")
                                if isinstance(ids, list) and len(ids) <= 3:
                                    print(f"          IDs: {ids}")
                                elif isinstance(ids, list):
                                    print(f"          IDs: {ids[:3]}... ({len(ids)} total)")
                                else:
                                    print(f"          IDs: {ids}")
                        
                        # Show edge structure
                        if trapi_validation.get('edges_analysis'):
                            for edge_id, edge_analysis in trapi_validation['edges_analysis'].items():
                                predicates = edge_analysis.get('predicates', [])
                                print(f"        Edge {edge_id}:")
                                print(f"          Predicates: {predicates}")
                                print(f"          From: {edge_analysis.get('subject')} To: {edge_analysis.get('object')}")
                
                # Detailed analysis for API execution steps
                elif step.step_type == 'api_execution' and hasattr(step, 'output_data'):
                    results = step.output_data.get('results', [])
                    if results:
                        print(f"    🔗 API RESULTS ANALYSIS:")
                        results_validation = debugger.validate_results(results)
                        debugger.log_step(f"results_validation_step_{i}", results_validation)
                        
                        print(f"      Total results: {results_validation['total_results']}")
                        print(f"      Valid relationships: {results_validation['valid_relationships']}")
                        
                        # Show confidence distribution
                        conf_dist = results_validation['confidence_distribution']
                        print(f"      Confidence distribution:")
                        print(f"        High (>0.7): {conf_dist['high']}")
                        print(f"        Medium (0.4-0.7): {conf_dist['medium']}")
                        print(f"        Low (<0.4): {conf_dist['low']}")
                        
                        # Show entity name resolution success
                        name_res = results_validation['entity_name_resolution']
                        total_names = name_res['resolved'] + name_res['unresolved']
                        resolution_rate = (name_res['resolved'] / max(total_names, 1)) * 100
                        print(f"      Entity name resolution: {resolution_rate:.1f}% ({name_res['resolved']}/{total_names})")
                        
                        # Show sample relationships with readable names
                        print(f"      📋 Sample relationships:")
                        for j, rel in enumerate(results_validation['sample_relationships'], 1):
                            print(f"        {j}. {rel['subject']} → {rel['predicate']} → {rel['object']}")
                            print(f"           Confidence: {rel['confidence']:.3f}")
        
        # Step 4: Domain Expertise Analysis
        print("\n🧠 STEP 4: DOMAIN EXPERTISE ANALYSIS")
        print("-" * 40)
        
        if hasattr(result, 'final_answer') and result.final_answer:
            answer = result.final_answer
            print(f"Final answer length: {len(answer)} characters")
            
            # Analyze domain expertise integration
            expertise_indicators = {
                'Pharmaceutical Context': ['brucellosis', 'infectious disease', 'bacteria'],
                'Mechanistic Explanation': ['translation', 'protein synthesis', 'ribosome'],
                'Drug Classification': ['antibiotic', 'tetracycline', 'aminoglycoside', 'chloramphenicol'],
                'Expert Inference': ['expertise', 'medicinal chemistry', 'drug class', 'infer'],
                'Specific Examples': ['doxycycline', 'streptomycin', 'rifampicin'],
                'Mechanism Details': ['30S ribosome', '50S ribosome', 'peptidyl transferase', 'tRNA']
            }
            
            print("🔬 Domain expertise integration analysis:")
            total_indicators = 0
            present_indicators = 0
            
            for category, terms in expertise_indicators.items():
                found_terms = [term for term in terms if term.lower() in answer.lower()]
                is_present = len(found_terms) > 0
                total_indicators += 1
                if is_present:
                    present_indicators += 1
                
                status = "✅" if is_present else "❌"
                print(f"  {status} {category}: {found_terms if found_terms else 'Not found'}")
            
            expertise_score = (present_indicators / total_indicators) * 100
            print(f"\n📊 Domain Expertise Score: {expertise_score:.1f}% ({present_indicators}/{total_indicators})")
            
            if expertise_score >= 80:
                print("🏆 EXCELLENT: High-level pharmaceutical sciences expertise demonstrated!")
            elif expertise_score >= 60:
                print("✅ GOOD: Solid biomedical domain knowledge present")
            else:
                print("⚠️  NEEDS IMPROVEMENT: Limited domain expertise integration")
        
        # Step 5: Show Enhanced Final Answer
        print("\n📝 STEP 5: ENHANCED FINAL ANSWER")
        print("-" * 35)
        
        if hasattr(result, 'final_answer') and result.final_answer:
            print("🎯 SCIENTIFICALLY-ENHANCED RESPONSE:")
            print("-" * 40)
            print(result.final_answer)
        else:
            print("❌ No final answer generated")
        
        # Step 6: Scientific Validation Summary
        print(f"\n📈 STEP 6: SCIENTIFIC VALIDATION SUMMARY")
        print("-" * 45)
        
        print("✅ VALIDATION RESULTS:")
        validation_summary = {
            'entity_extraction_success': len(entities) > 0,
            'trapi_queries_valid': any('is_valid_trapi' in step.get('details', {}) and step['details']['is_valid_trapi'] 
                                     for step in debugger.execution_log),
            'name_resolution_success': any('entity_name_resolution' in step.get('details', {}) 
                                         for step in debugger.execution_log),
            'domain_expertise_integration': expertise_score >= 60 if 'expertise_score' in locals() else False,
            'scientific_relationships_found': result.total_results > 0 if hasattr(result, 'total_results') else False
        }
        
        for validation_type, success in validation_summary.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {validation_type.replace('_', ' ').title()}: {status}")
        
        passes = sum(validation_summary.values())
        total = len(validation_summary)
        overall_score = (passes / total) * 100
        
        print(f"\n🎯 OVERALL SCIENTIFIC VALIDATION: {overall_score:.1f}% ({passes}/{total})")
        
        if overall_score >= 80:
            print("🏆 SYSTEM STATUS: Production-ready with high scientific rigor!")
        elif overall_score >= 60:
            print("✅ SYSTEM STATUS: Good scientific foundation, minor improvements needed")
        else:
            print("⚠️  SYSTEM STATUS: Requires significant improvements for scientific accuracy")
        
        # Cleanup
        await optimizer.close()
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"\n💥 SYSTEM ERROR after {execution_time:.1f}s:")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🏁 DEBUGGING DEMONSTRATION COMPLETE")
    print("=" * 45)
    print(f"Total execution time: {execution_time:.1f} seconds")
    print(f"Execution log entries: {len(debugger.execution_log)}")


async def demonstrate_step_by_step_debugging():
    """Demonstrate individual MCP tool debugging"""
    
    print("\n" + "="*60)
    print("🔧 INDIVIDUAL COMPONENT DEBUGGING")
    print("="*60)
    
    query = "What drugs can treat Brucellosis by targeting translation?"
    
    # Test 1: Entity Extraction
    print("\n1️⃣  ENTITY EXTRACTION DEBUGGING")
    print("-" * 35)
    
    try:
        print(f"Query: \"{query}\"")
        entity_response = await call_mcp_tool("bio_ner", query=query)
        
        print("📊 Raw Response Structure:")
        print(f"  Response type: {type(entity_response)}")
        print(f"  Keys: {list(entity_response.keys()) if isinstance(entity_response, dict) else 'Not a dict'}")
        
        if 'entities' in entity_response:
            entities = entity_response['entities']
            print(f"  Entities found: {len(entities)}")
            
            for i, entity in enumerate(entities[:3], 1):
                print(f"    Entity {i}:")
                print(f"      Name: {entity.get('name', 'No name')}")
                print(f"      Type: {entity.get('type', 'No type')}")
                print(f"      ID: {entity.get('id', 'No ID')}")
                print(f"      Confidence: {entity.get('confidence', 'No confidence')}")
        else:
            print("  ❌ No 'entities' key in response")
            
    except Exception as e:
        print(f"❌ Entity extraction error: {e}")
    
    # Test 2: TRAPI Query Building
    print("\n2️⃣  TRAPI QUERY BUILDING DEBUGGING")
    print("-" * 40)
    
    try:
        # Use entities from above or create mock entities
        entity_data = {"Brucellosis": "MONDO:0005683", "translation": "GO:0006412"}
        
        trapi_response = await call_mcp_tool(
            "build_trapi_query",
            query=query,
            entity_data=entity_data
        )
        
        print("📊 TRAPI Response Structure:")
        print(f"  Response type: {type(trapi_response)}")
        print(f"  Keys: {list(trapi_response.keys()) if isinstance(trapi_response, dict) else 'Not a dict'}")
        
        if 'query' in trapi_response:
            trapi_query = trapi_response['query']
            print(f"  TRAPI query created: {bool(trapi_query)}")
            
            if trapi_query and 'message' in trapi_query:
                message = trapi_query['message']
                if 'query_graph' in message:
                    qg = message['query_graph']
                    print(f"  Query graph nodes: {len(qg.get('nodes', {}))}")
                    print(f"  Query graph edges: {len(qg.get('edges', {}))}")
                    
                    # Show readable structure
                    print("  📋 Query Graph Structure:")
                    for node_id, node_data in qg.get('nodes', {}).items():
                        categories = node_data.get('categories', [])
                        ids = node_data.get('ids', [])
                        print(f"    Node {node_id}:")
                        print(f"      Categories: {categories}")
                        if isinstance(ids, list) and len(ids) <= 2:
                            print(f"      IDs: {ids}")
                        elif isinstance(ids, list):
                            print(f"      IDs: {ids[:2]}... ({len(ids)} total)")
        else:
            print("  ❌ No 'query' key in response")
            
    except Exception as e:
        print(f"❌ TRAPI building error: {e}")
    
    # Test 3: BTE API Call
    print("\n3️⃣  BTE API EXECUTION DEBUGGING")
    print("-" * 35)
    
    try:
        # Use the TRAPI query from above or create a simple one
        simple_trapi = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"categories": ["biolink:Disease"], "ids": ["MONDO:0005683"]},
                        "n1": {"categories": ["biolink:SmallMolecule"]}
                    },
                    "edges": {
                        "e0": {"subject": "n1", "object": "n0", "predicates": ["biolink:treats"]}
                    }
                }
            }
        }
        
        print("📊 Executing BTE API call with simple TRAPI query...")
        bte_response = await call_mcp_tool(
            "call_bte_api",
            json_query=simple_trapi,
            k=5,
            maxresults=20
        )
        
        print("📊 BTE Response Structure:")
        print(f"  Response type: {type(bte_response)}")
        print(f"  Keys: {list(bte_response.keys()) if isinstance(bte_response, dict) else 'Not a dict'}")
        
        if 'results' in bte_response:
            results = bte_response['results']
            print(f"  Results found: {len(results)}")
            
            # Show sample results with name resolution check
            for i, result in enumerate(results[:3], 1):
                print(f"    Result {i}:")
                print(f"      Subject: {result.get('subject', 'No subject')}")
                print(f"      Predicate: {result.get('predicate', 'No predicate')}")
                print(f"      Object: {result.get('object', 'No object')}")
                print(f"      Score: {result.get('score', 'No score')}")
                
                # Check if names are resolved (not raw IDs)
                subject = result.get('subject', '')
                obj = result.get('object', '')
                subject_resolved = not any(prefix in subject for prefix in ['UMLS:', 'MONDO:', 'CHEBI:', 'NCBIGene:'])
                obj_resolved = not any(prefix in obj for prefix in ['UMLS:', 'MONDO:', 'CHEBI:', 'NCBIGene:'])
                
                print(f"      Subject resolved: {'✅' if subject_resolved else '❌'}")
                print(f"      Object resolved: {'✅' if obj_resolved else '❌'}")
        
        if 'metadata' in bte_response:
            metadata = bte_response['metadata']
            print(f"  Metadata: {metadata}")
            
    except Exception as e:
        print(f"❌ BTE API error: {e}")


if __name__ == "__main__":
    print("Starting comprehensive debugging demonstration...")
    asyncio.run(demonstrate_enhanced_got_with_debugging())
    
    # Uncomment to run individual component testing
    # asyncio.run(demonstrate_step_by_step_debugging())