#!/usr/bin/env python3
"""
Comprehensive demonstration of the agentic-bte system with both fixes applied

This demonstrates:
1. Complex biomedical query processing
2. Entity extraction and linking
3. TRAPI query building with generic entity mapping
4. BTE API execution
5. Entity name resolution in final answers
6. Complete final answer generation
"""

import asyncio
import time
import json
from agentic_bte.core.queries.production_got_optimizer import execute_biomedical_query

async def demonstrate_complete_system():
    print("🚀 COMPREHENSIVE AGENTIC-BTE SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    # Complex biomedical query that will test both fixes
    complex_query = """
    What are the specific genetic variants and pharmacogenomic factors that influence 
    the efficacy and toxicity of antipsychotic medications like haloperidol and risperidone 
    in treating schizophrenia? Include information about drug metabolism pathways, 
    dopamine receptor variations, and personalized dosing strategies.
    """
    
    print("📋 COMPLEX BIOMEDICAL QUERY:")
    print("-" * 35)
    print(complex_query.strip())
    
    print("\n🔍 WHAT THIS QUERY TESTS:")
    print("-" * 30)
    print("✓ Generic entity mapping: 'genetic variants' → specific genes")
    print("✓ Entity name resolution: UMLS IDs → human-readable names")  
    print("✓ Complex multi-entity relationships")
    print("✓ Pharmacogenomic knowledge integration")
    print("✓ Personalized medicine applications")
    
    print("\n⏳ EXECUTING QUERY...")
    print("-" * 25)
    
    start_time = time.time()
    
    try:
        # Execute the biomedical query using the production optimizer
        result, presentation = await execute_biomedical_query(complex_query)
        
        execution_time = time.time() - start_time
        
        print(f"✅ EXECUTION COMPLETED in {execution_time:.1f} seconds")
        print("=" * 50)
        
        # Display comprehensive results
        print(f"\n📊 EXECUTION SUMMARY:")
        print("-" * 25)
        print(f"Success: {result.success}")
        print(f"Total Results: {result.total_results}")
        print(f"Quality Score: {result.quality_score:.3f}")
        print(f"Entities Found: {len(result.entities_found)}")
        
        if result.success and result.total_results > 0:
            print(f"\n🧬 ENTITIES EXTRACTED:")
            print("-" * 25)
            for i, entity in enumerate(result.entities_found[:8], 1):
                name = entity.get('name', 'Unknown')
                entity_type = entity.get('type', 'Unknown')
                confidence = entity.get('confidence', 0.0)
                print(f"{i:2d}. {name} ({entity_type}) - {confidence:.2f}")
            
            print(f"\n⚡ PERFORMANCE METRICS:")
            print("-" * 25)
            if hasattr(result, 'got_metrics'):
                metrics = result.got_metrics
                print(f"Execution Time: {metrics.get('total_execution_time', 0):.1f}s")
                print(f"Total Thoughts: {metrics.get('total_thoughts', 0)}")
                print(f"Average Confidence: {metrics.get('average_confidence', 0):.3f}")
                print(f"Quality Improvement: {metrics.get('quality_improvement', 0):.2f}x")
            
            print(f"\n📄 FINAL ANSWER PREVIEW:")
            print("-" * 30)
            
            # Show first part of final answer to demonstrate fixes
            if hasattr(result, 'final_answer'):
                final_answer = result.final_answer
                # Show first 500 characters
                preview = final_answer[:500]
                if len(final_answer) > 500:
                    preview += "..."
                print(preview)
                
                print(f"\n🔍 CHECKING FOR FIXES:")
                print("-" * 25)
                
                # Check for Fix 1: No more UMLS IDs in final answer
                umls_count = final_answer.count('UMLS:C')
                if umls_count == 0:
                    print("✅ Fix 1 SUCCESS: No unresolved UMLS IDs found")
                else:
                    print(f"⚠️  Fix 1 PARTIAL: Found {umls_count} unresolved UMLS IDs")
                
                # Check for Fix 2: Specific entities instead of generic terms
                generic_terms = ['genetic variations', 'genetic factors']
                specific_genes = ['DRD2', 'CYP2D6', 'CYP3A4', 'COMT', 'HTR2A']
                
                generic_found = any(term in final_answer for term in generic_terms)
                specific_found = any(gene in final_answer for gene in specific_genes)
                
                if specific_found and not generic_found:
                    print("✅ Fix 2 SUCCESS: Found specific gene names, no generic terms")
                elif specific_found and generic_found:
                    print("✅ Fix 2 PARTIAL: Found specific genes alongside some generic terms")
                elif not specific_found and generic_found:
                    print("⚠️  Fix 2 NEEDED: Still showing generic terms without specific genes")
                else:
                    print("ℹ️  Fix 2 N/A: No relevant genetic terms found in this answer")
            
            print(f"\n📁 RESULT FILE SAVED:")
            print("-" * 25)
            
            # Find the most recent result file
            import glob
            import os
            result_files = glob.glob("got_result_*.json")
            if result_files:
                latest_file = max(result_files, key=os.path.getctime)
                print(f"📄 {latest_file}")
                print(f"📏 File size: {os.path.getsize(latest_file)} bytes")
                
                # Show some key relationships from the results
                with open(latest_file, 'r') as f:
                    saved_result = json.load(f)
                
                final_answer_content = saved_result.get('final_answer', '')
                
                print(f"\n🔗 KEY RELATIONSHIPS FOUND:")
                print("-" * 30)
                
                # Extract key relationships from the answer
                lines = final_answer_content.split('\n')
                relationship_lines = []
                for line in lines:
                    if '→' in line or 'affects' in line or 'metabolizes' in line:
                        relationship_lines.append(line.strip())
                
                for i, rel in enumerate(relationship_lines[:5], 1):
                    if rel:
                        print(f"{i}. {rel}")
            
        else:
            print(f"\n❌ EXECUTION FAILED OR NO RESULTS")
            print("This might indicate:")
            print("- Network issues with BTE API")
            print("- Query complexity exceeding system limits")
            print("- Local BTE instance not running properly")
            
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"\n💥 EXECUTION ERROR after {execution_time:.1f}s:")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🏁 DEMONSTRATION COMPLETE")
    print("=" * 35)
    
    print(f"\n💡 NEXT STEPS:")
    print("- Review the final answer for specific gene names")
    print("- Check that UMLS IDs are resolved to readable names")
    print("- Examine the saved result file for detailed evidence")
    print("- Try additional complex queries to test system capabilities")

if __name__ == "__main__":
    asyncio.run(demonstrate_complete_system())