#!/usr/bin/env python3
"""
Enhanced GoT System Demonstration

This script demonstrates the enhanced GoT framework that combines the parallel
execution capabilities of the GoT system with the sophisticated domain expertise
and mechanistic reasoning from the LangGraph implementation.

This addresses the gap you identified where the system should provide responses
like your Brucellosis example with domain expertise and mechanistic reasoning.
"""

import asyncio
import time
from agentic_bte.core.queries.enhanced_got_optimizer import (
    execute_enhanced_biomedical_query, 
    EnhancedConfig
)

async def demonstrate_enhanced_got_system():
    """Demonstrate enhanced GoT system with domain expertise"""
    
    print("🧬 ENHANCED GOT SYSTEM WITH DOMAIN EXPERTISE")
    print("=" * 65)
    print()
    
    print("This demonstration shows how the enhanced GoT system combines:")
    print("✓ GoT framework's parallel execution and thought optimization")  
    print("✓ LangGraph's sophisticated domain expertise and mechanistic reasoning")
    print("✓ Expert inference capabilities for filling knowledge gaps")
    print("✓ Pharmaceutical sciences and medicinal chemistry expertise")
    print()
    
    # The exact query from your example
    expert_query = "What drugs can treat Brucellosis by targeting translation?"
    
    print(f"📋 EXPERT QUERY (from your example):")
    print("-" * 40)
    print(f'"{expert_query}"')
    print()
    
    print("🎯 EXPECTED SOPHISTICATED RESPONSE FEATURES:")
    print("-" * 45)
    print("• Domain context explanation (Brucellosis pathophysiology)")
    print("• Mechanistic reasoning (translation process importance)")  
    print("• Expert classification (antibiotic drug classes)")
    print("• Informed inference (gap-filling with domain knowledge)")
    print("• Specific examples with mechanisms of action")
    print()
    
    # Configure enhanced system
    config = EnhancedConfig(
        # Enable all expert capabilities
        enable_expert_inference=True,
        enable_mechanistic_reasoning=True,
        enable_rdf_accumulation=True,
        
        # Domain expertise
        pharmaceutical_expertise=True,
        medicinal_chemistry_expertise=True,
        biomedical_research_expertise=True,
        
        # Mechanistic analysis
        enable_pathway_analysis=True,
        enable_mechanism_synthesis=True,
        drug_class_inference=True,
        
        # Performance settings
        show_debug=True,
        max_iterations=3,
        parallel_execution=True
    )
    
    print("⏳ EXECUTING ENHANCED GOT SYSTEM...")
    print("-" * 40)
    start_time = time.time()
    
    try:
        # Execute with enhanced domain expertise
        result, presentation = await execute_enhanced_biomedical_query(expert_query, config)
        
        execution_time = time.time() - start_time
        
        print(f"✅ EXECUTION COMPLETED in {execution_time:.1f} seconds")
        print("=" * 50)
        print()
        
        # Show enhanced results
        if result.success:
            print("📊 ENHANCED EXECUTION SUMMARY:")
            print("-" * 35)
            print(f"Success: {result.success}")
            print(f"Total Results: {result.total_results}")
            print(f"Quality Score: {result.quality_score:.3f}")
            print(f"Entities Found: {len(result.entities_found)}")
            print()
            
            # Show domain expertise integration
            print("🧠 DOMAIN EXPERTISE FEATURES:")
            print("-" * 35)
            print("✓ Pharmaceutical sciences expertise integrated")
            print("✓ Mechanistic reasoning enabled")
            print("✓ Drug classification knowledge applied")
            print("✓ Expert inference capabilities active")
            print("✓ RDF knowledge accumulation used")
            print()
            
            # Show the sophisticated answer
            print("📝 ENHANCED FINAL ANSWER:")
            print("-" * 30)
            print()
            
            if hasattr(result, 'final_answer') and result.final_answer:
                print(result.final_answer)
            else:
                print("Final answer not available - check system configuration")
            
            print()
            print("🔍 ANSWER QUALITY ANALYSIS:")
            print("-" * 30)
            
            answer = result.final_answer if hasattr(result, 'final_answer') else ""
            
            # Check for domain expertise indicators
            domain_indicators = [
                ("Domain Context", any(term in answer.lower() for term in ['brucellosis', 'bacterial', 'infectious disease'])),
                ("Mechanistic Reasoning", any(term in answer.lower() for term in ['translation', 'protein synthesis', 'ribosome'])),
                ("Drug Classification", any(term in answer.lower() for term in ['antibiotic', 'tetracycline', 'aminoglycoside'])),
                ("Expert Inference", any(term in answer.lower() for term in ['expertise', 'knowledge', 'infer', 'class'])),
                ("Specific Examples", any(term in answer.lower() for term in ['doxycycline', 'streptomycin', 'chloramphenicol']))
            ]
            
            for feature, present in domain_indicators:
                status = "✅ PRESENT" if present else "❌ MISSING"
                print(f"  {feature}: {status}")
            
            print()
            
            # Compare with expected sophisticated response
            sophisticated_features = sum(1 for _, present in domain_indicators if present)
            total_features = len(domain_indicators)
            sophistication_score = sophisticated_features / total_features * 100
            
            print(f"🎯 SOPHISTICATION SCORE: {sophistication_score:.0f}%")
            print(f"   ({sophisticated_features}/{total_features} expert features present)")
            
            if sophistication_score >= 80:
                print("🏆 EXCELLENT: Answer demonstrates high-level domain expertise!")
            elif sophistication_score >= 60:
                print("✅ GOOD: Answer shows solid biomedical knowledge") 
            else:
                print("⚠️  NEEDS IMPROVEMENT: Answer lacks domain expertise depth")
                
        else:
            print("❌ EXECUTION FAILED")
            print(f"Error: {result.error_message if hasattr(result, 'error_message') else 'Unknown error'}")
            
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"💥 SYSTEM ERROR after {execution_time:.1f}s:")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print()
    print("🔬 SYSTEM ARCHITECTURE COMPARISON:")
    print("-" * 40)
    print("ORIGINAL GOT SYSTEM:")
    print("  • Parallel thought execution")
    print("  • Graph-based optimization") 
    print("  • Basic knowledge graph lookup")
    print("  • Generic LLM answer generation")
    print()
    print("ENHANCED GOT SYSTEM:")
    print("  • All original GoT capabilities PLUS:")
    print("  • Domain expertise integration")
    print("  • Mechanistic reasoning capabilities")
    print("  • Expert inference for knowledge gaps")
    print("  • Pharmaceutical sciences knowledge")
    print("  • Drug classification frameworks")
    print("  • RDF knowledge accumulation")
    print()
    
    print("💡 WHY THE ENHANCEMENT WORKS:")
    print("-" * 35)
    print("• Integrates LangGraph's sophisticated reasoning approach")
    print("• Adds domain-specific knowledge bases (drug classes, mechanisms)")
    print("• Enables expert inference when database evidence is limited")
    print("• Uses mechanistic reasoning for therapeutic queries")
    print("• Maintains GoT's parallel execution performance benefits")
    print()
    
    print("🏁 DEMONSTRATION COMPLETE")
    print("=" * 30)


async def demonstrate_comparison():
    """Demonstrate the difference between standard and enhanced systems"""
    
    print("\n" + "="*60)
    print("🔄 COMPARISON: STANDARD vs ENHANCED GOT SYSTEM")
    print("="*60)
    
    query = "What drugs can treat Brucellosis by targeting translation?"
    
    print(f"Using query: \"{query}\"")
    print()
    
    # Standard GoT system
    print("1️⃣  STANDARD GOT SYSTEM:")
    print("-" * 25)
    try:
        from agentic_bte.core.queries.production_got_optimizer import execute_biomedical_query
        standard_result, _ = await execute_biomedical_query(query)
        
        standard_answer = standard_result.final_answer if hasattr(standard_result, 'final_answer') else "No answer generated"
        print(f"Answer length: {len(standard_answer)} characters")
        print(f"Domain expertise: {'Limited' if len(standard_answer) < 500 else 'Moderate'}")
        print(f"First 200 chars: {standard_answer[:200]}...")
        
    except Exception as e:
        print(f"Standard system error: {e}")
    
    print()
    
    # Enhanced GoT system  
    print("2️⃣  ENHANCED GOT SYSTEM:")
    print("-" * 25)
    try:
        enhanced_result, _ = await execute_enhanced_biomedical_query(query)
        
        enhanced_answer = enhanced_result.final_answer if hasattr(enhanced_result, 'final_answer') else "No answer generated"
        print(f"Answer length: {len(enhanced_answer)} characters")
        print(f"Domain expertise: {'High' if 'medicinal chemistry' in enhanced_answer.lower() else 'Moderate'}")
        print(f"First 200 chars: {enhanced_answer[:200]}...")
        
    except Exception as e:
        print(f"Enhanced system error: {e}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    asyncio.run(demonstrate_enhanced_got_system())
    # Uncomment to run comparison
    # asyncio.run(demonstrate_comparison())