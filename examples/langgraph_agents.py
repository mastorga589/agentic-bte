#!/usr/bin/env python3
"""
LangGraph Multi-Agent Workflow Example

This script demonstrates the advanced multi-agent system for complex
biomedical research using iterative query decomposition and knowledge
graph accumulation.

Usage:
    python examples/langgraph_agents.py
"""

import os
import asyncio
from agentic_bte.agents import BiomedicalOrchestrator, execute_biomedical_research


def demo_simple_research():
    """Demonstrate simple biomedical research query"""
    print("🔬 Simple Research Example")
    print("=" * 50)
    
    query = "What drugs can treat diabetes?"
    print(f"Research Query: {query}")
    
    try:
        orchestrator = BiomedicalOrchestrator()
        result = orchestrator.execute_research(
            query=query,
            maxresults=20,
            k=3,
            confidence_threshold=0.7
        )
        
        if result["success"]:
            print("\n✅ Research completed successfully!")
            
            summary = result["execution_summary"]
            print(f"  • Total subqueries: {summary['total_subqueries']}")
            print(f"  • Successful subqueries: {summary['successful_subqueries']}")
            print(f"  • Knowledge triples: {summary['rdf_triples_count']}")
            print(f"  • Execution time: {summary['total_execution_time']:.2f}s")
            print(f"  • Average confidence: {summary['average_confidence']:.2f}")
            
            print(f"\n📝 Final Answer:")
            print(result["final_answer"][:500] + "..." if len(result["final_answer"]) > 500 else result["final_answer"])
            
            # Show RDF graph sample
            rdf_graph = result["rdf_graph"]
            if rdf_graph:
                print(f"\n🕸️ Knowledge Graph (sample):")
                lines = rdf_graph.split('\n')[:5]
                for line in lines:
                    if line.strip():
                        print(f"    {line}")
                if len(lines) > 5:
                    print("    ...")
        else:
            print(f"❌ Research failed: {result['error']}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def demo_complex_research():
    """Demonstrate complex multi-step biomedical research"""
    print("\n🧠 Complex Research Example")
    print("=" * 50)
    
    query = "Which drugs can treat Alzheimer's disease by targeting amyloid beta?"
    print(f"Complex Research Query: {query}")
    
    try:
        # Use the convenience function for simpler usage
        result = execute_biomedical_research(
            query=query,
            maxresults=30,
            k=5,
            confidence_threshold=0.6
        )
        
        if result["success"]:
            print("\n✅ Complex research completed successfully!")
            
            summary = result["execution_summary"]
            graph_stats = result["graph_statistics"]
            
            print(f"  • Research approach: Multi-agent iterative decomposition")
            print(f"  • Total subqueries: {summary['total_subqueries']}")
            print(f"  • Knowledge sources: BioThings Explorer")
            print(f"  • Entities processed: {graph_stats.get('unique_subjects', 0)}")
            print(f"  • Relationships found: {graph_stats.get('total_triples', 0)}")
            print(f"  • Execution time: {summary['total_execution_time']:.2f}s")
            
            print(f"\n🎯 Final Research Answer:")
            final_answer = result["final_answer"]
            # Show first paragraph
            paragraphs = final_answer.split('\n\n')
            if paragraphs:
                print(paragraphs[0][:400] + "..." if len(paragraphs[0]) > 400 else paragraphs[0])
            
            # Show subquery execution details
            subquery_results = result.get("subquery_results", [])
            if subquery_results:
                print(f"\n📋 Subquery Execution Summary:")
                for i, subquery in enumerate(subquery_results):
                    status = "✅" if subquery.get("success", False) else "❌"
                    query_text = subquery.get("subquery", "Unknown")[:60]
                    results_count = subquery.get("results_count", 0)
                    confidence = subquery.get("confidence", 0)
                    print(f"    {i+1}. {status} {query_text}... ({results_count} results, {confidence:.2f} confidence)")
        else:
            print(f"❌ Research failed: {result['error']}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def demo_research_with_rdf_exploration():
    """Demonstrate research with RDF graph exploration"""
    print("\n🕸️ Research with Knowledge Graph Exploration")
    print("=" * 50)
    
    query = "What are the mechanisms of action of metformin?"
    print(f"Mechanistic Research Query: {query}")
    
    try:
        orchestrator = BiomedicalOrchestrator()
        result = orchestrator.execute_research(
            query=query,
            maxresults=25,
            k=4,
            confidence_threshold=0.5
        )
        
        if result["success"]:
            print("\n✅ Mechanistic research completed!")
            
            # Show the RDF graph structure
            graph_stats = result["graph_statistics"]
            print(f"\n🔍 Knowledge Graph Analysis:")
            print(f"  • Total relationships: {graph_stats.get('total_triples', 0)}")
            print(f"  • Unique entities: {graph_stats.get('unique_subjects', 0)}")
            print(f"  • Relationship types: {graph_stats.get('unique_predicates', 0)}")
            
            # Show the accumulated RDF knowledge
            rdf_graph = result["rdf_graph"]
            if rdf_graph:
                print(f"\n📊 Sample Knowledge Relationships:")
                # Parse some triples for display
                lines = [line.strip() for line in rdf_graph.split('\n') if line.strip() and not line.startswith('@')]
                for i, line in enumerate(lines[:8]):
                    if '<' in line and '>' in line:
                        # Simplify RDF display
                        parts = line.split(' ')
                        if len(parts) >= 3:
                            subject = parts[0].replace('<', '').replace('>', '').split('/')[-1]
                            predicate = parts[1].replace('<', '').replace('>', '').split('/')[-1]
                            obj = ' '.join(parts[2:]).replace('<', '').replace('>', '').split('/')[-1].replace('.', '')
                            print(f"    {i+1}. {subject} → {predicate} → {obj}")
            
            print(f"\n💡 Research Insights:")
            final_answer = result["final_answer"]
            # Extract key insights (look for mechanism-related content)
            sentences = final_answer.split('. ')
            mechanism_sentences = [s for s in sentences if any(word in s.lower() for word in ['mechanism', 'target', 'pathway', 'inhibit', 'activate'])][:3]
            for i, sentence in enumerate(mechanism_sentences):
                print(f"    {i+1}. {sentence.strip()}.")
        else:
            print(f"❌ Research failed: {result['error']}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def demo_research_comparison():
    """Compare research approaches for the same question"""
    print("\n⚖️ Research Approach Comparison")
    print("=" * 50)
    
    base_query = "What genes are associated with breast cancer?"
    print(f"Research Question: {base_query}")
    
    print("\n🔍 Approach 1: High Confidence, Focused Results")
    try:
        result1 = execute_biomedical_research(
            query=base_query,
            maxresults=15,
            k=3,
            confidence_threshold=0.8  # High confidence
        )
        
        if result1["success"]:
            summary1 = result1["execution_summary"]
            print(f"    • Execution time: {summary1['total_execution_time']:.2f}s")
            print(f"    • Subqueries: {summary1['total_subqueries']}")
            print(f"    • Knowledge triples: {summary1['rdf_triples_count']}")
            print(f"    • Average confidence: {summary1['average_confidence']:.2f}")
        else:
            print(f"    ❌ Failed: {result1['error']}")
    except Exception as e:
        print(f"    ❌ Error: {str(e)}")
    
    print("\n🔍 Approach 2: Broader Search, More Results")
    try:
        result2 = execute_biomedical_research(
            query=base_query,
            maxresults=40,
            k=6,
            confidence_threshold=0.5  # Lower confidence for broader results
        )
        
        if result2["success"]:
            summary2 = result2["execution_summary"]
            print(f"    • Execution time: {summary2['total_execution_time']:.2f}s")
            print(f"    • Subqueries: {summary2['total_subqueries']}")
            print(f"    • Knowledge triples: {summary2['rdf_triples_count']}")
            print(f"    • Average confidence: {summary2['average_confidence']:.2f}")
        else:
            print(f"    ❌ Failed: {result2['error']}")
    except Exception as e:
        print(f"    ❌ Error: {str(e)}")


def main():
    """Run all LangGraph multi-agent demonstrations"""
    print("🤖 Agentic BTE Multi-Agent Research Demo")
    print("=" * 70)
    print("This demo showcases the LangGraph-based multi-agent system")
    print("for sophisticated biomedical research workflows")
    print("=" * 70)
    
    # Check environment
    if not os.getenv("AGENTIC_BTE_OPENAI_API_KEY"):
        print("⚠️ Warning: AGENTIC_BTE_OPENAI_API_KEY not set")
        print("The multi-agent system requires an OpenAI API key for LLM operations")
        print("Please set your API key: export AGENTIC_BTE_OPENAI_API_KEY='your-key'")
        print()
        return
    
    print("🔑 OpenAI API key detected - proceeding with demos")
    print()
    
    try:
        # Demo 1: Simple research
        demo_simple_research()
        
        # Demo 2: Complex multi-step research
        demo_complex_research()
        
        # Demo 3: RDF graph exploration
        demo_research_with_rdf_exploration()
        
        # Demo 4: Approach comparison
        demo_research_comparison()
        
        print("\n🎉 All multi-agent demos completed!")
        print("\n🚀 What you just saw:")
        print("  • Automatic query decomposition into single-hop subqueries")
        print("  • Iterative knowledge graph accumulation using RDF")
        print("  • Dynamic agent orchestration (Annotator → Planner → BTE Search → Summary)")
        print("  • Intelligent entity linking and name resolution")
        print("  • Biomedical expertise synthesis for final answers")
        
        print("\n📚 Next steps:")
        print("  • Explore the agent source code in agentic_bte/agents/")
        print("  • Customize agent behavior and prompts")
        print("  • Integrate with your own biomedical workflows")
        print("  • Scale up for batch processing of research questions")
        
    except Exception as e:
        print(f"\n❌ Demo suite failed: {str(e)}")
        print("Check your environment configuration and dependencies")


if __name__ == "__main__":
    main()