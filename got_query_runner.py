#!/usr/bin/env python3
"""
GoT Query Runner - User-friendly command interface

A simple command-line interface for executing biomedical queries using the
production-ready Graph of Thoughts (GoT) framework with comprehensive result
presentation and debugging capabilities.

Usage:
    python got_query_runner.py "What genes are associated with diabetes?"
    python got_query_runner.py --interactive
    python got_query_runner.py --debug --graphs "How does TP53 interact with other proteins?"
"""

import argparse
import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Optional

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_bte.core.queries.production_got_optimizer import (
    ProductionGoTOptimizer, 
    ProductionConfig, 
    execute_biomedical_query,
    run_biomedical_query
)


def setup_logging(debug: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'got_query_{os.getpid()}.log')
        ]
    )


def create_config(args) -> ProductionConfig:
    """Create configuration from command line arguments"""
    return ProductionConfig(
        show_debug=args.debug,
        show_graphs=args.graphs,
        save_results=not args.no_save,
        mcp_timeout=args.timeout,
        mcp_max_retries=args.retries,
        max_iterations=args.max_iterations,
        confidence_threshold=args.confidence_threshold,
        quality_threshold=args.quality_threshold,
        parallel_execution=not args.no_parallel,
        max_concurrent=args.max_concurrent
    )


def print_banner():
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GRAPH OF THOUGHTS (GoT) QUERY RUNNER                     ║
║                                                                              ║
║    Production-ready biomedical query optimization using Graph of Thoughts    ║
║    framework with comprehensive result presentation and debugging.           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_help():
    """Print usage help"""
    help_text = """
EXAMPLES:

  Basic query:
    python got_query_runner.py "What genes are associated with diabetes?"

  Interactive mode:
    python got_query_runner.py --interactive

  Debug mode with graphs:
    python got_query_runner.py --debug --graphs "How does TP53 interact with other proteins?"

  Complex research query:
    python got_query_runner.py --debug --save-results "Construct a comprehensive analysis of the molecular interactions between oxidative stress, inflammation, and neurodegeneration in Parkinson's disease"

COMMON BIOMEDICAL QUERIES:

  • "What genes are associated with [disease]?"
  • "How does [protein] interact with [other proteins]?"
  • "What are the pathways connecting [biological process] to [disease]?"
  • "What drugs target [protein] for treating [condition]?"
  • "What is the molecular mechanism of [biological process]?"

CONFIGURATION OPTIONS:

  --debug          Enable detailed debugging output and TRAPI query display
  --graphs         Generate visual graph representations of query execution
  --save-results   Save results to JSON and formatted text files
  --timeout        MCP tool timeout in seconds (default: 60)
  --retries        Maximum MCP tool retries (default: 3)
  --max-iterations Maximum GoT iterations (default: 5)
  --confidence-threshold  Minimum confidence for results (default: 0.7)
  --quality-threshold     Quality threshold for refinement (default: 0.1)
  --max-concurrent        Maximum concurrent operations (default: 3)
"""
    print(help_text)


def interactive_mode(config: ProductionConfig):
    """Run in interactive mode"""
    print("\n🧬 INTERACTIVE BIOMEDICAL QUERY MODE")
    print("Enter your biomedical queries below. Type 'quit', 'exit', or 'q' to stop.")
    print("Type 'help' for examples and guidance.")
    print("-" * 80)
    
    while True:
        try:
            query = input("\n🔍 Query: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
                
            if query.lower() == 'help':
                print_help()
                continue
            
            print(f"\n🚀 Executing query: {query}")
            print("⏳ Processing with GoT framework...")
            
            # Execute the query
            result, presentation = run_biomedical_query(query, config)
            
            # Print the presentation
            print("\n" + presentation)
            
            # Offer to continue
            if not input("\n📝 Press Enter to continue, or 'q' to quit: ").strip():
                continue
            else:
                break
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("💡 Try a different query or check your configuration.")


def single_query_mode(query: str, config: ProductionConfig):
    """Execute a single query"""
    print(f"\n🚀 Executing query: {query}")
    print("⏳ Processing with GoT framework...")
    
    try:
        # Execute the query
        result, presentation = run_biomedical_query(query, config)
        
        # Print the presentation
        print("\n" + presentation)
        
        # Print summary
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        print(f"\n📊 EXECUTION SUMMARY:")
        print(f"Status: {status}")
        print(f"Execution Time: {result.total_execution_time:.3f}s")
        print(f"Results Found: {result.total_results}")
        print(f"Quality Score: {result.quality_score:.3f}")
        
        if config.save_results:
            print("💾 Results saved to files (check current directory)")
            
    except Exception as e:
        print(f"\n❌ Error executing query: {str(e)}")
        print("💡 Try using --debug flag for more detailed error information.")
        return 1
    
    return 0


def validate_query(query: str) -> bool:
    """Basic query validation"""
    if len(query) < 10:
        print("⚠️  Query seems too short. Please provide a more detailed biomedical question.")
        return False
    
    # Check for biomedical keywords
    biomedical_keywords = [
        'gene', 'genes', 'protein', 'proteins', 'disease', 'drug', 'drugs',
        'pathway', 'pathways', 'molecule', 'molecules', 'cell', 'cells',
        'cancer', 'diabetes', 'treatment', 'therapy', 'interaction', 'mechanism',
        'biomarker', 'mutation', 'expression', 'regulation', 'signaling'
    ]
    
    query_lower = query.lower()
    has_biomedical_context = any(keyword in query_lower for keyword in biomedical_keywords)
    
    if not has_biomedical_context:
        print("⚠️  Query doesn't appear to be biomedical. This system is optimized for biomedical queries.")
        print("    Examples: 'What genes are associated with diabetes?', 'How does TP53 interact with p21?'")
        response = input("    Continue anyway? (y/N): ").strip().lower()
        return response in ['y', 'yes']
    
    return True


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Execute biomedical queries using Graph of Thoughts (GoT) framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What genes are associated with diabetes?"
  %(prog)s --interactive
  %(prog)s --debug --graphs "How does TP53 interact with other proteins?"
        """
    )
    
    # Positional argument for query
    parser.add_argument('query', nargs='?', 
                       help='Biomedical query to execute')
    
    # Mode options
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Run in interactive mode')
    
    # Output options
    parser.add_argument('-d', '--debug', action='store_true',
                       help='Enable debug output and TRAPI query display')
    parser.add_argument('-g', '--graphs', action='store_true',
                       help='Generate visual execution graphs')
    parser.add_argument('--no-save', action='store_true',
                       help='Don\'t save results to files')
    
    # Performance options
    parser.add_argument('--timeout', type=int, default=60,
                       help='MCP tool timeout in seconds (default: 60)')
    parser.add_argument('--retries', type=int, default=3,
                       help='Maximum MCP tool retries (default: 3)')
    parser.add_argument('--max-iterations', type=int, default=5,
                       help='Maximum GoT iterations (default: 5)')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                       help='Minimum confidence threshold (default: 0.7)')
    parser.add_argument('--quality-threshold', type=float, default=0.1,
                       help='Quality threshold for refinement (default: 0.1)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel execution')
    parser.add_argument('--max-concurrent', type=int, default=3,
                       help='Maximum concurrent operations (default: 3)')
    
    # Utility options
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--examples', action='store_true',
                       help='Show usage examples and exit')
    
    args = parser.parse_args()
    
    # Handle special cases
    if args.examples:
        print_help()
        return 0
    
    # Setup logging
    setup_logging(args.debug or args.verbose)
    
    # Print banner
    print_banner()
    
    # Create configuration
    config = create_config(args)
    
    # Determine mode
    if args.interactive:
        interactive_mode(config)
        return 0
    elif args.query:
        # Validate query
        if not validate_query(args.query):
            return 1
        
        return single_query_mode(args.query, config)
    else:
        print("❌ Error: No query provided and not in interactive mode.")
        print("💡 Use --interactive for interactive mode or provide a query as argument.")
        print("   Example: python got_query_runner.py \"What genes are associated with diabetes?\"")
        return 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("💡 Please report this issue with the full error details.")
        sys.exit(1)