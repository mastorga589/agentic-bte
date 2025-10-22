#!/usr/bin/env python3
"""
Graph of Thoughts (GoT) MCP Tool

This module provides an MCP tool that integrates the advanced Graph of Thoughts
biomedical query optimization system with MCP clients like Warp and Claude Desktop.
"""

import logging
from typing import Dict, Any, List
import asyncio
import json
from datetime import datetime
import nest_asyncio

# Import the production GoT system
from ....core.queries.production_got_optimizer import (
    ProductionGoTOptimizer,
    ProductionConfig,
    execute_biomedical_query  # Use async version directly
)
from ....core.queries.result_presenter import QueryResult

# Apply nest_asyncio patch for MCP compatibility
nest_asyncio.apply()

logger = logging.getLogger(__name__)


def get_got_tool_definition() -> Dict[str, Any]:
    """Get the MCP tool definition for the GoT optimizer"""
    return {
        "name": "got_biomedical_query",
        "description": """
Advanced Graph of Thoughts (GoT) biomedical query optimizer with comprehensive analysis.

This tool uses the Graph of Thoughts framework for sophisticated biomedical question answering,
providing detailed execution plans, TRAPI query debugging, performance metrics, and comprehensive
final answers. Ideal for complex research questions requiring in-depth analysis.

Features:
- Graph-based reasoning with parallel thought execution
- TRAPI query visualization and debugging
- Comprehensive result aggregation and refinement
- Performance metrics tracking (volume, latency, quality)
- Executive summaries and detailed breakdowns
- Research-grade final answer synthesis

Best for: Complex biomedical research questions, drug discovery, pathway analysis, 
gene-disease relationships, molecular mechanism studies.
""".strip(),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Complex biomedical research query to process using GoT optimization"
                },
                "show_debug": {
                    "type": "boolean",
                    "description": "Enable detailed debugging output including TRAPI queries (default: true)",
                    "default": True
                },
                "save_results": {
                    "type": "boolean", 
                    "description": "Save detailed results to timestamped files (default: false for MCP)",
                    "default": False
                },
                "confidence_threshold": {
                    "type": "number",
                    "description": "Minimum confidence threshold for results (default: 0.7)",
                    "default": 0.7,
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Maximum GoT framework iterations (default: 5)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10
                },
                "enable_refinement": {
                    "type": "boolean",
                    "description": "Enable iterative result refinement (default: true)",
                    "default": True
                },
                "output_format": {
                    "type": "string",
                    "description": "Output format preference",
                    "enum": ["comprehensive", "summary", "debug"],
                    "default": "comprehensive"
                }
            },
            "required": ["query"]
        }
    }


async def handle_got_query(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle GoT biomedical query processing
    
    Args:
        arguments: Dictionary containing query and configuration parameters
        
    Returns:
        Dictionary with structured MCP response including content array
    """
    try:
        # Extract parameters with defaults
        query = arguments.get("query", "").strip()
        show_debug = arguments.get("show_debug", True)
        save_results = arguments.get("save_results", False)  # Default False for MCP
        confidence_threshold = arguments.get("confidence_threshold", 0.7)
        max_iterations = arguments.get("max_iterations", 5)
        enable_refinement = arguments.get("enable_refinement", True)
        output_format = arguments.get("output_format", "comprehensive")
        
        # Validate required parameters
        if not query:
            return {
                "error": "Query parameter is required",
                "content": [{
                    "type": "text",
                    "text": "❌ Error: Query parameter is required for GoT processing"
                }]
            }
        
        logger.info(f"Processing GoT query: {query[:100]}...")
        logger.info(f"Configuration - debug:{show_debug}, confidence:{confidence_threshold}, iterations:{max_iterations}")
        
        # Create GoT configuration
        config = ProductionConfig(
            show_debug=show_debug,
            show_graphs=False,  # Disable graphs for MCP compatibility
            save_results=save_results,
            confidence_threshold=confidence_threshold,
            max_iterations=max_iterations,
            quality_threshold=0.1,  # Enable refinement
            enable_caching=True,
            parallel_execution=True
        )
        
        # Execute the GoT query
        start_time = datetime.now()
        
        try:
            # Use the production GoT system directly (already in async context)
            result, presentation = await execute_biomedical_query(query, config)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"GoT query completed in {execution_time:.3f}s - Success: {result.success}")
            
            # Format response based on output preference
            if output_format == "summary":
                content_text = _format_summary_response(result, query)
            elif output_format == "debug":
                content_text = _format_debug_response(result, presentation)
            else:  # comprehensive
                content_text = _format_comprehensive_response(result, presentation, query)
            
            return {
                "content": [{
                    "type": "text",
                    "text": content_text
                }],
                "metadata": {
                    "query": query,
                    "success": result.success,
                    "execution_time": execution_time,
                    "entities_found": len(result.entities_found),
                    "total_results": result.total_results,
                    "quality_score": result.quality_score,
                    "got_metrics": result.got_metrics,
                    "timestamp": start_time.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"GoT execution failed: {str(e)}", exc_info=True)
            return {
                "error": f"GoT execution failed: {str(e)}",
                "content": [{
                    "type": "text", 
                    "text": f"❌ **GoT Query Processing Failed**\n\n**Error**: {str(e)}\n\n**Query**: {query}\n\n💡 **Suggestions**:\n- Check if the query is biomedical in nature\n- Verify OpenAI API connectivity\n- Try a simpler query to test system functionality"
                }]
            }
        
    except Exception as e:
        logger.error(f"GoT tool error: {str(e)}", exc_info=True)
        return {
            "error": f"Tool error: {str(e)}",
            "content": [{
                "type": "text",
                "text": f"❌ **GoT Tool Error**\n\n{str(e)}\n\nPlease check the query parameters and try again."
            }]
        }


def _format_summary_response(result: QueryResult, query: str) -> str:
    """Format a concise summary response for the MCP client"""
    
    status_icon = "✅" if result.success else "❌"
    
    summary = f"""🧠 **GRAPH OF THOUGHTS (GoT) BIOMEDICAL ANALYSIS**
{status_icon} **Query**: {query}

📋 **EXECUTIVE SUMMARY**
• **Status**: {"SUCCESS" if result.success else "FAILED"}
• **Execution Time**: {result.total_execution_time:.3f}s
• **Results Found**: {result.total_results}
• **Quality Score**: {result.quality_score:.3f}
• **Entities Identified**: {len(result.entities_found)}

🎯 **FINAL ANSWER**
{result.final_answer}

📈 **GoT PERFORMANCE METRICS**
"""
    
    if result.got_metrics:
        summary += f"• **Volume**: {result.got_metrics.get('volume', 0)} thoughts\n"
        summary += f"• **Latency**: {result.got_metrics.get('latency', 0)} hops\n"
        summary += f"• **Quality Improvement**: {result.got_metrics.get('quality_improvement', 1.0):.2f}x\n"
    
    if not result.success and result.error_message:
        summary += f"\n❌ **Error**: {result.error_message}"
    
    return summary


def _format_debug_response(result: QueryResult, presentation: str) -> str:
    """Format a detailed debug response with full presentation"""
    
    debug_header = f"""🔬 **GoT DEBUG OUTPUT**
==========================================

**Execution Details**:
• Success: {result.success}
• Time: {result.total_execution_time:.3f}s
• Steps: {len(result.execution_steps)}
• Results: {result.total_results}

---

**FULL GoT PRESENTATION**:
"""
    
    return debug_header + "\n" + presentation


def _format_comprehensive_response(result: QueryResult, presentation: str, query: str) -> str:
    """Format a comprehensive response balancing detail and readability"""
    
    status_icon = "✅" if result.success else "❌"
    
    # Extract key sections from the full presentation
    lines = presentation.split('\n')
    
    # Find key sections
    final_answer_start = None
    exec_summary_start = None
    got_metrics_start = None
    step_results_start = None
    
    for i, line in enumerate(lines):
        if "🎯 FINAL ANSWER" in line:
            final_answer_start = i
        elif "📋 EXECUTIVE SUMMARY" in line:
            exec_summary_start = i
        elif "📈 GOT FRAMEWORK METRICS" in line:
            got_metrics_start = i
        elif "📖 STEP-BY-STEP RESULTS" in line:
            step_results_start = i
    
    # Build comprehensive response
    response = f"""🧠 **GRAPH OF THOUGHTS BIOMEDICAL ANALYSIS**
═══════════════════════════════════════════════

{status_icon} **Query**: {query}
⏱️ **Processed in**: {result.total_execution_time:.3f} seconds

"""
    
    # Add executive summary
    if exec_summary_start is not None:
        summary_end = min(len(lines), exec_summary_start + 15)  # Limit summary length
        for i in range(exec_summary_start, summary_end):
            if lines[i].startswith("🎯") or lines[i].startswith("🗂️"):  # Stop at next section
                break
            response += lines[i] + "\n"
    
    response += "\n"
    
    # Add final answer
    if final_answer_start is not None:
        answer_end = min(len(lines), final_answer_start + 10)
        for i in range(final_answer_start, answer_end):
            if lines[i].startswith("🗂️") or lines[i].startswith("📖"):  # Stop at next section
                break
            response += lines[i] + "\n"
    
    response += "\n"
    
    # Add key execution steps summary
    if step_results_start is not None and len(result.execution_steps) > 0:
        response += "🔄 **EXECUTION PIPELINE**\n"
        response += "─" * 30 + "\n"
        
        for i, step in enumerate(result.execution_steps[:5], 1):  # Show first 5 steps
            status = "✅" if step.success else "❌"
            step_name = step.step_type.replace('_', ' ').title()
            response += f"{i}. {status} {step_name} ({step.execution_time:.3f}s)\n"
            if step.confidence > 0:
                response += f"   └─ Confidence: {step.confidence:.2f}\n"
        
        if len(result.execution_steps) > 5:
            response += f"   ... and {len(result.execution_steps) - 5} more steps\n"
        
        response += "\n"
    
    # Add GoT metrics
    if got_metrics_start is not None and result.got_metrics:
        response += "📈 **GOT PERFORMANCE METRICS**\n"
        response += "─" * 30 + "\n"
        
        metrics = result.got_metrics
        response += f"• **Volume**: {metrics.get('volume', 0)} thoughts\n"
        response += f"• **Latency**: {metrics.get('latency', 0)} hops\n"
        response += f"• **Total Thoughts**: {metrics.get('total_thoughts', 0)}\n"
        response += f"• **Quality Improvement**: {metrics.get('quality_improvement', 1.0):.2f}x\n"
        
        if 'thought_distribution' in metrics:
            response += "\n**Thought Distribution**:\n"
            for thought_type, count in metrics['thought_distribution'].items():
                response += f"  └─ {thought_type.replace('_', ' ').title()}: {count}\n"
    
    # Add error information if failed
    if not result.success and result.error_message:
        response += f"\n❌ **ERROR DETAILS**\n{result.error_message}\n"
    
    return response


# Alias for backward compatibility
handle_got_biomedical_query = handle_got_query