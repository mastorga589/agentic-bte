"""
Result Presentation System for GoT Framework

This module provides comprehensive result presentation with graph visualization,
TRAPI query debugging, and user-friendly output formatting.
"""

import json
import time
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import StringIO
import textwrap

logger = logging.getLogger(__name__)


@dataclass
class QueryStep:
    """Represents a single step in query execution"""
    step_id: str
    step_type: str  # 'entity_extraction', 'query_building', 'api_execution', 'aggregation'
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    trapi_query: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    confidence: float = 0.0


@dataclass
class QueryResult:
    """Comprehensive query result with all execution details"""
    query: str
    final_answer: str
    execution_steps: List[QueryStep] = field(default_factory=list)
    total_execution_time: float = 0.0
    success: bool = True
    entities_found: List[Dict[str, Any]] = field(default_factory=list)
    total_results: int = 0
    quality_score: float = 0.0
    got_metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class ResultPresenter:
    """
    Comprehensive result presentation system
    
    Provides formatted output, graph visualization, and debugging information
    for GoT framework query execution results.
    """
    
    def __init__(self, show_debug: bool = True, show_graphs: bool = False):
        """
        Initialize result presenter
        
        Args:
            show_debug: Whether to show detailed debugging information
            show_graphs: Whether to generate graph visualizations
        """
        self.show_debug = show_debug
        self.show_graphs = show_graphs
        self.output_buffer = StringIO()
        
    def present_results(self, result: QueryResult) -> str:
        """
        Present comprehensive query results
        
        Args:
            result: QueryResult object containing all execution details
            
        Returns:
            Formatted string representation of results
        """
        self.output_buffer = StringIO()
        
        # Header
        self._write_header(result.query)
        
        # Executive Summary
        self._write_executive_summary(result)
        
        # Final Answer
        self._write_final_answer(result.final_answer)
        
        # Query Execution Plan
        self._write_execution_plan(result)
        
        # Step-by-step Results
        self._write_step_results(result)
        
        # TRAPI Query Debugging (if enabled)
        if self.show_debug:
            self._write_trapi_debugging(result)
        
        # GoT Framework Metrics
        self._write_got_metrics(result)
        
        # Graph Visualization (if enabled)
        if self.show_graphs:
            self._write_graph_visualization(result)
        
        # Footer
        self._write_footer(result)
        
        return self.output_buffer.getvalue()
    
    def _write_header(self, query: str):
        """Write formatted header"""
        self.output_buffer.write("=" * 100 + "\n")
        self.output_buffer.write("           GRAPH OF THOUGHTS (GoT) BIOMEDICAL QUERY RESULTS           \n")
        self.output_buffer.write("=" * 100 + "\n\n")
        
        wrapped_query = textwrap.fill(query, width=80, initial_indent="Query: ", 
                                     subsequent_indent="       ")
        self.output_buffer.write(f"{wrapped_query}\n")
        self.output_buffer.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def _write_executive_summary(self, result: QueryResult):
        """Write executive summary"""
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        
        self.output_buffer.write("📋 EXECUTIVE SUMMARY\n")
        self.output_buffer.write("-" * 50 + "\n")
        self.output_buffer.write(f"Status: {status}\n")
        self.output_buffer.write(f"Execution Time: {result.total_execution_time:.3f}s\n")
        self.output_buffer.write(f"Entities Found: {len(result.entities_found)}\n")
        self.output_buffer.write(f"Total Results: {result.total_results}\n")
        self.output_buffer.write(f"Quality Score: {result.quality_score:.3f}\n")
        self.output_buffer.write(f"Execution Steps: {len(result.execution_steps)}\n")
        
        if result.got_metrics:
            volume = result.got_metrics.get('volume', 0)
            latency = result.got_metrics.get('latency', 0)
            self.output_buffer.write(f"GoT Volume: {volume} thoughts\n")
            self.output_buffer.write(f"GoT Latency: {latency} hops\n")
        
        if result.error_message:
            self.output_buffer.write(f"Error: {result.error_message}\n")
        
        self.output_buffer.write("\n")
    
    def _write_final_answer(self, answer: str):
        """Write final answer section"""
        self.output_buffer.write("🎯 FINAL ANSWER\n")
        self.output_buffer.write("-" * 50 + "\n")
        
        wrapped_answer = textwrap.fill(answer, width=80)
        self.output_buffer.write(f"{wrapped_answer}\n\n")
    
    def _write_execution_plan(self, result: QueryResult):
        """Write query execution plan as a graph"""
        self.output_buffer.write("🗂️  QUERY EXECUTION PLAN\n")
        self.output_buffer.write("-" * 50 + "\n")
        
        if not result.execution_steps:
            self.output_buffer.write("No execution steps recorded.\n\n")
            return
        
        # Create visual execution flow
        step_types = {}
        for step in result.execution_steps:
            if step.step_type not in step_types:
                step_types[step.step_type] = []
            step_types[step.step_type].append(step)
        
        # Draw execution graph
        self.output_buffer.write("Execution Flow:\n")
        self.output_buffer.write("\n")
        
        flow_symbols = {
            'entity_extraction': '🔍',
            'query_building': '🛠️ ',
            'api_execution': '🌐',
            'aggregation': '📊'
        }
        
        for i, step in enumerate(result.execution_steps):
            symbol = flow_symbols.get(step.step_type, '⚙️')
            status = "✅" if step.success else "❌"
            
            indent = "    " if i > 0 else ""
            connector = "    ↓\n" if i < len(result.execution_steps) - 1 else ""
            
            self.output_buffer.write(f"{indent}{symbol} {step.step_type.replace('_', ' ').title()} {status}\n")
            self.output_buffer.write(f"{indent}   Time: {step.execution_time:.3f}s")
            
            if step.confidence > 0:
                self.output_buffer.write(f", Confidence: {step.confidence:.2f}")
            
            self.output_buffer.write("\n")
            
            if connector:
                self.output_buffer.write(connector)
        
        self.output_buffer.write("\n")
    
    def _write_step_results(self, result: QueryResult):
        """Write detailed step-by-step results"""
        self.output_buffer.write("📖 STEP-BY-STEP RESULTS\n")
        self.output_buffer.write("-" * 50 + "\n")
        
        for i, step in enumerate(result.execution_steps, 1):
            status_icon = "✅" if step.success else "❌"
            
            self.output_buffer.write(f"\n{i}. {step.step_type.replace('_', ' ').title()} {status_icon}\n")
            self.output_buffer.write(f"   Step ID: {step.step_id}\n")
            self.output_buffer.write(f"   Execution Time: {step.execution_time:.3f}s\n")
            
            if step.confidence > 0:
                self.output_buffer.write(f"   Confidence: {step.confidence:.2f}\n")
            
            # Input summary
            if step.input_data:
                input_summary = self._summarize_data(step.input_data)
                self.output_buffer.write(f"   Input: {input_summary}\n")
            
            # Output summary
            if step.output_data:
                output_summary = self._summarize_data(step.output_data)
                self.output_buffer.write(f"   Output: {output_summary}\n")
            
            # Error information
            if step.error_message:
                self.output_buffer.write(f"   Error: {step.error_message}\n")
            
            # Special handling for different step types
            if step.step_type == 'entity_extraction':
                self._write_entity_details(step)
            elif step.step_type == 'api_execution':
                self._write_api_details(step)
            elif step.step_type == 'aggregation':
                self._write_aggregation_details(step)
        
        self.output_buffer.write("\n")
    
    def _write_entity_details(self, step: QueryStep):
        """Write detailed entity extraction results"""
        entities = step.output_data.get('entities', [])
        if entities:
            self.output_buffer.write("   📋 Entities Found:\n")
            for entity in entities[:5]:  # Show top 5
                name = entity.get('name', 'Unknown')
                entity_type = entity.get('type', 'Unknown')
                confidence = entity.get('confidence', 0.0)
                entity_id = entity.get('id', 'N/A')
                self.output_buffer.write(f"      • {name} ({entity_type}) - ID: {entity_id}, Conf: {confidence:.2f}\n")
    
    def _write_api_details(self, step: QueryStep):
        """Write detailed API execution results"""
        results = step.output_data.get('results', [])
        total_results = step.output_data.get('total_results', len(results))
        
        self.output_buffer.write(f"   📊 API Results: {total_results} total\n")
        
        # Show top results with confidence scores
        for i, result in enumerate(results[:3], 1):
            score = result.get('score', 0.0)
            result_id = result.get('id', f'result_{i}')
            
            # Extract meaningful information from knowledge graph
            kg = result.get('knowledge_graph', {})
            nodes = kg.get('nodes', {})
            edges = kg.get('edges', {})
            
            self.output_buffer.write(f"      Result {i}: {result_id} (score: {score:.3f})\n")
            self.output_buffer.write(f"        Nodes: {len(nodes)}, Edges: {len(edges)}\n")
    
    def _write_aggregation_details(self, step: QueryStep):
        """Write detailed aggregation results"""
        ranked_results = step.output_data.get('ranked_results', [])
        conflicts_resolved = step.output_data.get('conflicts_resolved', 0)
        duplicates_removed = step.output_data.get('duplicates_removed', 0)
        
        self.output_buffer.write(f"   🔄 Aggregation Summary:\n")
        self.output_buffer.write(f"      Conflicts Resolved: {conflicts_resolved}\n")
        self.output_buffer.write(f"      Duplicates Removed: {duplicates_removed}\n")
        self.output_buffer.write(f"      Final Results: {len(ranked_results)}\n")
        
        if ranked_results:
            top_result = ranked_results[0]
            top_score = top_result.get('aggregate_score', 0.0)
            self.output_buffer.write(f"      Top Result Score: {top_score:.3f}\n")
    
    def _write_trapi_debugging(self, result: QueryResult):
        """Write TRAPI query debugging information"""
        self.output_buffer.write("🐛 TRAPI QUERY DEBUGGING\n")
        self.output_buffer.write("-" * 50 + "\n")
        
        trapi_steps = [step for step in result.execution_steps if step.trapi_query]
        
        if not trapi_steps:
            self.output_buffer.write("No TRAPI queries recorded.\n\n")
            return
        
        for i, step in enumerate(trapi_steps, 1):
            self.output_buffer.write(f"\nTRAPI Query {i} ({step.step_type}):\n")
            self.output_buffer.write(f"Step ID: {step.step_id}\n")
            
            if step.trapi_query:
                # Format TRAPI query for readability
                formatted_query = json.dumps(step.trapi_query, indent=2)
                
                # Truncate if too long
                if len(formatted_query) > 1000:
                    lines = formatted_query.split('\n')
                    truncated = '\n'.join(lines[:20]) + '\n    ... (truncated) ...\n' + '\n'.join(lines[-5:])
                    formatted_query = truncated
                
                self.output_buffer.write("```json\n")
                self.output_buffer.write(formatted_query)
                self.output_buffer.write("\n```\n")
            
            # Query graph summary
            if step.trapi_query and 'message' in step.trapi_query:
                qg = step.trapi_query.get('message', {}).get('query_graph', {})
                nodes = qg.get('nodes', {})
                edges = qg.get('edges', {})
                
                self.output_buffer.write(f"Query Graph: {len(nodes)} nodes, {len(edges)} edges\n")
                
                # Node summary
                if nodes:
                    self.output_buffer.write("Nodes:\n")
                    for node_id, node_data in nodes.items():
                        categories = node_data.get('categories', ['Unknown'])
                        name = node_data.get('name', 'Unnamed')
                        self.output_buffer.write(f"  {node_id}: {name} ({', '.join(categories)})\n")
                
                # Edge summary
                if edges:
                    self.output_buffer.write("Edges:\n")
                    for edge_id, edge_data in edges.items():
                        subject = edge_data.get('subject', 'unknown')
                        obj = edge_data.get('object', 'unknown')
                        predicates = edge_data.get('predicates', ['unknown'])
                        self.output_buffer.write(f"  {edge_id}: {subject} → {obj} ({', '.join(predicates)})\n")
        
        self.output_buffer.write("\n")
    
    def _write_got_metrics(self, result: QueryResult):
        """Write GoT framework metrics"""
        self.output_buffer.write("📈 GOT FRAMEWORK METRICS\n")
        self.output_buffer.write("-" * 50 + "\n")
        
        metrics = result.got_metrics
        
        if not metrics:
            self.output_buffer.write("No GoT metrics available.\n\n")
            return
        
        # Core metrics
        volume = metrics.get('volume', 0)
        latency = metrics.get('latency', 0)
        total_thoughts = metrics.get('total_thoughts', 0)
        
        self.output_buffer.write(f"Volume: {volume} thoughts\n")
        self.output_buffer.write(f"Latency: {latency} hops\n")
        self.output_buffer.write(f"Total Thoughts: {total_thoughts}\n")
        
        # Performance metrics
        quality_improvement = metrics.get('quality_improvement', 0.0)
        cost_reduction = metrics.get('cost_reduction', 0.0)
        parallel_speedup = metrics.get('parallel_speedup', 0.0)
        
        if quality_improvement > 0:
            self.output_buffer.write(f"Quality Improvement: {quality_improvement:.2f}x\n")
        if cost_reduction > 0:
            self.output_buffer.write(f"Cost Reduction: {cost_reduction:.2f}x\n")
        if parallel_speedup > 0:
            self.output_buffer.write(f"Parallel Speedup: {parallel_speedup:.2f}x\n")
        
        # Efficiency metrics
        if volume > 0 and latency > 0:
            efficiency = volume / latency
            self.output_buffer.write(f"Volume-Latency Efficiency: {efficiency:.2f}\n")
        
        # Thought distribution
        thought_types = metrics.get('thought_distribution', {})
        if thought_types:
            self.output_buffer.write("\nThought Type Distribution:\n")
            for thought_type, count in thought_types.items():
                percentage = (count / total_thoughts) * 100 if total_thoughts > 0 else 0
                self.output_buffer.write(f"  {thought_type}: {count} ({percentage:.1f}%)\n")
        
        self.output_buffer.write("\n")
    
    def _write_graph_visualization(self, result: QueryResult):
        """Write graph visualization information"""
        self.output_buffer.write("📊 GRAPH VISUALIZATION\n")
        self.output_buffer.write("-" * 50 + "\n")
        
        if self.show_graphs:
            try:
                graph_path = self._generate_execution_graph(result)
                if graph_path:
                    self.output_buffer.write(f"Execution graph saved to: {graph_path}\n")
                else:
                    self.output_buffer.write("Could not generate execution graph.\n")
            except Exception as e:
                self.output_buffer.write(f"Graph generation failed: {str(e)}\n")
        else:
            self.output_buffer.write("Graph visualization disabled.\n")
        
        # Text-based graph representation
        self._write_text_graph(result)
        self.output_buffer.write("\n")
    
    def _write_text_graph(self, result: QueryResult):
        """Write text-based graph representation"""
        self.output_buffer.write("\nText-based Execution Graph:\n")
        
        if not result.execution_steps:
            self.output_buffer.write("No steps to visualize.\n")
            return
        
        # Create ASCII art representation
        self.output_buffer.write("\n")
        
        for i, step in enumerate(result.execution_steps):
            # Step box
            step_name = step.step_type.replace('_', ' ').title()
            status = "✓" if step.success else "✗"
            time_str = f"{step.execution_time:.2f}s"
            
            # Create box
            box_content = f" {status} {step_name} ({time_str}) "
            box_width = len(box_content) + 2
            
            self.output_buffer.write("┌" + "─" * (box_width - 2) + "┐\n")
            self.output_buffer.write(f"│{box_content}│\n")
            self.output_buffer.write("└" + "─" * (box_width - 2) + "┘\n")
            
            # Connection arrow
            if i < len(result.execution_steps) - 1:
                self.output_buffer.write("      │\n")
                self.output_buffer.write("      ▼\n")
    
    def _generate_execution_graph(self, result: QueryResult) -> Optional[str]:
        """Generate visual execution graph using matplotlib"""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            # Create directed graph
            G = nx.DiGraph()
            
            # Add nodes
            for i, step in enumerate(result.execution_steps):
                node_id = f"step_{i}"
                step_name = step.step_type.replace('_', '\n').title()
                status = "✓" if step.success else "✗"
                label = f"{status} {step_name}\n{step.execution_time:.2f}s"
                
                color = 'lightgreen' if step.success else 'lightcoral'
                G.add_node(node_id, label=label, color=color)
            
            # Add edges (sequential flow)
            for i in range(len(result.execution_steps) - 1):
                G.add_edge(f"step_{i}", f"step_{i+1}")
            
            # Create plot
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Draw nodes
            node_colors = [G.nodes[node]['color'] for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.9)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, alpha=0.6)
            
            # Draw labels
            labels = {node: G.nodes[node]['label'] for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8)
            
            plt.title(f"GoT Query Execution Graph\n{result.query[:60]}...")
            plt.axis('off')
            plt.tight_layout()
            
            # Save graph
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"got_execution_graph_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except ImportError:
            logger.warning("matplotlib or networkx not available for graph generation")
            return None
        except Exception as e:
            logger.error(f"Error generating execution graph: {str(e)}")
            return None
    
    def _write_footer(self, result: QueryResult):
        """Write formatted footer"""
        self.output_buffer.write("=" * 100 + "\n")
        if result.success:
            self.output_buffer.write("                      🎉 QUERY EXECUTION COMPLETED SUCCESSFULLY 🎉                     \n")
        else:
            self.output_buffer.write("                      ⚠️  QUERY EXECUTION COMPLETED WITH ERRORS ⚠️                      \n")
        self.output_buffer.write("=" * 100 + "\n\n")
    
    def _summarize_data(self, data: Dict[str, Any]) -> str:
        """Create a brief summary of data"""
        if isinstance(data, dict):
            if 'entities' in data:
                entities = data['entities']
                return f"{len(entities)} entities"
            elif 'results' in data:
                results = data['results']
                return f"{len(results)} results"
            elif 'query' in data:
                return "TRAPI query"
            else:
                return f"dict with {len(data)} keys"
        elif isinstance(data, list):
            return f"list with {len(data)} items"
        else:
            return str(type(data).__name__)


def format_final_answer(results: List[Dict[str, Any]], entities: List[Dict[str, Any]], 
                       query: str) -> str:
    """
    Generate a comprehensive, research-quality final answer from query results with improved
    handling of sparse data and zero confidence scores
    
    Args:
        results: List of API results
        entities: List of extracted entities  
        query: Original query string
        
    Returns:
        Formatted final answer string with detailed explanations
    """
    if not results:
        return f"No results found for the query: {query}"
    
    # Extract therapeutic compounds and other entities with improved parsing
    compounds = set()
    pathways = set()
    mechanisms = set()
    diseases = set()
    genes = set()
    relationships = []
    
    # Track result quality with better handling of sparse data
    valid_results = 0
    total_confidence = 0.0
    results_with_nodes = 0
    results_with_edges = 0
    
    # Process each result with comprehensive entity extraction
    for result in results:
        score = result.get('score', 0.0)
        if isinstance(score, (int, float)) and score > 0:
            total_confidence += score
            valid_results += 1
        
        # Extract from TRAPI-style knowledge graphs
        kg = result.get('knowledge_graph', {})
        if isinstance(kg, dict):
            nodes = kg.get('nodes', {})
            edges = kg.get('edges', {})
            
            if nodes:
                results_with_nodes += 1
            if edges:
                results_with_edges += 1
            
            # Extract entities from nodes
            for node_id, node_data in nodes.items():
                if isinstance(node_data, dict):
                    node_name = node_data.get('name', '').strip()
                    node_categories = node_data.get('categories', [])
                    
                    if node_name and isinstance(node_categories, list):
                        for category in node_categories:
                            if isinstance(category, str):
                                cat_lower = category.lower()
                                if any(term in cat_lower for term in ['chemical', 'drug', 'compound', 'small_molecule']):
                                    compounds.add(node_name)
                                elif any(term in cat_lower for term in ['disease', 'disorder', 'condition']):
                                    diseases.add(node_name)
                                elif any(term in cat_lower for term in ['gene', 'protein']):
                                    genes.add(node_name)
                                elif any(term in cat_lower for term in ['pathway', 'biological_process']):
                                    pathways.add(node_name)
            
            # Extract relationships from edges
            for edge_id, edge_data in edges.items():
                if isinstance(edge_data, dict):
                    subject_id = edge_data.get('subject')
                    object_id = edge_data.get('object')
                    predicate = edge_data.get('predicate', 'related_to')
                    
                    if subject_id in nodes and object_id in nodes:
                        subject_name = nodes[subject_id].get('name', '')
                        object_name = nodes[object_id].get('name', '')
                        
                        if subject_name and object_name:
                            relationships.append({
                                'subject': subject_name[:50],  # Truncate long names
                                'predicate': predicate,
                                'object': object_name[:50],
                                'score': score
                            })
        
        # Extract from direct result fields (fallback for non-TRAPI results)
        if not kg:  # If no knowledge graph, try direct extraction
            for field in ['compound', 'chemical', 'drug', 'small_molecule', 'substance']:
                if field in result and isinstance(result[field], str):
                    name = result[field].strip()
                    if name:
                        compounds.add(name)
            
            # Extract from subject/object structure
            if 'subject' in result and 'object' in result:
                subject = result.get('subject', {})
                obj = result.get('object', {})
                predicate = result.get('predicate', 'related_to')
                
                for entity in [subject, obj]:
                    if isinstance(entity, dict):
                        entity_type = entity.get('type', '').lower()
                        entity_name = entity.get('name', '').strip()
                        
                        if entity_name:
                            if any(term in entity_type for term in ['chemical', 'drug', 'compound']):
                                compounds.add(entity_name)
                            elif any(term in entity_type for term in ['disease', 'disorder']):
                                diseases.add(entity_name)
                            elif any(term in entity_type for term in ['gene', 'protein']):
                                genes.add(entity_name)
    
    # Calculate data quality metrics
    avg_confidence = total_confidence / max(1, valid_results) if valid_results > 0 else 0.0
    data_completeness = (results_with_nodes + results_with_edges) / (2 * len(results)) if results else 0.0
    
    # Build comprehensive answer with quality assessment
    answer_parts = []
    
    # Query type detection
    query_lower = query.lower()
    is_therapeutic_query = any(word in query_lower for word in ['treat', 'drug', 'medicine', 'therapeutic'])
    is_genetic_query = any(word in query_lower for word in ['gene', 'protein', 'genetic'])
    is_mechanism_query = any(word in query_lower for word in ['mechanism', 'pathway', 'how', 'target'])
    
    # Main findings section
    if is_therapeutic_query:
        answer_parts.append(f"**🔬 Therapeutic Compounds Analysis**")
        if compounds:
            answer_parts.append(f"Found {len(compounds)} therapeutic compounds relevant to your query:")
            # Show compounds with improved formatting
            compound_list = sorted(list(compounds))
            for i, compound in enumerate(compound_list[:10], 1):
                answer_parts.append(f"  {i}. {compound}")
            if len(compounds) > 10:
                answer_parts.append(f"  ... and {len(compounds) - 10} additional compounds")
        else:
            answer_parts.append("⚠️ No specific therapeutic compounds were identified with sufficient confidence.")
            answer_parts.append("This may indicate that:")
            answer_parts.append("  • The query requires more specific terminology")
            answer_parts.append("  • The knowledge graph has limited coverage for this area")
            answer_parts.append("  • The relationships exist but have low confidence scores")
    
    elif is_genetic_query:
        answer_parts.append(f"**🧬 Genetic Factors Analysis**")
        if genes:
            answer_parts.append(f"Identified {len(genes)} genes/proteins:")
            gene_list = sorted(list(genes))
            for i, gene in enumerate(gene_list[:8], 1):
                answer_parts.append(f"  {i}. {gene}")
            if len(genes) > 8:
                answer_parts.append(f"  ... and {len(genes) - 8} additional genes")
        else:
            answer_parts.append("⚠️ No specific genes or proteins were identified.")
    
    else:
        # General biomedical analysis
        answer_parts.append(f"**📊 Biomedical Analysis Results**")
        
        findings = []
        if compounds:
            findings.append(f"{len(compounds)} therapeutic compounds")
        if diseases:
            findings.append(f"{len(diseases)} diseases/conditions")
        if genes:
            findings.append(f"{len(genes)} genes/proteins")
        if pathways:
            findings.append(f"{len(pathways)} biological pathways")
        
        if findings:
            answer_parts.append(f"Identified: {', '.join(findings)}")
        else:
            answer_parts.append("⚠️ Limited specific entities were extracted from the results.")
    
    # Associated entities section
    if diseases and not is_therapeutic_query:
        answer_parts.append(f"\n**🏥 Associated Medical Conditions:**")
        disease_list = sorted(list(diseases))
        for disease in disease_list[:5]:
            answer_parts.append(f"  • {disease}")
        if len(diseases) > 5:
            answer_parts.append(f"  • ... and {len(diseases) - 5} more conditions")
    
    # Relationship insights
    if relationships:
        answer_parts.append(f"\n**🔗 Key Biomedical Relationships:**")
        # Show highest confidence relationships
        top_relationships = sorted(relationships, key=lambda x: x.get('score', 0), reverse=True)[:5]
        for rel in top_relationships:
            score_str = f" (confidence: {rel['score']:.2f})" if rel['score'] > 0 else ""
            predicate_clean = rel['predicate'].replace('biolink:', '').replace('_', ' ')
            answer_parts.append(f"  • {rel['subject']} {predicate_clean} {rel['object']}{score_str}")
    
    # Quality and methodology section
    answer_parts.append(f"\n**📈 Analysis Quality Assessment:**")
    answer_parts.append(f"  • Total results processed: {len(results)}")
    answer_parts.append(f"  • Results with structured data: {results_with_nodes}/{len(results)} ({100*results_with_nodes/len(results):.0f}%)")
    
    if valid_results > 0:
        confidence_rating = "High" if avg_confidence > 0.7 else "Medium" if avg_confidence > 0.3 else "Low"
        answer_parts.append(f"  • Average confidence score: {avg_confidence:.2f} ({confidence_rating})")
    else:
        answer_parts.append(f"  • Confidence scores: Not available (may require manual validation)")
    
    answer_parts.append(f"  • Data completeness: {100*data_completeness:.0f}%")
    
    # Recommendations section based on data quality
    if avg_confidence < 0.3 or not compounds and is_therapeutic_query:
        answer_parts.append(f"\n**💡 Recommendations:**")
        if avg_confidence < 0.3:
            answer_parts.append(f"  • Results have low confidence - consider refining query specificity")
        if not compounds and is_therapeutic_query:
            answer_parts.append(f"  • No therapeutic compounds found - try alternative terminology or broader search terms")
        if results_with_nodes < len(results) * 0.5:
            answer_parts.append(f"  • Limited structured data available - results may need expert interpretation")
        answer_parts.append(f"  • Consider consulting additional biomedical databases or literature")
    
    # Research context
    if compounds or genes or diseases:
        answer_parts.append(f"\n**🔬 Research Context:**")
        answer_parts.append(f"This analysis is based on curated biomedical knowledge graphs and peer-reviewed data sources. ")
        answer_parts.append(f"Results represent current scientific understanding but should be validated through additional research.")
    
    return "\n".join(answer_parts) if answer_parts else "Unable to generate detailed analysis from available results."


def _extract_entity_details_from_results(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Extract detailed entity information from knowledge graph results
    
    Returns:
        Dictionary mapping entity IDs to their detailed information
    """
    entity_details = {}
    
    for result in results[:10]:  # Process top 10 results
        score = result.get('score', 0)
        kg = result.get('knowledge_graph', {})
        nodes = kg.get('nodes', {})
        
        for node_id, node_data in nodes.items():
            if node_id not in entity_details:
                # Extract key information
                name = node_data.get('name', '').strip()
                categories = node_data.get('categories', [])
                attributes = node_data.get('attributes', [])
                
                # Skip generic or empty nodes
                if not name or 'Result' in name or len(name) < 3:
                    continue
                
                # Extract additional attributes
                synonyms = []
                descriptions = []
                identifiers = {}
                
                for attr in attributes:
                    attr_type = attr.get('attribute_type_id', '')
                    value = attr.get('value', '')
                    
                    if 'synonym' in attr_type.lower():
                        synonyms.append(value)
                    elif 'description' in attr_type.lower():
                        descriptions.append(value)
                    elif any(id_type in attr_type for id_type in ['CHEBI', 'MONDO', 'HGNC', 'NCBIGene']):
                        identifiers[attr_type] = value
                
                entity_details[node_id] = {
                    'name': name,
                    'categories': categories,
                    'synonyms': synonyms,
                    'descriptions': descriptions,
                    'identifiers': identifiers,
                    'confidence_scores': [score]  # Track all scores this entity appears in
                }
            else:
                # Update confidence scores
                entity_details[node_id]['confidence_scores'].append(score)
    
    return entity_details


def _extract_relationships_from_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract relationship information from knowledge graph edges
    
    Returns:
        List of relationship dictionaries with subject, object, and predicate info
    """
    relationships = []
    
    for result in results[:10]:  # Process top 10 results
        kg = result.get('knowledge_graph', {})
        edges = kg.get('edges', {})
        nodes = kg.get('nodes', {})
        
        for edge_id, edge_data in edges.items():
            subject_id = edge_data.get('subject')
            object_id = edge_data.get('object')
            predicates = edge_data.get('predicates', [])
            attributes = edge_data.get('attributes', [])
            
            if subject_id in nodes and object_id in nodes:
                subject_name = nodes[subject_id].get('name', subject_id)
                object_name = nodes[object_id].get('name', object_id)
                
                relationships.append({
                    'subject': {'id': subject_id, 'name': subject_name},
                    'object': {'id': object_id, 'name': object_name},
                    'predicates': predicates,
                    'attributes': attributes,
                    'confidence': result.get('score', 0)
                })
    
    return relationships


def _generate_query_introduction(query: str, entities: List[Dict[str, Any]], 
                                entity_details: Dict[str, Dict[str, Any]]) -> str:
    """
    Generate contextual introduction based on query type and entities
    """
    query_lower = query.lower()
    
    # Detect query type patterns
    if any(word in query_lower for word in ['treat', 'drug', 'medicine', 'therapeutic']):
        if any(word in query_lower for word in ['disorder', 'disease', 'condition']):
            return f"To identify therapeutic compounds that address the specified condition, we analyzed the biomedical knowledge graph for established drug-disease relationships and their underlying mechanisms."
    
    elif any(word in query_lower for word in ['gene', 'protein']):
        if any(word in query_lower for word in ['associated', 'related', 'involved']):
            return f"To identify genes and proteins associated with the specified condition, we examined genetic associations, protein interactions, and pathway involvement from curated biomedical databases."
    
    elif any(word in query_lower for word in ['pathway', 'mechanism', 'interaction']):
        return f"To understand the molecular mechanisms and pathways involved, we analyzed protein interactions, signaling cascades, and regulatory networks from integrated biomedical knowledge sources."
    
    elif any(word in query_lower for word in ['symptom', 'phenotype', 'manifestation']):
        return f"To characterize the clinical manifestations and phenotypic associations, we examined disease-symptom relationships and phenotypic profiles from clinical and research databases."
    
    # Generic introduction
    return f"Based on comprehensive analysis of biomedical knowledge graphs and curated databases, we identified the following key findings related to your query."


def _build_detailed_findings(entity_details: Dict[str, Dict[str, Any]], 
                            relationships: List[Dict[str, Any]], 
                            query: str) -> List[str]:
    """
    Build detailed findings with rich explanations for each key entity
    """
    findings = []
    
    # Sort entities by confidence and relevance
    sorted_entities = sorted(
        entity_details.items(),
        key=lambda x: (max(x[1]['confidence_scores']), len(x[1]['confidence_scores'])),
        reverse=True
    )
    
    # Process top entities (limit to avoid overwhelming output)
    for i, (entity_id, details) in enumerate(sorted_entities[:8], 1):
        name = details['name']
        categories = details['categories']
        descriptions = details['descriptions']
        identifiers = details['identifiers']
        avg_confidence = sum(details['confidence_scores']) / len(details['confidence_scores'])
        
        # Skip very low confidence results
        if avg_confidence < 0.3:
            continue
        
        # Build detailed explanation for this entity
        explanation_parts = []
        
        # Add identifier information
        identifier_info = ""
        if identifiers:
            # Format identifiers nicely
            id_strings = []
            for id_type, id_value in identifiers.items():
                if 'CHEBI' in id_type:
                    id_strings.append(f"CHEBI:{id_value}")
                elif 'MONDO' in id_type:
                    id_strings.append(f"MONDO:{id_value}")
                elif 'HGNC' in id_type:
                    id_strings.append(f"HGNC:{id_value}")
                else:
                    id_strings.append(f"{id_value}")
            
            if id_strings:
                identifier_info = f" ({', '.join(id_strings)})"
        
        # Main entity description
        entity_type = _get_primary_category(categories)
        explanation_parts.append(f"**{name}{identifier_info}**:")
        
        # Add biological context based on entity type and relationships
        context = _generate_entity_context(name, entity_type, relationships, query)
        if context:
            explanation_parts.append(context)
        
        # Add description if available
        if descriptions:
            # Use the most relevant description
            best_description = descriptions[0][:200] + "..." if len(descriptions[0]) > 200 else descriptions[0]
            explanation_parts.append(best_description)
        
        # Add mechanism/relationship information
        entity_relationships = _get_entity_relationships(entity_id, name, relationships)
        if entity_relationships:
            mechanism_text = _format_relationship_mechanisms(entity_relationships, query)
            if mechanism_text:
                explanation_parts.append(mechanism_text)
        
        # Combine explanation parts
        finding_text = f"{i}. " + " ".join(explanation_parts)
        findings.append(finding_text)
    
    return findings


def _generate_entity_context(name: str, entity_type: str, relationships: List[Dict[str, Any]], query: str) -> str:
    """
    Generate biological/medical context for an entity based on its type and query context
    """
    query_lower = query.lower()
    
    if entity_type == 'drug':
        if any(word in query_lower for word in ['treat', 'therapeutic']):
            return f"{name} is a therapeutic compound used in clinical treatment."
        else:
            return f"{name} is a pharmacological agent."
    
    elif entity_type == 'disease':
        return f"{name} is a medical condition that affects normal physiological function."
    
    elif entity_type == 'gene':
        if any(word in query_lower for word in ['associated', 'related']):
            return f"{name} is a gene that has been associated with the specified condition through genetic studies."
        else:
            return f"{name} is a gene that plays a role in normal cellular function."
    
    elif entity_type == 'protein':
        return f"{name} is a protein that participates in cellular processes and molecular interactions."
    
    elif entity_type == 'pathway':
        return f"{name} is a biological pathway involved in cellular signaling or metabolic processes."
    
    return ""  # No specific context available


def _get_primary_category(categories: List[str]) -> str:
    """
    Extract the most relevant category from a list of categories
    """
    if not categories:
        return 'entity'
    
    # Priority mapping for common biomedical categories
    priority_categories = {
        'biolink:Drug': 'drug',
        'biolink:ChemicalEntity': 'chemical',
        'biolink:Disease': 'disease',
        'biolink:Gene': 'gene',
        'biolink:Protein': 'protein',
        'biolink:Pathway': 'pathway',
        'biolink:BiologicalProcess': 'process'
    }
    
    # Find highest priority category
    for cat in categories:
        if cat in priority_categories:
            return priority_categories[cat]
    
    # Fallback to first category, cleaned up
    return categories[0].replace('biolink:', '').lower()


def _get_entity_relationships(entity_id: str, entity_name: str, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Get all relationships involving a specific entity
    """
    entity_rels = []
    
    for rel in relationships:
        if (rel['subject']['id'] == entity_id or rel['subject']['name'] == entity_name or
            rel['object']['id'] == entity_id or rel['object']['name'] == entity_name):
            entity_rels.append(rel)
    
    return entity_rels


def _format_relationship_mechanisms(relationships: List[Dict[str, Any]], query: str) -> str:
    """
    Format relationship information as mechanism descriptions
    """
    if not relationships:
        return ""
    
    mechanism_parts = []
    query_lower = query.lower()
    
    for rel in relationships[:3]:  # Limit to avoid overwhelming text
        predicates = rel.get('predicates', [])
        subject_name = rel['subject']['name']
        object_name = rel['object']['name']
        
        # Format predicate into readable text
        if predicates:
            predicate = predicates[0].replace('biolink:', '').replace('_', ' ')
            
            # Generate mechanism text based on predicate type
            if 'treat' in predicate or 'therapeutic' in predicate:
                mechanism_parts.append(f"It serves as a therapeutic intervention for {object_name}.")
            elif 'interact' in predicate:
                mechanism_parts.append(f"It interacts with {object_name} in cellular processes.")
            elif 'regulate' in predicate or 'modulate' in predicate:
                mechanism_parts.append(f"It regulates the activity of {object_name}.")
            elif 'associate' in predicate:
                mechanism_parts.append(f"It is associated with {object_name} through molecular mechanisms.")
            elif 'target' in predicate:
                mechanism_parts.append(f"It specifically targets {object_name} for its biological effects.")
    
    return " ".join(mechanism_parts) if mechanism_parts else ""


def _generate_contextual_conclusion(entity_details: Dict[str, Dict[str, Any]], 
                                  relationships: List[Dict[str, Any]], 
                                  query: str) -> str:
    """
    Generate a contextual conclusion that ties findings together
    """
    query_lower = query.lower()
    num_entities = len(entity_details)
    num_relationships = len(relationships)
    
    conclusion_parts = []
    
    # Query-specific conclusions
    if any(word in query_lower for word in ['treat', 'drug', 'therapeutic']):
        conclusion_parts.append(f"These {num_entities} therapeutic compounds represent established treatment options with well-documented mechanisms of action.")
        
        if any(word in query_lower for word in ['mechanism', 'target', 'pathway']):
            conclusion_parts.append("Each agent works through specific molecular targets and pathways to achieve therapeutic efficacy.")
            
    elif any(word in query_lower for word in ['gene', 'protein']):
        conclusion_parts.append(f"These {num_entities} genetic factors have been identified through comprehensive genomic and proteomic analyses.")
        
        if any(word in query_lower for word in ['disease', 'disorder']):
            conclusion_parts.append("Understanding these genetic associations is crucial for developing targeted therapies and personalized treatment approaches.")
            
    elif any(word in query_lower for word in ['pathway', 'mechanism']):
        conclusion_parts.append(f"These molecular pathways and mechanisms represent key regulatory networks involved in the biological processes of interest.")
    
    # Add research context
    if num_relationships > 5:
        conclusion_parts.append(f"The {num_relationships} documented interactions highlight the complex interconnected nature of these biomedical entities.")
    
    # General conclusion
    conclusion_parts.append("Further research and clinical validation may be needed to fully understand the therapeutic implications of these findings.")
    
    return " ".join(conclusion_parts)
