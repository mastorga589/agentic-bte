"""
Unified Knowledge Manager Demo

This script demonstrates the capabilities of the unified knowledge manager
including TRAPI query building, evidence scoring, RDF graph management,
and knowledge consistency validation.
"""

import asyncio
import logging
import json
from typing import Dict, List, Any

from ..config import UnifiedConfig
from ..types import EntityContext, Entity, EntityType, ExecutionContext
from ..knowledge_manager import UnifiedKnowledgeManager, KnowledgeSource
from ...core.knowledge.predicate_strategy import QueryIntent


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class KnowledgeManagerDemo:
    """Demo class for showcasing knowledge manager capabilities"""
    
    def __init__(self):
        # Initialize configuration
        self.config = UnifiedConfig()
        
        # Configure for demo purposes
        self.config.domain.max_queries_per_strategy = 15
        self.config.domain.max_predicates_per_query = 5
        self.config.quality.min_results_threshold = 5
        self.config.quality.fallback_threshold = 2
        
        # Initialize knowledge manager
        self.knowledge_manager = UnifiedKnowledgeManager(self.config)
        
        logger.info("Demo knowledge manager initialized")
    
    async def run_trapi_query_building_demo(self):
        """Demonstrate TRAPI query building from natural language"""
        logger.info("=" * 60)
        logger.info("TRAPI QUERY BUILDING DEMO")
        logger.info("=" * 60)
        
        # Sample biomedical queries with entities
        test_cases = [
            {
                'query': "What drugs can treat type 2 diabetes?",
                'entities': [
                    Entity("diabetes", "MONDO:0005015", EntityType.DISEASE, 0.9),
                    Entity("drugs", None, EntityType.SMALL_MOLECULE, 0.7)
                ]
            },
            {
                'query': "What genes are associated with Alzheimer's disease?",
                'entities': [
                    Entity("Alzheimer's disease", "MONDO:0004975", EntityType.DISEASE, 0.85),
                    Entity("genes", None, EntityType.GENE, 0.8)
                ]
            },
            {
                'query': "How does metformin affect glucose metabolism?",
                'entities': [
                    Entity("metformin", "CHEBI:6801", EntityType.SMALL_MOLECULE, 0.9),
                    Entity("glucose metabolism", "GO:0006006", EntityType.BIOLOGICAL_PROCESS, 0.8)
                ]
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            logger.info(f"\nProcessing: {test_case['query']}")
            
            # Create entity context
            entity_context = EntityContext(
                query=test_case['query'],
                entities=test_case['entities'],
                confidence=0.8
            )
            
            # Create execution context
            execution_context = ExecutionContext(
                query=test_case['query'],
                strategy=None,  # Not needed for this demo
                entity_context=entity_context,
                knowledge_graph=None,
                config=self.config
            )
            
            # Generate TRAPI queries
            trapi_queries = await self.knowledge_manager.process_biomedical_query(
                test_case['query'],
                entity_context,
                execution_context
            )
            
            logger.info(f"Generated {len(trapi_queries)} TRAPI queries")
            
            # Show first few queries
            for i, trapi_query in enumerate(trapi_queries[:3]):
                logger.info(f"Query {i+1}: {trapi_query.predicate} (confidence: {trapi_query.confidence:.3f})")
                logger.info(f"  Entities: {' -> '.join(trapi_query.entities)}")
                logger.info(f"  Estimated results: {trapi_query.estimated_results}")
                logger.info(f"  Intent: {trapi_query.source_intent.value}")
            
            results.append({
                'query': test_case['query'],
                'trapi_queries': trapi_queries,
                'query_count': len(trapi_queries)
            })
        
        return results
    
    def run_evidence_scoring_demo(self):
        """Demonstrate evidence-based scoring of biomedical results"""
        logger.info("=" * 60)
        logger.info("EVIDENCE SCORING DEMO")
        logger.info("=" * 60)
        
        # Sample BTE/TRAPI results
        sample_results = [
            {
                "analyses": [
                    {
                        "edge_bindings": {
                            "e01": [{"id": "edge_high_evidence"}]
                        },
                        "node_bindings": {
                            "n0": [{"id": "MONDO:0005015"}],
                            "n1": [{"id": "CHEBI:6801"}]
                        }
                    }
                ]
            },
            {
                "analyses": [
                    {
                        "edge_bindings": {
                            "e01": [{"id": "edge_medium_evidence"}]
                        },
                        "node_bindings": {
                            "n0": [{"id": "MONDO:0005015"}],
                            "n1": [{"id": "CHEBI:71193"}]
                        }
                    }
                ]
            },
            {
                "analyses": [
                    {
                        "edge_bindings": {
                            "e01": [{"id": "edge_low_evidence"}]
                        },
                        "node_bindings": {
                            "n0": [{"id": "MONDO:0005015"}],
                            "n1": [{"id": "CHEBI:12345"}]
                        }
                    }
                ]
            }
        ]
        
        # Sample edges data with different evidence levels
        edges_data = {
            "edge_high_evidence": {
                "subject": "MONDO:0005015",
                "object": "CHEBI:6801",
                "predicate": "biolink:treated_by",
                "attributes": [
                    {"attribute_type_id": "biolink:max_research_phase", "value": 4},
                    {"attribute_type_id": "biolink:clinical_approval_status", "value": "approved_for_condition"}
                ],
                "sources": [
                    {"resource_id": "infores:drugbank"},
                    {"resource_id": "infores:clinicaltrials"},
                    {"resource_id": "infores:aact"}
                ]
            },
            "edge_medium_evidence": {
                "subject": "MONDO:0005015",
                "object": "CHEBI:71193",
                "predicate": "biolink:treated_by",
                "attributes": [
                    {"attribute_type_id": "biolink:max_research_phase", "value": 3}
                ],
                "sources": [
                    {"resource_id": "infores:clinicaltrials"}
                ]
            },
            "edge_low_evidence": {
                "subject": "MONDO:0005015",
                "object": "CHEBI:12345",
                "predicate": "biolink:related_to",
                "attributes": [
                    {"attribute_type_id": "biolink:supporting_study", "value": "preliminary"}
                ],
                "sources": [
                    {"resource_id": "infores:ctd"}
                ]
            }
        }
        
        logger.info(f"Scoring {len(sample_results)} biomedical results...")
        
        # Score the results
        scored_results = self.knowledge_manager.score_knowledge_results(
            sample_results,
            edges_data,
            "biolink:treated_by",
            QueryIntent.THERAPEUTIC
        )
        
        logger.info("\nEvidence Scoring Results:")
        for i, (result, score) in enumerate(scored_results):
            edge_id = result['analyses'][0]['edge_bindings']['e01'][0]['id']
            logger.info(f"Result {i+1}: {edge_id}")
            logger.info(f"  Score: {score:.3f}")
            logger.info(f"  Evidence level: {'High' if score > 0.8 else 'Medium' if score > 0.5 else 'Low'}")
        
        return scored_results
    
    def run_knowledge_graph_building_demo(self):
        """Demonstrate knowledge graph construction"""
        logger.info("=" * 60)
        logger.info("KNOWLEDGE GRAPH BUILDING DEMO")
        logger.info("=" * 60)
        
        # Use results from evidence scoring demo
        scored_results = self.run_evidence_scoring_demo()
        
        # Sample nodes data
        nodes_data = {
            "MONDO:0005015": {
                "name": "diabetes mellitus",
                "categories": ["biolink:Disease"]
            },
            "CHEBI:6801": {
                "name": "Metformin",
                "categories": ["biolink:SmallMolecule"]
            },
            "CHEBI:71193": {
                "name": "Liraglutide", 
                "categories": ["biolink:SmallMolecule"]
            },
            "CHEBI:12345": {
                "name": "Experimental Drug X",
                "categories": ["biolink:SmallMolecule"]
            }
        }
        
        # Sample edges data (same as scoring demo)
        edges_data = {
            "edge_high_evidence": {
                "subject": "MONDO:0005015",
                "object": "CHEBI:6801", 
                "predicate": "biolink:treated_by",
                "attributes": [
                    {"attribute_type_id": "biolink:max_research_phase", "value": 4}
                ],
                "sources": [
                    {"resource_id": "infores:drugbank"},
                    {"resource_id": "infores:clinicaltrials"}
                ]
            },
            "edge_medium_evidence": {
                "subject": "MONDO:0005015",
                "object": "CHEBI:71193",
                "predicate": "biolink:treated_by",
                "attributes": [
                    {"attribute_type_id": "biolink:max_research_phase", "value": 3}
                ],
                "sources": [
                    {"resource_id": "infores:clinicaltrials"}
                ]
            },
            "edge_low_evidence": {
                "subject": "MONDO:0005015",
                "object": "CHEBI:12345",
                "predicate": "biolink:related_to",
                "attributes": [],
                "sources": [
                    {"resource_id": "infores:ctd"}
                ]
            }
        }
        
        # Build knowledge graph
        knowledge_graph = self.knowledge_manager.build_knowledge_graph(
            scored_results,
            edges_data,
            nodes_data
        )
        
        logger.info("\nKnowledge Graph Summary:")
        logger.info(f"  Entities: {knowledge_graph.entity_count}")
        logger.info(f"  Relationships: {knowledge_graph.relationship_count}")
        logger.info(f"  Confidence Distribution: {knowledge_graph.confidence_distribution}")
        logger.info(f"  Provenance Summary: {knowledge_graph.provenance_summary}")
        
        # Show knowledge assertions
        logger.info(f"\nKnowledge Assertions: {len(self.knowledge_manager.knowledge_assertions)}")
        for i, (key, assertion) in enumerate(list(self.knowledge_manager.knowledge_assertions.items())[:3]):
            logger.info(f"Assertion {i+1}: {assertion.subject} -> {assertion.predicate} -> {assertion.object}")
            logger.info(f"  Confidence: {assertion.aggregated_confidence:.3f}")
            logger.info(f"  Evidence count: {len(assertion.evidence)}")
        
        return knowledge_graph
    
    def run_entity_knowledge_demo(self):
        """Demonstrate entity-specific knowledge retrieval"""
        logger.info("=" * 60)
        logger.info("ENTITY KNOWLEDGE RETRIEVAL DEMO")
        logger.info("=" * 60)
        
        # Build some knowledge first
        self.run_knowledge_graph_building_demo()
        
        # Query knowledge about specific entities
        entities_to_query = ["diabetes", "metformin", "Alzheimer"]
        
        for entity_name in entities_to_query:
            logger.info(f"\nQuerying knowledge about: {entity_name}")
            
            entity_knowledge = self.knowledge_manager.get_entity_knowledge(entity_name)
            
            logger.info(f"  RDF relationships: {entity_knowledge['relationship_count']}")
            logger.info(f"  Knowledge assertions: {entity_knowledge['assertion_count']}")
            
            # Show some relationships
            if entity_knowledge['knowledge_assertions']:
                logger.info("  Sample assertions:")
                for assertion in entity_knowledge['knowledge_assertions'][:2]:
                    logger.info(f"    {assertion.subject} -{assertion.predicate}-> {assertion.object} "
                              f"(conf: {assertion.aggregated_confidence:.3f})")
        
        return {entity: self.knowledge_manager.get_entity_knowledge(entity) 
                for entity in entities_to_query}
    
    async def run_knowledge_consistency_demo(self):
        """Demonstrate knowledge consistency validation"""
        logger.info("=" * 60)
        logger.info("KNOWLEDGE CONSISTENCY VALIDATION DEMO")
        logger.info("=" * 60)
        
        # Build knowledge graph first
        self.run_knowledge_graph_building_demo()
        
        # Add some conflicting assertions for demo
        from ..knowledge_manager import KnowledgeAssertion, KnowledgeEvidence
        
        # Add a conflicting assertion (drug that both treats and causes a disease)
        conflicting_assertion = KnowledgeAssertion(
            subject="aspirin",
            predicate="biolink:causes",  # Conflicts with treats
            object="gastric ulcer",
            subject_type="biolink:SmallMolecule",
            object_type="biolink:Disease",
            evidence=[KnowledgeEvidence(
                source=KnowledgeSource.BTE_API,
                confidence=0.6,
                provenance=["infores:ctd"],
                attributes={}
            )],
            aggregated_confidence=0.6
        )
        
        treating_assertion = KnowledgeAssertion(
            subject="aspirin", 
            predicate="biolink:treats",  # Conflicts with causes
            object="gastric ulcer",
            subject_type="biolink:SmallMolecule",
            object_type="biolink:Disease",
            evidence=[KnowledgeEvidence(
                source=KnowledgeSource.BTE_API,
                confidence=0.4,
                provenance=["infores:pubmed"],
                attributes={}
            )],
            aggregated_confidence=0.4
        )
        
        # Add low confidence assertion
        low_conf_assertion = KnowledgeAssertion(
            subject="unknown_compound",
            predicate="biolink:affects",
            object="rare_disease",
            subject_type="biolink:SmallMolecule",
            object_type="biolink:Disease",
            evidence=[],  # No evidence
            aggregated_confidence=0.1
        )
        
        # Add to knowledge manager
        self.knowledge_manager.knowledge_assertions["conflict1"] = conflicting_assertion
        self.knowledge_manager.knowledge_assertions["conflict2"] = treating_assertion
        self.knowledge_manager.knowledge_assertions["low_conf"] = low_conf_assertion
        
        logger.info("Added conflicting and low-confidence assertions for testing...")
        
        # Run validation
        validation_report = await self.knowledge_manager.validate_knowledge_consistency()
        
        logger.info("\nKnowledge Consistency Report:")
        logger.info(f"  Total assertions: {validation_report['total_assertions']}")
        logger.info(f"  Conflicts detected: {validation_report['conflicts_detected']}")
        logger.info(f"  Low confidence assertions: {validation_report['low_confidence_assertions']}")
        logger.info(f"  Missing evidence assertions: {validation_report['missing_evidence_assertions']}")
        
        if validation_report['conflicts']:
            logger.info("\n  Conflicts found:")
            for conflict in validation_report['conflicts']:
                logger.info(f"    {conflict['entity_pair']}: {conflict['conflicting_predicates']}")
        
        if validation_report['recommendations']:
            logger.info("\n  Recommendations:")
            for rec in validation_report['recommendations']:
                logger.info(f"    - {rec}")
        
        return validation_report
    
    def run_knowledge_export_demo(self):
        """Demonstrate knowledge graph export capabilities"""
        logger.info("=" * 60)
        logger.info("KNOWLEDGE EXPORT DEMO")
        logger.info("=" * 60)
        
        # Build knowledge first
        self.run_knowledge_graph_building_demo()
        
        # Export as Turtle (RDF)
        logger.info("Exporting as Turtle (RDF):")
        turtle_export = self.knowledge_manager.export_knowledge_graph("turtle")
        logger.info(f"  Turtle export length: {len(turtle_export)} characters")
        logger.info(f"  Sample: {turtle_export[:200]}...")
        
        # Export as JSON
        logger.info("\nExporting as JSON:")
        json_export = self.knowledge_manager.export_knowledge_graph("json")
        json_data = json.loads(json_export)
        logger.info(f"  JSON export contains {len(json_data)} assertions")
        
        if json_data:
            logger.info("  Sample assertion:")
            sample = json_data[0]
            logger.info(f"    {sample['subject']} -{sample['predicate']}-> {sample['object']}")
            logger.info(f"    Confidence: {sample['confidence']:.3f}")
            logger.info(f"    Evidence count: {sample['evidence_count']}")
        
        return {
            'turtle': turtle_export,
            'json': json_export,
            'assertion_count': len(json_data) if json_data else 0
        }
    
    def run_sparql_query_demo(self):
        """Demonstrate SPARQL querying of the RDF knowledge graph"""
        logger.info("=" * 60)
        logger.info("SPARQL QUERY DEMO")
        logger.info("=" * 60)
        
        # Build knowledge first
        self.run_knowledge_graph_building_demo()
        
        # Sample SPARQL queries
        sparql_queries = [
            {
                'name': 'Find all treatments',
                'query': '''
                PREFIX biolink: <https://w3id.org/biolink/vocab/>
                SELECT ?subject ?object
                WHERE {
                    ?subject biolink:treated_by ?object .
                }
                '''
            },
            {
                'name': 'Find entities related to diabetes',
                'query': '''
                PREFIX biolink: <https://w3id.org/biolink/vocab/>
                SELECT ?predicate ?object
                WHERE {
                    ?subject ?predicate ?object .
                    FILTER(CONTAINS(LCASE(STR(?subject)), "diabetes"))
                }
                '''
            }
        ]
        
        results = {}
        
        for query_info in sparql_queries:
            logger.info(f"\nExecuting SPARQL query: {query_info['name']}")
            
            try:
                query_results = self.knowledge_manager.query_knowledge_graph(query_info['query'])
                logger.info(f"  Results found: {len(query_results)}")
                
                # Show sample results
                for i, result in enumerate(query_results[:3]):
                    logger.info(f"    Result {i+1}: {result}")
                
                results[query_info['name']] = query_results
                
            except Exception as e:
                logger.error(f"  SPARQL query failed: {str(e)}")
                results[query_info['name']] = []
        
        return results
    
    def run_knowledge_statistics_demo(self):
        """Demonstrate comprehensive knowledge statistics"""
        logger.info("=" * 60)
        logger.info("KNOWLEDGE STATISTICS DEMO")
        logger.info("=" * 60)
        
        # Build knowledge first
        self.run_knowledge_graph_building_demo()
        
        # Get statistics
        stats = self.knowledge_manager.get_knowledge_statistics()
        
        logger.info("Knowledge System Statistics:")
        logger.info(f"  Knowledge assertions: {stats['knowledge_assertions']}")
        logger.info(f"  TRAPI queries built: {stats['trapi_queries_built']}")
        logger.info(f"  Cached queries: {stats['cached_queries']}")
        logger.info(f"  Results scored: {stats['results_scored']}")
        logger.info(f"  Average confidence: {stats['avg_confidence']:.3f}")
        
        logger.info("\n  RDF Graph Statistics:")
        rdf_stats = stats['rdf_graph']
        for key, value in rdf_stats.items():
            logger.info(f"    {key}: {value}")
        
        logger.info("\n  Query Intent Distribution:")
        for intent, count in stats['intent_distribution'].items():
            logger.info(f"    {intent}: {count}")
        
        logger.info("\n  Predicate Usage:")
        for predicate, count in stats['predicate_usage'].items():
            logger.info(f"    {predicate}: {count}")
        
        return stats
    
    async def run_full_demo(self):
        """Run all knowledge manager demo scenarios"""
        logger.info("🚀 Starting Unified Knowledge Manager Demo")
        logger.info("=" * 80)
        
        demo_results = {}
        
        try:
            # TRAPI query building
            demo_results['trapi_building'] = await self.run_trapi_query_building_demo()
            
            # Evidence scoring
            demo_results['evidence_scoring'] = self.run_evidence_scoring_demo()
            
            # Knowledge graph building
            demo_results['knowledge_graph'] = self.run_knowledge_graph_building_demo()
            
            # Entity knowledge retrieval
            demo_results['entity_knowledge'] = self.run_entity_knowledge_demo()
            
            # Knowledge consistency validation
            demo_results['consistency_validation'] = await self.run_knowledge_consistency_demo()
            
            # Knowledge export
            demo_results['knowledge_export'] = self.run_knowledge_export_demo()
            
            # SPARQL querying
            demo_results['sparql_queries'] = self.run_sparql_query_demo()
            
            # Statistics
            demo_results['statistics'] = self.run_knowledge_statistics_demo()
            
            # Final summary
            self._print_demo_summary(demo_results)
            
        except Exception as e:
            logger.error(f"Demo failed with error: {str(e)}")
            raise
        
        return demo_results
    
    def _print_demo_summary(self, demo_results: Dict[str, Any]):
        """Print summary of all demo results"""
        logger.info("=" * 80)
        logger.info("DEMO SUMMARY")
        logger.info("=" * 80)
        
        logger.info("Unified Knowledge Manager demonstrated:")
        logger.info("  ✅ TRAPI query generation from natural language")
        logger.info("  ✅ Evidence-based result scoring")
        logger.info("  ✅ Knowledge graph construction and accumulation")
        logger.info("  ✅ Entity-specific knowledge retrieval")
        logger.info("  ✅ Knowledge consistency validation")
        logger.info("  ✅ Multiple export formats (RDF/Turtle, JSON)")
        logger.info("  ✅ SPARQL querying capabilities")
        logger.info("  ✅ Comprehensive statistics and monitoring")
        
        # Summary statistics
        stats = demo_results.get('statistics', {})
        if stats:
            logger.info(f"\nFinal Statistics:")
            logger.info(f"  Total knowledge assertions: {stats.get('knowledge_assertions', 0)}")
            logger.info(f"  Total TRAPI queries built: {stats.get('trapi_queries_built', 0)}")
            logger.info(f"  Average confidence: {stats.get('avg_confidence', 0):.3f}")
            logger.info(f"  RDF triples: {stats.get('rdf_graph', {}).get('total_triples', 0)}")
        
        logger.info("\n🎉 Knowledge Manager Demo completed successfully!")


async def main():
    """Main demo function"""
    try:
        demo = KnowledgeManagerDemo()
        await demo.run_full_demo()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())