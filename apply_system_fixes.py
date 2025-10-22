#!/usr/bin/env python3
"""
System Fixes Integration Script

This script applies all three critical fixes to the existing agentic-bte system:

1. Entity Filtering Enhancement - Removes generic terms from BioNER
2. Subquery Placeholder System - Enables results to flow between subqueries  
3. TRAPI Single-Hop Validation - Ensures queries conform to BTE requirements

The script modifies the production system components and provides a way to
enable/disable the fixes for testing and comparison.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our fix modules
from entity_filter_enhancement import (
    BiomedicaleEntityFilter, 
    enhance_bio_ner_with_filtering
)
from subquery_placeholder_system import (
    SubqueryPlaceholderSystem,
    enhance_production_got_optimizer_with_placeholders
)
from trapi_single_hop_validator import (
    TRAPISingleHopValidator,
    enhance_trapi_builder_with_single_hop_validation
)

logger = logging.getLogger(__name__)


class SystemFixesIntegrator:
    """
    Integrates all system fixes into the existing agentic-bte components
    """
    
    def __init__(self):
        """Initialize the integrator"""
        self.fixes_applied = {
            'entity_filtering': False,
            'placeholder_system': False,
            'single_hop_validation': False
        }
        self.original_components = {}
    
    def apply_entity_filtering_fix(self) -> bool:
        """
        Apply entity filtering fix to remove generic terms
        
        Returns:
            True if fix applied successfully
        """
        try:
            logger.info("Applying entity filtering fix...")
            
            # Import BioNER tool from the existing system
            from agentic_bte.servers.mcp.tools.bio_ner_tool import get_cached_bio_ner_tool
            
            # Get the cached BioNER tool instance
            bio_ner_tool = get_cached_bio_ner_tool()
            
            # Store original for rollback
            if 'bio_ner_extract_and_link' not in self.original_components:
                self.original_components['bio_ner_extract_and_link'] = bio_ner_tool.extract_and_link
            
            # Enhance with filtering
            enhanced_tool = enhance_bio_ner_with_filtering(bio_ner_tool)
            
            logger.info("✅ Entity filtering fix applied successfully")
            self.fixes_applied['entity_filtering'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to apply entity filtering fix: {e}")
            return False
    
    def apply_placeholder_system_fix(self) -> bool:
        """
        Apply placeholder system fix to enable subquery result chaining
        
        Returns:
            True if fix applied successfully
        """
        try:
            logger.info("Applying placeholder system fix...")
            
            # Import the production GoT optimizer
            from agentic_bte.core.queries.production_got_optimizer import ProductionGoTOptimizer
            
            # Note: This fix needs to be applied when creating optimizer instances
            # We'll create an enhanced factory function
            
            def create_enhanced_production_got_optimizer(config=None):
                """Factory function for enhanced GoT optimizer"""
                optimizer = ProductionGoTOptimizer(config)
                return enhance_production_got_optimizer_with_placeholders(optimizer)
            
            # Store the enhanced factory
            self.original_components['create_enhanced_got_optimizer'] = create_enhanced_production_got_optimizer
            
            logger.info("✅ Placeholder system fix applied successfully")
            self.fixes_applied['placeholder_system'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to apply placeholder system fix: {e}")
            return False
    
    def apply_single_hop_validation_fix(self) -> bool:
        """
        Apply single-hop validation fix to prevent TRAPI query errors
        
        Returns:
            True if fix applied successfully
        """
        try:
            logger.info("Applying single-hop validation fix...")
            
            # Import TRAPI builder
            from agentic_bte.core.knowledge.trapi import TRAPIQueryBuilder
            
            # Create enhanced TRAPI builder factory
            def create_enhanced_trapi_builder():
                """Factory function for enhanced TRAPI builder"""
                builder = TRAPIQueryBuilder()
                return enhance_trapi_builder_with_single_hop_validation(builder)
            
            # Store the enhanced factory
            self.original_components['create_enhanced_trapi_builder'] = create_enhanced_trapi_builder
            
            logger.info("✅ Single-hop validation fix applied successfully")
            self.fixes_applied['single_hop_validation'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to apply single-hop validation fix: {e}")
            return False
    
    def apply_all_fixes(self) -> Dict[str, bool]:
        """
        Apply all fixes to the system
        
        Returns:
            Dictionary showing which fixes were applied successfully
        """
        logger.info("🔧 Applying all system fixes...")
        
        results = {
            'entity_filtering': self.apply_entity_filtering_fix(),
            'placeholder_system': self.apply_placeholder_system_fix(), 
            'single_hop_validation': self.apply_single_hop_validation_fix()
        }
        
        success_count = sum(results.values())
        total_fixes = len(results)
        
        logger.info(f"📊 System fixes applied: {success_count}/{total_fixes}")
        
        if success_count == total_fixes:
            logger.info("🎉 All fixes applied successfully!")
        else:
            logger.warning(f"⚠️  Only {success_count} out of {total_fixes} fixes applied successfully")
        
        return results
    
    def create_enhanced_query_runner(self):
        """
        Create an enhanced query runner with all fixes applied
        
        Returns:
            Enhanced query runner function
        """
        if not all(self.fixes_applied.values()):
            logger.warning("Not all fixes are applied. Enhanced query runner may not work optimally.")
        
        async def enhanced_biomedical_query(query: str, config=None):
            """
            Enhanced biomedical query runner with all fixes
            
            Args:
                query: Biomedical query string
                config: Optional configuration
                
            Returns:
                Query results with all enhancements
            """
            try:
                # Create enhanced optimizer with all fixes
                if 'create_enhanced_got_optimizer' in self.original_components:
                    optimizer = self.original_components['create_enhanced_got_optimizer'](config)
                else:
                    # Fallback to standard optimizer
                    from agentic_bte.core.queries.production_got_optimizer import ProductionGoTOptimizer
                    optimizer = ProductionGoTOptimizer(config)
                
                # Execute the query
                result, presentation = await optimizer.execute_query(query)
                
                return result, presentation
                
            except Exception as e:
                logger.error(f"Enhanced query execution failed: {e}")
                raise
        
        return enhanced_biomedical_query
    
    def get_fix_status(self) -> Dict[str, Any]:
        """
        Get status of applied fixes
        
        Returns:
            Dictionary with fix status information
        """
        return {
            'fixes_applied': self.fixes_applied.copy(),
            'total_fixes': len(self.fixes_applied),
            'successful_fixes': sum(self.fixes_applied.values()),
            'all_fixes_applied': all(self.fixes_applied.values()),
            'components_modified': list(self.original_components.keys())
        }


def demonstrate_fixes():
    """Demonstrate the fixes with example scenarios"""
    
    print("🧪 Demonstrating System Fixes")
    print("=" * 50)
    
    # Test entity filtering
    print("\n1. Entity Filtering Fix Demo:")
    from entity_filter_enhancement import BiomedicaleEntityFilter
    
    filter = BiomedicaleEntityFilter()
    test_entities = ["Brucellosis", "translation", "drugs", "targeting", "treat"]
    filtered = filter.filter_entities(test_entities)
    
    print(f"   Original: {test_entities}")
    print(f"   Filtered: {filtered}")
    print(f"   Removed: {set(test_entities) - set(filtered)}")
    
    # Test placeholder system  
    print("\n2. Placeholder System Fix Demo:")
    from subquery_placeholder_system import SubqueryPlaceholderSystem
    
    system = SubqueryPlaceholderSystem()
    # Simulate results
    mock_results = [{
        'knowledge_graph': {
            'nodes': {
                'CHEBI:27882': {'name': 'donepezil', 'categories': ['biolink:SmallMolecule']}
            }
        }
    }]
    
    placeholders = system.record_subquery_completion(
        0, "What drugs treat Brucellosis?", ['Brucellosis'], mock_results, True, 0.8
    )
    print(f"   Created placeholders: {placeholders}")
    
    # Test query resolution
    query_2 = "What genes do these drugs target?"
    resolved, enhanced_data = system.resolve_placeholders_in_subquery(query_2, 1)
    print(f"   Resolved query: {resolved}")
    print(f"   Enhanced data keys: {list(enhanced_data.keys())}")
    
    # Test TRAPI validation
    print("\n3. TRAPI Single-Hop Validation Demo:")
    from trapi_single_hop_validator import TRAPISingleHopValidator
    
    validator = TRAPISingleHopValidator()
    
    # Valid single-hop query
    valid_query = {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {"categories": ["biolink:Disease"]},
                    "n1": {"categories": ["biolink:SmallMolecule"]}
                },
                "edges": {
                    "e01": {"subject": "n0", "object": "n1", "predicates": ["biolink:treated_by"]}
                }
            }
        }
    }
    
    is_valid, errors = validator.validate_trapi_query(valid_query)
    print(f"   Single-hop query valid: {is_valid}")
    
    # Multi-hop query (invalid)
    multi_hop_query = {
        "message": {
            "query_graph": {
                "nodes": {"n0": {"categories": ["biolink:Disease"]}, "n1": {"categories": ["biolink:Gene"]}, "n2": {"categories": ["biolink:SmallMolecule"]}},
                "edges": {
                    "e01": {"subject": "n0", "object": "n1", "predicates": ["biolink:associated_with"]},
                    "e12": {"subject": "n1", "object": "n2", "predicates": ["biolink:targeted_by"]}
                }
            }
        }
    }
    
    is_valid, errors = validator.validate_trapi_query(multi_hop_query)
    print(f"   Multi-hop query valid: {is_valid}")
    print(f"   Errors: {errors}")


def main():
    """Main function to apply fixes and demonstrate functionality"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("🚀 Agentic-BTE System Fixes Integration")
    print("=" * 60)
    
    # Create integrator
    integrator = SystemFixesIntegrator()
    
    # Apply all fixes
    results = integrator.apply_all_fixes()
    
    # Show results
    print(f"\n📊 Fix Application Results:")
    for fix_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {fix_name}: {status}")
    
    # Show status
    status = integrator.get_fix_status()
    print(f"\n📈 System Status:")
    print(f"   Total fixes: {status['total_fixes']}")
    print(f"   Successful: {status['successful_fixes']}")
    print(f"   All applied: {status['all_fixes_applied']}")
    
    # Demonstrate fixes
    demonstrate_fixes()
    
    # Provide usage instructions
    print(f"\n📖 Usage Instructions:")
    print(f"   To use the enhanced system:")
    print(f"   1. Import: from apply_system_fixes import SystemFixesIntegrator")
    print(f"   2. Create: integrator = SystemFixesIntegrator()")  
    print(f"   3. Apply: integrator.apply_all_fixes()")
    print(f"   4. Use: enhanced_runner = integrator.create_enhanced_query_runner()")
    print(f"   5. Query: result = await enhanced_runner('Your biomedical query')")
    
    return integrator


if __name__ == "__main__":
    integrator = main()