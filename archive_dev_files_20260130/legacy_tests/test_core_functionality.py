#!/usr/bin/env python3
"""
Comprehensive test of core functionality after cleanup

This script tests the essential components that should work after removing
redundant code and fixing import issues.
"""

import sys
import logging
import time
from typing import Dict, Any

# Add project root to path
sys.path.append('/Users/mastorga/Documents/agentic-bte')

# Set up clean logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s: %(message)s'
)

# Suppress verbose loggers
logging.getLogger('nmslib').setLevel(logging.ERROR)
logging.getLogger('spacy').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)

def test_mcp_integration():
    """Test MCP tool integration"""
    print("🔧 Testing MCP Integration...")
    
    try:
        from call_mcp_tool import call_mcp_tool
        
        # Test bio_ner
        ner_result = call_mcp_tool("bio_ner", query="What drugs treat diabetes?")
        assert "entities" in ner_result
        print("  ✅ bio_ner tool working")
        
        # Test build_trapi_query
        trapi_result = call_mcp_tool("build_trapi_query", query="What drugs treat diabetes?", entity_data={})
        assert "query" in trapi_result
        print("  ✅ build_trapi_query tool working")
        
        # Test call_bte_api
        bte_result = call_mcp_tool("call_bte_api", json_query={}, k=5, maxresults=50)
        assert "message" in bte_result
        print("  ✅ call_bte_api tool working")
        
        return True
    except Exception as e:
        print(f"  ❌ MCP integration failed: {e}")
        return False

def test_simple_working_optimizer():
    """Test SimpleWorkingOptimizer"""
    print("\n🎯 Testing SimpleWorkingOptimizer...")
    
    try:
        from agentic_bte.core.queries.simple_working_optimizer import SimpleWorkingOptimizer
        
        # Create optimizer
        optimizer = SimpleWorkingOptimizer(openai_api_key="test_key")
        print("  ✅ SimpleWorkingOptimizer created")
        
        # Test query execution
        query = "What drugs treat diabetes?"
        result = optimizer.optimize_query(query)
        
        # Validate result
        assert result.success == True
        assert len(result.results) > 0
        assert len(result.entities) > 0
        assert result.metrics.execution_time >= 0
        assert result.final_answer != ""
        
        print(f"  ✅ Query executed successfully")
        print(f"    - Success: {result.success}")
        print(f"    - Results: {len(result.results)}")
        print(f"    - Entities: {len(result.entities)}")
        print(f"    - Execution time: {result.metrics.execution_time:.3f}s")
        
        # Test stats
        stats = optimizer.get_stats()
        assert stats["success_rate"] == 1.0
        print(f"    - Success rate: {stats['success_rate']:.1%}")
        
        return True
    except Exception as e:
        print(f"  ❌ SimpleWorkingOptimizer failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hybrid_intelligent_optimizer():
    """Test HybridIntelligentOptimizer"""
    print("\n🧠 Testing HybridIntelligentOptimizer...")
    
    try:
        from agentic_bte.core.queries.hybrid_optimizer import HybridIntelligentOptimizer
        
        # Create optimizer
        optimizer = HybridIntelligentOptimizer(openai_api_key="test_key")
        print("  ✅ HybridIntelligentOptimizer created")
        
        # Test query execution
        query = "What drugs treat diabetes?"
        result = optimizer.optimize_query(query)
        
        # Validate result
        assert result.success == True
        assert len(result.results) > 0
        assert len(result.entities) > 0
        assert result.metrics.execution_time >= 0
        assert result.final_answer != ""
        assert len(result.reasoning_chain) > 0
        
        print(f"  ✅ Query executed successfully")
        print(f"    - Success: {result.success}")
        print(f"    - Results: {len(result.results)}")
        print(f"    - Entities: {len(result.entities)}")
        print(f"    - Execution time: {result.metrics.execution_time:.3f}s")
        print(f"    - Reasoning steps: {len(result.reasoning_chain)}")
        
        # Test performance summary
        perf_summary = optimizer.get_strategy_performance_summary()
        print(f"    - Strategies tracked: {len(perf_summary)}")
        
        return True
    except Exception as e:
        print(f"  ❌ HybridIntelligentOptimizer failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_core_imports():
    """Test that core imports work without errors"""
    print("\n📦 Testing Core Imports...")
    
    imports_to_test = [
        ("interfaces", "agentic_bte.core.queries.interfaces"),
        ("config_manager", "agentic_bte.core.queries.config_manager"),
        ("performance_monitor", "agentic_bte.core.queries.performance_monitor"),
        ("error_handling", "agentic_bte.core.queries.error_handling"),
        ("caching", "agentic_bte.core.queries.caching"),
        ("model_manager", "agentic_bte.core.models.model_manager"),
    ]
    
    failed_imports = []
    
    for name, module_path in imports_to_test:
        try:
            __import__(module_path)
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed_imports.append((name, str(e)))
    
    return len(failed_imports) == 0

def test_configuration_system():
    """Test configuration management"""
    print("\n⚙️  Testing Configuration System...")
    
    try:
        from agentic_bte.core.queries.config_manager import get_config_manager
        from agentic_bte.core.queries.interfaces import OptimizationStrategy
        
        config_manager = get_config_manager()
        print("  ✅ Config manager created")
        
        # Test getting config
        config = config_manager.get_config("basic_adaptive")
        assert config is not None
        assert hasattr(config, 'max_results')
        assert hasattr(config, 'k')
        print("  ✅ Config retrieval working")
        
        return True
    except Exception as e:
        print(f"  ❌ Configuration system failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring"""
    print("\n📊 Testing Performance Monitoring...")
    
    try:
        from agentic_bte.core.queries.performance_monitor import get_performance_monitor
        
        monitor = get_performance_monitor()
        print("  ✅ Performance monitor created")
        
        # Test recording a result with mock optimization result
        from agentic_bte.core.queries.interfaces import OptimizationResult, OptimizationStrategy
        
        # Create a mock result object
        mock_result = OptimizationResult(
            query="test query",
            strategy=OptimizationStrategy.BASIC_ADAPTIVE,
            start_time=time.time()
        )
        mock_result.success = True
        mock_result.metrics.execution_time = 0.1
        mock_result.metrics.total_results = 3
        mock_result.finalize()
        
        monitor.record_optimization_result(mock_result)
        print("  ✅ Performance recording working")
        
        return True
    except Exception as e:
        print(f"  ❌ Performance monitoring failed: {e}")
        return False

def run_all_tests():
    """Run all core functionality tests"""
    print("🚀 COMPREHENSIVE CORE FUNCTIONALITY TEST")
    print("="*60)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {
        "MCP Integration": test_mcp_integration(),
        "Core Imports": test_core_imports(),
        "Configuration System": test_configuration_system(),
        "Performance Monitoring": test_performance_monitoring(),
        "SimpleWorkingOptimizer": test_simple_working_optimizer(),
        "HybridIntelligentOptimizer": test_hybrid_intelligent_optimizer(),
    }
    
    print(f"\n{'='*60}")
    print("📋 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"   Passed: {passed}/{len(test_results)}")
    print(f"   Failed: {failed}/{len(test_results)}")
    print(f"   Success Rate: {passed/len(test_results)*100:.1f}%")
    
    if failed == 0:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("✅ Core functionality is working correctly")
        print("✅ System is ready for use")
    else:
        print(f"\n⚠️  {failed} TEST(S) FAILED")
        print("❌ Some core functionality needs attention")
    
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)