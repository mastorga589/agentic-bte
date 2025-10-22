#!/usr/bin/env python3
"""
Test script to verify local BTE instance connectivity and functionality
"""

import requests
import json
import sys
import time
from typing import Dict, Any

LOCAL_BTE_URL = "http://localhost:3000"

def test_connection() -> bool:
    """Test basic connectivity to local BTE instance"""
    try:
        print("🔗 Testing connection to local BTE instance...")
        response = requests.get(f"{LOCAL_BTE_URL}", timeout=5)
        print(f"✅ Connection successful! Status: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed: Local BTE instance not running on localhost:3000")
        return False
    except requests.exceptions.Timeout:
        print("❌ Connection timeout: Local BTE instance not responding")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def test_meta_knowledge_graph() -> bool:
    """Test meta knowledge graph endpoint"""
    try:
        print("📊 Testing meta knowledge graph endpoint...")
        response = requests.get(f"{LOCAL_BTE_URL}/v1/meta_knowledge_graph", timeout=10)
        
        if response.status_code == 200:
            meta_kg = response.json()
            edges_count = len(meta_kg.get('edges', []))
            print(f"✅ Meta KG retrieved successfully! {edges_count} edges found")
            return True
        else:
            print(f"❌ Meta KG endpoint failed: Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Meta KG test error: {e}")
        return False

def test_simple_trapi_query() -> bool:
    """Test a simple TRAPI query"""
    try:
        print("🧪 Testing simple TRAPI query...")
        
        # Simple test query: gene to chemical relationships
        test_query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "ids": ["NCBIGene:1956"],  # EGFR gene
                            "categories": ["biolink:Gene"]
                        },
                        "n1": {
                            "categories": ["biolink:SmallMolecule"]
                        }
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"]
                        }
                    }
                }
            }
        }
        
        response = requests.post(
            f"{LOCAL_BTE_URL}/v1/query",
            json=test_query,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if we got results
            knowledge_graph = result.get("message", {}).get("knowledge_graph", {})
            nodes_count = len(knowledge_graph.get("nodes", {}))
            edges_count = len(knowledge_graph.get("edges", {}))
            
            print(f"✅ TRAPI query successful! {nodes_count} nodes, {edges_count} edges")
            return True
        else:
            print(f"❌ TRAPI query failed: Status {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ TRAPI query test error: {e}")
        return False

def check_local_bte_version() -> Dict[str, Any]:
    """Check the version and details of local BTE instance"""
    try:
        print("ℹ️  Checking local BTE version...")
        
        # Try common BTE info endpoints
        for endpoint in ["/", "/meta", "/version", "/info"]:
            try:
                response = requests.get(f"{LOCAL_BTE_URL}{endpoint}", timeout=5)
                if response.status_code == 200:
                    data = response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
                    print(f"✅ Found info at {endpoint}:")
                    if isinstance(data, dict):
                        print(json.dumps(data, indent=2)[:500])
                    else:
                        print(str(data)[:500])
                    return {"endpoint": endpoint, "data": data}
            except:
                continue
        
        print("⚠️  Could not retrieve version information")
        return {}
        
    except Exception as e:
        print(f"❌ Version check error: {e}")
        return {}

def wait_for_bte_startup(max_wait_seconds: int = 30) -> bool:
    """Wait for BTE instance to start up"""
    print(f"⏳ Waiting up to {max_wait_seconds} seconds for BTE to start...")
    
    for i in range(max_wait_seconds):
        try:
            response = requests.get(f"{LOCAL_BTE_URL}", timeout=2)
            if response.status_code in [200, 404]:  # 404 is also OK for some BTE setups
                print(f"✅ BTE is responding after {i+1} seconds!")
                return True
        except:
            pass
        
        time.sleep(1)
        if i % 5 == 4:  # Print progress every 5 seconds
            print(f"   Still waiting... ({i+1}/{max_wait_seconds}s)")
    
    print(f"❌ BTE did not start within {max_wait_seconds} seconds")
    return False

def main():
    """Main test function"""
    print("🧬 LOCAL BTE INSTANCE VERIFICATION")
    print("=" * 50)
    
    # First try immediate connection
    if not test_connection():
        print("\n🔄 BTE not immediately available. Waiting for startup...")
        if not wait_for_bte_startup():
            print("\n❌ RESULT: Local BTE instance is not running or not accessible")
            print("\nℹ️  TO START LOCAL BTE:")
            print("   1. Navigate to your BTE directory")
            print("   2. Run: npm start (or equivalent)")
            print("   3. Ensure it's running on port 3000")
            print("   4. Run this script again")
            sys.exit(1)
    
    # Run comprehensive tests
    print("\n🧪 RUNNING COMPREHENSIVE TESTS...")
    print("-" * 40)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Meta Knowledge Graph
    if test_meta_knowledge_graph():
        tests_passed += 1
    
    # Test 2: Simple TRAPI Query
    if test_simple_trapi_query():
        tests_passed += 1
    
    # Test 3: Version/Info Check
    version_info = check_local_bte_version()
    if version_info:
        tests_passed += 1
    
    # Results
    print(f"\n📊 TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 SUCCESS: Local BTE instance is fully functional!")
        print("\n🔧 TO USE LOCAL BTE WITH AGENTIC-BTE:")
        print("   Set environment variable: AGENTIC_BTE_BTE_API_BASE_URL=http://localhost:3000/v1")
        print("   Or run: export AGENTIC_BTE_BTE_API_BASE_URL=http://localhost:3000/v1")
    elif tests_passed > 0:
        print("⚠️  PARTIAL SUCCESS: Local BTE is running but some features may not work correctly")
    else:
        print("❌ FAILURE: Local BTE instance has issues")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()