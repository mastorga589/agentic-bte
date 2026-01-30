#!/usr/bin/env python3
"""
Configuration script to switch agentic-bte to use local BTE instance
"""

import os
import sys
from pathlib import Path

def set_environment_variable():
    """Set the environment variable for local BTE"""
    local_bte_url = "http://localhost:3000/v1"
    
    print("🔧 CONFIGURING AGENTIC-BTE TO USE LOCAL BTE")
    print("=" * 50)
    
    # Set environment variable for current session
    os.environ["AGENTIC_BTE_BTE_API_BASE_URL"] = local_bte_url
    print(f"✅ Set AGENTIC_BTE_BTE_API_BASE_URL={local_bte_url}")
    
    # Create/update .env file
    env_file = Path(".env")
    env_content = []
    
    # Read existing .env if it exists
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if not line.strip().startswith("AGENTIC_BTE_BTE_API_BASE_URL"):
                    env_content.append(line.rstrip())
    
    # Add the new setting
    env_content.append(f"AGENTIC_BTE_BTE_API_BASE_URL={local_bte_url}")
    
    # Write back to .env
    with open(env_file, 'w') as f:
        f.write('\n'.join(env_content) + '\n')
    
    print(f"✅ Updated .env file with local BTE configuration")
    
    return local_bte_url

def test_configuration():
    """Test the new configuration"""
    print("\n🧪 TESTING CONFIGURATION...")
    print("-" * 30)
    
    try:
        # Import and test the settings
        sys.path.insert(0, str(Path.cwd()))
        from agentic_bte.config.settings import reload_settings
        
        # Reload settings to pick up environment changes
        settings = reload_settings()
        
        print(f"✅ BTE API Base URL: {settings.bte_api_base_url}")
        
        if "localhost:3000" in settings.bte_api_base_url:
            print("✅ Configuration successfully updated to use local BTE!")
            return True
        else:
            print("❌ Configuration not properly updated")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test error: {e}")
        return False

def test_bte_client():
    """Test the BTE client with new configuration"""
    print("\n🔌 TESTING BTE CLIENT...")
    print("-" * 25)
    
    try:
        from agentic_bte.core.knowledge.bte_client import BTEClient
        
        # Create client with local configuration
        client = BTEClient()
        print(f"✅ BTE Client initialized with URL: {client.base_url}")
        
        # Test meta knowledge graph retrieval
        print("📊 Testing meta knowledge graph retrieval...")
        meta_kg = client.get_meta_knowledge_graph()
        edges_count = len(meta_kg.get('edges', []))
        
        print(f"✅ Successfully retrieved meta-KG with {edges_count} edges!")
        return True
        
    except Exception as e:
        print(f"❌ BTE Client test error: {e}")
        return False

def create_local_test_script():
    """Create a test script for local BTE functionality"""
    test_script = '''#!/usr/bin/env python3
"""
Quick test script for local BTE functionality with agentic-bte
"""

import asyncio
from agentic_bte.core.queries.production_got_optimizer import execute_biomedical_query

async def test_local_bte():
    print("🧬 TESTING LOCAL BTE WITH AGENTIC-BTE")
    print("=" * 50)
    
    # Simple test query
    query = "What genes are related to aspirin?"
    
    try:
        print(f"🎯 Query: {query}")
        print("⚡ Executing with local BTE...")
        
        result, presentation = await execute_biomedical_query(query)
        
        print(f"✅ Success: {result.success}")
        print(f"📊 Results found: {result.total_results}")
        print(f"🔍 Entities: {len(result.entities_found)}")
        
        # Show first part of answer
        if result.final_answer:
            print("\\n📋 ANSWER PREVIEW:")
            print(result.final_answer[:300] + "..." if len(result.final_answer) > 300 else result.final_answer)
        
        print("\\n🎉 LOCAL BTE IS WORKING WITH AGENTIC-BTE!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\\n💡 TROUBLESHOOTING:")
        print("   1. Ensure local BTE is running on localhost:3000")
        print("   2. Check BTE logs for errors")
        print("   3. Try restarting the local BTE instance")

if __name__ == "__main__":
    asyncio.run(test_local_bte())
'''
    
    with open("test_local_integration.py", "w") as f:
        f.write(test_script)
    
    print("✅ Created test_local_integration.py for integrated testing")

def main():
    """Main configuration function"""
    
    # Set environment variable and update .env
    local_url = set_environment_variable()
    
    # Test configuration
    if test_configuration():
        print("✅ Configuration test passed")
    else:
        print("❌ Configuration test failed")
        return
    
    # Test BTE client
    if test_bte_client():
        print("✅ BTE client test passed")
    else:
        print("❌ BTE client test failed - local BTE may not be fully ready")
    
    # Create test script
    create_local_test_script()
    
    print("\n🎉 CONFIGURATION COMPLETE!")
    print("=" * 30)
    print(f"📍 Local BTE URL: {local_url}")
    print("📁 Configuration saved to .env")
    print("🧪 Test script: test_local_integration.py")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Ensure your local BTE is fully started and responding")
    print("2. Run: python test_local_integration.py")
    print("3. If successful, all your agentic-bte queries will now use the local BTE!")
    
    print("\n🔄 TO SWITCH BACK TO PRODUCTION BTE:")
    print("   Remove AGENTIC_BTE_BTE_API_BASE_URL from .env file")

if __name__ == "__main__":
    main()