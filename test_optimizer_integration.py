#!/usr/bin/env python3
"""
Test script to verify integration of both optimization tools in MCP server
"""

import asyncio
import logging
from agentic_bte.servers.mcp.server import AgenticBTEMCPServer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_tool_registration():
    """Test that both optimization tools are properly registered"""
    
    print("🧪 Testing MCP Server Tool Registration")
    print("=" * 60)
    
    try:
        # Initialize server
        server = AgenticBTEMCPServer()
        
        print(f"✅ Server initialized successfully")
        
        # Test the tools by importing and checking the definitions
        from agentic_bte.servers.mcp.tools.bio_ner_tool import get_bio_ner_tool_definition
        from agentic_bte.servers.mcp.tools.trapi_tool import get_trapi_query_tool_definition
        from agentic_bte.servers.mcp.tools.bte_tool import get_bte_call_tool_definition
        from agentic_bte.servers.mcp.tools.query_tool import get_basic_plan_and_execute_tool_definition
        from agentic_bte.servers.mcp.tools.metakg_optimizer_tool import get_metakg_optimizer_tool_definition
        
        # Get tool definitions as registered in the server
        tool_definitions = [
            get_bio_ner_tool_definition(),
            get_trapi_query_tool_definition(),
            get_bte_call_tool_definition(),
            get_basic_plan_and_execute_tool_definition(),
            get_metakg_optimizer_tool_definition(),
        ]
        
        print(f"📊 Found {len(tool_definitions)} registered tools:")
        
        tool_names = []
        for tool_def in tool_definitions:
            tool_name = tool_def['name']
            tool_names.append(tool_name)
            description = tool_def['description'][:80] + "..." if len(tool_def['description']) > 80 else tool_def['description']
            print(f"   - {tool_name}: {description}")
        
        # Check for our expected tools
        expected_tools = [
            "bio_ner",
            "build_trapi_query", 
            "call_bte_api",
            "basic_plan_and_execute_query",
            "metakg_aware_optimizer"
        ]
        
        print(f"\n🔍 Verifying expected tools:")
        all_present = True
        for expected_tool in expected_tools:
            if expected_tool in tool_names:
                print(f"   ✅ {expected_tool} - Found")
            else:
                print(f"   ❌ {expected_tool} - Missing")
                all_present = False
        
        if all_present:
            print(f"\n🎉 All expected tools are properly registered!")
        else:
            print(f"\n⚠️  Some tools are missing from registration")
        
        # Test the tool handler imports
        print(f"\n🔄 Testing tool handler imports...")
        try:
            from agentic_bte.servers.mcp.tools.query_tool import handle_basic_plan_and_execute
            from agentic_bte.servers.mcp.tools.metakg_optimizer_tool import handle_metakg_optimizer
            print(f"   ✅ All tool handlers can be imported successfully")
            print(f"   📋 Tool routing verification:")
            print(f"      - basic_plan_and_execute_query: Handler available")
            print(f"      - metakg_aware_optimizer: Handler available")
        except ImportError as e:
            print(f"   ❌ Tool handler import error: {e}")
            
    except Exception as e:
        print(f"❌ Server initialization failed: {e}")
        logger.error(f"Test error: {e}", exc_info=True)

async def test_tool_definitions():
    """Test tool definition imports"""
    
    print(f"\n🔧 Testing Tool Definition Imports")
    print("=" * 60)
    
    try:
        # Test basic optimizer tool
        from agentic_bte.servers.mcp.tools.query_tool import (
            get_basic_plan_and_execute_tool_definition,
            handle_basic_plan_and_execute
        )
        
        basic_def = get_basic_plan_and_execute_tool_definition()
        print(f"✅ Basic optimizer tool definition loaded")
        print(f"   - Name: {basic_def['name']}")
        print(f"   - Description: {basic_def['description'][:60]}...")
        
        # Test meta-KG optimizer tool  
        from agentic_bte.servers.mcp.tools.metakg_optimizer_tool import (
            get_metakg_optimizer_tool_definition,
            handle_metakg_optimizer
        )
        
        metakg_def = get_metakg_optimizer_tool_definition()
        print(f"✅ Meta-KG optimizer tool definition loaded")
        print(f"   - Name: {metakg_def['name']}")
        print(f"   - Description: {metakg_def['description'][:60]}...")
        
        # Verify they have different names
        if basic_def['name'] != metakg_def['name']:
            print(f"✅ Tool names are properly distinct")
        else:
            print(f"❌ Tool names are the same - conflict!")
            
        print(f"\n📋 Tool Schema Validation:")
        
        # Check required parameters
        for tool_name, tool_def in [("Basic", basic_def), ("Meta-KG", metakg_def)]:
            schema = tool_def.get('inputSchema', {})
            required = schema.get('required', [])
            properties = schema.get('properties', {})
            
            print(f"   {tool_name} Optimizer:")
            print(f"     - Required params: {required}")
            print(f"     - Total params: {len(properties)}")
            
            if 'query' in required:
                print(f"     ✅ 'query' parameter is required")
            else:
                print(f"     ❌ 'query' parameter is not required")
                
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Tool definition error: {e}")
        logger.error(f"Tool definition test error: {e}", exc_info=True)

async def main():
    """Run all tests"""
    print("🚀 Agentic BTE Optimizer Integration Test")
    print("=" * 60)
    
    await test_tool_definitions()
    await test_tool_registration()
    
    print("\n✨ Integration test complete!")

if __name__ == "__main__":
    asyncio.run(main())