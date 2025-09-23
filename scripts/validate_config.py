#!/usr/bin/env python3
"""
Configuration Validation Script for Agentic BTE MCP Server

This script validates that all configurations are properly set up
and the MCP server can start correctly.
"""

import os
import sys
import json
import subprocess
from pathlib import Path


def check_console_script():
    """Check if the agentic-bte-mcp console script is available."""
    print("🔧 Checking console script...")
    try:
        result = subprocess.run(['which', 'agentic-bte-mcp'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Console script found: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError:
        print("❌ Console script not found. Run: pip install -e .")
        return False


def check_environment():
    """Check if required environment variables are set."""
    print("\n🌍 Checking environment variables...")
    
    # Load from .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        print(f"✅ Found .env file: {env_file}")
        with open(env_file) as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"')
                    os.environ[key] = value
    
    # Check required variables
    required_vars = [
        'AGENTIC_BTE_OPENAI_API_KEY',
        'AGENTIC_BTE_OPENAI_MODEL',
    ]
    
    all_good = True
    for var in required_vars:
        value = os.environ.get(var, '')
        if value:
            if 'API_KEY' in var:
                print(f"✅ {var}: {'*' * 20}...{value[-4:]}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
            all_good = False
    
    return all_good


def check_config_files():
    """Check if configuration files exist and are valid."""
    print("\n📄 Checking configuration files...")
    
    config_files = [
        "warp-mcp-config.json",
        "claude-desktop-config.json", 
        "mcp-config.json"
    ]
    
    all_good = True
    for config_file in config_files:
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    
                if 'mcpServers' in config and 'agentic-bte' in config['mcpServers']:
                    server_config = config['mcpServers']['agentic-bte']
                    command = server_config.get('command', '')
                    
                    print(f"✅ {config_file}: Valid (command: {command})")
                else:
                    print(f"❌ {config_file}: Missing agentic-bte server config")
                    all_good = False
                    
            except json.JSONDecodeError as e:
                print(f"❌ {config_file}: Invalid JSON - {e}")
                all_good = False
        else:
            print(f"⚠️  {config_file}: Not found (optional)")
    
    return all_good


def check_server_import():
    """Check if the server can be imported without errors."""
    print("\n🔗 Checking server imports...")
    
    try:
        # Test import
        import agentic_bte.servers.mcp.server
        print("✅ Server module imports successfully")
        
        # Test server creation
        from agentic_bte.servers.mcp.server import AgenticBTEMCPServer
        server = AgenticBTEMCPServer()
        print("✅ Server instance creates successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Server creation error: {e}")
        return False


def check_dependencies():
    """Check if key dependencies are available."""
    print("\n📦 Checking dependencies...")
    
    dependencies = [
        ('mcp', 'MCP framework'),
        ('langchain_openai', 'LangChain OpenAI'),
        ('spacy', 'SpaCy (optional)'),
        ('scispacy', 'SciSpacy (optional)'),
        ('requests', 'HTTP requests'),
        ('pydantic', 'Data validation')
    ]
    
    all_good = True
    for package, description in dependencies:
        try:
            __import__(package)
            print(f"✅ {description}")
        except ImportError:
            if package in ['spacy', 'scispacy']:
                print(f"⚠️  {description}: Not available (optional)")
            else:
                print(f"❌ {description}: Missing")
                all_good = False
    
    return all_good


def test_server_start():
    """Test that the server can start (briefly)."""
    print("\n🚀 Testing server startup...")
    
    try:
        # Start server with timeout
        process = subprocess.Popen(
            ['agentic-bte-mcp'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy()
        )
        
        # Wait a bit for startup
        try:
            stdout, stderr = process.communicate(timeout=5)
            print(f"❌ Server exited unexpectedly")
            if stderr:
                print(f"Error: {stderr}")
            return False
        except subprocess.TimeoutExpired:
            # Server is still running - good!
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
            
            print("✅ Server starts successfully")
            return True
            
    except FileNotFoundError:
        print("❌ agentic-bte-mcp command not found")
        return False
    except Exception as e:
        print(f"❌ Server startup error: {e}")
        return False


def main():
    """Run all validation checks."""
    print("🧪 Agentic BTE MCP Server Configuration Validation")
    print("=" * 50)
    
    checks = [
        check_console_script,
        check_environment,
        check_config_files,
        check_dependencies,
        check_server_import,
        test_server_start,
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"❌ Check failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Validation Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print(f"✅ All checks passed ({passed}/{total})!")
        print("\n🎉 Your Agentic BTE MCP server is ready to use!")
        print("\n📋 Available Tools (4):")
        print("  - bio_ner: Biomedical entity extraction")
        print("  - build_trapi_query: TRAPI query construction")
        print("  - call_bte_api: BTE API execution")
        print("  - plan_and_execute_query: Complete query processing with optimization")
        print("\nNext steps:")
        print("1. Restart Warp to load the new configuration")
        print("2. Test the MCP tools in Warp")
        print("3. Check WARP_SETUP.md for usage examples")
        return 0
    else:
        print(f"⚠️  Some checks failed ({passed}/{total})")
        print("\nPlease fix the issues above and run this script again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())