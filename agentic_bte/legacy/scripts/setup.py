#!/usr/bin/env python3
"""
Setup Script for Agentic BTE

This script handles the installation of required spaCy biomedical models
and other setup tasks.
"""

import subprocess
import sys
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(cmd: List[str], description: str) -> bool:
    """
    Run a command and return success status
    
    Args:
        cmd: Command to run as list
        description: Description for logging
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Running: {description}")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True
        )
        
        if result.stdout:
            logger.debug(f"Output: {result.stdout.strip()}")
        
        logger.info(f"✓ {description} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed")
        logger.error(f"Error: {e.stderr.strip() if e.stderr else str(e)}")
        return False
    except Exception as e:
        logger.error(f"✗ {description} failed: {e}")
        return False


def install_spacy_models() -> bool:
    """
    Install required spaCy biomedical models
    
    Returns:
        True if all models installed successfully
    """
    logger.info("Installing spaCy biomedical models...")
    
    models = [
        {
            "name": "en_core_sci_lg",
            "description": "Large scientific spaCy model",
            "cmd": [sys.executable, "-m", "spacy", "download", "en_core_sci_lg"]
        },
        {
            "name": "en_ner_bc5cdr_md",
            "description": "Drug/disease NER spaCy model",
            "cmd": [
                sys.executable, "-m", "pip", "install", 
                "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz"
            ]
        }
    ]
    
    success_count = 0
    for model in models:
        if run_command(model["cmd"], f"Installing {model['name']} ({model['description']})"):
            success_count += 1
    
    if success_count == len(models):
        logger.info("✓ All spaCy models installed successfully")
        return True
    else:
        logger.error(f"✗ Only {success_count}/{len(models)} models installed successfully")
        return False


def verify_installation() -> bool:
    """
    Verify that the installation is working correctly
    
    Returns:
        True if verification successful
    """
    logger.info("Verifying installation...")
    
    try:
        # Test basic imports
        logger.info("Testing basic imports...")
        from agentic_bte.config.settings import get_settings
        from agentic_bte.core.entities.bio_ner import BioNERTool
        
        # Test settings
        settings = get_settings()
        logger.info(f"Settings loaded: OpenAI model = {settings.openai_model}")
        
        # Test spaCy model availability
        bio_ner = BioNERTool()
        models = bio_ner.get_available_models()
        logger.info(f"Available models: {models}")
        
        if models["large_scientific_model"] and models["drug_disease_model"]:
            logger.info("✓ All required spaCy models are available")
        else:
            logger.warning("⚠ Some spaCy models are not available - will use fallback extraction")
        
        logger.info("✓ Installation verification completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Installation verification failed: {e}")
        return False


def main():
    """
    Main setup function
    """
    print("🧬 Agentic BTE Setup")
    print("=" * 50)
    
    success = True
    
    # Install spaCy models
    if not install_spacy_models():
        success = False
    
    print("\n" + "="*50)
    
    # Verify installation
    if not verify_installation():
        success = False
    
    print("\n" + "="*50)
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and configure your OpenAI API key")
        print("2. Run the MCP server: agentic-bte-mcp")
        print("3. Or use the Python API directly in your code")
    else:
        print("\n❌ Setup completed with errors")
        print("\nSome components may not work correctly.")
        print("Check the error messages above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()