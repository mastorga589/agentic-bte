#!/usr/bin/env python3
"""
Clean Jupyter notebooks by removing personal paths and making them portable.
"""
import json
import os
import re
from pathlib import Path

def clean_notebook(notebook_path):
    """Clean a single notebook file."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    modified = False
    
    # Process each cell
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                new_source = []
                for line in source:
                    original_line = line
                    
                    # Replace hardcoded .env path with relative path
                    line = re.sub(
                        r'load_dotenv\("/Users/mastorga/Documents/BTE-LLM/\.env"\)',
                        'load_dotenv("../.env")',
                        line
                    )
                    
                    # Replace hardcoded data paths with relative paths
                    line = re.sub(
                        r'/Users/mastorga/Documents/BTE-LLM/Prototype/data/',
                        './data/',
                        line
                    )
                    
                    # Replace hardcoded output/log paths with relative paths
                    line = re.sub(
                        r'/Users/mastorga/Documents/BTE-LLM/Prototype/logs/',
                        './logs/',
                        line
                    )
                    
                    # Replace any remaining /Users/mastorga paths with relative
                    line = re.sub(
                        r'/Users/mastorga/Documents/[^"\']+/Prototype/',
                        './',
                        line
                    )
                    
                    if line != original_line:
                        modified = True
                    
                    new_source.append(line)
                
                cell['source'] = new_source
    
    # Clear outputs to reduce file size and remove any potential sensitive output
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    
    if modified:
        # Save the cleaned notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        return True
    return False

def main():
    """Clean all notebooks in the Prototype directory."""
    prototype_dir = Path(__file__).parent / "Prototype"
    
    if not prototype_dir.exists():
        print(f"Error: {prototype_dir} does not exist")
        return
    
    # Find all notebooks
    notebooks = list(prototype_dir.glob("*.ipynb"))
    
    print(f"Found {len(notebooks)} notebooks to clean")
    
    cleaned_count = 0
    for notebook_path in notebooks:
        print(f"Cleaning: {notebook_path.name}")
        if clean_notebook(notebook_path):
            cleaned_count += 1
    
    print(f"\nCleaned {cleaned_count} notebooks")
    print("✓ Removed hardcoded personal paths")
    print("✓ Cleared all outputs")
    print("✓ Reset execution counts")

if __name__ == "__main__":
    main()
