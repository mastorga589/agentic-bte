"""
DMDB Dataset Utilities for Benchmark Testing

Utilities for loading and sampling from the DMDB (Drug-Disease-Biological Process)
dataset used in the 50-question benchmark experiments.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import os

logger = logging.getLogger(__name__)

# Path to the DMDB dataset - configurable via environment variable
# Default: look in project data directory
DMDB_DATASET_PATH = os.getenv(
    'AGENTIC_BTE_DMDB_DATASET_PATH',
    './data/DMDB_go_bp_filtered_jjoy_05_08_2025.csv'
)


def load_dmdb_dataset() -> pd.DataFrame:
    """
    Load the DMDB dataset containing drug-disease-biological process triplets.
    
    Returns:
        DataFrame with columns: drug, drug_name, disease, disease_name, bp, bp_name
    """
    dataset_path = Path(DMDB_DATASET_PATH)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"DMDB dataset not found at {DMDB_DATASET_PATH}")
    
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded DMDB dataset with {len(df)} rows")
    
    return df


def sample_dmdb_questions(
    n_samples: int = 50,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Sample drug-disease-biological process questions from DMDB dataset.
    
    Args:
        n_samples: Number of samples to draw
        random_seed: Random seed for reproducible sampling
    
    Returns:
        DataFrame with sampled rows and generated questions
    """
    df = load_dmdb_dataset()
    
    # Sample deterministically
    sampled = df.sample(n=n_samples, random_state=random_seed)
    
    # Generate questions without IDs
    sampled['question_without_id'] = (
        "Which drugs can treat " + 
        sampled['disease_name'] + 
        " by targeting " + 
        sampled['bp_name'] + 
        "? Return a comprehensive list of all drugs that target the correct biological process"
    )
    
    # Generate questions with IDs
    sampled['question_with_id'] = (
        "Which drugs can treat " + 
        sampled['disease_name'] + 
        " (ID: " + sampled['disease'] + 
        ") by targeting " + 
        sampled['bp_name'] + 
        " (ID: " + sampled['bp'] + 
        ")? Return a comprehensive list of all drugs that target the correct biological process"
    )
    
    logger.info(f"Generated {n_samples} drug-disease-BP questions")
    
    return sampled.reset_index(drop=True)


def get_ground_truth_drugs(row: pd.Series) -> List[str]:
    """
    Extract ground truth drug(s) for a given DMDB row.
    
    Args:
        row: A row from the DMDB dataset
    
    Returns:
        List of drug names/IDs that are correct answers
    """
    # The DMDB dataset represents known drug-disease-BP associations,
    # so the drug in the row is a correct answer
    # DMDB columns: id, Drug_MeshID, disease, bp, drug_name, disease_name, bp_name
    return [row['drug_name'], row['Drug_MeshID']]


def calculate_retrieval_metrics(
    predicted_drugs: List[str],
    ground_truth_drugs: List[str]
) -> Dict[str, float]:
    """
    Calculate retrieval metrics for drug predictions.
    
    Args:
        predicted_drugs: List of predicted drug names/IDs
        ground_truth_drugs: List of ground truth drug names/IDs
    
    Returns:
        Dictionary with precision, recall, and F1 metrics
    """
    if not predicted_drugs:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "found_ground_truth": False
        }
    
    # Normalize for case-insensitive matching
    predicted_set = {str(drug).lower().strip() for drug in predicted_drugs}
    ground_truth_set = {str(drug).lower().strip() for drug in ground_truth_drugs}
    
    # Calculate intersection
    true_positives = len(predicted_set.intersection(ground_truth_set))
    
    # Precision: What fraction of predictions are correct?
    precision = true_positives / len(predicted_set) if predicted_set else 0.0
    
    # Recall: What fraction of ground truth was found?
    recall = true_positives / len(ground_truth_set) if ground_truth_set else 0.0
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "found_ground_truth": true_positives > 0
    }


def parse_drug_list_from_response(response_text: str) -> List[str]:
    """
    Parse drug names from an LLM response.
    
    Uses multiple strategies to extract drug names:
    1. Numbered/bulleted list items
    2. Drug names with IDs in parentheses
    3. Capitalized words that look like drug names
    4. Common drug name patterns
    
    Args:
        response_text: The response text from LLM or system
    
    Returns:
        List of extracted drug names (deduplicated)
    """
    import re
    
    drugs = []
    
    # Strategy 1: Extract from numbered/bulleted lists
    # Matches: "1. Metformin", "- Aspirin", "* Ibuprofen (CHEBI:123)"
    list_pattern = r'^[\s]*(?:[\d]+[\.):]|[-*•])[\s]+([A-Z][a-zA-Z0-9\-]+(?:[\s][a-zA-Z0-9\-]+){0,2})'
    lines = response_text.split('\n')
    
    for line in lines:
        # Extract from list items
        match = re.match(list_pattern, line)
        if match:
            drug_name = match.group(1).strip()
            # Remove trailing punctuation and parentheticals
            drug_name = re.sub(r'[\s]*\([^)]+\)[\s]*', '', drug_name)  # Remove (ID: xxx)
            drug_name = re.sub(r'[,;:\.]$', '', drug_name)  # Remove trailing punctuation
            if drug_name and 2 < len(drug_name) < 50:
                drugs.append(drug_name)
    
    # Strategy 2: Extract drug names with IDs
    # Matches: "Metformin (CHEBI:6801)", "Aspirin (MESH:D001241)"
    id_pattern = r'\b([A-Z][a-zA-Z0-9\-]+(?:[\s][a-zA-Z0-9\-]+){0,2})[\s]*\([A-Z]+:[A-Z0-9]+\)'
    for match in re.finditer(id_pattern, response_text):
        drug_name = match.group(1).strip()
        if drug_name and 2 < len(drug_name) < 50:
            drugs.append(drug_name)
    
    # Strategy 3: Extract capitalized drug-like names
    # More conservative - only if we haven't found many drugs yet
    if len(drugs) < 5:
        # Look for capitalized words that might be drug names
        # Avoid common English words
        common_words = {'The', 'This', 'That', 'These', 'Those', 'There', 'Here', 
                       'What', 'Which', 'When', 'Where', 'Why', 'How',
                       'Based', 'However', 'Therefore', 'Additionally', 'Furthermore'}
        
        cap_pattern = r'\b([A-Z][a-z]{2,}(?:in|ol|ide|one|ate|ine|ane)\b)'
        for match in re.finditer(cap_pattern, response_text):
            drug_name = match.group(1)
            if drug_name not in common_words and len(drug_name) > 4:
                drugs.append(drug_name)
    
    # Deduplicate while preserving order and normalize
    seen = set()
    unique_drugs = []
    for drug in drugs:
        drug_lower = drug.lower().strip()
        if drug_lower not in seen and drug_lower:
            seen.add(drug_lower)
            unique_drugs.append(drug)
    
    logger.debug(f"Extracted {len(unique_drugs)} unique drugs from response")
    
    return unique_drugs[:30]  # Return up to 30 drugs
