"""
Semantic Query Classification - LLM-based query type classification

This module provides robust semantic classification of biomedical queries using LLMs
instead of simple keyword matching, improving query decomposition accuracy.

Migrated and enhanced from the original BTE-LLM implementation.
"""

import logging
from typing import Dict, List, Any, Optional
from enum import Enum

from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing_extensions import TypedDict, Literal

from .types import QueryType
from ...config.settings import get_settings
from ...exceptions.base import ExternalServiceError

logger = logging.getLogger(__name__)


class QueryClassificationResult(TypedDict):
    """Result of query classification with confidence and reasoning"""
    query_type: Literal[
        "drug_mechanism", "disease_treatment", "gene_function", 
        "pathway_analysis", "drug_target", "disease_gene", 
        "phenotype_gene", "drug_discovery", "biomarker_discovery", "unknown"
    ]
    confidence: float
    reasoning: str
    entities_mentioned: List[str]


class SemanticQueryClassifier:
    """
    LLM-based semantic query classifier for biomedical queries
    
    Uses structured output and few-shot examples to accurately classify
    complex biomedical research questions into appropriate query types.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.settings = get_settings()
        self.openai_api_key = openai_api_key or self.settings.openai_api_key
        
        if not self.openai_api_key:
            raise ExternalServiceError("OpenAI API key is required for semantic classification")
        
        self.llm = ChatOpenAI(
            temperature=0.1,  # Low temperature for consistent classification
            model=self.settings.openai_model,
            api_key=self.openai_api_key
        )
        
        # Create structured output LLM
        self.classifier_llm = self.llm.with_structured_output(QueryClassificationResult)
    
    def classify_query(self, query: str, entities: Dict[str, str] = None) -> QueryType:
        """
        Classify a biomedical query into the most appropriate query type
        
        Args:
            query: The biomedical query to classify
            entities: Optional pre-extracted entities for context
            
        Returns:
            QueryType enum value for the classified query type
        """
        try:
            # Use semantic classification if enabled
            if self.settings.enable_semantic_classification:
                result = self._semantic_classification(query, entities)
                query_type = QueryType.from_string(result["query_type"])
                
                logger.info(f"Query classified as: {query_type.value} "
                           f"(confidence: {result['confidence']:.2f})")
                logger.debug(f"Classification reasoning: {result['reasoning']}")
                
                return query_type
            else:
                # Fall back to keyword-based classification
                return self._fallback_keyword_classification(query)
                
        except Exception as e:
            logger.error(f"Error in semantic classification: {e}")
            # Fallback to keyword-based classification
            return self._fallback_keyword_classification(query)
    
    def _semantic_classification(self, query: str, entities: Dict[str, str] = None) -> QueryClassificationResult:
        """
        Perform semantic classification using LLM
        
        Args:
            query: Query to classify
            entities: Optional entities for context
            
        Returns:
            Classification result dictionary
        """
        # Create classification prompt with examples
        classification_prompt = self._build_classification_prompt(query, entities)
        
        # Get structured classification result
        result = self.classifier_llm.invoke(classification_prompt)
        return result
    
    def _build_classification_prompt(self, query: str, entities: Dict[str, str] = None) -> str:
        """Build the classification prompt with examples and context"""
        
        entity_context = ""
        if entities:
            entity_context = f"\n\nExtracted entities: {entities}"
        
        prompt = f"""
You are a biomedical research query classifier. Your task is to classify research questions 
into the most appropriate query type for optimal decomposition and execution.

Query Types and Definitions:

1. **drug_mechanism**: Questions asking HOW a drug works, its mechanism of action, molecular pathways, or mode of action
   - Focus: Understanding the biological mechanism by which a drug produces its effects
   - Examples: "How does aspirin prevent heart attacks?", "What is the mechanism of action of metformin?"

2. **disease_treatment**: Questions asking WHAT treats a disease, therapeutic options, or treatment approaches
   - Focus: Finding treatments, therapies, or interventions for a specific condition
   - Examples: "What drugs treat diabetes?", "What are the treatment options for cancer?"

3. **drug_target**: Questions asking what molecular targets a drug interacts with
   - Focus: Direct drug-target interactions, binding sites, molecular targets
   - Examples: "What proteins does ibuprofen target?", "What receptors does morphine bind to?"

4. **gene_function**: Questions about gene roles, functions, or biological activities
   - Focus: Understanding what genes do, their biological roles
   - Examples: "What does the BRCA1 gene do?", "What is the function of TP53?"

5. **disease_gene**: Questions about genes associated with or causing diseases
   - Focus: Gene-disease associations, genetic causes
   - Examples: "What genes cause Alzheimer's disease?", "Which genes are associated with diabetes?"

6. **pathway_analysis**: Questions about biological pathways, processes, or networks
   - Focus: Understanding biological pathways and their components
   - Examples: "What genes are involved in apoptosis?", "What pathways regulate cell cycle?"

7. **phenotype_gene**: Questions linking observable traits to genetic factors
   - Focus: Phenotype-genotype relationships
   - Examples: "What genes affect height?", "Which genes influence eye color?"

8. **drug_discovery**: Questions about finding new drugs or drug candidates
   - Focus: Drug discovery, compound screening, drug development
   - Examples: "What compounds could be developed as new diabetes drugs?", "What are potential drug targets for cancer?"

9. **biomarker_discovery**: Questions about identifying biomarkers for diseases or conditions
   - Focus: Biomarker identification, diagnostic markers
   - Examples: "What biomarkers predict response to chemotherapy?", "Which proteins are biomarkers for heart disease?"

CRITICAL CLASSIFICATION RULES:
- Questions asking "HOW does [drug] treat/affect [disease]" are ALWAYS **drug_mechanism**
- Questions asking "WHAT [drugs] treat [disease]" are **disease_treatment**  
- Questions asking "WHICH gene plays [role] in HOW [drug] treats [disease]" are **drug_mechanism**
- Pay attention to question words: HOW = mechanism focus, WHAT/WHICH = entity identification focus
- Consider the PRIMARY intent of the question, not just keywords present

Examples:

Query: "How does metformin treat diabetes?"
Classification: drug_mechanism
Reasoning: Asks HOW a specific drug works - mechanism focus

Query: "What drugs treat diabetes?" 
Classification: disease_treatment
Reasoning: Asks WHAT treatments exist - treatment identification focus

Query: "Which gene plays the most significant role in how chlorphenamine treats allergic rhinitis?"
Classification: drug_mechanism  
Reasoning: Despite mentioning "treats", the core question is about HOW the drug works through genetic mechanisms

Query: "What genes are associated with breast cancer?"
Classification: disease_gene
Reasoning: Asks about gene-disease associations

Query: "What does the BRCA1 gene do?"
Classification: gene_function
Reasoning: Asks about gene function/role

Now classify this query:

Query: "{query}"{entity_context}

Provide your classification with confidence (0.0-1.0), reasoning, and any key entities mentioned.
        """
        
        return prompt
    
    def _fallback_keyword_classification(self, query: str) -> QueryType:
        """Fallback keyword-based classification if LLM fails"""
        query_lower = query.lower()
        
        # Improved keyword classification with better precedence
        if ("how" in query_lower and any(word in query_lower for word in ["treat", "work", "affect", "impact"])) or \
           any(word in query_lower for word in ["mechanism", "pathway"]):
            return QueryType.DRUG_MECHANISM
        elif any(word in query_lower for word in ["target", "targets", "interact", "bind"]):
            return QueryType.DRUG_TARGET
        elif any(word in query_lower for word in ["treat", "therapy", "therapeutic"]) and "how" not in query_lower:
            return QueryType.DISEASE_TREATMENT
        elif any(word in query_lower for word in ["discover", "development", "candidates"]) and "drug" in query_lower:
            return QueryType.DRUG_DISCOVERY
        elif any(word in query_lower for word in ["biomarker", "marker", "diagnostic"]):
            return QueryType.BIOMARKER_DISCOVERY
        elif any(word in query_lower for word in ["associated", "linked", "related"]):
            if "gene" in query_lower and ("disease" in query_lower or "cancer" in query_lower):
                return QueryType.DISEASE_GENE
            elif "gene" in query_lower:
                return QueryType.GENE_FUNCTION
            else:
                return QueryType.GENE_FUNCTION
        elif any(word in query_lower for word in ["pathway", "process", "network"]):
            return QueryType.PATHWAY_ANALYSIS
        elif "function" in query_lower or "role" in query_lower:
            return QueryType.GENE_FUNCTION
        else:
            return QueryType.UNKNOWN
    
    def get_classification_confidence(self, query: str, entities: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Get detailed classification results including confidence and reasoning
        
        Args:
            query: The biomedical query to classify
            entities: Optional pre-extracted entities
            
        Returns:
            Dictionary with classification details
        """
        try:
            if self.settings.enable_semantic_classification:
                result = self._semantic_classification(query, entities)
                
                return {
                    "query_type": QueryType.from_string(result["query_type"]),
                    "confidence": result["confidence"],
                    "reasoning": result["reasoning"],
                    "entities_mentioned": result["entities_mentioned"],
                    "method": "llm_semantic"
                }
            else:
                # Use keyword classification
                query_type = self._fallback_keyword_classification(query)
                return {
                    "query_type": query_type,
                    "confidence": 0.7,
                    "reasoning": "Keyword-based classification (semantic classification disabled)",
                    "entities_mentioned": [],
                    "method": "keyword_direct"
                }
                
        except Exception as e:
            logger.error(f"Error getting detailed classification: {e}")
            fallback_type = self._fallback_keyword_classification(query)
            return {
                "query_type": fallback_type,
                "confidence": 0.5,
                "reasoning": "Fallback keyword-based classification due to LLM error",
                "entities_mentioned": [],
                "method": "keyword_fallback"
            }


# Convenience functions
def classify_query(query: str, entities: Dict[str, str] = None, openai_api_key: Optional[str] = None) -> QueryType:
    """
    Classify a query using semantic understanding
    
    Args:
        query: The biomedical query to classify
        entities: Optional pre-extracted entities
        openai_api_key: Optional OpenAI API key
        
    Returns:
        QueryType enum value
    """
    classifier = SemanticQueryClassifier(openai_api_key)
    return classifier.classify_query(query, entities)


def get_detailed_classification(query: str, entities: Dict[str, str] = None, 
                              openai_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Get detailed classification results
    
    Args:
        query: The biomedical query to classify
        entities: Optional pre-extracted entities
        openai_api_key: Optional OpenAI API key
        
    Returns:
        Dictionary with detailed classification information
    """
    classifier = SemanticQueryClassifier(openai_api_key)
    return classifier.get_classification_confidence(query, entities)