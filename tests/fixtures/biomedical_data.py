"""
Biomedical Test Data Fixtures

Sample data for testing biomedical entity recognition, TRAPI queries,
BTE API responses, and multi-agent workflows.
"""

# Sample biomedical queries for testing
SAMPLE_QUERIES = {
    "diabetes_treatment": "What drugs can treat type 2 diabetes?",
    "gene_disease": "What genes are associated with Alzheimer's disease?", 
    "drug_mechanism": "How does aspirin work as a drug?",
    "complex_query": "Which drugs can treat Crohn's disease by targeting the inflammatory response?",
    "simple_entity": "diabetes",
    "no_entities": "Hello world, how are you today?",
}

# Sample entity extraction results
SAMPLE_ENTITIES = {
    "diabetes_entities": {
        "entities": {
            "diabetes": {
                "id": "MONDO:0005015",
                "type": "disease",
                "name": "diabetes mellitus"
            },
            "drugs": {
                "id": "biolink:ChemicalEntity", 
                "type": "general",
                "name": "drugs"
            }
        },
        "entity_ids": {
            "diabetes": "MONDO:0005015",
            "drugs": "biolink:ChemicalEntity"
        }
    },
    "alzheimer_entities": {
        "entities": {
            "Alzheimer's disease": {
                "id": "MONDO:0004975",
                "type": "disease",
                "name": "Alzheimer disease"
            },
            "genes": {
                "id": "biolink:Gene",
                "type": "general", 
                "name": "genes"
            }
        },
        "entity_ids": {
            "Alzheimer's disease": "MONDO:0004975",
            "genes": "biolink:Gene"
        }
    }
}

# Sample TRAPI queries
SAMPLE_TRAPI_QUERIES = {
    "diabetes_treatment": {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {
                        "categories": ["biolink:Disease"],
                        "ids": ["MONDO:0005015"]
                    },
                    "n1": {
                        "categories": ["biolink:SmallMolecule"]
                    }
                },
                "edges": {
                    "e01": {
                        "subject": "n0",
                        "object": "n1", 
                        "predicates": ["biolink:treated_by"]
                    }
                }
            }
        }
    },
    "gene_disease": {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {
                        "categories": ["biolink:Disease"],
                        "ids": ["MONDO:0004975"] 
                    },
                    "n1": {
                        "categories": ["biolink:Gene"]
                    }
                },
                "edges": {
                    "e01": {
                        "subject": "n0",
                        "object": "n1",
                        "predicates": ["biolink:condition_associated_with_gene"]
                    }
                }
            }
        }
    }
}

# Sample BTE API responses
SAMPLE_BTE_RESPONSES = {
    "diabetes_treatment": {
        "message": {
            "results": [
                {
                    "node_bindings": {
                        "n0": [{"id": "MONDO:0005015"}],
                        "n1": [{"id": "CHEBI:6801"}]
                    },
                    "edge_bindings": {
                        "e01": [{"id": "edge_1"}]
                    }
                },
                {
                    "node_bindings": {
                        "n0": [{"id": "MONDO:0005015"}],
                        "n1": [{"id": "CHEBI:71193"}]
                    },
                    "edge_bindings": {
                        "e01": [{"id": "edge_2"}]
                    }
                }
            ],
            "knowledge_graph": {
                "nodes": {
                    "MONDO:0005015": {
                        "categories": ["biolink:Disease"],
                        "name": "diabetes mellitus"
                    },
                    "CHEBI:6801": {
                        "categories": ["biolink:SmallMolecule"],
                        "name": "Metformin"
                    },
                    "CHEBI:71193": {
                        "categories": ["biolink:SmallMolecule"],
                        "name": "Liraglutide"
                    }
                },
                "edges": {
                    "edge_1": {
                        "subject": "MONDO:0005015",
                        "predicate": "biolink:tested_by_clinical_trials_of",
                        "object": "CHEBI:6801"
                    },
                    "edge_2": {
                        "subject": "MONDO:0005015", 
                        "predicate": "biolink:tested_by_clinical_trials_of",
                        "object": "CHEBI:71193"
                    }
                }
            }
        }
    }
}

# Sample processed BTE results
SAMPLE_BTE_RESULTS = {
    "diabetes_drugs": [
        {
            "subject": "MONDO:0005015",
            "subject_type": "biolink:Disease",
            "predicate": "biolink:tested_by_clinical_trials_of",
            "object": "CHEBI:6801",
            "object_type": "biolink:SmallMolecule"
        },
        {
            "subject": "MONDO:0005015",
            "subject_type": "biolink:Disease", 
            "predicate": "biolink:tested_by_clinical_trials_of",
            "object": "CHEBI:71193",
            "object_type": "biolink:SmallMolecule"
        }
    ]
}

# Sample entity mappings (ID to name)
SAMPLE_ENTITY_MAPPINGS = {
    "diabetes_mapping": {
        "diabetes mellitus": "MONDO:0005015",
        "Metformin": "CHEBI:6801",
        "Liraglutide": "CHEBI:71193",
        "type 2 diabetes mellitus": "MONDO:0005148"
    },
    "alzheimer_mapping": {
        "Alzheimer disease": "MONDO:0004975",
        "APOE": "NCBIGene:348",
        "APP": "NCBIGene:351",
        "PSEN1": "NCBIGene:5663"
    }
}

# Sample RDF triples for testing
SAMPLE_RDF_TRIPLES = [
    {
        "subject": "diabetes mellitus",
        "subject_type": "biolink:Disease",
        "predicate": "biolink:tested_by_clinical_trials_of",
        "object": "Metformin",
        "object_type": "biolink:SmallMolecule"
    },
    {
        "subject": "diabetes mellitus",
        "subject_type": "biolink:Disease",
        "predicate": "biolink:condition_associated_with_gene", 
        "object": "PDX1",
        "object_type": "biolink:Gene"
    }
]

# Sample agent state for LangGraph testing
SAMPLE_AGENT_STATE = {
    "messages": [("human", "What drugs treat diabetes?")],
    "query": "What drugs treat diabetes?",
    "subQuery": [],
    "entity_data": [],
    "maxresults": 50,
    "k": 5,
    "final_answer": "",
    "next": "orchestrator",
    "execution_metadata": {
        "start_time": None,
        "subquery_results": [],
        "total_api_calls": 0,
        "total_results": 0
    },
    "failed_trapis": [],
    "confidence_threshold": 0.7
}

# Error responses for testing error handling
SAMPLE_ERROR_RESPONSES = {
    "api_error": {
        "error": "API request failed",
        "status_code": 500,
        "message": "Internal server error"
    },
    "validation_error": {
        "error": "Invalid input",
        "details": "Query parameter is required"
    },
    "timeout_error": {
        "error": "Request timeout", 
        "timeout_seconds": 30.0
    }
}