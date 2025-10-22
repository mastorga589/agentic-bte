"""
Direct port of original LangGraph BioNER Tool

This is a direct port of /Users/mastorga/Documents/BTE-LLM/Prototype/tools/BioNER.py
to maintain exact compatibility with the original implementation.
"""
from langchain_openai import ChatOpenAI
from scispacy.linking import EntityLinker
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from typing import Annotated, Literal, Union
from typing_extensions import TypedDict
from os.path import dirname, join
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import spacy
import requests
import time
import json
import pandas as pd
import getpass
import os
import logging
import traceback
from datetime import datetime

# Setup comprehensive logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create formatter for detailed logging
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)

# Add handler if not already present
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Global models exactly as in original - with instrumentation
logger.info("Initializing spaCy models and linkers...")
try:
    logger.debug("Loading en_core_sci_lg model...")
    nlp = spacy.load("en_core_sci_lg")
    logger.info(f"Successfully loaded en_core_sci_lg model with {len(nlp.pipe_names)} pipes: {nlp.pipe_names}")
    
    logger.debug("Loading en_ner_bc5cdr_md model...")
    drug_disease_nlp = spacy.load("en_ner_bc5cdr_md")
    logger.info(f"Successfully loaded en_ner_bc5cdr_md model with {len(drug_disease_nlp.pipe_names)} pipes: {drug_disease_nlp.pipe_names}")
    
    logger.debug("Adding scispacy_linker to main nlp pipeline...")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    logger.info(f"Successfully added scispacy_linker. Updated pipeline: {nlp.pipe_names}")
    
except Exception as e:
    logger.error(f"Failed to initialize spaCy models: {e}")
    logger.error(f"Full traceback: {traceback.format_exc()}")
    raise

# Direct port of original modifiedBioNERTool function
def modifiedBioNERTool(query: str, openai_api_key: str = None):
    """Extract biological entities from a query and returns them along with their ID
    
    Direct port of the original LangGraph implementation.
    """
    start_time = datetime.now()
    logger.info(f"=== Starting modifiedBioNERTool with query: '{query}' ===")
    logger.debug(f"Query length: {len(query)} characters")
    logger.debug(f"OpenAI API key provided: {bool(openai_api_key)}")
    
    try:
        # Step 1: Biomedical entity extraction - LLM setup
        logger.debug("Setting up LLM for entity extraction...")
        if openai_api_key:
            logger.debug("Using provided OpenAI API key")
            llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=openai_api_key)
        else:
            try:
                logger.debug("Attempting to load OpenAI API key from settings...")
                from ...config.settings import get_settings
                settings = get_settings()
                llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=settings.openai_api_key)
                logger.debug("Successfully loaded OpenAI API key from settings")
            except Exception as settings_error:
                logger.warning(f"Failed to load settings: {settings_error}")
                logger.debug("Using default OpenAI API key from environment")
                llm = ChatOpenAI(temperature=0, model="gpt-4o")
        
        logger.info("LLM initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise
    
    global nlp, drug_disease_nlp

    def extractEnts(query: str):
        logger.info(f"=== Starting extractEnts for query: '{query}' ===")
        entities = []
        
        try:
            # Process with both spaCy models
            logger.debug("Processing query with spaCy models...")
            docs = [
                nlp(query),
                drug_disease_nlp(query)
            ]
            logger.debug(f"Created {len(docs)} spaCy docs for processing")

            # Extract entity texts from each doc
            for i, doc in enumerate(docs):
                model_name = "en_core_sci_lg" if i == 0 else "en_ner_bc5cdr_md"
                logger.debug(f"Processing entities from {model_name}...")
                doc_entities = []
                for ent in doc.ents:
                    entity_text = ent.text.strip()
                    doc_entities.append(entity_text)
                    logger.debug(f"Found entity: '{entity_text}' (label: {ent.label_}) from {model_name}")
                
                entities.extend(doc_entities)
                logger.info(f"Extracted {len(doc_entities)} entities from {model_name}")

            logger.info(f"Total spaCy entities extracted: {len(entities)}")
            logger.debug(f"spaCy entities: {entities}")

            # Extract biological process terms using LLM
            logger.debug("Extracting biological processes using LLM...")
            bp_prompt = f"""You are a helpful assistant that can extract biological processes from a given query. 
                        These might include concepts such as "cholesterol biosynthesis", "Aminergic neurotransmitter loading into synaptic vesicle", etc. Each entity should be a noun.

                        You must always return the full phrase/long form of each biomedical entity 
                        
                        Return results as a list. Return "" if no biological processes are in the query. DO NOT INCLUDE YOUR THOUGHTS
                        Here is your query: {query}"""

            logger.debug("Sending biological process extraction prompt to LLM...")
            bp_response = llm.invoke(bp_prompt)
            bp_list = bp_response.content.strip()
            logger.debug(f"LLM biological process response: '{bp_list}'")
            
            if bp_list and bp_list != "":
                entities.append(bp_list)
                logger.info(f"Added biological process list: '{bp_list}'")
            else:
                logger.info("No biological processes found by LLM")
        
            # Deduplicate while preserving order
            original_count = len(entities)
            entities = list(dict.fromkeys(entities))
            logger.info(f"Deduplicated entities: {original_count} -> {len(entities)}")
            logger.info(f"Final extracted entities: {entities}")

            return entities
            
        except Exception as e:
            logger.error(f"Error in extractEnts: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
        
    # Step 2: Linking each entity into a semantic type
    def classifyEnt(query: str, ent: str):
        logger.debug(f"=== Classifying entity: '{ent}' ===")
        logger.debug(f"Query context: '{query}'")
        
        try:
            supportedEntTypes = ["biologicalProcess", "general"]
            logger.debug(f"Supported entity types: {supportedEntTypes}")
            
            class entType(TypedDict):
                """The most appropriate entity type given a specific entity"""
                entType: Literal[*supportedEntTypes] # type: ignore

            classify_prompt = f"""
            Classify the following biomedical entity into one of: 
            {supportedEntTypes}

            Biomedical entity: {ent}

            For context, here is the query that the entity was extracted from: {query}
            """
            
            logger.debug("Sending classification prompt to LLM...")
            logger.debug(f"Classification prompt: {classify_prompt}")
            
            chosen_type_response = llm.with_structured_output(entType).invoke(classify_prompt)
            logger.debug(f"LLM classification response: {chosen_type_response}")

            chosen_type = str(chosen_type_response["entType"])
            
            logger.info(f"Entity '{ent}' classified as: {chosen_type}")
            
            return chosen_type
            
        except Exception as e:
            logger.error(f"Error classifying entity '{ent}': {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Default to 'general' if classification fails
            logger.warning(f"Defaulting entity '{ent}' to 'general' type")
            return "general"


    # Step 3: Entity-linking using different linkers based on entity type mapping
    def linkEnt(entList: list, query: str):
        logger.info(f"=== Starting linkEnt for {len(entList)} entities ===")
        logger.debug(f"Entity list: {entList}")
        logger.debug(f"Query context: '{query}'")
        
        idList = {}

        def sriNameResolver(ent: str, base_url = "https://name-lookup.ci.transltr.io/lookup", is_bp: bool = False, k = 50) -> list:    
            logger.debug(f"=== Starting sriNameResolver for entity: '{ent}' ===")
            logger.debug(f"Parameters - is_bp: {is_bp}, k: {k}, base_url: {base_url}")
            
            try:
                candidates = []
                autoComplete = True
                
                processed_ent = ent.replace(" ", "%20")
                logger.debug(f"URL-encoded entity: '{processed_ent}'")

                final_url = base_url + "?string=" + processed_ent + "&autocomplete=" + str(autoComplete).lower() + "&limit=" + str(k)

                if is_bp:
                    final_url = final_url + "&only_prefixes=GO&biolink_type=BiologicalProcess"
                    
                logger.info(f"Making SRI Name Resolver request to: {final_url}")
                
                response = requests.get(final_url, headers={"accept": "application/json"})
                logger.debug(f"SRI response status: {response.status_code}")
                logger.debug(f"SRI response headers: {dict(response.headers)}")
            
                if response.status_code != 200:
                    logger.error(f"SRI Name Resolver request failed with status {response.status_code}")
                    logger.error(f"Response content: {response.content}")
                    return []
                
                candidate_list = json.loads(response.content.decode("utf-8"))
                logger.debug(f"SRI returned {len(candidate_list)} candidates")
                logger.debug(f"Raw candidate list: {candidate_list[:3]}...")  # Show first 3 for debugging
            
                for i, item in enumerate(candidate_list):
                    parsed = {
                        "label": item.get("label", ""),
                        "curie": item.get("curie", ""),
                        "score": item.get("score", "")
                    }
                    candidates.append(parsed)
                    if i < 3:  # Log first few for debugging
                        logger.debug(f"Parsed candidate {i}: {parsed}")
            
                logger.info(f"Successfully processed {len(candidates)} candidates from SRI Name Resolver")
                return candidates
                
            except Exception as e:
                logger.error(f"Error in sriNameResolver for entity '{ent}': {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                return []

        def remove_TUI(text):
            parts = text.split("TUI", 1)
            return parts[0]
        
        def selectID(doc: object):
            logger.debug(f"=== Starting selectID for doc ===")
            
            try:
                bioID = ""
                candidates = {}

                linker = nlp.get_pipe("scispacy_linker")   
                logger.debug(f"Retrieved scispacy_linker from pipeline")
                
                for ent in doc.ents:
                    logger.debug(f"Processing entity: '{ent.text}' (label: {ent.label_})")
                    
                    if ent._.kb_ents:  # Check if entity has linked knowledge base IDs
                        logger.debug(f"Entity has {len(ent._.kb_ents)} linked knowledge base IDs")
                        
                        for i, id in enumerate(ent._.kb_ents):
                            try:
                                cui = id[0]
                                definition = remove_TUI(str(linker.kb.cui_to_entity[cui]))
                                candidates[cui] = definition
                                logger.debug(f"Candidate {i}: CUI={cui}, definition={definition[:100]}...")
                            except Exception as cui_error:
                                logger.warning(f"Error processing CUI {cui}: {cui_error}")
                    else:
                        logger.debug(f"Entity '{ent.text}' has no linked knowledge base IDs")
            
                    if not candidates:
                        logger.info(f"No candidates found for entity '{ent.text}' in UMLS linker")
                        return ""
                    
                    logger.debug(f"Found {len(candidates)} candidates for entity '{ent.text}'")
                    
                    select_prompt = f"""You are a smart biomedical assistant that can understand the context and the intent behind a query. 
                                Be careful when choosing IDs for entities that can refer to different concepts (for example, HIV can refer either to the virus or the disease; you MUST choose the most appropriate concept/definition based on the query). 
                                Use the context and the intent behind the query to choose the most appropriate ID. 
                                Here is the complete query: {query}
                                Select the one most appropriate ID/CUI for {ent.text} from the list below:
                                {candidates}
                                If none of the choices are appropriate, return "".
                                Otherwise, return only the ID/CUI.
                                """
        
                    logger.debug("Sending UMLS candidate selection prompt to LLM...")
                    logger.debug(f"Selection prompt for '{ent.text}': {select_prompt[:200]}...")
                    
                    # LLM selects most appropriate ID from list
                    selectedID = llm.invoke(select_prompt).content.strip()
                    logger.debug(f"LLM selected ID response: '{selectedID}'")
        
                    # Extract just the UMLS CUI using regex
                    match = re.search(r"C\d{7}", selectedID)
                    if match:
                        bioID = "UMLS:" + match.group(0)
                        definition = candidates.get(match.group(0), "")
                        logger.info(f"Successfully extracted UMLS ID: {bioID}")
                        logger.debug(f"Definition: {definition[:100]}...")
                    else:
                        bioID = ""
                        definition = ""
                        logger.info(f"No valid UMLS CUI found in LLM response: '{selectedID}'")
        
                    # Printing chosen ID + definition, if any
                    logger.info(f"Final selectID result - Entity: '{ent.text}', ID: '{bioID}'")
                    if definition:
                        logger.debug(f"Definition: {definition}")

                return bioID
                
            except Exception as e:
                logger.error(f"Error in selectID: {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                return ""

        def selectIDbp(entity: str, candidates: list):
            logger.debug(f"=== Starting selectIDbp for entity: '{entity}' ===")
            logger.debug(f"Received {len(candidates)} candidates")
            
            try:
                if not candidates:
                    logger.warning(f"No candidates provided for entity '{entity}'")
                    return ""
                    
                choices = [""]

                for i, entry in enumerate(candidates):
                    curie = entry.get("curie")
                    choices.append(curie)
                    if i < 5:  # Log first few for debugging
                        logger.debug(f"Candidate {i}: {entry}")
                
                logger.debug(f"Created {len(choices)} choices for structured output")

                class selectedID(TypedDict):
                    """The most appropriate CUI/ID from the given candidates"""
                    selectedID: Literal[*choices] # type: ignore
                
                select_prompt = f"""You are a smart biomedical assistant that can understand the context and the intent behind a query. 
                            Be careful when choosing IDs for entities that can refer to different concepts (for example, HIV can refer either to the virus or the disease; you MUST choose the most appropriate concept/definition based on the query). 
                            Use the context and the intent behind the query to choose the most appropriate ID. 
                            Here is the complete query: {query}
                            Select the one most appropriate ID/CUI for {entity} from the list below:
                            {candidates}
                            If none of the choices are appropriate, return "".
                            Otherwise, return only the ID/CUI.
                            """
        
                logger.debug("Sending SRI candidate selection prompt to LLM...")
                logger.debug(f"Selection prompt for '{entity}': {select_prompt[:200]}...")
                
                # LLM selects most appropriate ID from list
                selectedID_response = llm.with_structured_output(selectedID).invoke(select_prompt)
                logger.debug(f"LLM structured response: {selectedID_response}")

                bioID = str(selectedID_response["selectedID"])
                
                if bioID and bioID != "":
                    logger.info(f"Successfully selected SRI ID: '{bioID}' for entity '{entity}'")
                else:
                    logger.info(f"No suitable SRI ID selected for entity '{entity}'")

                # Printing chosen ID + definition, if any
                logger.info(f"Final selectIDbp result - Entity: '{entity}', ID: '{bioID}'")

                return bioID
                
            except Exception as e:
                logger.error(f"Error in selectIDbp for entity '{entity}': {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                return ""

        # General Pipeline            
        try:
            for i, entity in enumerate(entList):
                logger.info(f"=== Processing entity {i+1}/{len(entList)}: '{entity}' ===")
                chosenID = ""
                
                try:
                    # Step 1: Classify entity type
                    logger.debug(f"Classifying entity: '{entity}'")
                    entclass = classifyEnt(query, entity)
                    logger.info(f"Entity '{entity}' classified as: {entclass}")
                    
                    # Step 2: Link entity based on its type
                    if entclass == "biologicalProcess":
                        logger.debug(f"Processing biological process: '{entity}'")
                        sri_candidates = sriNameResolver(entity, is_bp=True)
                        logger.debug(f"Got {len(sri_candidates)} SRI candidates for biological process")
                        chosenID = selectIDbp(entity, sri_candidates)
                        logger.info(f"Biological process linking result: '{chosenID}'")
                    else:
                        logger.debug(f"Processing general entity: '{entity}'")
                        # First try UMLS linker
                        doc = nlp(entity)
                        logger.debug(f"Created spaCy doc with {len(doc.ents)} entities")
                        chosenID = selectID(doc)
                        logger.info(f"UMLS linking result: '{chosenID}'")
                        
                        if chosenID == "":
                            logger.debug(f"UMLS linking failed, trying SRI Name Resolver")
                            sri_candidates = sriNameResolver(entity, is_bp=False)
                            logger.debug(f"Got {len(sri_candidates)} SRI candidates for general entity")
                            chosenID = selectIDbp(entity, sri_candidates)
                            logger.info(f"SRI linking fallback result: '{chosenID}'")
                            
                            if chosenID == "":
                                logger.warning(f"No ID found for entity '{entity}' - skipping")
                                continue
                    
                    # Step 3: Store result
                    if chosenID and chosenID != "":
                        idList[entity] = chosenID
                        logger.info(f"Successfully linked entity: '{entity}' -> '{chosenID}'")
                    else:
                        logger.warning(f"Empty ID returned for entity '{entity}' - skipping")
                        
                except Exception as entity_error:
                    logger.error(f"Error processing entity '{entity}': {entity_error}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    continue
            
            logger.info(f"=== linkEnt completed: {len(idList)} entities successfully linked ===")
            logger.info(f"Linked entities: {list(idList.keys())}")
            return idList
            
        except Exception as e:
            logger.error(f"Fatal error in linkEnt: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {}
    
    try:
        # Execute the main pipeline
        logger.info("=== Starting entity extraction phase ===")
        entityList = extractEnts(query)
        logger.info(f"Entity extraction completed: {len(entityList)} entities found")
        
        logger.info("=== Starting entity linking phase ===")
        bioIDs = linkEnt(entityList, query)
        logger.info(f"Entity linking completed: {len(bioIDs)} entities linked")
        
        # Calculate timing
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"=== modifiedBioNERTool completed in {duration:.2f} seconds ===")
        
        # Final results
        if bioIDs:
            logger.info(f"SUCCESS: BioNER extracted and linked {len(bioIDs)} entities: {bioIDs}")
            return bioIDs
        else:
            logger.warning("No entities were successfully extracted and linked")
            return {"message": "No entities found"}
            
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.error(f"FAILURE: modifiedBioNERTool failed after {duration:.2f} seconds: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


# Wrapper class for compatibility
class BioNERTool:
    def __init__(self, openai_api_key=None):
        logger.debug(f"Initializing BioNERTool wrapper with API key: {bool(openai_api_key)}")
        self.openai_api_key = openai_api_key
    
    def extract_and_link(self, query: str, include_types: bool = True):
        """Extract and link biomedical entities - wrapper for original function"""
        logger.info(f"=== BioNERTool.extract_and_link called with query: '{query}' ===")
        logger.debug(f"include_types parameter: {include_types}")
        
        try:
            # Call the original function
            logger.debug("Calling modifiedBioNERTool...")
            bio_ids = modifiedBioNERTool(query, self.openai_api_key)
            logger.debug(f"modifiedBioNERTool returned: {bio_ids}")
            
            if isinstance(bio_ids, dict) and "message" in bio_ids:
                logger.info("No entities found by modifiedBioNERTool")
                return {"entities": []}
            
            # Convert to expected format
            logger.debug(f"Converting {len(bio_ids)} bio_ids to expected format...")
            entities_list = []
            for entity_text, entity_id in bio_ids.items():
                entity_obj = {
                    "name": entity_text,
                    "id": entity_id,
                    "type": "general",
                    "synonyms": [],
                    "definition": ""
                }
                entities_list.append(entity_obj)
                logger.debug(f"Converted entity: {entity_obj}")
            
            result = {"entities": entities_list}
            logger.info(f"BioNERTool.extract_and_link returning {len(entities_list)} entities")
            logger.debug(f"Final result: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"BioNER extraction failed in wrapper: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "error": str(e),
                "entities": []
            }
