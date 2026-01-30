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

nlp = spacy.load("en_core_sci_lg")
drug_disease_nlp = spacy.load("en_ner_bc5cdr_md")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

class NERInput(BaseModel):
    query: str = Field(description = "Query to extract biological entities from")

@tool("modifiedBioNERTool", args_schema=NERInput)
def modifiedBioNERTool(query: str):
    """Extract biological entities from a query and returns them along with their ID"""


### Step 1: Biomedical entity extraction
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    global nlp, drug_disease_nlp

    def extractEnts(query: str):
        entities = []
        
        docs = [
        nlp(query),
        drug_disease_nlp(query)
        ]

        # Extract entity texts from each doc
        for doc in docs:
            for ent in doc.ents:
                entities.append(ent.text.strip())

        # Extract biological process terms using LLM
        bp_prompt = f"""You are a helpful assistant that can extract biological processes from a given query. 
                    These might include concepts such as "cholesterol biosynthesis", "Aminergic neurotransmitter loading into synaptic vesicle", etc. Each entity should be a noun.

                    You must always return the full phrase/long form of each biomedical entity 
                    
                    Return results as a list. Return "" if no biological processes are in the query. DO NOT INCLUDE YOUR THOUGHTS
                    Here is your query: {query}"""

        bp_list = llm.invoke(bp_prompt).content.strip()
        entities.append(bp_list)
    
        # Deduplicate while preserving order
        entities = list(dict.fromkeys(entities))

        return entities
        
### Step 2: Linking each entity into a semantic type
    def classifyEnt(query: str, ent: str):
        supportedEntTypes = ["biologicalProcess", "general"]
        
        class entType(TypedDict):
            """The most appropriate entity type given a specific entity"""
            entType: Literal[*supportedEntTypes] # type: ignore

        classify_prompt = f"""
        Classify the following biomedical entity into one of: 
        {supportedEntTypes}

        Biomedical entity: {ent}

        For context, here is the query that the entity was extracted from: {query}
        """
        chosen_type = llm.with_structured_output(entType).invoke(classify_prompt)

        chosen_type = str(chosen_type["entType"])

        print(ent + " - " + chosen_type)
        
        return chosen_type


### Step 3: Entity-linking using different linkers based on entity type mapping
    def linkEnt(entList: list, query: str):
        idList = {}

        def sriNameResolver(ent: str, base_url = "https://name-lookup.ci.transltr.io/lookup", is_bp: bool = False, k = 50) -> list:    
            candidates = []
            autoComplete = True
            
            processed_ent = ent.replace(" ", "%20")

            final_url = base_url + "?string=" + processed_ent + "&autocomplete=" + str(autoComplete).lower() + "&limit=" + str(k)

            if is_bp:
                final_url = final_url + "&only_prefixes=GO&biolink_type=BiologicalProcess"
                
            response = requests.get(final_url, headers={"accept": "application/json"})
        
            candidate_list = json.loads(response.content.decode("utf-8"))
        
            for item in candidate_list:
                parsed = {
                    "label": item.get("label", ""),
                    "curie": item.get("curie", ""),
                    "score": item.get("score", "")
                }
        
                candidates.append(parsed)
        
            return candidates

        def remove_TUI(text):
            parts = text.split("TUI", 1)
            return parts[0]
        
        def selectID(doc: object):
            bioID = ""
            candidates = {}

            linker = nlp.get_pipe("scispacy_linker")   
            
            for ent in doc.ents:
                if ent._.kb_ents:  # Check if entity has linked knowledge base IDs
                    for id in ent._.kb_ents:
                        candidates[id[0]] = remove_TUI(str(linker.kb.cui_to_entity[id[0]]))
        
                select_prompt = f"""You are a smart biomedical assistant that can understand the context and the intent behind a query. 
                            Be careful when choosing IDs for entities that can refer to different concepts (for example, HIV can refer either to the virus or the disease; you MUST choose the most appropriate concept/definition based on the query). 
                            Use the context and the intent behind the query to choose the most appropriate ID. 
                            Here is the complete query: {query}
                            Select the one most appropriate ID/CUI for {ent.text} from the list below:
                            {candidates}
                            If none of the choices are appropriate, return "".
                            Otherwise, return only the ID/CUI.
                            """
    
                # LLM selects most appropriate ID from list
                selectedID = llm.invoke(select_prompt).content.strip()
    
                # Extract just the UMLS CUI using regex
                match = re.search(r"C\d{7}", selectedID)
                if match:
                    bioID = "UMLS:" + match.group(0)
                    definition = candidates[match.group(0)]
                else:
                    bioID = ""
                    definition = ""
    
                # Printing chosen ID + definition, if any
                print(ent.text + " - " + bioID + '\n' + definition)

            return bioID

        def selectIDbp(entity: str, candidates: list):
            choices = [""]

            for entry in candidates:
                curie = entry.get("curie")
                choices.append(curie)

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
    
            # LLM selects most appropriate ID from list
            selectedID = llm.with_structured_output(selectedID).invoke(select_prompt)

            bioID = str(selectedID["selectedID"])

            # Printing chosen ID + definition, if any
            print(entity + " - " + bioID + '\n')

            return bioID

        # General Pipeline            
        for entity in entList:
            chosenID = ""
            entclass = classifyEnt(query, entity)
            if entclass == "biologicalProcess":
                chosenID = selectIDbp(entity, sriNameResolver(entity, is_bp=True))
            else:
                doc = nlp(entity)
                chosenID = selectID(doc)
                if chosenID == "":
                    chosenID = selectIDbp(entity, sriNameResolver(entity, is_bp=False))
                    if chosenID == "":
                        continue

            idList[entity] = chosenID
            

        return idList
    
    entityList = extractEnts(query)
    bioIDs = linkEnt(entityList, query)

    print(bioIDs)
    
    return bioIDs if bioIDs else {"message": "No entities found"}

def PubtatorParse(results: str):
    entities = []

    lines = results.strip().split('\n')

    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 6:  # Ensures the line has enough elements for parsing
            entity_data = {
                "name": parts[3],
                "type": parts[4],
                "id": parts[5]
            }
            if entity_data["type"].lower() == "gene":
                entity_data["id"] = "NCBIGene:" + entity_data["id"]
            entities.append(entity_data)

    return entities

def PubtatorTool(query: str):
    """Extract biological entities from a query and returns them along with their ID"""

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {
        "text": query
    }

    request = requests.post('https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/request.cgi', headers=headers, data=data)

    max_attempts = 10

    attempts = 0

    while attempts < max_attempts:
        try:
            response = requests.post('https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/retrieve.cgi', headers=headers, data=request.json())
            if response.status_code == 200:
                bioEnts = PubtatorParse(response.text)
                return bioEnts
            elif response.status_code == 400:
                print(f"Attempt {attempts + 1}/{max_attempts}: Response not ready yet (400). Retrying in 60 seconds...")
                time.sleep(60)
            else:  # Unexpected error codes
                response.raise_for_status()

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempts + 1}/{max_attempts}: Network error - {e}")
            time.sleep(5)

        attempts += 1

    return {"message": "No entities found"}