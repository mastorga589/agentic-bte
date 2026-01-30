from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Any, Literal
from typing_extensions import TypedDict
from pprint import pprint
from time import sleep
import json
import requests

metaKG = requests.get("https://bte.transltr.io/v1/meta_knowledge_graph")

def extractDict(rawstring: str):
    # Finding positions of the brackets
    start_index = rawstring.find("{")
    end_index = rawstring.find("}\n`")

    # Extracting dict object
    if start_index != -1 and end_index != -1:
        extracted_dict = rawstring[start_index:end_index + 1].strip()

    if extracted_dict:
        return json.loads(extracted_dict)
    else:
        # Returns raw dict if already valid
        try:
            if json.loads(rawstring):
                return rawstring
        except:
            return {"error": "could not parse dict"}  
        

@tool("TRAPIQuery")
def TRAPIQuery(query: str, entity_data: dict, failed_trapis: list = []):
    """Use this tool to build a TRAPI JSON query"""
    
    MAX_RETRIES = 3
    llm = ChatOpenAI(temperature=0, model="gpt-4o")

    def findPredicate(subject: str, object: str):
        global metaKG

        predicates = []

        metaEdges = json.loads(metaKG.content).get("edges")

        for edge in metaEdges:
            if (edge.get("subject").lower() == subject.lower()) and (edge.get("object").lower() == object.lower()):
                p = edge.get("predicate")
                predicates.append(p)

        return predicates
    
    def choosePredicate(predicateList: List, query: str):
        class predicateChoice(TypedDict):
            """The most appropriate predicate chosen for the TRAPI query"""

            predicate: Literal[*predicateList]

        llm = ChatOpenAI(temperature=0, model="gpt-4o")

        predicatePrompt = """Choose the most specific predicate by examining the query closely and choosing the closest answer.  
            
            Here is the query: {query}
            """

        chosen_predicate = llm.with_structured_output(predicateChoice).invoke(predicatePrompt)

        return str(chosen_predicate["predicate"])

    def invoke_with_retries(prompt, parser_func, max_retries=3, delay=5):
        for attempt in range(max_retries):
            try:
                response = llm.invoke(prompt)
                return parser_func(response.content)
            except Exception as e:
                print(f"Retry {attempt+1}/{max_retries} failed: {e}")
                sleep(delay)
        return None

    def identify_nodes(query: str):
        nodeprompt = f"""
        Your task is to help build a TRAPI query by identifying the correct subject and object nodes from the list of nodes below. 

        Subject: The entity that initiates or is the focus of the relationship in your question.
        Object: The entity that is affected or related to the subject in your question.
        
        Each of these must have the prefix "biolink:". You must use the context and intent behind the user query to make the correct choice.

        Nodes: Disease, PhysiologicalProcess, BiologicalEntity, Gene, PathologicalProcess, Polypeptide, SmallMolecule, PhenotypicFeature

        Here is the user query: {query}

        Be as specific as possible as downstream results will be affected by your node choices.
        Your response MUST ONLY CONTAIN a dictionary object with "subject" and "object" as keys along with their corresponding values.
        """
        
        return invoke_with_retries(nodeprompt, extractDict)
    
    def build_TRAPI(query: str, subject_object: dict, predicate: str, entity_data: dict, failed_trapis: list):
        print("\n\n Building trapi....")

        # TRAPI example that could be included in the prompt
        trapi_ex = {
                "message": {
                "query_graph": {
                "nodes": {
                    "n0": {
                    "categories": [
                        "biolink:Disease"
                    ],
                    "ids": [
                        "MONDO:0016575"
                    ]
                    },
                    "n1": {
                    "categories": [
                        "biolink:PhenotypicFeature"
                    ]
                    }
                },
                "edges": {
                    "e01": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": [
                        "biolink:has_phenotype"
                    ]
                    }
                }
                }
        }
        }

        trapiPrompt = f"""
            You are a smart assistant that has access to a knowledge graph. 
            You are tasked with correctly parsing the user prompt into a TRAPI query. 
            Here's an example TRAPI query:
            {json.dumps(trapi_ex, indent=2)}

            Here is the actual user query: "{query}"

            Here are the biological entities and their IDs extracted from the query:
            {json.dumps(entity_data, indent=2)}

            Here are the nodes you MUST use:
            {json.dumps(subject_object, indent=2)}

            Here is the chosen predicate
            {predicate}

            Do NOT use the following TRAPI queries as they have failed to get results: 
            {failed_trapis}

            Some predicates have directionality ("treated_by" is NOT the same as "treats"). When defining the edges, you MUST use the correct predicate with the correct directionality.
            (e.g., biolink:Disease is "treated_by" (NOT "treats") biolink:SmallMolecule; 
            biolink:Disease as subject and biolink:Gene as object should result in "condition_associated_with_gene", NOT "gene_associated_with_condition")

            In this KG, the subject of the predicate is always the source (domain), and the object is the target (range). For example, if 'X treats Y', then X is the subject and Y is the object.
            
            Please make sure that only one node in the TRAPI query has the "ids" field (either "n0" or "n1" but NOT both); 
            decide which one would result in more useful results (for "Which of these genes are involved in a physiological process?", including the IDs for Gene would result in more useful results. In this case DO NOT include the ID for Physiological Process)

            Using these, output a JSON object containing the completed TRAPI query. 
        """

        return invoke_with_retries(trapiPrompt, extractDict)
    
    # Main dataflow
    subject_object = identify_nodes(query)
    if not subject_object:
        print("Could not determine subject/object nodes")
        return "Could not determine subject/object nodes"
    
    print(f"\n\nIdentified subject/object: {json.dumps(subject_object)}")
    
    predicateList = findPredicate(subject_object.get("subject"), subject_object.get("object"))

    # Extract all failed predicates from the failed TRAPI queries
    failed_predicates = set()
    for trapi_query in failed_trapis:
        try:
            preds = trapi_query.get("message", {}).get("query_graph", {}).get("edges", {}).get("e01", {}).get("predicates", [])
            failed_predicates.update(preds)
        except Exception as e:
            print(f"Error extracting predicates from failed_trapis: {e}")

    # Remove failed predicates from predicateList
    predicateList = [p for p in predicateList if p not in failed_predicates]

    predicate = choosePredicate(predicateList, query)

    trapi = build_TRAPI(query, subject_object, predicate, entity_data, failed_trapis)
    
    if trapi:
        return trapi

    # Retry if initial attempt fails
    for _ in range(MAX_RETRIES):
        trapi = build_TRAPI(query, subject_object, predicate, entity_data)
        if trapi:
            return trapi

    return "Invalid TRAPI"
            



    
def parseBTE(jsonr: dict, k: int, maxresults: int):
    # Extract all nodes, edges, and results (these contain IDs and names)
    nodes = jsonr.get("message", {}).get("knowledge_graph", {}).get("nodes", {})
    edges = jsonr.get("message", {}).get("knowledge_graph", {}).get("edges", {})
    results = jsonr.get("message", {}).get("results", [])

    # Parsing node data into a list
    node_data = {}
    for node_id, node_info in nodes.items():
        node_data[node_id] = {
            "ID": node_id,
            "name": node_info.get("name", ["Unknown"]),
            "category": node_info.get("categories", ["Unknown"])[0]
        }

    # Extracting edge data
    edge_data = {}
    for edge_id, edge_info in edges.items():
        edge_data[edge_id] = {
            "subject": edge_info.get("subject"),
            "predicate": edge_info.get("predicate"),
            "object": edge_info.get("object")
        }

    # Parsing through results to match IDs
    parsed_results = []
    results_id = {}

    # Track how many results have been collected per input ID
    results_per_id = {}

    for result in results:
        bindings = result.get("node_bindings", {})
        subject_id = bindings.get("n0", [{}])[0].get("id", "Unknown")
        object_id = bindings.get("n1", [{}])[0].get("id", "Unknown")
        
        subject_name = node_data.get(subject_id, {}).get("name", "Unknown")
        object_name = node_data.get(object_id, {}).get("name", "Unknown")
        relationship = None

        if results_per_id:
            subject_count = results_per_id.get(subject_id, {}).get("result count (k)", 0)
            object_count = results_per_id.get(object_id, {}).get("result count (k)", 0)

            if subject_count >= k or object_count >= k:
                continue

        for edge_id, edge_info in edge_data.items():
            if (edge_data[edge_id].get("subject") == subject_id) and (edge_data[edge_id].get("object") == object_id):
                relationship = edge_data[edge_id].get("predicate")
        
        if relationship:
            parsed_results.append({
                "subject": subject_id,
                "subject_type": node_data.get(subject_id, {}).get("category", "Unknown"),
                "predicate": relationship,
                "object": object_id, 
                "object_type": node_data.get(object_id, {}).get("category", "Unknown")
            })
            if "by" not in relationship:
                if subject_id not in results_per_id:
                    results_per_id[subject_id] = {"result count (k)": 0, "name": subject_name, "results": []}

                results_per_id[subject_id]["result count (k)"] += 1

                results_per_id[subject_id]["results"].append({
                    "predicate": relationship,
                    "target_name": object_name,
                    "target_type": node_data.get(object_id, {}).get("category", "Unknown")
                })
            else:
                if object_id not in results_per_id:
                    results_per_id[object_id] = {"result count (k)": 0, "name": object_name, "results": []}

                results_per_id[object_id]["result count (k)"] += 1

                results_per_id[object_id]["results"].append({
                    "predicate": relationship,
                    "target_name": object_name,
                    "target_type": node_data.get(object_id, {}).get("category", "Unknown")
                })
        else:
            for edge_id, edge_info in edge_data.items():
                if (edge_data[edge_id].get("subject") == object_id) and (edge_data[edge_id].get("object") == subject_id):
                    relationship = edge_data[edge_id].get("predicate")

            parsed_results.append({
                "subject": object_id,
                "subject_type": node_data.get(object_id, {}).get("category", "Unknown"),
                "predicate": relationship,
                "object": subject_id, 
                "object_type": node_data.get(subject_id, {}).get("category", "Unknown")
            })

            if object_id not in results_per_id:
                results_per_id[object_id] = {"result count (k)": 0, "name": object_name}

            results_per_id[object_id]["result count (k)"] += 1
        if len(results_id) == 0:
            results_id.update({
                f"{subject_name}": subject_id,
                f"{object_name}": object_id
            })
        else:            
            if subject_id not in results_id.values():
                results_id.update({f"{subject_name}": subject_id})
            
            if object_id not in results_id.values():
                results_id.update({f"{object_name}": object_id})
        
        if len(results_per_id) >= maxresults:
            break

    
    pprint(results_per_id, indent=4)
    return parsed_results, results_id

@tool("BTECall")
def BTECall(json_query: dict, maxresults: int, k: int):
    """Makes an API request to BioThings Explorer using a JSON object containing the query graph and returns the results"""

    api_url = "https://bte.transltr.io/v1/query"

    if not isinstance(json_query, dict):
            return {"error": "Invalid JSON format"}

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    num_ids = 0

    # Counting the number of IDs/entities being queried
    nodes = json_query.get("message", {}).get("query_graph", {}).get("nodes", {})
    for node_id, node_data in nodes.items():
        ids = node_data.get("ids")
        if ids and isinstance(ids, list):
            num_ids += len(ids)
    
    try:
        response = requests.post(api_url, json=json_query, headers=headers)
        response.raise_for_status()  # Raise an error for non-200 responses

        results, ids = parseBTE(response.json(), k, maxresults)

        message = response.json().get("description")

        if results: # Return API response as JSON
            return results, ids, message
        else:
            return None, None, {"error": message}
    except requests.exceptions.RequestException as e:
        return None, None, {"error": f"API request failed: {str(e)}, {response.json().get("description")}"}