from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from os.path import dirname, join
from dotenv import load_dotenv
from typing import Literal, List, Dict
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.types import Command
from operator import add
from copy import deepcopy
import getpass
import os
import pandas as pd
from tools.BioNER import modifiedBioNERTool
from tools.BTECall import TRAPIQuery, BTECall
from rdflib import Graph, URIRef, Namespace
from rdflib.namespace import RDF, RDFS


# Set your OpenAI API key
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

if not os.environ.get("OPENAI_API_KEY"): #field to ask for OpenAI API key
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Please enter OpenAI API Key: ")

# Create an LLM-based agent
llm = ChatOpenAI(temperature=0, model="gpt-4.1")  # Change model if needed

# Create an RDFlib graph where results will be stored
resultGraph = Graph()

members = ["annotator", "planner", "BTE_search"]
options = members + ["FINISH"]

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options]

class State(MessagesState):
    next: str
    query: str
    subQuery: Annotated[list[str], add]
    entity_data: Annotated[list[dict], add]
    maxresults: int
    k: int
    final_answer: str

def RDFgraphUpdater(triples: list):
    global resultGraph

    # Define namespaces
    BIOLINK = Namespace("https://w3id.org/biolink/vocab/")
    EX = Namespace("http://example.org/entity/")
    GENE = Namespace("https://biolink.github.io/biolink-model/Gene/")
    DISEASE = Namespace("https://biolink.github.io/biolink-model/Disease/")
    PHYSPROCESS = Namespace("https://biolink.github.io/biolink-model/PhysiologicalProcess/")
    BIOENT = Namespace("https://biolink.github.io/biolink-model/BiologicalEntity/")
    PATHPROCESS = Namespace("https://biolink.github.io/biolink-model/PathologicalProcess/")
    SMALLMOL = Namespace("https://biolink.github.io/biolink-model/SmallMolecule/")
    PHENFEATURE = Namespace("https://biolink.github.io/biolink-model/PhenotypicFeature/")
    POLYPEPTIDE = Namespace("https://biolink.github.io/biolink-model/Polypeptide/")

    # Bind namespaces (for nicer Turtle output)
    resultGraph.bind("biolink", BIOLINK)
    resultGraph.bind("ex", EX)
    resultGraph.bind("gene", GENE)
    resultGraph.bind("disease", DISEASE)
    resultGraph.bind("physprocess", PHYSPROCESS)
    resultGraph.bind("phenprocess", PHENFEATURE)
    resultGraph.bind("bioent", BIOENT)
    resultGraph.bind("pathprocess", PATHPROCESS)
    resultGraph.bind("smallmol", SMALLMOL)
    resultGraph.bind("polypeptide", POLYPEPTIDE)

    # Map biolink entity types to their namespaces
    ENTITY_NAMESPACE_MAP = {
        "biolink:Gene": GENE,
        "biolink:Disease": DISEASE,
        "biolink:PhysiologicalProcess": PHYSPROCESS,
        "biolink:BiologicalEntity": BIOENT,
        "biolink:PathologicalProcess": PATHPROCESS,
        "biolink:SmallMolecule": SMALLMOL,
        "biolink:PhenotypicFeature": PHENFEATURE,
        "biolink:Polypeptide": POLYPEPTIDE,
    }

    # Helper function to make a URI from a name
    def make_entity_uri(name: str, entity_type: str):
        ns = ENTITY_NAMESPACE_MAP.get(entity_type, EX)
        return ns[name.replace(" ", "_").replace(":", "-").lower()]
    
    for triple in triples:
        subj = make_entity_uri(triple['subject'], triple.get("subject_type", ""))
        pred = URIRef(BIOLINK + triple['predicate'].split(":")[1])
        obj = make_entity_uri(triple['object'], triple.get("object_type", ""))
        resultGraph.add((subj, pred, obj))

def BTE_search(state: State) -> Command[Literal["orchestrator"]]:
    MAX_ATTEMPTS = 3
    subquery = str(state["subQuery"][-1])
    maxresults = state["maxresults"]
    k = state["k"]
    failed_trapis = []
    bte_results = []
    entity_data = {}
    messages = {}
    ent_update = {}
    
    # Converting from list of dicts to unified dict
    for i in state["entity_data"]:
        entity_data.update(i)

    def split_trapi_query(trapi_query: Dict, batch_limit: int = 50) -> List[Dict]:
        query_graph = trapi_query.get("message", {}).get("query_graph", {})
        nodes = query_graph.get("nodes", {})

        # Identify nodes that need to be split
        split_nodes = {k: v for k, v in nodes.items() if "ids" in v and len(v["ids"]) > batch_limit}

        if not split_nodes:
            return [trapi_query]  # No need to split

        # For simplicity, assume only ONE node needs splitting (common case)
        node_id, node_data = next(iter(split_nodes.items()))
        id_chunks = [
            node_data["ids"][i:i + batch_limit]
            for i in range(0, len(node_data["ids"]), batch_limit)
        ]

        # Create a new query for each chunk
        queries = []
        for chunk in id_chunks:
            new_query = deepcopy(trapi_query)
            new_query["message"]["query_graph"]["nodes"][node_id]["ids"] = chunk
            queries.append(new_query)

        return queries
    
    def remove_existing_ents(target: dict, reference: list):
        # Build a set of key-value pairs from the reference list
        ref_pairs = set()
        for d in reference:
            ref_pairs.update(d.items())

        # Filter out key-value pairs in the target that exist in the reference set
        return {k: v for k, v in target.items() if (k, v) not in ref_pairs}

    def run_trapi_and_bte(subquery, entity_data, failed_trapis):
        results = []
        updates = {}
        msg_log = {}

        trapi = TRAPIQuery.invoke(input={"query": subquery, "entity_data": entity_data, "failed_trapis": failed_trapis})
        if trapi == "Invalid TRAPI":
            print("Retrying TRAPI generation...")
            trapi = TRAPIQuery.invoke(input={"query": subquery, "entity_data": entity_data, "failed_trapis": failed_trapis})

        trapi_list = split_trapi_query(trapi)
        print(f"\n\nNo. of TRAPI queries (batch size 50): {len(trapi_list)}")

        for index, trapi in enumerate(trapi_list):
            try:
                print(f"\n\nTRAPI query #{index + 1}:\n{trapi}\n")
                result, ids, update = BTECall.invoke(input={
                    "json_query": trapi, 
                    "maxresults": maxresults,
                    "k": k})

                if result:
                    results.extend(result)
                else:
                    msg_log["Query:"] = update
                    failed_trapis.append(trapi)
                    break

                if ids:
                    filtered_ids = remove_existing_ents(ids, state["entity_data"])
                    updates.update(filtered_ids)

                msg_log[f"Query #{index + 1}"] = update
            except Exception as e:
                print(f"Error during BTE call: {e}")
                continue

        return results, updates, msg_log

    # Retry TRAPI generation and BTE execution until results are found or max attempts are reached
    for attempt in range(MAX_ATTEMPTS):
        print(f"\nAttempt #{attempt + 1} to generate and run TRAPI query.")
        bte_results, ent_update, messages = run_trapi_and_bte(subquery, entity_data, failed_trapis)

        if bte_results:
            break
        else:
            print("No results found. Retrying...\n")

    if not bte_results:
        return Command(
            update={
                "messages": [
                    HumanMessage(content=str(bte_results), name="BTE_search")
                ]
                },
            goto="planner"
        )
    
    # If results are found
    RDFgraphUpdater(bte_results)

    return Command(
        update={
            "messages": [
                HumanMessage(content=f"{messages}", name = "BTE_search")
            ],
            "entity_data": [{k: v} for k, v in ent_update.items()]
        },
        goto="orchestrator"
    )

def annotator_node(state: State) -> Command[Literal["orchestrator"]]:
    response = modifiedBioNERTool.invoke(state["query"])
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=str(response), name="annotator"
                )
            ],
            "entity_data": [response]
        },
        goto="orchestrator"
    )

def planner_node(state: State) -> Command[Literal["orchestrator"]]:
    planner_prompt = f"""Your role is to create a plan to answer the main query by dividing it into several simpler single hop subqueries. 
    The system has access to a knowledge graph where necessary data will be retrieved from to answer the query.

    This will be an iterative process where answers to previous subqueries might be relevant in constructing future queries. 
    Initial subqueries should encompass the overall query path; each subsequent subquery should be more refined. A mechanistic approach should be used in developing the query plan.
    For example, for "Which drugs can treat Crohn's disease by targeting the inflammatory response?", 
    the first subquery might be "Which drugs can treat Crohn's disease?" followed by "Which genes do these drugs target?", 
    then "Which of these genes are related to the inflammatory response?".
    
    Here are the node types present in the knowledge graph: Disease, PhysiologicalProcess, BiologicalEntity, Gene, PathologicalProcess, Polypeptide, SmallMolecule, PhenotypicFeature

    The predicates in the knowledge graph can be grouped into causality, treatment, genetics & biomarkers, interactions, phenotypes & diagnostics, responses & effects, associations, and structural/hierarchical relationships.
    Restrict your queries to within these relationships.

    Here are the current results so far expressed in Turtle:
    {resultGraph.serialize(format="turtle")}

    Make sure that each subquery interrogates discrete relationships between node types. 
    For example, instead of directly asking "What are the mechanisms of action of these drugs?", 
    you must create subqueries that can help form a reasoning chain to answer the question ("What genes do these drugs interact with?" and "Which physiological processes are these genes involved in?)

    If a subquery results in "No results found", rephrase/reframe the question or explore the question from a different angle.

    Do NOT prescribe which nodes to use in your subquery, the nodes and predicates are only for your reference.
    You need to determine what the next single-hop subquery is and formulate it into a natural language subquestion. 

    Your response MUST ONLY CONTAIN the single-hop natural language subquestion (for example, "What genes does doxorubicin target?"). 
    Please DO NOT include your thoughts or anything else as it will interfere with downstream processes.
"""

    messages = [
        {"role": "assistant", "content": planner_prompt},
    ] + state["messages"]
    response = llm.invoke(messages)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response.content, name="planner"
                )
            ],
            "subQuery": [response.content]
        },
        goto="orchestrator"
    )

def summary_node(state: State) -> Command[Literal["orchestrator"]]:
    summary_llm = ChatOpenAI(temperature=0, model="gpt-4.1")

    final_prompt = f""" You are an expert proficient in the pharmaceutical sciences, medicinal chemistry and biomedical research.
        Your team has access to a biomedical knowledge graph, and have provided you with the following results based on the user's query.
        However, the knowledge graph might not be perfect and have gaps in its data, and some relationships between the entities might be implicit.
        Your job is to answer the user's query using your team's results and your own biomedical expertise.
        Your summary of the results MUST be comprehensive while still avoiding redundancy.
        Expound on the logical steps you took to form your final answer.

        Make sure to organize your answer around the user query:
        {state["query"]}

        Here are the findings of your team expressed in Turtle:
        {resultGraph.serialize(format="turtle")}

        Here are the entities each ID corresponds to:
        {state["entity_data"]}

        You MUST base your answer on the evidence/context included in the prompt; 
        however, you can use your expertise to contextualize the results. 
        For example, if the results only show target genes for dirithromycin, knowing that dirithromycin and erythromycin are part of the same drug class, you can infer that they are likely to share similar properties and targets.
        
        Maintain complete transparency about your problem-solving process; the relationships between each entity should be clear in your answer.
        Do NOT list down all entities; rather, choose to most important ones to illustrate your point (for example, "the BRCA family of proteins is involved in breast cancer").
        Remember, prioritize accuracy, explainability, and user understanding.
        Only include relevant results in the final answer.
        """
    
    summarize = [
        {"role": "system", "content": final_prompt},
    ]

    summary = summary_llm.invoke(summarize)

    print(summary.content)

    print("\n\n These were the results: \n")
    print(resultGraph.serialize(format="turtle"))
    print("\n\n")

    return Command(
        update={
            "final_answer": summary.content
        },
        goto=END
    )



# Setting system prompt
system_prompt = """
You are a biomedical research supervisor with access to a biomedical knowledge graph. 
You have access to an annotator ("annotator") which can annotate biomedical entities with their IDs, 
a planner ("planner") which can tell you which single-hop subquery would help you answer the user query, and 
a knowledge graph tool ("BTE_search") which can only answer single-hop queries.

In no particular order, your job is to:
1. If necessary, annotate the biomedical entities within the user query with their IDs using the annotator. ONLY use the annotator for this.
2. Construct a plan to answer the user query by deconstructing it to subqueries. These subquestions MUST be single-hop questions, and you MUST use the planner to help you with this. Use a mechanistic approach.
3. You MUST answer the subquery prescribed by the planner before asking the planner for the next one. This will result in a step-wise approach.
4. Make sure that each subquestion is answered before you provide your final answer.
5. Given the subquestions and the user query, respond with the team to act next. Each team will perform a task and respond with their results and status.
6. You are responsible for making sure that the user query is fully answered by the subqueries and their answers before giving your final answer.
6. Provide the user with a summary of your findings.
7. When finished, respond with FINISH. Only respond with FINISH if all subqueries have been answered.

CORE PRINCIPLES:
- Analyze the user's request thoroughly
- Select teams strategically and efficiently
- Provide clear reasoning for worker selection
- Maintain complete transparency about your problem-solving process
- Prioritize accuracy and user understanding
"""

def orchestrator(state: State) -> Command[Literal[*members, "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]

    next = llm.with_structured_output(Router).invoke(messages)
    goto = next["next"]
    if goto == "FINISH":
        goto = "summary"

    return Command(goto=goto, update={"next": goto})

def main(query, maxresults: int = 50, k: int = 5):
    builder = StateGraph(State)
    builder.add_node("orchestrator", orchestrator)
    builder.add_node("planner", planner_node)
    builder.add_node("annotator", annotator_node)
    builder.add_node("BTE_search", BTE_search)
    builder.add_node("summary", summary_node)

    builder.add_edge(START, "orchestrator")

    graph = builder.compile()

    print(query)
    print("\n")

    input = {"messages": [
        ("system", system_prompt),
        ("human", query)],
        "query": query,
        "maxresults": maxresults,
        "k": k}

    try:
        for s in graph.stream(input, {"recursion_limit": 50}, subgraphs=True):
                print(s)
                print("------------------")

    except Exception as e:
        print(f"Unfortunately, there has been an error.\n {e} \n Continuing onto the next query\n\n")
    finally:
        # Clears results of previous query
        resultGraph.remove((None, None, None))

    print("\n-------------Next Query-------------\n")

    try:
        if s[-1].get("summary", "").get("final_answer", ""):
            return s[-1].get("summary", "").get("final_answer", "")
    except:
        return None
    
def BTEx(query, maxresults: int = 50, k: int = 5):
    return main(query, maxresults, k)

if __name__ == "__main__":
    main()
    