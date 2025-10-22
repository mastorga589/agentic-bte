from __future__ import annotations

"""
Optional LangGraph-based orchestrator that uses local tools (BioNER, BTECall).
This mirrors the original Prototype Agent at a high level while keeping imports
within this package. It is not required for the GoT planner to function.
"""

from typing import Literal, Annotated
from typing_extensions import TypedDict
from operator import add

from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .tools.BioNER import modifiedBioNERTool
from .tools.BTECall import TRAPIQuery, BTECall

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS

# RDFlib graph for results accumulation (compat with original)
_result_graph = Graph()


class Router(TypedDict):
    next: Literal["annotator", "planner", "BTE_search", "FINISH"]


class State(MessagesState):
    next: str
    query: str
    subQuery: Annotated[list[str], add]
    entity_data: Annotated[list[dict], add]
    maxresults: int
    k: int
    final_answer: str


def _rdf_update(triples: list[dict]):
    BIOLINK = Namespace("https://w3id.org/biolink/vocab/")
    EX = Namespace("http://example.org/entity/")
    _result_graph.bind("biolink", BIOLINK)
    _result_graph.bind("ex", EX)

    def make_uri(name: str) -> URIRef:
        return EX[name.replace(" ", "_").replace(":", "-").lower()]

    for t in triples:
        s = make_uri(t['subject'])
        p = URIRef(BIOLINK + t['predicate'].split(":")[1]) if ":" in t['predicate'] else BIOLINK[t['predicate']]
        o = make_uri(t['object'])
        _result_graph.add((s, p, o))


def BTE_search(state: State) -> Command[Literal["orchestrator"]]:
    subquery = str(state["subQuery"][-1])
    maxresults = state["maxresults"]
    k = state["k"]
    failed_trapis: list = []

    # Flatten entity_data list of dicts
    entity_data: dict = {}
    for d in state["entity_data"]:
        entity_data.update(d)

    trapi = TRAPIQuery(subquery, entity_data, failed_trapis)
    bte_results, ent_update, message = BTECall(trapi, maxresults, k)

    if not bte_results:
        return Command(
            update={"messages": [HumanMessage(content=str(message), name="BTE_search")]},
            goto="orchestrator"
        )

    _rdf_update(bte_results)

    # Convert ent_update (name->id) into list of dicts
    ent_list = [{k: v} for k, v in (ent_update or {}).items()]

    return Command(
        update={
            "messages": [HumanMessage(content=str(message), name="BTE_search")],
            "entity_data": ent_list,
        },
        goto="orchestrator",
    )


def annotator_node(state: State) -> Command[Literal["orchestrator"]]:
    response = modifiedBioNERTool.invoke({"query": state["query"]})
    return Command(
        update={
            "messages": [HumanMessage(content=str(response), name="annotator")],
            "entity_data": [response],
        },
        goto="orchestrator",
    )


def planner_node(state: State) -> Command[Literal["orchestrator"]]:
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    prompt = f"""
Your job is to propose the next single-hop subquery to help answer the main query.
Return ONLY the natural language subquestion, no commentary.
Main query: {state['query']}
Current RDF (Turtle):
{_result_graph.serialize(format='turtle')}
"""
    response = llm.invoke([{ "role": "user", "content": prompt}])
    subq = (response.content or "").strip()
    return Command(
        update={
            "messages": [HumanMessage(content=subq, name="planner")],
            "subQuery": [subq],
        },
        goto="orchestrator",
    )


def summary_node(state: State) -> Command[Literal["orchestrator"]]:
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    final_prompt = f"""
Summarize findings to answer the user's question.
Question: {state['query']}
RDF (Turtle):
{_result_graph.serialize(format='turtle')}
Entities: {state['entity_data']}
Be concise and mechanistic, grounded in evidence.
"""
    out = llm.invoke([{ "role": "user", "content": final_prompt }])
    return Command(update={"final_answer": out.content or ""}, goto=END)


def orchestrator(state: State) -> Command[Literal["annotator", "planner", "BTE_search", "summary"]]:
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    system_prompt = """
You coordinate annotator, planner, and BTE_search to step-wise solve the query.
Route to FINISH only when all necessary subqueries are answered.
"""
    msgs = [{"role": "system", "content": system_prompt}] + state["messages"]
    nxt = llm.with_structured_output(Router).invoke(msgs)
    goto = nxt.get("next", "planner")
    if goto == "FINISH":
        goto = "summary"
    return Command(goto=goto, update={"next": goto})


def build_graph() -> StateGraph:
    g = StateGraph(State)
    g.add_node("orchestrator", orchestrator)
    g.add_node("planner", planner_node)
    g.add_node("annotator", annotator_node)
    g.add_node("BTE_search", BTE_search)
    g.add_node("summary", summary_node)
    g.add_edge(START, "orchestrator")
    return g.compile()


def run(query: str, maxresults: int = 50, k: int = 5) -> str:
    graph = build_graph()
    output = None
    for s in graph.stream({
        "messages": [("human", query)],
        "query": query,
        "maxresults": maxresults,
        "k": k,
    }, {"recursion_limit": 30}, subgraphs=True):
        output = s
    try:
        return output[-1].get("summary", {}).get("final_answer", "")
    except Exception:
        return ""
