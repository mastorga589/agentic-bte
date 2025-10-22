# Stand-alone Prototype

This directory contains a stand-alone integration of:
- Graph of Thoughts planner (DAG of subqueries, parallel execution of independent nodes)
- TRAPI builder (LLM-based with meta-KG validation)
- BTE client (TRAPI execution and parsing)
- Predicate selection (top-3 candidates executed concurrently per subquery)
- Ported LangGraph tools (BioNER and BTECall) for compatibility

Requirements
- Python 3.10+
- Packages: langchain-openai, networkx, rdflib, requests
- Optional (for BioNER): spaCy models en_core_sci_lg, en_ner_bc5cdr_md

Environment variables
- OPENAI_API_KEY: required for LLM operations
- OPENAI_MODEL (optional): defaults to gpt-4o-mini
- BTE_API_BASE_URL (optional): defaults to https://bte.transltr.io/v1

Usage
- GoT planner CLI
  python -m Prototype.run_got "Which drugs can treat Crohn's disease by modifying the immune response?"

- Programmatic API
  from Prototype.got_planner import GoTPlanner
  import asyncio
  answer = asyncio.run(GoTPlanner().execute("...question..."))
  print(answer)

Notes
- The planner decomposes the query into atomic single-hop subqueries using an LLM with BTE meta-KG constraints.
- Non-dependent subqueries (per the DAG edges) are executed in parallel.
- Each subquery runs top-3 predicate variants concurrently to maximize answer coverage.
- Results are accumulated as RDF (rdflib) and a final answer is synthesized by the LLM.
