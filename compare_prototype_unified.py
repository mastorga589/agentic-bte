#!/usr/bin/env python3
import sys
import os
import io
import re
import json
import asyncio
import contextlib
from typing import List, Dict, Any, Tuple

# Paths
PROTOTYPE_PATH = "/Users/mastorga/Documents/BTE-LLM/Prototype"
UNIFIED_ROOT = "/Users/mastorga/Documents/agentic-bte"

# Queries to compare
QUERIES = [
    "Which drugs can treat Spinal muscular atrophy by targeting Alternative mRNA splicing, via spliceosome?",
    "Which drugs can treat Crohn's disease by targeting inflammatory response?",
    "What genes are associated with Alternative mRNA splicing, via spliceosome?",
]

# ---------- Prototype runner and parsers ----------

def parse_prototype_output(text: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "subqueries": [],
        "trapi_predicates": [],
        "trapi_nodes": [],
        "result_counts": [],
        "final_answer": None,
    }

    # Subqueries from planner
    # Look for lines containing "subQuery': ['...']" or planner message content
    subq_regex = re.compile(r"subQuery\]?: \['([^']+)']")
    for m in subq_regex.finditer(text):
        summary["subqueries"].append(m.group(1).strip())

    # TRAPI dicts printed as: TRAPI query #n:\n{...}
    # Extract JSON-like dict blocks after that marker and parse pred/nodes
    trapi_block_pattern = re.compile(r"TRAPI query #[^\n]+:\n(\{[\s\S]*?\})\n", re.MULTILINE)
    for m in trapi_block_pattern.finditer(text):
        raw = m.group(1)
        try:
            trapi = json.loads(raw.replace("'", '"'))  # tolerate single quotes
        except Exception:
            # Try a more careful replace only for keys/strings
            try:
                trapi = eval(raw)  # last resort; trusted local code
            except Exception:
                continue
        qg = trapi.get("message", {}).get("query_graph", {})
        edges = qg.get("edges", {})
        e01 = edges.get("e01", {})
        preds = e01.get("predicates", [])
        summary["trapi_predicates"].append(preds)
        nodes = qg.get("nodes", {})
        summary["trapi_nodes"].append({k: v.get("categories", []) for k, v in nodes.items()})

    # Result counts lines: 'Query processed successfully, retrieved N results.'
    rc_pattern = re.compile(r"retrieved (\d+) results")
    summary["result_counts"] = [int(x) for x in rc_pattern.findall(text)]

    # Final answer: starts with 'To answer the question' then block until next delimiter
    fa_idx = text.find("To answer the question")
    if fa_idx != -1:
        summary["final_answer"] = text[fa_idx:].strip()

    return summary


def run_prototype(query: str) -> Dict[str, Any]:
    sys.path.insert(0, PROTOTYPE_PATH)
    from Agent import BTEx  # type: ignore

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            BTEx(query, maxresults=50, k=5)
        except Exception as e:
            print(f"[PROTO ERROR] {e}")
    out = buf.getvalue()
    return parse_prototype_output(out)

# ---------- Unified runner and parsers ----------

# Configure logging capture from unified modules
import logging

class BufferHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.buf = io.StringIO()
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        self.buf.write(msg + "\n")


def parse_unified_logs(text: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "subqueries": [],
        "trapi_predicates": [],
        "trapi_nodes": [],
        "result_counts": [],
    }
    # Subqueries: lines like "Node Q1: <content>..."
    node_line = re.compile(r"Node\s+([A-Za-z0-9_]+):\s+(.+?)\.\.\.")
    for m in node_line.finditer(text):
        summary["subqueries"].append(m.group(2).strip())

    # Robust JSON extraction after known markers by balancing braces
    def extract_json_blocks(src: str, markers: List[str]) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        n = len(src)
        for mk in markers:
            start = 0
            while True:
                idx = src.find(mk, start)
                if idx == -1:
                    break
                brace_idx = src.find("{", idx)
                if brace_idx == -1:
                    start = idx + len(mk)
                    continue
                i = brace_idx
                depth = 0
                in_str = False
                esc = False
                while i < n:
                    ch = src[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                    else:
                        if ch == '"':
                            in_str = True
                        elif ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                end = i + 1
                                raw = src[brace_idx:end]
                                obj = None
                                try:
                                    obj = json.loads(raw)
                                except Exception:
                                    try:
                                        obj = eval(raw, {}, {})
                                    except Exception:
                                        obj = None
                                if obj is not None:
                                    blocks.append(obj)
                                start = end
                                break
                    i += 1
                else:
                    # Unmatched braces; move past marker
                    start = idx + len(mk)
        return blocks

    json_blocks = extract_json_blocks(
        text,
        [
            "TRAPI query for node",
            "LLM TRAPI output:",
            "Full TRAPI query:",
        ],
    )

    for trapi in json_blocks:
        qg = trapi.get("message", {}).get("query_graph", {})
        edges = qg.get("edges", {})
        preds_agg: List[str] = []
        for e in edges.values():
            preds_agg.extend(e.get("predicates", []) or [])
        if preds_agg:
            summary["trapi_predicates"].append(preds_agg)
        nodes = qg.get("nodes", {})
        if nodes:
            summary["trapi_nodes"].append({k: v.get("categories", []) for k, v in nodes.items()})

    # Result counts: "TRAPI response: X KG nodes, Y KG edges, Z results"
    rc_pattern = re.compile(r"TRAPI response: \d+ KG nodes, \d+ KG edges, (\d+) results")
    summary["result_counts"] = [int(x) for x in rc_pattern.findall(text)]

    return summary


async def run_unified_async(query: str) -> Tuple[Dict[str, Any], str]:
    # Import here to ensure path and environment are in place
    sys.path.insert(0, UNIFIED_ROOT)
    from agentic_bte.unified.agent import UnifiedBiomedicalAgent  # type: ignore
    from agentic_bte.unified.agent import QueryMode  # type: ignore

    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    buf_handler = BufferHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    buf_handler.setFormatter(formatter)

    # Attach only for our modules
    targets = [
        'agentic_bte.unified.got_planner',
        'agentic_bte.core.knowledge.trapi',
        'agentic_bte.core.knowledge.bte_client',
        'agentic_bte.unified.agent'
    ]
    attached_handlers: List[Tuple[logging.Logger, logging.Handler]] = []
    for name in targets:
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.addHandler(buf_handler)
        attached_handlers.append((lg, buf_handler))

    agent = UnifiedBiomedicalAgent()
    await agent.initialize()
    resp = await agent.process_query(text=query, query_mode=QueryMode.BALANCED, max_results=50)

    # Detach handlers
    for lg, h in attached_handlers:
        try:
            lg.removeHandler(h)
        except Exception:
            pass

    logs = buf_handler.buf.getvalue()
    parsed = parse_unified_logs(logs)
    final_answer = getattr(resp, 'final_answer', None)
    return parsed, final_answer or ""


def run_unified(query: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    final_answer = ""
    try:
        parsed, final_answer = asyncio.run(run_unified_async(query))
    except RuntimeError:
        # If event loop already running (e.g., in notebook), create a new loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        parsed, final_answer = loop.run_until_complete(run_unified_async(query))
        loop.close()
    parsed["final_answer"] = final_answer
    return parsed

# ---------- Compare runner ----------

def compare_and_report(queries: List[str]) -> str:
    rows = []
    report_md = [
        "# Prototype vs Unified Comparison Report\n",
        "This report summarizes subqueries, TRAPI predicates, results, and final answers for a small batch of queries.\n\n"
    ]

    for q in queries:
        prot = run_prototype(q)
        uni = run_unified(q)

        rows.append({
            "query": q,
            "prototype": prot,
            "unified": uni,
        })

        report_md.append(f"## Query\n{q}\n\n")
        # Subqueries
        report_md.append("### Subqueries\n")
        report_md.append(f"- Prototype: {json.dumps(prot.get('subqueries', []), ensure_ascii=False)}\n")
        report_md.append(f"- Unified: {json.dumps(uni.get('subqueries', []), ensure_ascii=False)}\n\n")

        # TRAPI predicates
        report_md.append("### TRAPI Predicates (per step in order encountered)\n")
        report_md.append(f"- Prototype: {json.dumps(prot.get('trapi_predicates', []), ensure_ascii=False)}\n")
        report_md.append(f"- Unified: {json.dumps(uni.get('trapi_predicates', []), ensure_ascii=False)}\n\n")

        # Result counts
        report_md.append("### Result Counts (per TRAPI call)\n")
        report_md.append(f"- Prototype: {prot.get('result_counts', [])}\n")
        report_md.append(f"- Unified: {uni.get('result_counts', [])}\n\n")

        # Final answers
        report_md.append("### Final Answers\n")
        pa = prot.get('final_answer') or ''
        ua = uni.get('final_answer') or ''
        report_md.append("- Prototype Final Answer:\n\n" + (pa[:1500] + ('...' if len(pa) > 1500 else '')) + "\n\n")
        report_md.append("- Unified Final Answer:\n\n" + (ua[:1500] + ('...' if len(ua) > 1500 else '')) + "\n\n")

        # Differences summary (basic heuristic)
        diffs = []
        if len(prot.get('result_counts', [])) != len(uni.get('result_counts', [])):
            diffs.append("Different number of TRAPI calls recorded")
        if prot.get('trapi_predicates') != uni.get('trapi_predicates'):
            diffs.append("Different predicates used in TRAPI edges")
        if bool(pa) != bool(ua):
            diffs.append("One system produced a final answer while the other did not")
        report_md.append("### Observed Differences\n")
        if diffs:
            for d in diffs:
                report_md.append(f"- {d}\n")
        else:
            report_md.append("- None major; sequences broadly align.\n")
        report_md.append("\n\n")

    # Save report
    out_path = os.path.join(UNIFIED_ROOT, "compare_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_md))

    return out_path


if __name__ == "__main__":
    path = compare_and_report(QUERIES)
    print(f"Wrote comparison to {path}")
