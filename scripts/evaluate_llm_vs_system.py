#!/usr/bin/env python3
"""
Evaluate LLM-only vs current Agentic BTE system on 20 disease→drug pairs.

Replicates the methodology from the legacy notebook by:
- Building questions from disease_name + bp_name
- Getting answers from two systems (LLM-only, Agentic BTE)
- Extracting answers tagged with **...** (for LLM baseline), or formatting Agentic BTE results to that format
- Converting answer strings to IDs via SRI Name Resolver (top-scoring candidate)
- Verifying correctness via SRI Node Normalization ID intersection vs ground-truth drug ID

Outputs per-item diagnostics and final accuracy for both systems.
"""

from __future__ import annotations

import os
import sys
import re
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import argparse
from urllib.parse import quote

import pandas as pd
import requests
import logging
import subprocess
import glob

# Configure root logging for visibility of INFO-level module logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Local project imports
from agentic_bte.config.settings import get_settings
from agentic_bte.core.queries.production_got_optimizer import ProductionConfig, run_biomedical_query

# Optional LLM baseline using LangChain's OpenAI wrapper (already used in repo)
try:
    from langchain_openai import ChatOpenAI
except Exception as _:
    ChatOpenAI = None  # Will raise if actually used

DATASET_PATH = \
    "/Users/mastorga/Documents/BTE-LLM/Prototype/data/DMDB_go_bp_filtered_jjoy_05_08_2025.csv"

# ------------- Utilities -------------

def extract_marked_entities(text: str) -> List[str]:
    """Extract entities wrapped in **...** from a string."""
    if not text:
        return []
    return [m.strip() for m in re.findall(r"\*\*(.*?)\*\*", text)]


def extract_primary_answer_block(text: str) -> str:
    """Return only the text within the '## Primary Answer' section (if present)."""
    if not text:
        return ""
    try:
        # Match from '## Primary Answer' to the next heading starting with '## '
        m = re.search(r"##\s*Primary\s*Answer\s*(.*?)(?:\n##\s|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
        return (m.group(1) if m else text).strip()
    except Exception:
        return text


def sri_name_resolver(
    name: str,
    is_bp: bool = False,
    k: int = 50,
    biolink_type: Optional[str] = None,
    allowed_prefixes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    base_url = "https://name-lookup.ci.transltr.io/lookup"
    params = {
        "string": name,
        "autocomplete": "true",
        "limit": str(k),
    }
    if is_bp:
        params["only_prefixes"] = "GO"
        params["biolink_type"] = "BiologicalProcess"
    if biolink_type:
        params["biolink_type"] = biolink_type
    if allowed_prefixes:
        params["only_prefixes"] = ",".join(allowed_prefixes)
    try:
        r = requests.get(base_url, params=params, headers={"accept": "application/json"}, timeout=30)
        r.raise_for_status()
        items = r.json()
        out = []
        for it in items:
            out.append({
                "label": it.get("label", ""),
                "curie": it.get("curie", ""),
                "score": it.get("score", 0.0),
            })
        return out
    except Exception:
        return []


def pick_best_curie_by_score(candidates: List[Dict[str, Any]]) -> str:
    if not candidates:
        return ""
    best = max(candidates, key=lambda x: (x.get("score", 0.0), bool(x.get("curie"))))
    return str(best.get("curie") or "")


def node_normalize(curie: str) -> List[str]:
    """Return a list of normalized curies (equivalents) for a given curie via SRI Node Normalization."""
    if not curie:
        return []
    url = "https://nodenormalization-sri.renci.org/1.5/get_normalized_nodes"
    try:
        r = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "curies": [curie],
                "conflate": True,
                "description": True,
                "drug_chemical_conflate": False,
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        entry = data.get(curie)
        if not entry:
            return []
        out = set()
        main_id = ((entry.get("id") or {}).get("identifier"))
        if main_id:
            out.add(main_id)
        for equiv in entry.get("equivalent_identifiers", []):
            ident = equiv.get("identifier")
            if ident:
                out.add(ident.strip("[]'\" "))
        return list(out)
    except Exception:
        return []


def normalization_match(ans_curie: str, gt_curie: str) -> bool:
    try:
        ans_ids = set(node_normalize(ans_curie))
        gt_ids = set(node_normalize(gt_curie))
        return bool(ans_ids & gt_ids)
    except Exception:
        return False


def normalize_simple(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def simple_name_match(ans_name: str, gt_name: str) -> bool:
    a = normalize_simple(ans_name)
    g = normalize_simple(gt_name)
    if not a or not g:
        return False
    # Require a bit of length to avoid trivial matches
    if len(g) < 4 and len(a) < 4:
        return a == g
    return (g in a) or (a in g)


def looks_like_chemical_prefix(curie: str) -> bool:
    if not curie:
        return False
    if curie.startswith("MESH:D"):
        return False
    prefixes = (
        "CHEBI:", "DRUGBANK:", "ChEMBL", "UNII:", "PUBCHEM", "NCIT:", "MESH:C", "INCHIKEY:",
        "RXNORM:", "UMLS:", "ATC:", "CAS:", "KEGG.COMPOUND:" 
    )
    return any(curie.startswith(p) for p in prefixes)


# ------------- Systems -------------

def get_llm(model: Optional[str] = None, temperature: float = 0.0):
    if ChatOpenAI is None:
        raise RuntimeError("langchain_openai is not available; cannot run LLM baseline")
    settings = get_settings()
    return ChatOpenAI(temperature=temperature, model=model or settings.openai_model, api_key=settings.openai_api_key)


def llm_only_answer(question: str, model: Optional[str] = None) -> str:
    """LLM-only baseline: ask directly; require 5 items, only **...** responses."""
    llm = get_llm(model=model, temperature=0.0)
    prompt = (
        question
        + "\n\nRules: Return exactly 5 distinct drug names, one per line, each wrapped like **Drug**."
        + " Do not include numbering, explanations, or anything else."
    )
    try:
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        # Ensure only 5 marked entities; if not, try to coerce by extracting first 5 '**'
        ents = extract_marked_entities(text)
        if len(ents) >= 5:
            return "\n".join([f"**{e}**" for e in ents[:5]])
        # Fallback: take first 5 non-empty lines and wrap
        lines = [ln.strip(" -*\t") for ln in str(text).splitlines() if ln.strip()]
        lines = [ln for ln in lines if ln]
        return "\n".join([f"**{ln}**" for ln in lines[:5]])
    except Exception as e:
        return ""


def agentic_bte_compute(question: str, max_items: int = 5, out_dir: Optional[str] = None, idx: Optional[int] = None, no_parallel: bool = False) -> Dict[str, Any]:
    """Run Production GoT optimizer end-to-end (run_biomedical_query) and return final answer plus results-derived list.

    If out_dir is provided, persist TRAPI queries and a compact result summary per question.
    """
    cfg = ProductionConfig(
        show_debug=False,
        save_results=False,
        max_iterations=5,
        parallel_execution=(not no_parallel),
        max_concurrent=(1 if no_parallel else 3),
        enable_parallel_predicates=(not no_parallel),
        max_concurrent_predicate_calls=(1 if no_parallel else 3),
        max_predicates_per_subquery=2,
    )
    result, presentation = run_biomedical_query(question, cfg)
    final_answer_text = getattr(result, 'final_answer', '') or ''

    # Persist TRAPI queries and compact results if requested
    if out_dir:
        import os, json, time
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        qtag = f"q{idx}" if idx is not None else "q"
        # Collect TRAPI queries from execution steps
        trapis = []
        try:
            for step in getattr(result, 'execution_steps', []) or []:
                tq = getattr(step, 'trapi_query', None)
                if tq:
                    trapis.append({
                        'step_id': getattr(step, 'step_id', ''),
                        'step_type': getattr(step, 'step_type', ''),
                        'query_graph': (tq.get('message', {}) or {}).get('query_graph', {}) if isinstance(tq, dict) else tq
                    })
        except Exception:
            pass
        try:
            with open(os.path.join(out_dir, f"trapi_queries_{qtag}_{ts}.json"), 'w') as f:
                json.dump({'question': question, 'trapi_queries': trapis}, f, indent=2)
        except Exception:
            pass
        # Save a compact result summary (without huge blobs)
        try:
            compact_steps = []
            for step in getattr(result, 'execution_steps', []) or []:
                compact_steps.append({
                    'step_id': getattr(step, 'step_id', ''),
                    'step_type': getattr(step, 'step_type', ''),
                    'execution_time': getattr(step, 'execution_time', 0.0),
                    'success': getattr(step, 'success', False),
                    'confidence': getattr(step, 'confidence', 0.0),
                    'has_trapi': bool(getattr(step, 'trapi_query', None))
                })
            with open(os.path.join(out_dir, f"compact_result_{qtag}_{ts}.json"), 'w') as f:
                json.dump({'question': question, 'steps': compact_steps, 'final_result_count': len((getattr(result, 'execution_steps', []) or []))}, f, indent=2)
            with open(os.path.join(out_dir, f"presentation_{qtag}_{ts}.txt"), 'w') as f:
                f.write(presentation or '')
        except Exception:
            pass

    # Prefer aggregated final_results; fallback to last successful api_execution results
    results: List[Dict[str, Any]] = []
    try:
        steps = list(getattr(result, 'execution_steps', []) or [])
        # Search for aggregation step
        for step in reversed(steps):
            if getattr(step, 'step_type', '') == 'aggregation' and getattr(step, 'success', False):
                results = (getattr(step, 'output_data', {}) or {}).get('final_results', []) or []
                if results:
                    break
        if not results:
            for step in reversed(steps):
                if getattr(step, 'step_type', '') == 'api_execution' and getattr(step, 'success', False):
                    results = (getattr(step, 'output_data', {}) or {}).get('results', []) or []
                    if results:
                        break
    except Exception:
        results = []

    # Build an ID->name mapping if results include names/ids
    name_to_id: Dict[str, str] = {}
    for rel in results:
        sid = rel.get('subject_id'); sname = rel.get('subject')
        oid = rel.get('object_id'); oname = rel.get('object')
        if sid and sname: name_to_id[sname] = sid
        if oid and oname: name_to_id[oname] = oid

    # Collect candidate drugs from results using both category and prefix filters
    id_to_name = {v: k for k, v in name_to_id.items() if v}
    drugs_from_results: List[str] = []
    allowed_prefix = re.compile(r"^(CHEBI:|DRUGBANK:|ChEMBL|UNII:|PUBCHEM|NCIT:|INCHIKEY:)")
    for rel in results:
        subj_id = rel.get("subject_id")
        obj_id = rel.get("object_id")
        subj_type = (rel.get("subject_type") or "").lower()
        obj_type = (rel.get("object_type") or "").lower()
        
        def to_name(eid: Optional[str]) -> Optional[str]:
            if not eid:
                return None
            return id_to_name.get(eid)
        # Subject as drug
        if subj_id and allowed_prefix.match(str(subj_id)):
            if 'smallmolecule' in subj_type or 'chemical' in subj_type or not subj_type:
                s_name = to_name(subj_id)
                if s_name:
                    drugs_from_results.append(s_name)
        # Object as drug
        if obj_id and allowed_prefix.match(str(obj_id)):
            if 'smallmolecule' in obj_type or 'chemical' in obj_type or not obj_type:
                o_name = to_name(obj_id)
                if o_name:
                    drugs_from_results.append(o_name)
        if len(drugs_from_results) >= max_items:
            break

    seen = set(); uniq_results: List[str] = []
    for d in drugs_from_results:
        if d and d not in seen:
            seen.add(d)
            uniq_results.append(d)
        if len(uniq_results) >= max_items:
            break

    return {
        "final_answer": final_answer_text,
        "drugs_from_results": uniq_results,
        "entity_mappings": name_to_id,
        "results": results,
    }


# ------------- Evaluation -------------

def extract_candidates_from_final_answer(text: str, max_items: int = 5) -> List[str]:
    """
    Extract candidates strictly from the 'Primary Answer' section only.
    Acceptance rules:
    - Only accept lines that are exactly of the form '**Name**' (no extra text on the line)
    - Reject tokens containing colons ':' or parentheses '()' within the '**...**'
    - Ignore bolded section headings or instructions
    """
    if not text:
        return []
    block = extract_primary_answer_block(text)
    if not block:
        return []
    out: List[str] = []
    for raw_line in str(block).splitlines():
        line = raw_line.strip()
        # Must be exactly a single '**...**' token on the line
        m = re.fullmatch(r"\*\*([^*]+)\*\*", line)
        if not m:
            continue
        token = m.group(1).strip()
        # Exclude headings/phrases and tokens with colon/parentheses
        bad = token.lower() in {"primary answer", "query execution plan", "key evidence by subquery", "scientific context", "quality transparency"}
        if bad or (":" in token) or ("(" in token) or (")" in token):
            continue
        out.append(token)
        if len(out) >= max_items:
            break
    return out


def evaluate_pair(question: str, gt_drug_id: str, gt_drug_name: str, systems: Dict[str, Any], persist_dir: Optional[str] = None, idx: Optional[int] = None, no_parallel: bool = False) -> Dict[str, Any]:
    per_system = {}

    # Precompute normalized GT IDs; if empty, allow name-based fallback matching
    gt_ids = set(node_normalize(gt_drug_id)) if gt_drug_id else set()

    # LLM-only baseline
    for sys_name, fn in systems.items():
        try:
            output = fn(question)
            answers = extract_marked_entities(output)
            best_ans_id = ""
            best_ans_name = ""
            correct = False
            for ans in answers[:5]:
                candidates = sri_name_resolver(ans, is_bp=False, k=50)
                ans_id = pick_best_curie_by_score(candidates)
                if not ans_id:
                    # If ID resolution fails, try simple name contains as secondary check
                    if simple_name_match(ans, gt_drug_name):
                        best_ans_name = best_ans_name or ans
                        correct = True
                        break
                    continue
                ans_ids = set(node_normalize(ans_id))
                if gt_ids and ans_ids and (ans_ids & gt_ids):
                    best_ans_id = ans_id
                    best_ans_name = ans
                    correct = True
                    break
                # Node normalization path failed or ambiguous; try simple character search fallback
                if simple_name_match(ans, gt_drug_name):
                    if not best_ans_id:
                        best_ans_id = ans_id
                    best_ans_name = best_ans_name or ans
                    correct = True
                    break
                if not best_ans_id:
                    best_ans_id = ans_id
                    best_ans_name = best_ans_name or ans
            per_system[sys_name] = {
                "raw_output": output,
                "answers": answers[:5],
                "picked_id": best_ans_id,
                "picked_name": best_ans_name,
                "correct": correct,
            }
        except Exception as e:
            per_system[sys_name] = {
                "raw_output": str(e),
                "answers": [],
                "picked_id": "",
                "picked_name": "",
                "correct": False,
            }
    
    # Agentic_BTE (final answer + results-derived)
    try:
        agentic = agentic_bte_compute(question, max_items=5, out_dir=persist_dir, idx=idx, no_parallel=no_parallel)
        final_text = agentic.get("final_answer", "")
        final_candidates = extract_candidates_from_final_answer(final_text, max_items=5)
        final_best = ""
        final_best_name = ""
        final_correct = False
        for ans in final_candidates:
            # Unfiltered lookup; filters were causing 0 results for valid drugs like Metformin
            candidates = sri_name_resolver(
                ans,
                is_bp=False,
                k=50
            )
            ans_id = pick_best_curie_by_score(candidates)
            if not ans_id:
                if simple_name_match(ans, gt_drug_name):
                    final_correct = True
                    break
                continue
            ans_ids = set(node_normalize(ans_id))
            if gt_ids and ans_ids and (ans_ids & gt_ids):
                final_best = ans_id
                final_best_name = ans
                final_correct = True
                break
            if simple_name_match(ans, gt_drug_name):
                if not final_best:
                    final_best = ans_id
                final_best_name = final_best_name or ans
                final_correct = True
                break
            if not final_best:
                final_best = ans_id
                final_best_name = final_best_name or ans
        # results-derived selection retained for reference
        res_candidates = agentic.get("drugs_from_results", [])
        res_best = ""
        res_best_name = ""
        res_correct = False
        for ans in res_candidates:
            # Unfiltered lookup; filters were causing 0 results for valid drugs like Metformin
            candidates = sri_name_resolver(
                ans,
                is_bp=False,
                k=50
            )
            ans_id = pick_best_curie_by_score(candidates)
            if not ans_id:
                if simple_name_match(ans, gt_drug_name):
                    res_correct = True
                    break
                continue
            ans_ids = set(node_normalize(ans_id))
            if gt_ids and ans_ids and (ans_ids & gt_ids):
                res_best = ans_id
                res_best_name = ans
                res_correct = True
                break
            if simple_name_match(ans, gt_drug_name):
                if not res_best:
                    res_best = ans_id
                res_best_name = res_best_name or ans
                res_correct = True
                break
            if not res_best:
                res_best = ans_id
                res_best_name = res_best_name or ans
        per_system["Agentic_BTE_final"] = {
            "raw_output": final_text,
            "answers": final_candidates,
            "picked_id": final_best,
            "picked_name": final_best_name,
            "correct": final_correct,
        }
        per_system["Agentic_BTE_results"] = {
            "raw_output": "\n".join([f"**{x}**" for x in res_candidates]),
            "answers": res_candidates,
            "picked_id": res_best,
            "picked_name": res_best_name,
            "correct": res_correct,
        }
    except Exception as e:
        per_system["Agentic_BTE_final"] = {
            "raw_output": str(e),
            "answers": [],
            "picked_id": "",
            "correct": False,
        }
        per_system["Agentic_BTE_results"] = {
            "raw_output": str(e),
            "answers": [],
            "picked_id": "",
            "correct": False,
        }
    return per_system


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM vs Agentic BTE")
    parser.add_argument("--single_question", type=str, default="", help="Run a single question string")
    parser.add_argument("--gt_id", type=str, default="", help="Ground-truth drug CURIE (e.g., MESH:C042705)")
    parser.add_argument("--gt_name", type=str, default="", help="Ground-truth drug name (for logging)")
    parser.add_argument("--skip_llm", action="store_true", help="Skip LLM baseline")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling (default: random)")
    parser.add_argument("--n", type=int, default=5, help="Number of pairs to evaluate (default: 5)")
    parser.add_argument("--outdir", type=str, default="logs/evaluations", help="Directory to store results/logs")
    parser.add_argument("--no_parallel", action="store_true", help="Disable parallel execution and predicate concurrency")
    parser.add_argument("--system_model", type=str, default="", help="OpenAI model for Agentic BTE system (overrides settings.openai_model)")
    parser.add_argument("--baseline_model", type=str, default="", help="OpenAI model for LLM-only baseline")
    parser.add_argument("--exclude_regex_disease", type=str, default="", help="Regex to exclude rows by disease_name (case-insensitive)")
    parser.add_argument("--exclude_regex_bp", type=str, default="", help="Regex to exclude rows by bp_name (case-insensitive)")
    parser.add_argument("--isolate_per_query", action="store_true", help="Run each query in a separate process for full isolation")
    args = parser.parse_args()
    # Single-question fast path
    if args.single_question:
        question = args.single_question.strip()
        if not question or not args.gt_id:
            print("--single_question requires --gt_id as ground truth")
            sys.exit(1)
        gt_id = args.gt_id.strip()
        gt_name = (args.gt_name or "").strip() or gt_id

        systems: Dict[str, Any] = {}
        if (ChatOpenAI is not None) and (not args.skip_llm):
            systems["LLM_only"] = lambda q: llm_only_answer(q)

        print(f"\nQuestion: {question}")
        print(f"Ground truth: {gt_id} - {gt_name}")
        per_system = evaluate_pair(question, gt_id, gt_name, systems)

        tallies = {k: 0 for k in [*systems.keys(), "Agentic_BTE_final"]}
        for sys_name, data in per_system.items():
            if sys_name == "Agentic_BTE_results":
                continue  # debug only; not part of evaluation
            if data.get("correct"):
                tallies[sys_name] += 1
            picked = data.get("picked_id") or ""
            print(f"{sys_name} picked: {picked or 'N/A'} - {'True' if data.get('correct') else 'False'}")

        # Save row
        out_dir = args.outdir or os.path.join("logs", "evaluations")
        os.makedirs(out_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"single_eval_{timestamp}.csv")
        row = {
            "question": question,
            "ground_truth_id": gt_id,
            "ground_truth_name": gt_name,
            **{f"{k}_picked": v.get("picked_id", "") for k, v in per_system.items()},
            **{f"{k}_correct": v.get("correct", False) for k, v in per_system.items()},
            **{f"{k}_output": v.get("raw_output", "") for k, v in per_system.items()},
            **{f"{k}_answers_extracted": ", ".join(v.get("answers", [])) for k, v in per_system.items()},
        }
        pd.DataFrame([row]).to_csv(out_path, index=False)
        print(f"Saved detailed results to {out_path}")
        return

    # Load data
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found: {DATASET_PATH}")
        sys.exit(1)
    df = pd.read_csv(DATASET_PATH)

    # Validate expected columns (from the notebook printout)
    expected_cols = {"drug_name", "Drug_MeshID", "disease_name", "bp_name"}
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print(f"Dataset missing columns: {missing}")
        print(f"Available columns: {list(df.columns)[:20]} ...")
        sys.exit(1)

    # Build questions and optionally filter out rows by regex
    df = df.copy()
    if args.exclude_regex_disease:
        df = df[~df["disease_name"].astype(str).str.contains(args.exclude_regex_disease, case=False, na=False)]
    if args.exclude_regex_bp:
        df = df[~df["bp_name"].astype(str).str.contains(args.exclude_regex_bp, case=False, na=False)]
    df["question"] = (
        "Which drugs can treat "
        + df["disease_name"].astype(str)
        + " by targeting "
        + df["bp_name"].astype(str)
        + "? Enumerate 5 drugs and do not include anything else in your response. Each of your answer entities MUST be tagged with ** at the start AND end of the phrase (**....**), otherwise it will not be assessed"
    )

    # Prepare output directory
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.outdir, f"llm_vs_system_{args.n}_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)

    # Random sampling each run unless a seed is provided
    if args.seed is not None:
        sample_df = df.sample(n=args.n, random_state=args.seed)
    else:
        sample_df = df.sample(n=args.n)

    # Optionally override the system model at runtime
    settings = get_settings()
    if args.system_model:
        settings.openai_model = args.system_model.strip()
        print(f"[INFO] Using system model: {settings.openai_model}")

    systems: Dict[str, Any] = {}
    if (ChatOpenAI is not None) and (not args.skip_llm):
        baseline_model = args.baseline_model.strip() or None
        systems["LLM_only"] = (lambda q, _m=baseline_model: llm_only_answer(q, model=_m))

    tallies = {k: 0 for k in [*systems.keys(), "Agentic_BTE_final"]}
    results_rows = []

    try:
        # Isolation mode: spawn a fresh process per query and aggregate outputs
        if args.isolate_per_query:
            run_ts = time.strftime("%Y%m%d_%H%M%S")
            base_run_dir = os.path.join(args.outdir, f"llm_vs_system_{args.n}_{run_ts}")
            os.makedirs(base_run_dir, exist_ok=True)
            for i, row in enumerate(sample_df.itertuples(index=False), 1):
                question = getattr(row, "question")
                raw_gt_id = getattr(row, "Drug_MeshID")
                gt_name = getattr(row, "drug_name")
                # Resolve GT ID as in non-isolated path
                gt_candidates = sri_name_resolver(
                    gt_name,
                    is_bp=False,
                    k=50,
                    biolink_type="ChemicalEntity",
                    allowed_prefixes=["CHEBI", "DRUGBANK", "ChEMBL", "UNII", "PUBCHEM", "NCIT", "INCHIKEY"]
                )
                name_based_gt = pick_best_curie_by_score(gt_candidates)
                if not name_based_gt:
                    broader = sri_name_resolver(gt_name, is_bp=False, k=50, biolink_type="ChemicalEntity")
                    name_based_gt = pick_best_curie_by_score(broader)
                if not name_based_gt:
                    unconstrained = sri_name_resolver(gt_name, is_bp=False, k=50)
                    chem_like = [c for c in unconstrained if looks_like_chemical_prefix(str(c.get("curie", "")))]
                    name_based_gt = pick_best_curie_by_score(chem_like) or pick_best_curie_by_score(unconstrained)
                def looks_like_chemical(curie: str) -> bool:
                    if not curie:
                        return False
                    if curie.startswith("MESH:D"):
                        return False
                    prefixes = ("CHEBI:", "DRUGBANK:", "ChEMBL", "UNII:", "PUBCHEM", "NCIT:", "MESH:C", "INCHIKEY:")
                    return any(curie.startswith(p) for p in prefixes)
                gt_id = raw_gt_id if looks_like_chemical(str(raw_gt_id)) else (name_based_gt or str(raw_gt_id))
                # Per-query outdir
                q_outdir = os.path.join(base_run_dir, f"q{i}")
                os.makedirs(q_outdir, exist_ok=True)
                # Build subprocess command
                cmd = [sys.executable, __file__, "--single_question", question, "--gt_id", str(gt_id), "--gt_name", str(gt_name), "--outdir", q_outdir]
                if args.no_parallel:
                    cmd.append("--no_parallel")
                if args.skip_llm:
                    cmd.append("--skip_llm")
                if args.system_model:
                    cmd.extend(["--system_model", args.system_model])
                if args.baseline_model:
                    cmd.extend(["--baseline_model", args.baseline_model])
                # Execute
                print("\n" + "="*70)
                print(f"### ISOLATED QUERY {i}/{args.n}")
                print("="*70)
                cp = subprocess.run(cmd, capture_output=True, text=True)
                # Find the single_eval CSV in q_outdir
                csvs = sorted(glob.glob(os.path.join(q_outdir, "single_eval_*.csv")))
                if not csvs:
                    print(f"[WARN] No single_eval CSV found in {q_outdir}. stderr: {cp.stderr[:200]}")
                    continue
                csv_path = csvs[-1]
                sdf = pd.read_csv(csv_path)
                r = sdf.iloc[0].to_dict()
                # Tally and print picks
                for sys_name in [*(systems.keys()), "Agentic_BTE_final"]:
                    picked = r.get(f"{sys_name}_picked", "")
                    ok = bool(r.get(f"{sys_name}_correct", False))
                    if sys_name in tallies and ok:
                        tallies[sys_name] += 1
                    print(f"{sys_name} picked: {picked or 'N/A'} - {'True' if ok else 'False'}")
                print(f"GT: {gt_name} ({gt_id})")
                print("-"*40)
                # Aggregate
                results_rows.append(r)
            # After loop, write aggregated outputs
            n = len(results_rows)
            print("\n=== Summary ===")
            for sys_name in tallies:
                acc = tallies[sys_name] / n if n else 0.0
                print(f"{sys_name}: {tallies[sys_name]} / {n} = {acc:.3f}")
            out_dir = base_run_dir
            timestamp = run_ts
            csv_path = os.path.join(out_dir, f"llm_vs_system_{args.n}_{timestamp}.csv")
            xlsx_path = os.path.join(out_dir, f"llm_vs_system_{args.n}_{timestamp}.xlsx")
            df_out = pd.DataFrame(results_rows)
            df_out.to_csv(csv_path, index=False)
            try:
                df_out.to_excel(xlsx_path, index=False)
                print(f"Saved detailed results to {csv_path} and {xlsx_path}")
            except Exception as e:
                print(f"Saved detailed results to {csv_path} (Excel failed: {e})")
            print("\n=== DETAILED COMPARISON (Names + IDs) ===")
            for r in results_rows:
                llm_name = r.get("LLM_only_picked_name", "")
                llm_id = r.get("LLM_only_picked", "")
                sys_name_p = r.get("Agentic_BTE_final_picked_name", "")
                sys_id = r.get("Agentic_BTE_final_picked", "")
                llm_ok = r.get("LLM_only_correct", False)
                sys_ok = r.get("Agentic_BTE_final_correct", False)
                print(f"GT: {r.get('ground_truth_name','')} ({r.get('ground_truth_id','')}) | LLM: {llm_name} ({llm_id}) -> {llm_ok} | System: {sys_name_p} ({sys_id}) -> {sys_ok}")
            return
        
        # Non-isolated path
        for i, row in enumerate(sample_df.itertuples(index=False), 1):
            question = getattr(row, "question")
            raw_gt_id = getattr(row, "Drug_MeshID")  # dataset field (may be inconsistent)
            gt_name = getattr(row, "drug_name")

            # Top marker for per-query logs
            print("\n" + "="*70)
            print(f"### QUERY {i}/{args.n}")
            print("="*70)

            # Determine a reliable ground-truth CURIE for the drug by resolving the name to chemical entities
            gt_candidates = sri_name_resolver(
                gt_name,
                is_bp=False,
                k=50,
                biolink_type="ChemicalEntity",
                allowed_prefixes=["CHEBI", "DRUGBANK", "ChEMBL", "UNII", "PUBCHEM", "NCIT", "INCHIKEY"]
            )
            name_based_gt = pick_best_curie_by_score(gt_candidates)

            # If name-based resolution fails, retry with broader settings and log
            if not name_based_gt:
                print("[WARN] GT chemical resolution failed with strict prefixes; retrying with ChemicalEntity only...")
                broader = sri_name_resolver(gt_name, is_bp=False, k=50, biolink_type="ChemicalEntity")
                name_based_gt = pick_best_curie_by_score(broader)
            if not name_based_gt:
                print("[WARN] GT chemical resolution still empty; retrying unconstrained and filtering by chemical-like prefixes...")
                unconstrained = sri_name_resolver(gt_name, is_bp=False, k=50)
                # pick first that looks like a chemical
                chem_like = [c for c in unconstrained if looks_like_chemical_prefix(str(c.get("curie", "")))]
                name_based_gt = pick_best_curie_by_score(chem_like) or pick_best_curie_by_score(unconstrained)

            # Prefer dataset GT if it looks like a chemical CURIE and not a disease (exclude MESH:D*)
            def looks_like_chemical(curie: str) -> bool:
                if not curie:
                    return False
                if curie.startswith("MESH:D"):
                    return False
                prefixes = ("CHEBI:", "DRUGBANK:", "ChEMBL", "UNII:", "PUBCHEM", "NCIT:", "MESH:C", "INCHIKEY:")
                return any(curie.startswith(p) for p in prefixes)

            gt_id = raw_gt_id if looks_like_chemical(str(raw_gt_id)) else (name_based_gt or str(raw_gt_id))

            if str(gt_id).startswith("MESH:D"):
                print(f"[WARN] Ground truth remains a disease CURIE ({gt_id}); using name-based fallback if needed.")

            print(f"\n{i}. Question: {question}")
            print(f"Ground truth: {gt_id} - {gt_name}")

            per_system = evaluate_pair(question, gt_id, gt_name, systems, persist_dir=run_dir, idx=i, no_parallel=args.no_parallel)
            # Artifacts are persisted during the main Agentic_BTE run inside evaluate_pair (no second run needed)

            for sys_name, data in per_system.items():
                if sys_name == "Agentic_BTE_results":
                    continue  # debug only; not part of evaluation
                if data.get("correct"):
                    tallies[sys_name] += 1
                picked = data.get("picked_id") or ""
                picked_name = data.get("picked_name") or ""
                print(f"{sys_name} picked: {picked_name or 'N/A'} ({picked or 'N/A'}) - {'True' if data.get('correct') else 'False'}")

            # Show ground truth line with names and IDs
            print(f"GT: {gt_name} ({gt_id})")
        
            # Bottom marker for per-query logs
            print("="*70 + "\n")

            results_rows.append({
                "question": question,
                "ground_truth_id": gt_id,
                "ground_truth_raw_id": raw_gt_id,
                "ground_truth_name": gt_name,
                "system_model_used": settings.openai_model,
                "baseline_model_used": baseline_model or settings.openai_model if 'baseline_model' in locals() else settings.openai_model,
                **{f"{k}_picked": v.get("picked_id", "") for k, v in per_system.items()},
                **{f"{k}_picked_name": v.get("picked_name", "") for k, v in per_system.items()},
                **{f"{k}_correct": v.get("correct", False) for k, v in per_system.items()},
                **{f"{k}_output": v.get("raw_output", "") for k, v in per_system.items()},
                **{f"{k}_answers_extracted": ", ".join(v.get("answers", [])) for k, v in per_system.items()},
            })
            print("-" * 40)
    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted by user (SIGINT). Saving partial results...")
    except Exception as e:
        print(f"\n[ERROR] Evaluation aborted due to unexpected error: {e}. Saving partial results...")

    n = len(results_rows)
    print("\n=== Summary ===")
    for sys_name in tallies:
        acc = tallies[sys_name] / n if n else 0.0
        print(f"{sys_name}: {tallies[sys_name]} / {n} = {acc:.3f}")

    # Save CSV/Excel log
    out_dir = run_dir
    os.makedirs(out_dir, exist_ok=True)
    timestamp = run_ts
    csv_path = os.path.join(out_dir, f"llm_vs_system_{args.n}_{timestamp}.csv")
    xlsx_path = os.path.join(out_dir, f"llm_vs_system_{args.n}_{timestamp}.xlsx")
    df_out = pd.DataFrame(results_rows)
    df_out.to_csv(csv_path, index=False)
    try:
        df_out.to_excel(xlsx_path, index=False)
        print(f"Saved detailed results to {csv_path} and {xlsx_path}")
    except Exception as e:
        print(f"Saved detailed results to {csv_path} (Excel failed: {e})")

    # Final detailed comparison with names and IDs
    print("\n=== DETAILED COMPARISON (Names + IDs) ===")
    for r in results_rows:
        llm_name = r.get("LLM_only_picked_name", "")
        llm_id = r.get("LLM_only_picked", "")
        sys_name = r.get("Agentic_BTE_final_picked_name", "")
        sys_id = r.get("Agentic_BTE_final_picked", "")
        llm_ok = r.get("LLM_only_correct", False)
        sys_ok = r.get("Agentic_BTE_final_correct", False)
        print(f"GT: {r.get('ground_truth_name','')} ({r.get('ground_truth_id','')}) | LLM: {llm_name} ({llm_id}) -> {llm_ok} | System: {sys_name} ({sys_id}) -> {sys_ok}")


if __name__ == "__main__":
    main()
