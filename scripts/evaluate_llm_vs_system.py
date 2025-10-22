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

import pandas as pd
import requests

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


def sri_name_resolver(name: str, is_bp: bool = False, k: int = 50) -> List[Dict[str, Any]]:
    base_url = "https://name-lookup.ci.transltr.io/lookup"
    params = {
        "string": name,
        "autocomplete": "true",
        "limit": str(k),
    }
    if is_bp:
        params["only_prefixes"] = "GO"
        params["biolink_type"] = "BiologicalProcess"
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


def agentic_bte_compute(question: str, max_items: int = 5) -> Dict[str, Any]:
    """Run Production GoT optimizer end-to-end (run_biomedical_query) and return final answer plus results-derived list."""
    cfg = ProductionConfig(show_debug=False, save_results=False, max_iterations=5, parallel_execution=True, max_concurrent=3)
    result, presentation = run_biomedical_query(question, cfg)
    final_answer_text = getattr(result, 'final_answer', '') or ''
    # Extract first available parsed results from execution steps
    results = []
    try:
        for step in getattr(result, 'execution_steps', []) or []:
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

    # Collect candidate drugs purely by ID prefix from results (fallback reference)
    id_to_name = {v: k for k, v in name_to_id.items() if v}
    drugs_from_results: List[str] = []
    for rel in results:
        subj = rel.get("subject_id") or rel.get("subject")
        obj = rel.get("object_id") or rel.get("object")
        def to_name(eid: Optional[str]) -> Optional[str]:
            if not eid:
                return None
            return id_to_name.get(eid)
        if subj and re.match(r"^(CHEBI:|DRUGBANK:|CHEMBL)", str(subj)):
            s_name = to_name(subj)
            if s_name:
                drugs_from_results.append(s_name)
        if obj and re.match(r"^(CHEBI:|DRUGBANK:|CHEMBL)", str(obj)):
            o_name = to_name(obj)
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
    Extract candidates from Agentic BTE final answer by reading all **...** spans.
    We intentionally ignore any heuristics and only trust explicit **tagging** to
    keep parity with the LLM baseline extraction and avoid false tokens.
    """
    if not text:
        return []
    ents = extract_marked_entities(text)
    return ents[:max_items]


def evaluate_pair(question: str, gt_drug_id: str, gt_drug_name: str, systems: Dict[str, Any]) -> Dict[str, Any]:
    per_system = {}
    # LLM-only baseline
    for sys_name, fn in systems.items():
        try:
            output = fn(question)
            answers = extract_marked_entities(output)
            best_ans_id = ""
            correct = False
            for ans in answers[:5]:
                candidates = sri_name_resolver(ans, is_bp=False, k=50)
                ans_id = pick_best_curie_by_score(candidates)
                if not ans_id:
                    continue
                if normalization_match(ans_id, gt_drug_id):
                    best_ans_id = ans_id
                    correct = True
                    break
                if not best_ans_id:
                    best_ans_id = ans_id
            per_system[sys_name] = {
                "raw_output": output,
                "answers": answers[:5],
                "picked_id": best_ans_id,
                "correct": correct,
            }
        except Exception as e:
            per_system[sys_name] = {
                "raw_output": str(e),
                "answers": [],
                "picked_id": "",
                "correct": False,
            }
    
    # Agentic_BTE (final answer + results-derived)
    try:
        agentic = agentic_bte_compute(question, max_items=5)
        final_text = agentic.get("final_answer", "")
        final_candidates = extract_candidates_from_final_answer(final_text, max_items=5)
        final_best = ""
        final_correct = False
        for ans in final_candidates:
            candidates = sri_name_resolver(ans, is_bp=False, k=50)
            ans_id = pick_best_curie_by_score(candidates)
            if not ans_id:
                continue
            if normalization_match(ans_id, gt_drug_id):
                final_best = ans_id
                final_correct = True
                break
            if not final_best:
                final_best = ans_id
        # results-derived selection retained for reference
        res_candidates = agentic.get("drugs_from_results", [])
        res_best = ""
        res_correct = False
        for ans in res_candidates:
            candidates = sri_name_resolver(ans, is_bp=False, k=50)
            ans_id = pick_best_curie_by_score(candidates)
            if not ans_id:
                continue
            if normalization_match(ans_id, gt_drug_id):
                res_best = ans_id
                res_correct = True
                break
            if not res_best:
                res_best = ans_id
        per_system["Agentic_BTE_final"] = {
            "raw_output": final_text,
            "answers": final_candidates,
            "picked_id": final_best,
            "correct": final_correct,
        }
        per_system["Agentic_BTE_results"] = {
            "raw_output": "\n".join([f"**{x}**" for x in res_candidates]),
            "answers": res_candidates,
            "picked_id": res_best,
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
        out_dir = os.path.join("logs", "evaluations")
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

    # Build questions and sample 20
    df = df.copy()
    df["question"] = (
        "Which drugs can treat "
        + df["disease_name"].astype(str)
        + " by targeting "
        + df["bp_name"].astype(str)
        + "? Enumerate 5 drugs and do not include anything else in your response. Each of your answer entities MUST be tagged with ** at the start AND end of the phrase (**....**), otherwise it will not be assessed"
    )

    sample_df = df.sample(n=5, random_state=42)

    systems: Dict[str, Any] = {}
    if (ChatOpenAI is not None):
        systems["LLM_only"] = lambda q: llm_only_answer(q)

    tallies = {k: 0 for k in [*systems.keys(), "Agentic_BTE_final"]}
    results_rows = []

    for i, row in enumerate(sample_df.itertuples(index=False), 1):
        question = getattr(row, "question")
        gt_id = getattr(row, "Drug_MeshID")  # ground-truth ID
        gt_name = getattr(row, "drug_name")

        print(f"\n{i}. Question: {question}")
        print(f"Ground truth: {gt_id} - {gt_name}")

        per_system = evaluate_pair(question, gt_id, gt_name, systems)

        for sys_name, data in per_system.items():
            if sys_name == "Agentic_BTE_results":
                continue  # debug only; not part of evaluation
            if data.get("correct"):
                tallies[sys_name] += 1
            picked = data.get("picked_id") or ""
            print(f"{sys_name} picked: {picked or 'N/A'} - {'True' if data.get('correct') else 'False'}")

        results_rows.append({
            "question": question,
            "ground_truth_id": gt_id,
            "ground_truth_name": gt_name,
            **{f"{k}_picked": v.get("picked_id", "") for k, v in per_system.items()},
            **{f"{k}_correct": v.get("correct", False) for k, v in per_system.items()},
            **{f"{k}_output": v.get("raw_output", "") for k, v in per_system.items()},
            **{f"{k}_answers_extracted": ", ".join(v.get("answers", [])) for k, v in per_system.items()},
        })
        print("-" * 40)

    n = len(sample_df)
    print("\n=== Summary ===")
    for sys_name in tallies:
        acc = tallies[sys_name] / n if n else 0.0
        print(f"{sys_name}: {tallies[sys_name]} / {n} = {acc:.3f}")

    # Save CSV log
    out_dir = os.path.join("logs", "evaluations")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"llm_vs_system_20_{timestamp}.csv")
    pd.DataFrame(results_rows).to_csv(out_path, index=False)
    print(f"Saved detailed results to {out_path}")


if __name__ == "__main__":
    main()
