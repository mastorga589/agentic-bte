import os
import time
import json
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List

from agentic_bte.core.queries.production_got_optimizer import ProductionGoTOptimizer, ProductionConfig

# Configure concise logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logging.getLogger("agentic_bte.core.knowledge.bte_client").setLevel(logging.WARNING)

QUERIES: List[str] = [
    # Q1
    (
        "Which drugs can treat type 2 diabetes by targeting AMPK signaling? "
        "Enumerate 5 drugs and do not include anything else in your response. "
        "Each of your answer entities MUST be tagged with ** at the start AND end of the phrase (**....**), otherwise it will not be assessed"
    ),
    # Q2
    (
        "Which drugs can treat rheumatoid arthritis by modulating NF-kappaB signaling? "
        "Enumerate 5 drugs and do not include anything else in your response. "
        "Each of your answer entities MUST be tagged with ** at the start AND end of the phrase (**....**), otherwise it will not be assessed"
    ),
    # Q3
    (
        "Which drugs can treat asthma by targeting IL-4 signaling? "
        "Enumerate 5 drugs and do not include anything else in your response. "
        "Each of your answer entities MUST be tagged with ** at the start AND end of the phrase (**....**), otherwise it will not be assessed"
    ),
    # Q4
    (
        "Which drugs can treat breast cancer by targeting EGFR signaling? "
        "Enumerate 5 drugs and do not include anything else in your response. "
        "Each of your answer entities MUST be tagged with ** at the start AND end of the phrase (**....**), otherwise it will not be assessed"
    ),
    # Q5
    (
        "Which drugs can treat Parkinson's disease by targeting dopamine receptor signaling? "
        "Enumerate 5 drugs and do not include anything else in your response. "
        "Each of your answer entities MUST be tagged with ** at the start AND end of the phrase (**....**), otherwise it will not be assessed"
    ),
]

CFG = ProductionConfig(
    show_debug=False,
    save_results=False,
    enable_parallel_predicates=True,
    max_predicates_per_subquery=2,
)

async def run_one(query: str) -> Dict[str, Any]:
    opt = ProductionGoTOptimizer(CFG)
    t0 = time.time()
    result, presentation = await opt.execute_query(query)
    dt = time.time() - t0
    # Extract kept subqueries and a compact step summary
    kept = getattr(opt, "subquery_info", [])
    steps = []
    for s in getattr(result, 'execution_steps', []) or []:
        steps.append({
            'step_id': getattr(s, 'step_id', ''),
            'step_type': getattr(s, 'step_type', ''),
            'success': getattr(s, 'success', False),
            'execution_time': getattr(s, 'execution_time', 0.0),
            'predicate': ((getattr(s, 'output_data', {}) or {}).get('predicate_used')
                          or (getattr(s, 'input_data', {}) or {}).get('predicate'))
        })
    # Collect TRAPI query_graphs
    trapis = []
    for s in getattr(result, 'execution_steps', []) or []:
        tq = getattr(s, 'trapi_query', None)
        if tq and isinstance(tq, dict):
            qg = (tq.get('message', {}) or {}).get('query_graph', {})
        else:
            qg = tq
        if qg:
            trapis.append({'step_id': getattr(s, 'step_id', ''), 'step_type': getattr(s, 'step_type', ''), 'query_graph': qg})
    await opt.close()
    return {
        'question': query,
        'success': bool(getattr(result, 'success', False)),
        'total_seconds': dt,
        'kept_subqueries': kept,
        'steps': steps,
        'trapi_queries': trapis,
        'final_answer': getattr(result, 'final_answer', ''),
    }

async def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join('logs', 'demo_5_queries', ts)
    os.makedirs(outdir, exist_ok=True)

    all_rows = []
    for i, q in enumerate(QUERIES, 1):
        print(f"\n[{i}/5] Running: {q[:80]}...")
        data = await run_one(q)
        all_rows.append(data)
        # Save per-query JSON and presentation
        with open(os.path.join(outdir, f'query_{i}.json'), 'w') as f:
            json.dump(data, f, indent=2)
        with open(os.path.join(outdir, f'final_{i}.txt'), 'w') as f:
            f.write(data.get('final_answer', ''))
        print(f"  → success={data['success']} time={data['total_seconds']:.2f}s kept_subqueries={len(data['kept_subqueries'])}")

    # Summary
    summary = {
        'total_queries': len(all_rows),
        'successes': sum(1 for r in all_rows if r.get('success')),
        'avg_seconds': sum(r.get('total_seconds', 0.0) for r in all_rows) / max(1, len(all_rows)),
        'details': [{'i': i+1, 'seconds': r.get('total_seconds', 0.0)} for i, r in enumerate(all_rows)],
    }
    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    asyncio.run(main())
