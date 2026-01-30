import time
import logging
import asyncio
from agentic_bte.core.queries.production_got_optimizer import ProductionGoTOptimizer, ProductionConfig

# Configure logging: show optimizer info/warnings, suppress very noisy client debug
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logging.getLogger("agentic_bte.core.knowledge.bte_client").setLevel(logging.WARNING)

q = (
    "Which drugs can treat type 2 diabetes by targeting AMPK signaling? "
    "Enumerate 5 drugs and do not include anything else in your response. "
    "Each of your answer entities MUST be tagged with ** at the start AND end of the phrase (**....**), otherwise it will not be assessed"
)

async def main():
    cfg = ProductionConfig(
        show_debug=False,
        save_results=False,
        enable_parallel_predicates=True,
        max_predicates_per_subquery=2,
    )
    opt = ProductionGoTOptimizer(cfg)
    t0 = time.time()
    result, _ = await opt.execute_query(q)
    dt = time.time() - t0
    kept = getattr(opt, "subquery_info", [])

    print("\n=== META-KG FILTER RESULT ===")
    print(f"Kept subqueries: {len(kept)}")
    for i, sq in enumerate(kept, 1):
        print(f"  {i}. {sq.get('subject_category')} -> {sq.get('object_category')} | {sq.get('query')}")
    print(f"TOTAL_SECONDS={dt:.3f}")
    print(f"SUCCESS={result.success} STEPS={len(result.execution_steps)}")
    await opt.close()

if __name__ == "__main__":
    asyncio.run(main())
