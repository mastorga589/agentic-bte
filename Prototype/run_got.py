#!/usr/bin/env python3
"""
CLI runner for the stand-alone GoT planner.

Usage:
  python -m Prototype.run_got "Which drugs treat Crohn's disease?"
"""

import asyncio
import sys

from .got_planner import GoTPlanner


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python -m Prototype.run_got \"<biomedical question>\"")
        return 1
    query = argv[1]
    planner = GoTPlanner()
    answer = asyncio.run(planner.execute(query))
    print("\n=== FINAL ANSWER ===\n")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
