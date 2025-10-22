#!/usr/bin/env python3
"""
CLI runner for the ported LangGraph Agent
Usage:
  python -m Prototype.run_agent "Which drugs treat Crohn's disease?"
"""
import sys
from .Agent import BTEx


def main(argv):
    if len(argv) < 2:
        print("Usage: python -m Prototype.run_agent \"<biomedical question>\"")
        return 1
    q = argv[1]
    ans = BTEx(q)
    print("\n=== FINAL ANSWER (LangGraph Agent) ===\n")
    print(ans)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
