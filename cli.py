"""Command-line demo for the research Agent."""

from __future__ import annotations

import argparse
import json

from agent import ResearchAgent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="需要研究的问题")
    parser.add_argument("--run-id", help="复用的运行 ID")
    parser.add_argument("--resume", action="store_true", help="读取已完成的检查点")
    args = parser.parse_args()

    result = ResearchAgent().run(args.query, args.run_id, args.resume)
    print(result["report"])
    print("\n--- Evaluation ---")
    print(json.dumps(result["evaluation"], ensure_ascii=False, indent=2))
    print(f"\nrun_id={result['run_id']}")


if __name__ == "__main__":
    main()

