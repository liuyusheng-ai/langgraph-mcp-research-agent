"""Evaluate the research agent with a real model and real web search on 30 questions.

    ollama pull qwen3.5:4b && ollama pull gemma3:4b && pip install ddgs
    python eval/run_real_eval.py                   # both configurations, writes eval/real/
    python eval/run_real_eval.py --agreement       # after filling the `human` column of spot_check.csv

Questions (eval/real/questions.json): 15 factual and 10 comparative questions with 3-5 reference
key points each, plus 5 unanswerable ones (future, private or non-existent information).

Configurations
- planner:    Planner splits the question into sub-queries, Researcher searches each, Writer cites.
- no_planner: ablation; the original question is the only search query.

Search results are cached by query in eval/real/search_cache.json, so both configurations see the
same results for the same query and reruns are reproducible.

Metrics (judge: a different model family from the agent, so the agent does not grade itself)
- key point recall: share of reference key points the report states (judge, one call per point;
  a refused answerable question scores 0)
- citation faithfulness: share of cited sentences whose cited sources support them (judge)
- citation coverage, unknown citations, tool success / effective result rates (rules)
- correct refusals on unanswerable questions and false refusals on answerable ones (rules)
- latency and tokens of the agent (judge calls excluded)
A random 20% of judge decisions go to spot_check.csv for a person to label; `--agreement`
reports how often the judge agrees with that person (`--column reviewer_claude` compares with
the review by a stronger model that is already filled in).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_agent import ResearchAgent  # noqa: E402
from research_agent.checkpoint import SQLiteCheckpointStore  # noqa: E402
from research_agent.llm import OllamaResearchModel  # noqa: E402
from research_agent.models import Evidence, EXECUTED_STATUSES  # noqa: E402
from research_agent.tools import MetaSearchTool, WebSearchTool  # noqa: E402

OUT = ROOT / "eval" / "real"
CONFIGS = ("planner", "no_planner")
MAX_CLAIMS = 8


class CachedSearch:
    """Wraps a search tool; results (including empty ones) are cached per query, errors are not."""

    def __init__(self, tool, cache_path: Path) -> None:
        self.tool, self.name, self.available = tool, tool.name, tool.available
        self.cache_path = cache_path
        self.cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        self.lock = threading.Lock()

    def search(self, query: str, limit: int = 3) -> list[Evidence]:
        key = f"{limit}:{query}"
        if key not in self.cache:
            results = self.tool.search(query, limit)
            with self.lock:
                self.cache[key] = [item.to_dict() for item in results]
                self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8")
        return [Evidence(**item) for item in self.cache[key]]


def yes(answer: str) -> bool:
    text = answer.strip().strip("“”\"'*。. ")
    if text.startswith("是"):
        return True
    if text.startswith("否") or text.startswith("不"):
        return False
    return "是" in text[:10] and "否" not in text[:10]


def report_body(report: str) -> str:
    lines = [line for line in report.split("## 参考来源")[0].splitlines() if not line.startswith("# ")]
    return "\n".join(lines).strip()


def cited_sentences(body: str) -> list[tuple[str, list[int]]]:
    sentences = re.split(r"(?<=[。！？!?])\s*|\n+", body)
    out = []
    for sentence in sentences:
        numbers = [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]
        text = re.sub(r"\[\d+\]", "", sentence).strip(" -*")
        if numbers and len(text) >= 6:
            out.append((text, sorted(set(numbers))))
    return out


def judge_key_point(judge: OllamaResearchModel, body: str, point: str) -> bool:
    prompt = (
        "下面是一份研究报告和一个要点。判断报告是否表达了这个要点的意思（措辞可以不同，但意思必须一致）。"
        "只回答“是”或“否”。\n\n"
        f"报告：\n{body}\n\n要点：{point}"
    )
    return yes(judge.chat("你是严格的评测员。", prompt, role="judge"))


def judge_claim(judge: OllamaResearchModel, sentence: str, sources: str) -> bool:
    prompt = (
        "下面是一句话和它引用的资料。判断资料是否支持这句话的主要内容；资料中没有出现的信息算不支持。"
        "只回答“是”或“否”。\n\n"
        f"句子：{sentence}\n\n资料：\n{sources}"
    )
    return yes(judge.chat("你是严格的事实核查员。", prompt, role="judge"))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered) + 0.5) - 1))
    return round(ordered[index], 1)


def run_config(name: str, questions: list[dict], search, args, judge: OllamaResearchModel, workdir: Path):
    model = OllamaResearchModel(model=args.model, plan_queries=name == "planner")
    agent = ResearchAgent(
        toolbox=type("Toolbox", (), {"tools": [search]})(),
        model=model,
        checkpoints=SQLiteCheckpointStore(str(workdir / f"{name}.db")),
        timeout_seconds=args.tool_timeout,
    )
    rows, judgements = [], []
    for number, q in enumerate(questions, 1):
        calls_before = len(model.calls)
        started = time.perf_counter()
        result = agent.run(q["question"])
        elapsed = (time.perf_counter() - started) * 1000
        calls = model.calls[calls_before:]
        metrics = result["evaluation"]
        body = report_body(result["report"])
        refused = result["status"] == "insufficient_evidence"
        row = {
            "id": q["id"], "type": q["type"], "question": q["question"], "status": result["status"],
            "refused": refused, "plan": result.get("plan", []), "report": result["report"],
            "evidence_count": metrics["evidence_count"], "citation_coverage": metrics["citation_coverage"],
            "unknown_citations": metrics["unknown_citations"],
            "tool_calls": [{k: c[k] for k in ("query", "status", "result_count", "elapsed_ms")} for c in result["tool_calls"]],
            "latency_ms": round(elapsed, 1),
            "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
            "completion_tokens": sum(c["completion_tokens"] for c in calls),
            "key_points": [], "claims": [],
        }
        if q["type"] != "unanswerable":
            for point in q["key_points"]:
                covered = False if refused else judge_key_point(judge, body, point)
                row["key_points"].append({"point": point, "covered": covered})
                if not refused:
                    judgements.append({"config": name, "id": q["id"], "kind": "key_point", "item": point,
                                       "context": body, "judge": "是" if covered else "否"})
            if not refused:
                evidence = result.get("evidence", [])
                for sentence, numbers in cited_sentences(body)[:MAX_CLAIMS]:
                    valid = [n for n in numbers if 1 <= n <= len(evidence)]
                    sources = "\n".join(f"[{n}] {evidence[n - 1]['content']}" for n in valid)
                    supported = bool(valid) and judge_claim(judge, sentence, sources)
                    row["claims"].append({"sentence": sentence, "citations": numbers, "supported": supported})
                    judgements.append({"config": name, "id": q["id"], "kind": "claim", "item": sentence,
                                       "context": sources, "judge": "是" if supported else "否"})
        rows.append(row)
        recall = (sum(k["covered"] for k in row["key_points"]) / len(row["key_points"])) if row["key_points"] else None
        faithful = (sum(c["supported"] for c in row["claims"]) / len(row["claims"])) if row["claims"] else None
        print(f"[{name} {number}/{len(questions)}] {q['id']} {result['status']:<21} "
              f"recall={'-' if recall is None else f'{recall:.0%}'} "
              f"faithful={'-' if faithful is None else f'{faithful:.0%}'} {elapsed / 1000:.1f}s", flush=True)
    return rows, judgements, model.calls


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    completed = [r for r in answerable if r["status"] == "completed"]
    points = [k for r in answerable for k in r["key_points"]]
    claims = [c for r in completed for c in r["claims"]]
    executed = [c for r in rows for c in r["tool_calls"] if c["status"] in EXECUTED_STATUSES]

    def by_type(kind: str) -> float:
        pts = [k for r in answerable if r["type"] == kind for k in r["key_points"]]
        return round(100 * sum(k["covered"] for k in pts) / len(pts), 1) if pts else 0.0

    return {
        "answerable": len(answerable),
        "completed": len(completed),
        "false_refusals": sum(r["refused"] for r in answerable),
        "key_point_recall": round(100 * sum(k["covered"] for k in points) / len(points), 1) if points else 0.0,
        "key_point_recall_factual": by_type("factual"),
        "key_point_recall_comparative": by_type("comparative"),
        "claims": len(claims),
        "citation_faithfulness": round(100 * sum(c["supported"] for c in claims) / len(claims), 1) if claims else 0.0,
        "citation_coverage": round(100 * sum(r["citation_coverage"] for r in completed) / len(completed), 1) if completed else 0.0,
        "unknown_citation_reports": sum(bool(r["unknown_citations"]) for r in completed),
        "unanswerable": len(unanswerable),
        "correct_refusals": sum(r["refused"] for r in unanswerable),
        "tool_calls": len(executed),
        "tool_success_rate": round(100 * sum(c["status"] in {"success", "empty"} for c in executed) / len(executed), 1) if executed else 0.0,
        "effective_result_rate": round(100 * sum(c["status"] == "success" for c in executed) / len(executed), 1) if executed else 0.0,
        "avg_evidence": round(sum(r["evidence_count"] for r in rows) / len(rows), 1),
        "latency_p50_s": round(percentile([r["latency_ms"] for r in rows], 50) / 1000, 1),
        "latency_p95_s": round(percentile([r["latency_ms"] for r in rows], 95) / 1000, 1),
        "avg_tokens": round(sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows) / len(rows)),
    }


def render_markdown(meta: dict, summaries: dict[str, dict], rows: dict[str, list[dict]]) -> str:
    names = {"planner": "Planner + Researcher + Writer", "no_planner": "去掉 Planner（只用原问题检索）"}
    metrics = [
        ("要点召回率（25 道可回答题）", lambda s: f"{s['key_point_recall']}%"),
        ("　事实题 / 对比题", lambda s: f"{s['key_point_recall_factual']}% / {s['key_point_recall_comparative']}%"),
        ("引用忠实度（被引资料支持该句）", lambda s: f"{s['citation_faithfulness']}%（{s['claims']} 句）"),
        ("引用覆盖率", lambda s: f"{s['citation_coverage']}%"),
        ("出现不存在的引用编号", lambda s: f"{s['unknown_citation_reports']} 份"),
        ("不可回答题正确拒答", lambda s: f"{s['correct_refusals']}/{s['unanswerable']}"),
        ("可回答题误拒答", lambda s: f"{s['false_refusals']}/{s['answerable']}"),
        ("工具调用成功率 / 有效结果率", lambda s: f"{s['tool_success_rate']}% / {s['effective_result_rate']}%（{s['tool_calls']} 次）"),
        ("平均证据数", lambda s: f"{s['avg_evidence']}"),
        ("延迟 P50 / P95", lambda s: f"{s['latency_p50_s']} s / {s['latency_p95_s']} s"),
        ("平均 token / 题", lambda s: f"{s['avg_tokens']}"),
    ]
    lines = [
        "# 真实模型评测结果",
        "",
        f"由 `python eval/run_real_eval.py` 生成。Agent 模型 `{meta['model']}`，裁判模型 `{meta['judge']}`"
        "（不同模型家族，避免自评），均为 Ollama 本地运行、temperature 0。"
        f"网页搜索：{meta['search']}，结果按查询缓存在 `search_cache.json`，两组配置对同一查询看到相同结果。",
        "",
        "| 指标 | " + " | ".join(names[c] for c in summaries) + " |",
        "| --- | " + " | ".join("---" for _ in summaries) + " |",
    ]
    for label, fmt in metrics:
        lines.append(f"| {label} | " + " | ".join(fmt(s) for s in summaries.values()) + " |")
    lines += [
        "",
        "要点召回率由裁判模型逐条判断报告是否表达了标准要点；被拒答的可回答题按 0 计。"
        "引用忠实度对每份报告最多取 8 个带引用的句子，判断所引资料是否支持该句。"
        f"随机 20% 的裁判结果（{meta['spot_check']} 条）导出到 `spot_check.csv` 供人工复核，"
        "填写 `human` 列后运行 `--agreement` 计算一致率。",
        "",
    ]
    for config, config_rows in rows.items():
        lines += [f"## 逐题明细：{names[config]}", "", "| ID | 状态 | 要点召回 | 引用忠实 | 证据数 | 耗时 s |", "| --- | --- | --- | --- | --- | --- |"]
        for r in config_rows:
            kp = f"{sum(k['covered'] for k in r['key_points'])}/{len(r['key_points'])}" if r["key_points"] else "-"
            cl = f"{sum(c['supported'] for c in r['claims'])}/{len(r['claims'])}" if r["claims"] else "-"
            lines.append(f"| {r['id']} | {r['status']} | {kp} | {cl} | {r['evidence_count']} | {r['latency_ms'] / 1000:.1f} |")
        lines.append("")
    return "\n".join(lines)


def agreement(column: str = "human") -> int:
    """Agreement between the judge and a reviewer column of spot_check.csv (`human` by default)."""
    path = OUT / "spot_check.csv"
    with path.open(encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle) if (r.get(column) or "").strip() in {"是", "否"}]
    if not rows:
        print(f"{path} 的 {column} 列还没有填写（填“是”或“否”）")
        return 1
    agree = sum(r[column].strip() == r["judge"] for r in rows)
    print(f"裁判与 {column} 一致：{agree}/{len(rows)} = {100 * agree / len(rows):.1f}%")
    for kind in ("key_point", "claim"):
        part = [r for r in rows if r["kind"] == kind]
        if part:
            strict = sum(r["judge"] == "否" and r[column].strip() == "是" for r in part)
            lenient = sum(r["judge"] == "是" and r[column].strip() == "否" for r in part)
            same = sum(r[column].strip() == r["judge"] for r in part)
            print(f"  {kind}: {same}/{len(part)}（裁判判否而复核为是 {strict} 条，裁判判是而复核为否 {lenient} 条）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--judge", default="gemma3:4b")
    parser.add_argument("--search", choices=("meta", "tavily"), default="meta")
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tool-timeout", type=float, default=20.0)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--agreement", action="store_true")
    parser.add_argument("--column", default="human", help="spot_check.csv column to compare with the judge")
    args = parser.parse_args()
    if args.agreement:
        return agreement(args.column)

    questions = json.loads((OUT / "questions.json").read_text(encoding="utf-8"))["questions"]
    if args.limit:
        questions = questions[: args.limit]
    tool = MetaSearchTool() if args.search == "meta" else WebSearchTool()
    if args.search == "tavily" and not tool.api_key:
        print("需要设置 TAVILY_API_KEY")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    search = CachedSearch(tool, args.out / f"search_cache_{args.search}.json" if args.search != "meta" else args.out / "search_cache.json")
    judge = OllamaResearchModel(model=args.judge)

    all_rows, all_judgements, summaries = {}, [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for config in args.configs:
            rows, judgements, _ = run_config(config, questions, search, args, judge, Path(tmp))
            all_rows[config], summaries[config] = rows, summarize(rows)
            all_judgements += judgements

    sample = random.Random(42).sample(all_judgements, max(1, len(all_judgements) // 5)) if all_judgements else []
    with (args.out / "spot_check.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config", "id", "kind", "item", "context", "judge", "human"])
        writer.writeheader()
        for row in sample:
            writer.writerow({**row, "human": ""})

    meta = {
        "model": args.model, "judge": args.judge, "questions": len(questions), "spot_check": len(sample),
        "search": "ddgs 免 Key 元搜索（中文走 Bing）" if args.search == "meta" else "Tavily",
    }
    (args.out / "results.json").write_text(
        json.dumps({"meta": meta, "summary": summaries, "rows": all_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out / "RESULTS.md").write_text(render_markdown(meta, summaries, all_rows), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
