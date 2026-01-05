"""
Autonomous multi-agent EDA runner that uses the core offline_agent components.

Agents:
- Planner (Phi-4 Mini Q6 via phi4_llm) -> JSON plan
- Executor (pandas/duckdb) -> runs actions
- Verifier -> guards against empty/invalid outputs
- Reflector -> asks model to improve plan after failed verification
- Memory -> JSONL long-term store of attempts/results

Usage:
  python auto_eda.py "auto explore the dataset" data.csv --rounds 3
  python auto_eda.py "find null-heavy columns and describe distributions" data.parquet
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from offline_agent import (
    ExecutionContext,
    execute_plan,
    infer_schema,
    load_file,
    plan_with_phi4,
    phi4_llm,
    verify_result,
)


# ---------------------------------------------------------------------------
# Memory store (JSONL)
# ---------------------------------------------------------------------------


class MemoryStore:
    def __init__(self, path: str):
        self.path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True) if os.path.dirname(self.path) else None

    def append(self, event: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def last(self, n: int = 5) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines[-n:]]


# ---------------------------------------------------------------------------
# Reflection and replanning
# ---------------------------------------------------------------------------


def summarize_result(res: Any, max_len: int = 1200) -> str:
    """Compact summary of executor output for reflection prompt."""
    try:
        if hasattr(res, "to_markdown"):
            return res.to_markdown()
        text = str(res)
        return text[:max_len]
    except Exception:
        return "unserializable_result"


def replan_with_reflection(
    user_query: str,
    schema: str,
    last_plan: Dict[str, Any],
    last_result: Any,
    last_feedback: str,
) -> Dict[str, Any]:
    prompt = f"""
You are a self-reflective data-analysis planner.
User request: {user_query}
Dataset schema: {schema}

Previous plan (JSON):
{json.dumps(last_plan, indent=2)}

Previous outcome summary:
{summarize_result(last_result)}

Verifier feedback:
{last_feedback}

Propose a revised JSON plan with the same action schema as before.
Return JSON only.
"""
    raw = phi4_llm(prompt)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Auto EDA Loop
# ---------------------------------------------------------------------------


@dataclass
class AutoEDAResult:
    ok: bool
    plan: Dict[str, Any]
    result: Any
    rounds_used: int


def auto_eda(
    query: str,
    file_path: str,
    rounds: int = 3,
    memory_path: str = ".autoeda_memory.jsonl",
) -> AutoEDAResult:
    data = load_file(file_path)
    schema = infer_schema(data)
    memory = MemoryStore(memory_path)

    last_plan: Optional[Dict[str, Any]] = None
    last_result: Any = None
    last_feedback = ""

    for i in range(1, rounds + 1):
        if last_plan is None:
            plan = plan_with_phi4(query, schema)
        else:
            plan = replan_with_reflection(query, schema, last_plan, last_result, last_feedback)

        ctx = ExecutionContext()
        if hasattr(data, "copy"):
            ctx.df = data.copy()  # avoid mutations
        else:
            ctx.df = None
        if isinstance(data, str):
            ctx.text = data

        result = execute_plan(plan, ctx)
        ok = verify_result(result)

        feedback = "result_ok" if ok else "result_invalid_or_empty"
        memory.append(
            {
                "round": i,
                "query": query,
                "schema": schema,
                "plan": plan,
                "ok": ok,
                "feedback": feedback,
                "result_preview": summarize_result(result, max_len=400),
            }
        )

        if ok:
            return AutoEDAResult(ok=True, plan=plan, result=result, rounds_used=i)

        last_plan, last_result, last_feedback = plan, result, feedback

    return AutoEDAResult(ok=False, plan=last_plan or {}, result=last_result, rounds_used=rounds)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous multi-agent EDA")
    parser.add_argument("query", help="Natural language request")
    parser.add_argument("file_path", help="Path to data file")
    parser.add_argument("--rounds", type=int, default=3, help="Max reflection rounds")
    parser.add_argument(
        "--memory",
        default=".autoeda_memory.jsonl",
        help="Path to JSONL memory log (appends across runs)",
    )
    args = parser.parse_args()

    res = auto_eda(args.query, args.file_path, rounds=args.rounds, memory_path=args.memory)
    print(f"Rounds used: {res.rounds_used}, ok={res.ok}")
    print("Plan:")
    print(json.dumps(res.plan, indent=2))
    print("Result:")
    print(summarize_result(res.result))


if __name__ == "__main__":
    main()

