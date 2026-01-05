import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from agentic import coder, executor, planner, reporter, reviewer
from agentic.config import (
    MAX_REFLECTION_ROUNDS,
    KNOWLEDGE_MEMORY_PATH,
    SESSION_MEMORY_PATH,
)
from agentic.memory import JSONLMemory
from agentic.tools.loaders import infer_schema, load_file


@dataclass
class OrchestrationResult:
    ok: bool
    plan: Dict[str, Any]
    code: str
    result: Any
    feedback: str
    rounds_used: int
    report: Optional[str] = None


class AgenticOrchestrator:
    def __init__(self, session_memory_path: str = SESSION_MEMORY_PATH):
        self.session_memory = JSONLMemory(session_memory_path)
        self.knowledge_memory = JSONLMemory(KNOWLEDGE_MEMORY_PATH)

    def run(
        self,
        query: str,
        file_path: str,
        rounds: int = MAX_REFLECTION_ROUNDS,
        auto_eda: bool = False,
    ) -> OrchestrationResult:
        data = load_file(file_path)
        schema = infer_schema(data)

        last_plan: Optional[Dict[str, Any]] = None
        last_result: Any = None
        last_feedback = ""
        code = ""

        for i in range(1, rounds + 1):
            plan = (
                planner.plan_auto_eda(file_path)
                if auto_eda and i == 1
                else planner.plan_query(query, schema)
            )
            code = coder.generate_code(plan)

            ctx = executor.ExecutionContext()
            if hasattr(data, "copy"):
                ctx.df = data.copy()
            elif isinstance(data, str):
                ctx.text = data

            result = executor.execute_plan(plan, ctx)
            ok, feedback = reviewer.review_result(result)

            self.session_memory.append(
                {
                    "round": i,
                    "query": query,
                    "schema": schema,
                    "plan": plan,
                    "code": code,
                    "ok": ok,
                    "feedback": feedback,
                    "result_preview": str(result)[:400],
                }
            )

            if ok:
                report = reporter.render_report(result, query, schema)
                return OrchestrationResult(
                    ok=True,
                    plan=plan,
                    code=code,
                    result=result,
                    feedback=feedback,
                    rounds_used=i,
                    report=report,
                )

            last_plan, last_result, last_feedback = plan, result, feedback

        return OrchestrationResult(
            ok=False,
            plan=last_plan or {},
            code=code,
            result=last_result,
            feedback=last_feedback or "Failed after retries.",
            rounds_used=rounds,
            report=None,
        )


def run_cli():
    parser = argparse.ArgumentParser(description="Agentic offline EDA platform")
    parser.add_argument("query", help="Natural language request")
    parser.add_argument("file_path", help="Path to data file")
    parser.add_argument("--rounds", type=int, default=MAX_REFLECTION_ROUNDS)
    parser.add_argument("--auto-eda", action="store_true", help="Run autonomous EDA plan first")
    args = parser.parse_args()

    orchestrator = AgenticOrchestrator()
    res = orchestrator.run(args.query, args.file_path, rounds=args.rounds, auto_eda=args.auto_eda)

    print(f"Rounds used: {res.rounds_used}, ok={res.ok}")
    print("Plan:")
    print(json.dumps(res.plan, indent=2))
    print("Feedback:", res.feedback)
    print("Result preview:")
    print(str(res.result)[:1200])
    if res.report:
        print("\nReport:")
        print(res.report)
