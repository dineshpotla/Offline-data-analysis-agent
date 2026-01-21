import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from agentic import coder, executor, planner, reporter, reviewer
from agentic.config import (
    MAX_REFLECTION_ROUNDS,
    KNOWLEDGE_MEMORY_PATH,
    PROFILE_CACHE_PATH,
    SESSION_MEMORY_PATH,
)
from agentic.memory import JSONLMemory, load_profile_cache, save_profile_cache
from agentic.tools.loaders import file_hash, infer_schema, load_file


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
        self.profile_cache = load_profile_cache(PROFILE_CACHE_PATH)

    def _build_memory_context(self, limit: int = 8) -> str:
        """Build a short conversation context from session memory."""
        items = self.session_memory.read_last(limit)
        lines = []
        for item in items:
            if item.get("type") == "message":
                role = item.get("role", "user")
                content = item.get("content", "")
                lines.append(f"{role}: {content}")
        return "\n".join(lines).strip()

    def run(
        self,
        query: str,
        file_path: str,
        rounds: int = MAX_REFLECTION_ROUNDS,
        auto_eda: bool = False,
    ) -> OrchestrationResult:
        data = load_file(file_path)
        schema = infer_schema(data)
        try:
            h = file_hash(file_path)
            if h not in self.profile_cache:
                self.profile_cache[h] = {"schema": schema, "path": file_path}
                save_profile_cache(PROFILE_CACHE_PATH, self.profile_cache)
        except Exception:
            pass

        last_plan: Optional[Dict[str, Any]] = None
        last_result: Any = None
        last_feedback = ""
        code = ""

        memory_context = self._build_memory_context()
        self.session_memory.append({"type": "message", "role": "user", "content": query})

        for i in range(1, rounds + 1):
            plan = (
                planner.plan_auto_eda(file_path)
                if auto_eda and i == 1
                else planner.plan_query(query, schema, memory_context=memory_context)
            )
            # Safety: force intent plan for count/columns/nulls queries
            intent_plan = planner._intent_plan(query, schema=schema)
            if intent_plan:
                plan = intent_plan
            # Fill in file_path for any intent-based plan that leaves it blank
            for step in plan.get("steps", []):
                if step.get("action") == "load_file":
                    args = step.setdefault("args", {})
                    if not args.get("path"):
                        args["path"] = file_path
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
                self.session_memory.append(
                    {"type": "message", "role": "assistant", "content": report}
                )
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
    parser.add_argument("query", nargs="?", help="Natural language request")
    parser.add_argument("file_path", nargs="?", help="Path to data file")
    parser.add_argument("--file", dest="file_path_opt", help="Path to data file")
    parser.add_argument("--rounds", type=int, default=MAX_REFLECTION_ROUNDS)
    parser.add_argument("--auto-eda", action="store_true", help="Run autonomous EDA plan first")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat loop")
    args = parser.parse_args()

    orchestrator = AgenticOrchestrator()
    file_path = args.file_path_opt or args.file_path
    if not file_path:
        raise ValueError("file_path is required (use positional or --file)")
    if args.chat:
        print("Interactive mode. Type 'exit' to quit.")
        auto_eda_pending = True
        while True:
            if auto_eda_pending and args.query:
                query = args.query
            else:
                query = input("> ").strip()
            if not query or query.lower() in {"exit", "quit"}:
                break
            res = orchestrator.run(
                query,
                file_path,
                rounds=args.rounds,
                auto_eda=args.auto_eda and auto_eda_pending,
            )
            auto_eda_pending = False
            print(f"Rounds used: {res.rounds_used}, ok={res.ok}")
            print("Plan:")
            print(json.dumps(res.plan, indent=2))
            print("Feedback:", res.feedback)
            print("Result preview:")
            print(str(res.result)[:1200])
            if res.report:
                print("\nReport:")
                print(res.report)
    else:
        if not args.query:
            raise ValueError("query is required unless --chat is set")
        res = orchestrator.run(args.query, file_path, rounds=args.rounds, auto_eda=args.auto_eda)

        print(f"Rounds used: {res.rounds_used}, ok={res.ok}")
        print("Plan:")
        print(json.dumps(res.plan, indent=2))
        print("Feedback:", res.feedback)
        print("Result preview:")
        print(str(res.result)[:1200])
        if res.report:
            print("\nReport:")
            print(res.report)
