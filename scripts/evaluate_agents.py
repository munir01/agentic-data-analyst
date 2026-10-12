#!/usr/bin/env python
"""
Evaluate agent_v1.py and agent_v2.py against eval/ground_truth.json.

Run from the project root:

    python scripts/evaluate_agents.py

Outputs:
- eval/evaluation_results.json
- eval/results_v1.json
- eval/results_v2.json
- eval/evaluation_summary.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import math
import numbers
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
EVAL_DIR = PROJECT_ROOT / "eval"
GROUND_TRUTH_PATH = EVAL_DIR / "ground_truth.json"

DEFAULT_AGENTS = ["agent_v1", "agent_v2"]
DEFAULT_PASS_THRESHOLD = 0.95
RESULT_PREVIEW_ROWS = 10

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]
    return repr(value)


def summarize_result(result: Any, max_rows: int = RESULT_PREVIEW_ROWS) -> Any:
    safe = make_json_safe(result)

    if isinstance(safe, dict) and "columns" in safe and "rows" in safe:
        rows = safe.get("rows") or []
        return {
            "columns": safe.get("columns"),
            "row_count": len(rows),
            "rows_preview": rows[:max_rows],
            "rows_truncated_in_log": len(rows) > max_rows,
        }

    if isinstance(safe, list) and len(safe) > max_rows:
        return {
            "item_count": len(safe),
            "items_preview": safe[:max_rows],
            "items_truncated_in_log": True,
        }

    return safe


def load_agent_module(agent_name: str):
    path = SRC_DIR / f"{agent_name}.py"

    if not path.exists():
        raise FileNotFoundError(f"Agent file not found: {path}")

    unique_name = f"_eval_{agent_name}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(unique_name, path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "ask_agent"):
        raise AttributeError(f"{path} must define ask_agent(question)")
    if not hasattr(module, "execute_tool"):
        raise AttributeError(f"{path} must define execute_tool(name, tool_input)")
    if not hasattr(module, "client"):
        raise AttributeError(f"{path} must define a Bedrock client named client")

    return module


class ConverseClientProxy:
    def __init__(self, client):
        self._client = client
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.bedrock_latency_ms = 0

    def converse(self, *args, **kwargs):
        self.calls += 1
        response = self._client.converse(*args, **kwargs)

        usage = response.get("usage", {}) or {}
        self.input_tokens += int(usage.get("inputTokens", 0) or 0)
        self.output_tokens += int(usage.get("outputTokens", 0) or 0)
        self.total_tokens += int(usage.get("totalTokens", 0) or 0)

        metrics = response.get("metrics", {}) or {}
        self.bedrock_latency_ms += int(metrics.get("latencyMs", 0) or 0)

        return response

    def __getattr__(self, name):
        return getattr(self._client, name)


def scalar_matches(expected: Any, candidate: Any) -> bool:
    if expected is None:
        return candidate is None

    if isinstance(expected, bool):
        return expected is candidate

    if isinstance(expected, numbers.Integral) and not isinstance(expected, bool):
        if isinstance(candidate, numbers.Integral) and not isinstance(candidate, bool):
            return int(expected) == int(candidate)
        if isinstance(candidate, numbers.Real) and math.isfinite(float(candidate)):
            return abs(float(candidate) - float(expected)) < 1e-9
        return False

    if isinstance(expected, numbers.Real) and not isinstance(expected, bool):
        if not isinstance(candidate, numbers.Real) or isinstance(candidate, bool):
            return False

        expected_f = float(expected)
        candidate_f = float(candidate)

        if not (math.isfinite(expected_f) and math.isfinite(candidate_f)):
            return expected_f == candidate_f

        tolerance = max(1e-3, abs(expected_f) * 1e-3)
        return abs(expected_f - candidate_f) <= tolerance

    return str(expected).strip().casefold() == str(candidate).strip().casefold()


def row_match_fraction(expected_values: list[Any], candidate_values: list[Any]) -> float:
    if not expected_values:
        return 1.0

    used = set()
    matched = 0

    for expected in expected_values:
        for idx, candidate in enumerate(candidate_values):
            if idx in used:
                continue
            if scalar_matches(expected, candidate):
                used.add(idx)
                matched += 1
                break

    return matched / len(expected_values)


def extract_query_rows(tool_calls: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []

    for call in tool_calls:
        if call.get("name") != "run_query" or call.get("status") != "success":
            continue

        result = call.get("_full_result")

        if isinstance(result, dict) and isinstance(result.get("rows"), list):
            for row in result["rows"]:
                if isinstance(row, (list, tuple)):
                    rows.append(list(row))
                else:
                    rows.append([row])

        elif isinstance(result, list):
            for row in result:
                if isinstance(row, (list, tuple)):
                    rows.append(list(row))

    return rows


def ground_truth_coverage(
    expected_rows: list[dict[str, Any]],
    candidate_rows: list[list[Any]],
) -> tuple[float, list[dict[str, Any]]]:
    if not expected_rows:
        return 1.0, []

    details = []
    scores = []

    for expected_row in expected_rows:
        expected_values = list(expected_row.values())
        best_score = 0.0
        best_candidate = None

        for candidate_row in candidate_rows:
            score = row_match_fraction(expected_values, candidate_row)
            if score > best_score:
                best_score = score
                best_candidate = candidate_row
            if best_score == 1.0:
                break

        scores.append(best_score)
        details.append(
            {
                "expected": make_json_safe(expected_row),
                "best_match_fraction": round(best_score, 4),
                "best_candidate_row": make_json_safe(best_candidate),
            }
        )

    return sum(scores) / len(scores), details


def run_agent_on_task(
    module,
    agent_name: str,
    task_id: str,
    task: dict[str, Any],
    pass_threshold: float,
) -> dict[str, Any]:
    question = task["question"]
    expected_rows = task.get("expected_rows", [])

    tool_calls: list[dict[str, Any]] = []

    original_execute_tool = module.execute_tool
    original_client = module.client
    client_proxy = ConverseClientProxy(original_client)

    def tracked_execute_tool(name, tool_input):
        record = {
            "name": name,
            "input": make_json_safe(tool_input),
            "status": "running",
        }

        started = time.perf_counter()

        try:
            result = original_execute_tool(name, tool_input)
            record["status"] = "success"
            record["_full_result"] = result
            record["result_preview"] = summarize_result(result)
            return result

        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            raise

        finally:
            record["latency_seconds"] = round(time.perf_counter() - started, 4)
            tool_calls.append(record)

    module.execute_tool = tracked_execute_tool
    module.client = client_proxy

    answer = None
    run_error = None
    started = time.perf_counter()
    captured_stdout = io.StringIO()

    try:
        with contextlib.redirect_stdout(captured_stdout):
            answer = module.ask_agent(question)
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"
    finally:
        wall_latency = time.perf_counter() - started
        module.execute_tool = original_execute_tool
        module.client = original_client

    candidate_rows = extract_query_rows(tool_calls)
    coverage, match_details = ground_truth_coverage(expected_rows, candidate_rows)

    tool_errors = sum(1 for call in tool_calls if call.get("status") == "error")
    sql_queries = [
        call.get("input", {}).get("sql")
        for call in tool_calls
        if call.get("name") == "run_query"
        and isinstance(call.get("input"), dict)
        and call.get("input", {}).get("sql")
    ]

    clean_tool_calls = []
    for call in tool_calls:
        clean = {k: v for k, v in call.items() if k != "_full_result"}
        clean_tool_calls.append(make_json_safe(clean))

    passed = run_error is None and coverage >= pass_threshold

    return {
        "agent": agent_name,
        "task_id": task_id,
        "question": question,
        "metric_definition": task.get("metric_definition"),
        "status": "PASS" if passed else ("ERROR" if run_error else "FAIL"),
        "ground_truth_coverage": round(coverage, 4),
        "pass_threshold": pass_threshold,
        "final_answer": answer,
        "run_error": run_error,
        "sql_queries": sql_queries,
        "tool_calls": clean_tool_calls,
        "tool_call_count": len(tool_calls),
        "tool_error_count": tool_errors,
        "model_calls": client_proxy.calls,
        "input_tokens": client_proxy.input_tokens,
        "output_tokens": client_proxy.output_tokens,
        "total_tokens": client_proxy.total_tokens,
        "wall_latency_seconds": round(wall_latency, 3),
        "bedrock_reported_latency_ms": client_proxy.bedrock_latency_ms,
        "ground_truth_match_details": match_details,
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(make_json_safe(data), indent=2), encoding="utf-8")


def save_outputs(results: list[dict[str, Any]], ground_truth_file: str) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    write_json(
        EVAL_DIR / "evaluation_results.json",
        {
            "ground_truth_file": ground_truth_file,
            "generated_at_unix": time.time(),
            "results": results,
        },
    )

    for agent_name in DEFAULT_AGENTS:
        agent_results = [r for r in results if r["agent"] == agent_name]
        if agent_results:
            write_json(
                EVAL_DIR / f"results_{agent_name.replace('agent_', '')}.json",
                {
                    "agent": agent_name,
                    "ground_truth_file": ground_truth_file,
                    "results": agent_results,
                },
            )

    summary_path = EVAL_DIR / "evaluation_summary.csv"
    fieldnames = [
        "agent",
        "task_id",
        "status",
        "ground_truth_coverage",
        "tool_call_count",
        "tool_error_count",
        "model_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "wall_latency_seconds",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key) for key in fieldnames})


def print_task_result(result: dict[str, Any]) -> None:
    print(
        f"  {result['status']:5}  "
        f"coverage={result['ground_truth_coverage']:.2f}  "
        f"tools={result['tool_call_count']}  "
        f"errors={result['tool_error_count']}  "
        f"model_calls={result['model_calls']}  "
        f"latency={result['wall_latency_seconds']:.1f}s"
    )


def print_final_summary(results: list[dict[str, Any]], task_ids: list[str]) -> None:
    by_key = {(r["agent"], r["task_id"]): r for r in results}
    task_width = max(22, max((len(task_id) for task_id in task_ids), default=4) + 2)

    print("\n\nEVALUATION SUMMARY")
    print("==================")
    print(f"{'Task':<{task_width}} {'V1':<16} {'V2':<16}")
    print("-" * (task_width + 34))

    for task_id in task_ids:
        cells = []
        for agent_name in DEFAULT_AGENTS:
            result = by_key.get((agent_name, task_id))
            if result is None:
                cells.append("N/A")
            else:
                cells.append(
                    f"{result['status']} ({result['ground_truth_coverage']:.2f})"
                )

        print(f"{task_id:<{task_width}} {cells[0]:<16} {cells[1]:<16}")

    print("\nSaved:")
    print("  eval/evaluation_results.json")
    print("  eval/results_v1.json")
    print("  eval/results_v2.json")
    print("  eval/evaluation_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate agent_v1 and agent_v2 against trusted SQL ground truth."
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=DEFAULT_AGENTS,
        help="Agent module names in src/ (default: agent_v1 agent_v2)",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional task IDs to run. Default: all tasks.",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=DEFAULT_PASS_THRESHOLD,
        help="Ground-truth coverage required for PASS (default: 0.95)",
    )
    args = parser.parse_args()

    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            f"Missing {GROUND_TRUTH_PATH}\n"
            "Run: python scripts/generate_ground_truth.py"
        )

    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    tasks = ground_truth.get("tasks", {})

    if not tasks:
        raise ValueError(f"No tasks found in {GROUND_TRUTH_PATH}")

    if args.tasks:
        missing = [task_id for task_id in args.tasks if task_id not in tasks]
        if missing:
            raise ValueError(f"Unknown task IDs: {', '.join(missing)}")
        task_ids = args.tasks
    else:
        task_ids = list(tasks.keys())

    print("Agent evaluation")
    print("================")
    print(f"Ground truth: {GROUND_TRUTH_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Tasks: {len(task_ids)}")
    print(f"Agents: {', '.join(args.agents)}")
    print("\nNote: this makes live Amazon Bedrock calls, so AWS usage charges may apply.\n")

    results: list[dict[str, Any]] = []

    for agent_name in args.agents:
        print(f"\nLoading {agent_name}...")

        try:
            module = load_agent_module(agent_name)
        except Exception as exc:
            print(f"  ERROR loading {agent_name}: {exc}")

            for task_id in task_ids:
                results.append(
                    {
                        "agent": agent_name,
                        "task_id": task_id,
                        "question": tasks[task_id]["question"],
                        "metric_definition": tasks[task_id].get("metric_definition"),
                        "status": "ERROR",
                        "ground_truth_coverage": 0.0,
                        "pass_threshold": args.pass_threshold,
                        "final_answer": None,
                        "run_error": f"Import error: {type(exc).__name__}: {exc}",
                        "sql_queries": [],
                        "tool_calls": [],
                        "tool_call_count": 0,
                        "tool_error_count": 0,
                        "model_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "wall_latency_seconds": 0.0,
                        "bedrock_reported_latency_ms": 0,
                        "ground_truth_match_details": [],
                    }
                )
            continue

        for index, task_id in enumerate(task_ids, start=1):
            print(f"[{agent_name}] {index}/{len(task_ids)} {task_id}")

            result = run_agent_on_task(
                module=module,
                agent_name=agent_name,
                task_id=task_id,
                task=tasks[task_id],
                pass_threshold=args.pass_threshold,
            )
            results.append(result)
            print_task_result(result)

            save_outputs(
                results,
                str(GROUND_TRUTH_PATH.relative_to(PROJECT_ROOT)),
            )

    print_final_summary(results, task_ids)


if __name__ == "__main__":
    main()
