import json
import os
import re
from typing import Any, TypedDict

import boto3
import mlflow
from botocore.config import Config
from langgraph.graph import END, StateGraph

from db_tools import list_tables, get_schema, run_query


MODEL_ID = "amazon.nova-lite-v1:0"

# Initial attempt + up to 2 retries for each independent path.
MAX_VERIFIER_RETRIES = 2
MAX_ANALYZER_RETRIES = 2

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)
MLFLOW_EXPERIMENT = os.getenv(
    "MLFLOW_EXPERIMENT",
    "olist-agent-v3",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)


client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    config=Config(
        connect_timeout=5,
        read_timeout=20,
        retries={"max_attempts": 1},
    ),
)


class AgentState(TypedDict, total=False):
    question: str
    metric_definition: str
    schema_context: dict[str, Any]
    schema_catalog: str
    join_hints: str

    # Independent verifier path
    verification_sql: str
    verification_rationale: str
    verification_result: dict[str, Any]
    verification_schema_error: str | None
    verification_sql_error: str | None
    verifier_retry_count: int
    verifier_retry_feedback: str

    # Analyzer path
    sql: str
    analysis_rationale: str
    query_result: dict[str, Any]
    analyzer_schema_error: str | None
    sql_error: str | None
    analyzer_retry_count: int
    analyzer_feedback: str

    # Final comparison state
    verifier_approved: bool
    verifier_feedback: str

    final_answer: str


def execute_tool(name: str, tool_input: dict[str, Any]):
    """
    Keep database access behind one dispatcher so the evaluator can wrap
    and capture tool calls.
    """
    if name == "list_tables":
        return list_tables()

    if name == "get_schema":
        return get_schema(tool_input["table_name"])

    if name == "run_query":
        return run_query(tool_input["sql"])

    raise ValueError(f"Unknown tool: {name}")


def _extract_text(response: dict[str, Any]) -> str:
    blocks = response["output"]["message"]["content"]
    return "\n".join(
        block["text"]
        for block in blocks
        if "text" in block
    ).strip()


def _call_model(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1400,
) -> str:
    response = client.converse(
        modelId=MODEL_ID,
        system=[{"text": system_prompt}],
        messages=[
            {
                "role": "user",
                "content": [{"text": user_prompt}],
            }
        ],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": 0,
        },
    )
    return _extract_text(response)


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "", 1)
        cleaned = cleaned.replace("```", "", 1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            f"Model did not return a JSON object: {text[:300]}"
        )

    return json.loads(cleaned[start:end + 1])


def _validate_read_only_sql(sql: str, actor: str) -> None:
    normalized = sql.lstrip().upper()

    if not (
        normalized.startswith("SELECT")
        or normalized.startswith("WITH")
    ):
        raise ValueError(
            f"{actor} must return a read-only SELECT/WITH query."
        )


def _split_mode_a_input(agent_input: str) -> tuple[str, str]:
    question_marker = "QUESTION:"
    metric_marker = "METRIC DEFINITION:"

    if (
        question_marker in agent_input
        and metric_marker in agent_input
    ):
        after_question = agent_input.split(
            question_marker,
            1,
        )[1]

        question, metric_part = after_question.split(
            metric_marker,
            1,
        )

        metric = metric_part.strip()

        suffix = "Follow the metric definition exactly."
        if metric.endswith(suffix):
            metric = metric[:-len(suffix)].strip()

        return question.strip(), metric.strip()

    return agent_input.strip(), (
        "Use a sensible analytical definition and state "
        "important assumptions."
    )



def _schema_column_map(
    schemas: dict[str, Any],
) -> dict[str, set[str]]:
    """
    Convert get_schema() results into:
        {table_name: {column1, column2, ...}}

    db_tools.get_schema() returns rows such as:
        {"column": "order_id", "type": "TEXT"}
    """
    result: dict[str, set[str]] = {}

    for table, schema_rows in schemas.items():
        columns: set[str] = set()

        if isinstance(schema_rows, list):
            for item in schema_rows:
                if isinstance(item, dict) and item.get("column"):
                    columns.add(str(item["column"]))

        result[table] = columns

    return result


def _build_schema_catalog(
    schemas: dict[str, Any],
) -> str:
    """
    Build a compact schema representation that is easier for a small LLM
    to ground on than a large nested JSON object.
    """
    lines = []

    for table in sorted(schemas):
        schema_rows = schemas[table]
        parts = []

        if isinstance(schema_rows, list):
            for item in schema_rows:
                if not isinstance(item, dict):
                    continue

                column = item.get("column")
                dtype = item.get("type", "")

                if column:
                    parts.append(
                        f"{column} {dtype}".strip()
                    )

        lines.append(
            f"{table}: " + ", ".join(parts)
        )

    return "\n".join(lines)


def _build_join_hints(
    schemas: dict[str, Any],
) -> str:
    """
    Derive candidate join paths from shared key-like column names.

    These are hints, not hard-coded Olist relationships. The model still
    has to decide whether a join is appropriate for the metric.
    """
    column_map = _schema_column_map(schemas)
    owners: dict[str, list[str]] = {}

    for table, columns in column_map.items():
        for column in columns:
            if column.endswith("_id") or column == "product_category_name":
                owners.setdefault(column, []).append(table)

    lines = []

    for column in sorted(owners):
        tables = sorted(set(owners[column]))

        if len(tables) < 2:
            continue

        qualified = " <-> ".join(
            f"{table}.{column}"
            for table in tables
        )
        lines.append(f"{column}: {qualified}")

    if not lines:
        return "No shared key-like columns were detected automatically."

    return "\n".join(lines)


def _extract_cte_names(sql: str) -> set[str]:
    """
    Best-effort extraction of CTE names. This is intentionally lightweight:
    it is only used to avoid treating CTEs as unknown base tables.
    """
    return {
        match.group(1)
        for match in re.finditer(
            r"(?:WITH|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(",
            sql,
            flags=re.IGNORECASE,
        )
    }


def _find_schema_reference_errors(
    sql: str,
    schemas: dict[str, Any],
) -> list[str]:
    """
    Deterministically catch obvious schema hallucinations before SQLite runs.

    Example caught:
        FROM orders o ... SELECT o.seller_id
    because seller_id is not a column in orders.

    This is deliberately conservative. It validates explicit base-table
    alias.column references and unknown FROM/JOIN tables, but does not try
    to fully parse SQL or infer CTE output schemas.
    """
    column_map = _schema_column_map(schemas)
    cte_names = _extract_cte_names(sql)

    alias_to_table: dict[str, str] = {}
    errors: list[str] = []

    # Capture FROM/JOIN <table> [AS] <alias>. Avoid swallowing SQL keywords
    # as aliases when no alias is present.
    keyword_stop = {
        "where", "join", "left", "right", "inner", "outer", "full",
        "cross", "on", "group", "order", "having", "limit", "union",
    }

    table_pattern = re.compile(
        r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)"
        r"(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
        flags=re.IGNORECASE,
    )

    for match in table_pattern.finditer(sql):
        table = match.group(1)
        alias = match.group(2)

        if alias and alias.casefold() in keyword_stop:
            alias = None

        if table in cte_names:
            # CTE output columns are not known from the physical schema.
            continue

        if table not in column_map:
            errors.append(
                f"Unknown base table '{table}'."
            )
            continue

        alias_to_table[table] = table

        if alias:
            alias_to_table[alias] = table

    for alias, column in re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b",
        sql,
    ):
        table = alias_to_table.get(alias)

        # Unknown alias may belong to a CTE/subquery; SQLite will validate it.
        if table is None:
            continue

        if column not in column_map.get(table, set()):
            errors.append(
                f"Column '{column}' does not exist in base table "
                f"'{table}' (referenced as {alias}.{column})."
            )

    # Keep messages stable and non-duplicated for retry feedback.
    return list(dict.fromkeys(errors))


@mlflow.trace(name="schema_inspection")
def schema_node(state: AgentState) -> dict[str, Any]:
    tables = execute_tool("list_tables", {})
    schemas = {}

    for table in tables:
        schemas[table] = execute_tool(
            "get_schema",
            {"table_name": table},
        )

    return {
        "schema_context": schemas,
        "schema_catalog": _build_schema_catalog(schemas),
        "join_hints": _build_join_hints(schemas),
    }


@mlflow.trace(name="independent_verifier_plan")
def verifier_plan_node(
    state: AgentState,
) -> dict[str, Any]:
    system_prompt = """
You are the Independent Verifier in a SQL analytics workflow.

Create ONE read-only SQLite query that independently computes the
AUTHORITATIVE METRIC DEFINITION.

Important:
1. Work only from the question, metric definition, SCHEMA CATALOG, and JOIN HINTS.
2. NEVER invent a column. A table may use a column only if that exact column is
   listed for that table in the SCHEMA CATALOG.
3. Before writing SQL, mentally trace every required field to the table that
   actually owns it. If a field lives in another table, join through the
   listed shared keys instead of pretending it exists in the current table.
4. JOIN HINTS are candidate paths derived from shared columns. Use only paths
   that are appropriate for the metric.
5. You have NOT seen the Analyzer's SQL. Do not assume any Analyzer approach.
6. Preserve the exact observation grain.
4. Determine the correct join path from the schema.
5. Prevent join multiplication.
6. Pre-aggregate or deduplicate whenever required by the metric definition.
7. If the metric says one distinct entity-pair, explicitly deduplicate that
   pair BEFORE counting or averaging.
8. Handle NULL values explicitly.
9. Do not convert missing measurements into zero unless explicitly required.
10. Do not add filters or business rules that are not in the metric definition
    unless they are strictly necessary to make the definition executable.
11. After joins, qualify potentially ambiguous columns with table aliases.
12. Every selected value in a grouped result must either be a grouping key or
    a proper aggregate.
13. Counts must represent the population defined by the metric.
14. Use SQLite-compatible SQL only.
15. Force floating-point arithmetic for rates/percentages, e.g. 100.0 * x / y.
16. Return enough columns to verify the important metric values and counts.
17. Before returning SQL, check every base-table alias.column reference
    against the SCHEMA CATALOG.
18. If the previous verifier attempt failed schema validation or SQL execution,
    fix that verifier error directly.
19. Return only JSON:
    {"verification_sql": "...", "rationale": "..."}
20. Do not include markdown fences.
"""

    user_prompt = f"""
QUESTION:
{state["question"]}

AUTHORITATIVE METRIC DEFINITION:
{state["metric_definition"]}

SCHEMA CATALOG:
{state["schema_catalog"]}

CANDIDATE JOIN HINTS:
{state["join_hints"]}

VERIFIER RETRY NUMBER:
{state.get("verifier_retry_count", 0)}

PREVIOUS VERIFIER ERROR:
{state.get("verifier_retry_feedback", "") or "None. This is the first verifier attempt."}

Generate an independent verification query now.
"""

    raw = _call_model(
        system_prompt,
        user_prompt,
    )
    parsed = _parse_json_object(raw)

    sql = str(
        parsed.get("verification_sql", "")
    ).strip()
    rationale = str(
        parsed.get("rationale", "")
    ).strip()

    if not sql:
        raise ValueError(
            "Independent Verifier returned an empty SQL query."
        )

    _validate_read_only_sql(
        sql,
        "Independent Verifier",
    )

    return {
        "verification_sql": sql,
        "verification_rationale": rationale,
        "verification_result": {},
        "verification_schema_error": None,
        "verification_sql_error": None,
    }



@mlflow.trace(name="independent_verifier_schema_check")
def verifier_schema_check_node(
    state: AgentState,
) -> dict[str, Any]:
    errors = _find_schema_reference_errors(
        state["verification_sql"],
        state["schema_context"],
    )

    if not errors:
        return {
            "verification_schema_error": None,
        }

    message = " ".join(errors)

    return {
        "verification_schema_error": message,
        "verifier_retry_feedback": (
            "The independent verifier SQL failed deterministic schema "
            f"validation before execution: {message} "
            "Use only columns owned by the tables shown in SCHEMA CATALOG "
            "and correct the join path."
        ),
        "verifier_retry_count": (
            state.get("verifier_retry_count", 0) + 1
        ),
    }


def route_after_verifier_schema_check(
    state: AgentState,
) -> str:
    if not state.get("verification_schema_error"):
        return "execute_verifier"

    if state.get("verifier_retry_count", 0) <= MAX_VERIFIER_RETRIES:
        return "retry_verifier"

    return "finalize"


@mlflow.trace(name="independent_verifier_execution")
def verifier_execute_node(
    state: AgentState,
) -> dict[str, Any]:
    try:
        result = execute_tool(
            "run_query",
            {"sql": state["verification_sql"]},
        )

        return {
            "verification_result": result,
            "verification_schema_error": None,
            "verification_sql_error": None,
            "verifier_retry_feedback": "",
        }

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return {
            "verification_result": {},
            "verification_sql_error": error,
            "verifier_retry_feedback": (
                "The independent verification SQL failed to execute. "
                f"Fix this verifier SQL error: {error}"
            ),
            "verifier_retry_count": (
                state.get("verifier_retry_count", 0) + 1
            ),
        }


def route_after_verifier_execution(
    state: AgentState,
) -> str:
    if not state.get("verification_sql_error"):
        return "analyze"

    if state.get("verifier_retry_count", 0) <= MAX_VERIFIER_RETRIES:
        return "retry_verifier"

    return "finalize"


@mlflow.trace(name="analyzer")
def analyzer_node(
    state: AgentState,
) -> dict[str, Any]:
    system_prompt = """
You are the Analyzer in a SQL analytics workflow.

Generate ONE read-only SQLite query that implements the authoritative metric
definition exactly.

Rules:
1. Treat the metric definition as authoritative.
2. NEVER invent a column. A table may use a column only if that exact column is
   listed for that table in the SCHEMA CATALOG.
3. Before writing SQL, mentally trace every required field to the table that
   actually owns it. If a field lives in another table, join through the
   listed shared keys instead of pretending it exists in the current table.
4. JOIN HINTS are candidate paths derived from shared columns. Use only paths
   that are appropriate for the metric.
5. Preserve the requested observation grain.
6. Use the correct join path.
5. Prevent join multiplication when it would change the observation grain.
6. Pre-aggregate or deduplicate when the metric definition requires it.
7. If the metric says one distinct entity-pair, explicitly deduplicate that
   pair BEFORE counting or averaging.
8. Handle NULL values explicitly when relevant.
9. Do not convert missing measurements into zero unless explicitly requested.
10. Do not add filters or business rules that are not in the metric definition.
11. After joins, qualify potentially ambiguous columns with table aliases.
12. In grouped output, every selected metric must be properly aggregated.
13. Use SQLite-compatible SQL only.
14. Force floating-point arithmetic for rates/percentages, e.g. 100.0 * x / y.
15. Return enough numerical evidence and counts to answer the question.
16. Before returning SQL, check every base-table alias.column reference
    against the SCHEMA CATALOG.
17. If previous Analyzer SQL failed schema validation, SQL execution, or
    disagreed with verification, correct that specific Analyzer problem.
18. Return only JSON:
    {"sql": "...", "rationale": "..."}
19. Do not include markdown fences.
"""

    user_prompt = f"""
QUESTION:
{state["question"]}

AUTHORITATIVE METRIC DEFINITION:
{state["metric_definition"]}

SCHEMA CATALOG:
{state["schema_catalog"]}

CANDIDATE JOIN HINTS:
{state["join_hints"]}

ANALYZER RETRY NUMBER:
{state.get("analyzer_retry_count", 0)}

ANALYZER FEEDBACK FROM PREVIOUS ATTEMPT:
{state.get("analyzer_feedback", "") or "None. This is the first Analyzer attempt."}

Generate the SQL now.
"""

    raw = _call_model(
        system_prompt,
        user_prompt,
    )
    parsed = _parse_json_object(raw)

    sql = str(
        parsed.get("sql", "")
    ).strip()
    rationale = str(
        parsed.get("rationale", "")
    ).strip()

    if not sql:
        raise ValueError(
            "Analyzer returned an empty SQL query."
        )

    _validate_read_only_sql(
        sql,
        "Analyzer",
    )

    return {
        "sql": sql,
        "analysis_rationale": rationale,
        "query_result": {},
        "analyzer_schema_error": None,
        "sql_error": None,
    }



@mlflow.trace(name="analyzer_schema_check")
def analyzer_schema_check_node(
    state: AgentState,
) -> dict[str, Any]:
    errors = _find_schema_reference_errors(
        state["sql"],
        state["schema_context"],
    )

    if not errors:
        return {
            "analyzer_schema_error": None,
        }

    message = " ".join(errors)

    return {
        "analyzer_schema_error": message,
        "analyzer_feedback": (
            "The Analyzer SQL failed deterministic schema validation "
            f"before execution: {message} "
            "Use only columns owned by the tables shown in SCHEMA CATALOG "
            "and correct the join path."
        ),
        "analyzer_retry_count": (
            state.get("analyzer_retry_count", 0) + 1
        ),
    }


def route_after_analyzer_schema_check(
    state: AgentState,
) -> str:
    if not state.get("analyzer_schema_error"):
        return "execute_analyzer"

    if state.get("analyzer_retry_count", 0) <= MAX_ANALYZER_RETRIES:
        return "retry_analyzer"

    return "finalize"


@mlflow.trace(name="sql_execution")
def execute_node(
    state: AgentState,
) -> dict[str, Any]:
    try:
        result = execute_tool(
            "run_query",
            {"sql": state["sql"]},
        )

        return {
            "query_result": result,
            "analyzer_schema_error": None,
            "sql_error": None,
        }

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return {
            "query_result": {},
            "sql_error": error,
            "analyzer_feedback": (
                "The Analyzer SQL failed to execute. "
                f"Fix this Analyzer SQL error: {error}"
            ),
            "analyzer_retry_count": (
                state.get("analyzer_retry_count", 0) + 1
            ),
        }


def route_after_analyzer_execution(
    state: AgentState,
) -> str:
    if not state.get("sql_error"):
        return "compare"

    if state.get("analyzer_retry_count", 0) <= MAX_ANALYZER_RETRIES:
        return "retry_analyzer"

    return "finalize"


@mlflow.trace(name="result_comparison")
def compare_node(
    state: AgentState,
) -> dict[str, Any]:
    system_prompt = """
You are the Comparison Verifier in a SQL analytics workflow.

You have:
- an authoritative metric definition
- the Analyzer SQL and its result
- an independently generated verification SQL and its result

Decide whether the Analyzer result is trustworthy.

Verification rules:
1. The Analyzer must implement the authoritative metric definition exactly.
2. Compare important numerical metrics and counts against the independent
   verification result.
3. Reject if the results differ materially, even if both SQL queries execute.
4. Reject incorrect observation grain, wrong join population, join
   multiplication, missing required deduplication, or incorrect NULL handling.
5. Reject zero imputation unless the metric explicitly requests it.
6. Reject grouped SQL that selects a non-grouped, non-aggregated metric value.
7. Counts must refer to the population defined by the metric definition.
8. Small harmless rounding differences are acceptable.
9. The two SQL queries do not need identical syntax or identical column names.
10. Approval means the Analyzer result is analytically equivalent to the
    independent verification result for the requested metric.

Return only JSON:
{
  "approved": true or false,
  "feedback": "specific explanation of agreement or the exact Analyzer problem"
}

Do not include markdown fences.
"""

    user_prompt = f"""
QUESTION:
{state["question"]}

AUTHORITATIVE METRIC DEFINITION:
{state["metric_definition"]}

ANALYZER SQL:
{state["sql"]}

ANALYZER RESULT:
{json.dumps(
    state["query_result"],
    default=str,
    indent=2,
)}

INDEPENDENT VERIFICATION SQL:
{state["verification_sql"]}

INDEPENDENT VERIFICATION RESULT:
{json.dumps(
    state["verification_result"],
    default=str,
    indent=2,
)}
"""

    raw = _call_model(
        system_prompt,
        user_prompt,
    )
    parsed = _parse_json_object(raw)

    approved = bool(
        parsed.get(
            "approved",
            False,
        )
    )
    feedback = str(
        parsed.get(
            "feedback",
            "",
        )
    ).strip()

    result = {
        "verifier_approved": approved,
        "verifier_feedback": feedback,
    }

    if approved:
        result["analyzer_feedback"] = ""
    else:
        result["analyzer_feedback"] = feedback
        result["analyzer_retry_count"] = (
            state.get("analyzer_retry_count", 0) + 1
        )

    return result


def route_after_comparison(
    state: AgentState,
) -> str:
    if state.get("verifier_approved", False):
        return "finalize"

    if state.get("analyzer_retry_count", 0) <= MAX_ANALYZER_RETRIES:
        return "retry_analyzer"

    return "finalize"


@mlflow.trace(name="finalizer")
def finalizer_node(
    state: AgentState,
) -> dict[str, Any]:
    approved = state.get(
        "verifier_approved",
        False,
    )

    verifier_schema_error = state.get("verification_schema_error")
    analyzer_schema_error = state.get("analyzer_schema_error")
    verifier_error = state.get("verification_sql_error")
    analyzer_error = state.get("sql_error")

    if verifier_schema_error:
        return {
            "final_answer": (
                "The result could not be fully verified because the "
                "independent verification SQL repeatedly violated the "
                f"database schema: {verifier_schema_error}"
            )
        }

    if analyzer_schema_error:
        return {
            "final_answer": (
                "The result could not be fully verified because the "
                "Analyzer SQL repeatedly violated the database schema: "
                f"{analyzer_schema_error}"
            )
        }

    if verifier_error:
        return {
            "final_answer": (
                "The result could not be fully verified because the "
                "independent verification SQL failed after retries: "
                f"{verifier_error}"
            )
        }

    if analyzer_error:
        return {
            "final_answer": (
                "The result could not be fully verified because the "
                "Analyzer SQL failed after retries: "
                f"{analyzer_error}"
            )
        }

    system_prompt = """
You are the Finalizer in a data-analysis workflow.

Answer the business question concisely using the supplied Analyzer result.

Rules:
1. Follow the authoritative metric definition.
2. Support the conclusion with numerical evidence.
3. Do not claim causation when the data only establishes association.
4. If VERIFIED is false, explicitly say that the result could not be fully
   verified and do not present the numbers as reliable.
"""

    user_prompt = f"""
QUESTION:
{state["question"]}

METRIC DEFINITION:
{state["metric_definition"]}

VERIFIED:
{approved}

VERIFIER FEEDBACK:
{state.get("verifier_feedback", "")}

FINAL ANALYZER SQL:
{state.get("sql", "")}

ANALYZER RESULT:
{json.dumps(
    state.get("query_result", {}),
    default=str,
    indent=2,
)}

INDEPENDENT VERIFICATION RESULT:
{json.dumps(
    state.get("verification_result", {}),
    default=str,
    indent=2,
)}
"""

    answer = _call_model(
        system_prompt,
        user_prompt,
        max_tokens=1000,
    )

    return {
        "final_answer": answer,
    }


def build_graph():
    workflow = StateGraph(
        AgentState,
    )

    workflow.add_node(
        "schema",
        schema_node,
    )
    workflow.add_node(
        "verifier_plan",
        verifier_plan_node,
    )
    workflow.add_node(
        "verifier_schema_check",
        verifier_schema_check_node,
    )
    workflow.add_node(
        "verifier_execute",
        verifier_execute_node,
    )
    workflow.add_node(
        "analyze",
        analyzer_node,
    )
    workflow.add_node(
        "analyzer_schema_check",
        analyzer_schema_check_node,
    )
    workflow.add_node(
        "execute",
        execute_node,
    )
    workflow.add_node(
        "compare",
        compare_node,
    )
    workflow.add_node(
        "finalize",
        finalizer_node,
    )

    workflow.set_entry_point(
        "schema",
    )

    workflow.add_edge(
        "schema",
        "verifier_plan",
    )
    workflow.add_edge(
        "verifier_plan",
        "verifier_schema_check",
    )

    # Catch hallucinated base-table columns before touching SQLite.
    workflow.add_conditional_edges(
        "verifier_schema_check",
        route_after_verifier_schema_check,
        {
            "retry_verifier": "verifier_plan",
            "execute_verifier": "verifier_execute",
            "finalize": "finalize",
        },
    )

    # A verifier SQL execution failure retries the VERIFIER, not the Analyzer.
    workflow.add_conditional_edges(
        "verifier_execute",
        route_after_verifier_execution,
        {
            "retry_verifier": "verifier_plan",
            "analyze": "analyze",
            "finalize": "finalize",
        },
    )

    workflow.add_edge(
        "analyze",
        "analyzer_schema_check",
    )

    # Catch hallucinated base-table columns before touching SQLite.
    workflow.add_conditional_edges(
        "analyzer_schema_check",
        route_after_analyzer_schema_check,
        {
            "retry_analyzer": "analyze",
            "execute_analyzer": "execute",
            "finalize": "finalize",
        },
    )

    # An Analyzer SQL execution failure retries the ANALYZER.
    workflow.add_conditional_edges(
        "execute",
        route_after_analyzer_execution,
        {
            "retry_analyzer": "analyze",
            "compare": "compare",
            "finalize": "finalize",
        },
    )

    # Only successfully executed verifier + Analyzer results are compared.
    workflow.add_conditional_edges(
        "compare",
        route_after_comparison,
        {
            "retry_analyzer": "analyze",
            "finalize": "finalize",
        },
    )

    workflow.add_edge(
        "finalize",
        END,
    )

    return workflow.compile()


graph = build_graph()


@mlflow.trace(name="agent_v3")
def ask_agent(
    agent_input: str,
) -> str:
    question, metric_definition = _split_mode_a_input(
        agent_input,
    )

    initial_state: AgentState = {
        "question": question,
        "metric_definition": metric_definition,
        "verifier_retry_count": 0,
        "verifier_retry_feedback": "",
        "verification_schema_error": None,
        "analyzer_retry_count": 0,
        "analyzer_schema_error": None,
        "analyzer_feedback": "",
        "verifier_approved": False,
        "verifier_feedback": "",
    }

    final_state = graph.invoke(
        initial_state,
    )

    print("\nV3 TRACE SUMMARY")
    print("================")
    print(
        "Verifier retries: "
        f"{final_state.get('verifier_retry_count', 0)}"
    )
    print(
        "Analyzer retries: "
        f"{final_state.get('analyzer_retry_count', 0)}"
    )
    print(
        "Verified: "
        f"{final_state.get('verifier_approved', False)}"
    )

    print("\nIndependent verification SQL:")
    print(
        final_state.get(
            "verification_sql",
            "",
        )
    )

    print("\nFinal Analyzer SQL:")
    print(
        final_state.get(
            "sql",
            "",
        )
    )

    print("\nVerifier feedback:")
    print(
        final_state.get(
            "verifier_feedback",
            "",
        )
    )

    if final_state.get("verification_schema_error"):
        print("\nVerifier schema error:")
        print(final_state["verification_schema_error"])

    if final_state.get("analyzer_schema_error"):
        print("\nAnalyzer schema error:")
        print(final_state["analyzer_schema_error"])

    if final_state.get("verification_sql_error"):
        print("\nVerifier SQL error:")
        print(final_state["verification_sql_error"])

    if final_state.get("sql_error"):
        print("\nAnalyzer SQL error:")
        print(final_state["sql_error"])

    return final_state["final_answer"]


if __name__ == "__main__":
    demo_input = """QUESTION:
Do late deliveries have a meaningful relationship with customer satisfaction?
Quantify the relationship using the data.

METRIC DEFINITION:
Observation unit: one order. Average multiple review rows per order first.
Include only orders with actual and estimated delivery timestamps.
Late means actual delivery is after estimated delivery.

Follow the metric definition exactly.
"""

    print("Running agent_v3...")

    answer = ask_agent(
        demo_input,
    )

    print("\nFINAL ANSWER")
    print("============")
    print(answer)
