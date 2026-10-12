#!/usr/bin/env python3
"""
Run all reference SQL queries and write eval/ground_truth.json.

Run from your project root:
    python scripts/generate_ground_truth.py
"""

import json
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "db" / "olist.db"
QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.json"
OUTPUT_PATH = PROJECT_ROOT / "eval" / "ground_truth.json"


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    output = {
        "database": "db/olist.db",
        "tasks": {}
    }

    db_uri = f"file:{DB_PATH}?mode=ro"

    with sqlite3.connect(db_uri, uri=True) as conn:
        for task in questions:
            sql_path = PROJECT_ROOT / task["reference_sql"]
            sql = sql_path.read_text(encoding="utf-8")

            cursor = conn.execute(sql)
            columns = [d[0] for d in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

            output["tasks"][task["id"]] = {
                "question": task["question"],
                "metric_definition": task["metric_definition"],
                "reference_sql": task["reference_sql"],
                "expected_rows": rows
            }

            print(f"\n[{task['id']}]")
            print(task["question"])
            for row in rows[:10]:
                print(" ", row)
            if len(rows) > 10:
                print(f"  ... {len(rows) - 10} more rows")

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
