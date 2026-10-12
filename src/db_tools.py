import sqlite3
import time
from pathlib import Path

DB_PATH = Path("db/olist.db")
QUERY_TIMEOUT_SECONDS = 10


def list_tables():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' ORDER BY name;"
        ).fetchall()

    return [row[0] for row in rows]


def get_schema(table_name):
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"PRAGMA table_info({table_name});"
        ).fetchall()

    return [
        {
            "column": row[1],
            "type": row[2],
        }
        for row in rows
    ]


def run_query(sql):
    db_uri = f"file:{DB_PATH}?mode=ro"
    deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS

    def cancel_if_timed_out():
        return 1 if time.monotonic() >= deadline else 0

    with sqlite3.connect(db_uri, uri=True) as conn:
        # Ask SQLite to periodically check whether this query has exceeded
        # the allowed runtime. Returning non-zero interrupts the query.
        conn.set_progress_handler(cancel_if_timed_out, 10_000)

        try:
            cursor = conn.execute(sql)

            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchmany(100)

        finally:
            conn.set_progress_handler(None, 0)

    return {
        "columns": columns,
        "rows": rows,
    }

if __name__ == "__main__":
    print("Tables:")
    print(list_tables())

    print("\nOrders schema:")
    print(get_schema("orders"))

    print("\nExample query:")
    print(run_query("SELECT COUNT(*) AS num_orders FROM orders;"))
