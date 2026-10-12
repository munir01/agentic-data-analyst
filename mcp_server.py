import sys
from pathlib import Path
from typing import Any

from mcp.server import MCPServer


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db_tools import get_schema as db_get_schema
from db_tools import list_tables as db_list_tables
from db_tools import run_query as db_run_query


mcp = MCPServer("olist-db", version="0.1.0")


@mcp.tool()
def list_tables() -> list[str]:
    """List the tables available in the local Olist SQLite database."""
    return db_list_tables()


@mcp.tool()
def get_schema(table_name: str) -> list[dict[str, str]]:
    """Return the schema for one database table."""
    return db_get_schema(table_name)


@mcp.tool()
def run_query(sql: str) -> dict[str, Any]:
    """Run a read-only SQL query against the local Olist SQLite database."""
    return db_run_query(sql)


if __name__ == "__main__":
    mcp.run(transport="stdio")
