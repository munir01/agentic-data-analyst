import asyncio
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters


PROJECT_ROOT = Path(__file__).resolve().parent

server = StdioServerParameters(
    command=sys.executable,
    args=[str(PROJECT_ROOT / "mcp_server.py")],
)


async def main():
    async with Client(server) as client:
        print("Connected to MCP server.")
        print(f"Protocol version: {client.protocol_version}\n")

        tools = await client.list_tools()

        print("Available tools:")
        for tool in tools.tools:
            print(f"  - {tool.name}: {tool.description}")

        print("\nCalling list_tables...")
        result = await client.call_tool("list_tables", {})

        print("\nResult:")
        if result.structured_content is not None:
            print(result.structured_content)
        else:
            for item in result.content:
                text = getattr(item, "text", None)
                print(text if text is not None else item)


if __name__ == "__main__":
    asyncio.run(main())
