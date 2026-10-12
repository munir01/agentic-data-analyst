import json
import boto3
from botocore.config import Config
from db_tools import list_tables, get_schema, run_query


MODEL_ID = "amazon.nova-lite-v1:0"

# client = boto3.client(
#     "bedrock-runtime",
#     region_name="us-east-1"
# )

client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    config=Config(
        connect_timeout=5,
        read_timeout=20,
        retries={"max_attempts": 1},
    ),
)

TOOLS = [
    {
        "toolSpec": {
            "name": "list_tables",
            "description": "List all tables available in the SQLite database.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "get_schema",
            "description": "Return the columns and data types for a database table.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table to inspect"
                        }
                    },
                    "required": ["table_name"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "run_query",
            "description": (
                "Run a read-only SQL SELECT query against the SQLite database. "
                "Use this to calculate statistics or retrieve evidence."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "A valid SQLite SELECT query"
                        }
                    },
                    "required": ["sql"]
                }
            }
        }
    }
]


def execute_tool(name, tool_input):
    if name == "list_tables":
        return list_tables()

    if name == "get_schema":
        return get_schema(tool_input["table_name"])

    if name == "run_query":
        return run_query(tool_input["sql"])

    raise ValueError(f"Unknown tool: {name}")


def make_json_safe(value):
    """
    Convert Python values such as tuples into structures
    that Bedrock can safely accept as JSON.
    """
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]

    if isinstance(value, list):
        return [make_json_safe(v) for v in value]

    if isinstance(value, dict):
        return {
            key: make_json_safe(val)
            for key, val in value.items()
        }

    return value


def ask_agent(question):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": question
                }
            ]
        }
    ]

    system = [
        {
            "text": (
                "You are a data analyst. "
                "Use the provided database tools to investigate the user's question. "
                "Do not guess database contents. "
                "Inspect the available tables and schemas as needed. "
                "Query the database to obtain numerical evidence. "
                "Support your conclusions with data. "
                "Do not claim causation when the data only demonstrates association."
            )
        }
    ]

    while True:

        response = client.converse(
            modelId=MODEL_ID,
            messages=messages,
            system=system,
            toolConfig={
                "tools": TOOLS
            },
            inferenceConfig={
                "maxTokens": 1000,
                "temperature": 0
            }
        )

        assistant_message = response["output"]["message"]

        messages.append(assistant_message)

        stop_reason = response["stopReason"]

        # If the model is finished, return its text response.
        if stop_reason != "tool_use":

            text_parts = []

            for block in assistant_message["content"]:
                if "text" in block:
                    text_parts.append(block["text"])

            return "\n".join(text_parts)

        # Otherwise, execute every requested tool.
        tool_results = []

        for block in assistant_message["content"]:

            if "toolUse" not in block:
                continue

            tool_use = block["toolUse"]

            tool_name = tool_use["name"]
            tool_input = tool_use["input"]
            tool_use_id = tool_use["toolUseId"]

            print("\n----------------------------------------")
            print(f"Calling tool: {tool_name}")
            print("Input:")
            print(json.dumps(tool_input, indent=2))

            try:

                result = execute_tool(
                    tool_name,
                    tool_input
                )

                result = make_json_safe(result)

                print("Result:")
                print(json.dumps(result, indent=2))

                # IMPORTANT:
                # Bedrock expects content[].json to contain
                # a JSON object, not a raw list.
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [
                                {
                                    "json": {
                                        "result": result
                                    }
                                }
                            ],
                            "status": "success"
                        }
                    }
                )

            except Exception as e:

                print(f"Tool error: {e}")

                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [
                                {
                                    "json": {
                                        "error": str(e)
                                    }
                                }
                            ],
                            "status": "error"
                        }
                    }
                )

        # Send the tool results back to the model.
        messages.append(
            {
                "role": "user",
                "content": tool_results
            }
        )


if __name__ == "__main__":

    question = (
        "Do late deliveries have a meaningful relationship with "
        "customer satisfaction? Quantify the relationship using the data."
    )

    print("QUESTION")
    print("========")
    print(question)

    answer = ask_agent(question)

    print("\n\nFINAL ANSWER")
    print("============")
    print(answer)