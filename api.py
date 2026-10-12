import os
import secrets
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

API_KEY = os.getenv("API_KEY")

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent_v4 import ask_agent


app = FastAPI(
    title="Olist Agent API",
    version="0.2.0",
)


class AskRequest(BaseModel):
    question: str
    metric_definition: str | None = None


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    x_api_key: str | None = Header(default=None)
):
    if (
        API_KEY is None
        or x_api_key is None
        or not secrets.compare_digest(x_api_key, API_KEY)
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    if request.metric_definition:
        agent_input = f"""QUESTION:
{request.question}

METRIC DEFINITION:
{request.metric_definition}

Follow the metric definition exactly.
"""
    else:
        agent_input = request.question

    try:
        answer = ask_agent(agent_input)
        return AskResponse(answer=answer)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
