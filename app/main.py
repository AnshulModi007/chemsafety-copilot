"""FastAPI backend exposing the ChemSafety Copilot agent over HTTP.

Run locally with: uvicorn app.main:app --reload
"""
import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FEEDBACK_LOG_PATH  # noqa: E402
from src.agent.copilot import ask as copilot_ask  # noqa: E402
from src.agent.copilot import stream_ask as copilot_stream_ask  # noqa: E402

app = FastAPI(
    title="ChemSafety Copilot API",
    description="Agentic RAG over CSB chemical incident reports, plus live PubChem lookups and PSV sizing.",
    version="0.1.0",
)

_feedback_lock = threading.Lock()


class AskRequest(BaseModel):
    query: str
    history: list[dict] = []


class FeedbackRequest(BaseModel):
    query: str
    resolved_query: str | None = None
    intent: str | None = None
    answer: str
    rating: Literal["up", "down"]
    comment: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    try:
        return copilot_ask(request.query, request.history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    def event_source():
        try:
            for kind, payload in copilot_stream_ask(request.query, request.history):
                if kind == "delta":
                    yield f"data: {json.dumps({'type': 'delta', 'text': payload})}\n\n"
                elif kind == "routing":
                    yield f"data: {json.dumps({'type': 'routing', **payload})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'done', **payload})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> dict:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **request.model_dump(),
    }
    with _feedback_lock:
        with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    return {"status": "ok"}
