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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FEEDBACK_LOG_PATH  # noqa: E402
from app.metrics import router as metrics_router  # noqa: E402
from src.agent.copilot import ask as copilot_ask  # noqa: E402
from src.agent.copilot import stream_ask as copilot_stream_ask  # noqa: E402
from src.telemetry.recorder import instrumented_stream, record_query  # noqa: E402
from src.telemetry.store import writer as telemetry_writer  # noqa: E402

app = FastAPI(
    title="ChemSafety Copilot API",
    description="Agentic RAG over CSB chemical incident reports, plus live PubChem lookups and PSV sizing.",
    version="0.1.0",
)

# The React frontend (frontend/) is served from a different origin than this
# API, so the browser preflights every /ask call. Streamlit needs none of this
# -- it calls the backend server-side -- so this exists purely for the SPA.
#
# ADD THE PRODUCTION ORIGIN HERE when deploying the frontend, e.g.
# "https://chemsafety-copilot.vercel.app". Keep this an explicit allowlist
# rather than ["*"]: allow_origins=["*"] is incompatible with credentialed
# requests and would leave the API callable from any page.
ALLOWED_ORIGINS = [
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Read-only aggregate metrics, mounted under /metrics/* and kept on its own
# router so the observability surface never mixes with the query API.
app.include_router(metrics_router)


@app.on_event("startup")
def _start_telemetry() -> None:
    """Bring the background writer thread up before the first query, so the
    first record does not pay the thread-start cost on the request path."""
    telemetry_writer.start()


@app.on_event("shutdown")
def _stop_telemetry() -> None:
    telemetry_writer.flush()
    telemetry_writer.stop()


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
    # record_query only observes: it re-raises whatever the pipeline raised, and
    # a telemetry failure inside it is swallowed rather than surfaced here.
    with record_query(request.query) as outcome:
        try:
            response = copilot_ask(request.query, request.history)
        except Exception as e:
            # Hand telemetry the real exception before translating it, so the
            # error breakdown shows "llm_provider", not "validation".
            outcome.set_error(e)
            raise HTTPException(status_code=500, detail=str(e)) from e
        outcome.set_response(response)
        return response


@app.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    def event_source():
        try:
            # instrumented_stream re-binds the telemetry scope around each step:
            # Starlette drives this generator one item at a time via
            # anyio.to_thread.run_sync, so a scope set on one step is gone by
            # the next and mid-stream LLM calls would go unattributed.
            stream = instrumented_stream(
                request.query, copilot_stream_ask(request.query, request.history)
            )
            for kind, payload in stream:
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
