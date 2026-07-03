"""FastAPI backend exposing the ChemSafety Copilot agent over HTTP.

Run locally with: uvicorn app.main:app --reload
"""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent.copilot import ask as copilot_ask  # noqa: E402

app = FastAPI(
    title="ChemSafety Copilot API",
    description="Agentic RAG over CSB chemical incident reports, plus live PubChem lookups and PSV sizing.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    query: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    try:
        return copilot_ask(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
