"""Shared pytest fixtures. `fake_groq` replaces the real Groq client used by
src.agent.router with an in-memory fake that returns pre-queued JSON strings
-- routing tests need to be deterministic and free, not dependent on a live
API call whose classification could vary between runs."""
from types import SimpleNamespace

import pytest


class FakeGroqClient:
    """Queue JSON strings to be returned, in order, from successive
    `.chat.completions.create(...)` calls. Records every call's kwargs so
    tests can assert on what was sent, if needed."""

    def __init__(self):
        self.responses: list[str] = []
        self.calls: list[dict] = []

    def queue(self, *json_strings: str) -> None:
        self.responses.extend(json_strings)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.responses.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def fake_groq(monkeypatch) -> FakeGroqClient:
    fake = FakeGroqClient()
    monkeypatch.setattr("src.agent.router._client.chat.completions.create", fake._create)
    return fake
