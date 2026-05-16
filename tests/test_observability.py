from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


def test_init_observability_loads_dotenv_before_langfuse(monkeypatch):
    calls = {"dotenv_path": None, "dotenv_loaded": False}
    expected_env_path = Path(__file__).resolve().parents[1] / ".env"

    def fake_load_dotenv(path):
        calls["dotenv_path"] = path
        calls["dotenv_loaded"] = True

    class FakeLangfuseClient:
        def auth_check(self):
            assert calls["dotenv_loaded"] is True
            return True

    class FakeInstrumentor:
        def instrument(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        types.SimpleNamespace(load_dotenv=fake_load_dotenv),
    )
    monkeypatch.setitem(
        sys.modules,
        "langfuse",
        types.SimpleNamespace(get_client=lambda: FakeLangfuseClient()),
    )
    monkeypatch.setitem(
        sys.modules,
        "openinference.instrumentation.claude_agent_sdk",
        types.SimpleNamespace(ClaudeAgentSDKInstrumentor=lambda: FakeInstrumentor()),
    )

    sys.modules.pop("arxiv_rag.observability", None)
    observability = importlib.import_module("arxiv_rag.observability")

    observability.init_observability()

    assert calls["dotenv_path"] == expected_env_path


def test_init_observability_authenticates_and_instruments_once(monkeypatch):
    calls = {"auth_check": 0, "instrument": 0}

    class FakeLangfuseClient:
        def auth_check(self):
            calls["auth_check"] += 1
            return True

    class FakeInstrumentor:
        def instrument(self):
            calls["instrument"] += 1

    monkeypatch.setitem(
        sys.modules,
        "langfuse",
        types.SimpleNamespace(get_client=lambda: FakeLangfuseClient()),
    )
    monkeypatch.setitem(
        sys.modules,
        "openinference.instrumentation.claude_agent_sdk",
        types.SimpleNamespace(ClaudeAgentSDKInstrumentor=lambda: FakeInstrumentor()),
    )

    sys.modules.pop("arxiv_rag.observability", None)
    observability = importlib.import_module("arxiv_rag.observability")

    observability.init_observability()
    observability.init_observability()

    assert calls == {"auth_check": 1, "instrument": 1}


def test_init_observability_raises_when_langfuse_auth_fails(monkeypatch):
    class FakeLangfuseClient:
        def auth_check(self):
            return False

    class FakeInstrumentor:
        def instrument(self):
            raise AssertionError("instrument should not run when auth fails")

    monkeypatch.setitem(
        sys.modules,
        "langfuse",
        types.SimpleNamespace(get_client=lambda: FakeLangfuseClient()),
    )
    monkeypatch.setitem(
        sys.modules,
        "openinference.instrumentation.claude_agent_sdk",
        types.SimpleNamespace(ClaudeAgentSDKInstrumentor=lambda: FakeInstrumentor()),
    )

    sys.modules.pop("arxiv_rag.observability", None)
    observability = importlib.import_module("arxiv_rag.observability")

    with pytest.raises(RuntimeError, match="Langfuse authentication failed"):
        observability.init_observability()
