import sys
import types

from arxiv_rag.tracing.langfuse import build_tracer


def test_enabled_tracer_reuses_one_trace_context(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.example")

    captured = {"updates": []}

    class FakeRootSpan:
        trace_id = "trace-123"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, **kwargs):
            captured["updates"].append(kwargs)

    class FakeLangfuse:
        def start_as_current_observation(self, **kwargs):
            captured["root_kwargs"] = kwargs
            return FakeRootSpan()

    monkeypatch.setitem(sys.modules, "langfuse", types.SimpleNamespace(Langfuse=lambda **kwargs: FakeLangfuse()))

    tracer = build_tracer(input_payload={"messages": [{"role": "user", "content": "hi"}]})

    assert tracer.enabled is True
    assert tracer.trace_id == "trace-123"


def test_build_tracer_disabled_when_env_missing(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)

    tracer = build_tracer(input_payload={"messages": []})

    assert tracer.enabled is False
    assert tracer.trace_id is None
