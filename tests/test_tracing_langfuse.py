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


def test_record_tool_appends_to_tool_events(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.example")

    class FakeRootSpan:
        trace_id = "trace-123"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, **kwargs):
            pass

    class FakeLangfuse:
        def start_as_current_observation(self, **kwargs):
            return FakeRootSpan()

    monkeypatch.setitem(sys.modules, "langfuse", types.SimpleNamespace(Langfuse=lambda **kwargs: FakeLangfuse()))

    tracer = build_tracer(input_payload={"messages": [{"role": "user", "content": "hi"}]})

    tracer.record_tool(
        name="search",
        input_payload={"query": "machine learning"},
        output_payload={"results": ["paper1", "paper2"]},
    )

    assert len(tracer.tool_events) == 1
    assert tracer.tool_events[0]["name"] == "search"
    assert tracer.tool_events[0]["input"] == {"query": "machine learning"}
    assert tracer.tool_events[0]["output"] == {"results": ["paper1", "paper2"]}


def test_record_result_sets_answer(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.example")

    class FakeRootSpan:
        trace_id = "trace-123"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, **kwargs):
            pass

    class FakeLangfuse:
        def start_as_current_observation(self, **kwargs):
            return FakeRootSpan()

    monkeypatch.setitem(sys.modules, "langfuse", types.SimpleNamespace(Langfuse=lambda **kwargs: FakeLangfuse()))

    tracer = build_tracer(input_payload={"messages": [{"role": "user", "content": "hi"}]})

    tracer.record_result(answer="The answer is 42")

    assert tracer.answer == "The answer is 42"


def test_close_calls_exit_on_root_observation(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.example")

    exit_called = {"count": 0}

    class FakeRootSpan:
        trace_id = "trace-123"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            exit_called["count"] += 1
            return False

        def update(self, **kwargs):
            pass

    class FakeLangfuse:
        def start_as_current_observation(self, **kwargs):
            return FakeRootSpan()

    monkeypatch.setitem(sys.modules, "langfuse", types.SimpleNamespace(Langfuse=lambda **kwargs: FakeLangfuse()))

    tracer = build_tracer(input_payload={"messages": [{"role": "user", "content": "hi"}]})

    tracer.close()

    assert exit_called["count"] == 1
    assert tracer.root_observation is None
