from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceHandle:
    """Trace handle for Langfuse v3 SDK."""

    enabled: bool
    trace_id: str | None = None
    tool_events: list[dict] = field(default_factory=list)
    answer: str | None = None
    _span: Any | None = None
    _langfuse_client: Any | None = None

    def record_tool(self, *, name: str, input_payload: dict, output_payload: dict) -> None:
        """Record a tool call to the trace as a nested span."""
        self.tool_events.append(
            {"name": name, "input": input_payload, "output": output_payload}
        )
        if self.enabled and self._langfuse_client:
            try:
                with self._langfuse_client.start_as_current_observation(
                    name=name,
                    as_type="span",
                    input=input_payload,
                    output=output_payload,
                ):
                    pass
            except Exception:
                pass

    def record_result(self, *, answer: str) -> None:
        """Record the final answer to the trace."""
        self.answer = answer
        if self.enabled and self._span:
            try:
                self._span.update(output=answer)
            except Exception:
                pass
        elif self.enabled and self._langfuse_client:
            try:
                self._langfuse_client.update_current_span(output=answer)
            except Exception:
                pass

    def close(self) -> None:
        """Close the trace and flush to Langfuse."""
        if self._span:
            try:
                self._span.__exit__(None, None, None)
            except Exception:
                pass
            self._span = None
        if self.enabled and self._langfuse_client:
            try:
                self._langfuse_client.flush()
            except Exception:
                pass

    def get_trace_url(self) -> str | None:
        """Get the URL to view this trace in Langfuse UI."""
        if not self.trace_id:
            return None
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        return f"{host}/project/{os.getenv('LANGFUSE_PUBLIC_KEY', '')}/traces/{self.trace_id}"

    @property
    def root_observation(self):
        """Backward compatibility property."""
        return self._span


def build_tracer(*, input_payload: dict | None = None) -> TraceHandle:
    """
    Build a Langfuse tracer using v3 SDK patterns.

    Returns a TraceHandle with tracing enabled if LANGFUSE_* env vars are set.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")

    if not (public_key and secret_key and host):
        return TraceHandle(enabled=False, trace_id=None)

    try:
        from langfuse import Langfuse

        langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=host,
        )

        context_manager = langfuse_client.start_as_current_observation(
            name="run_agent_turn",
            as_type="span",
            input=input_payload,
        )
        root = context_manager.__enter__()

        return TraceHandle(
            enabled=True,
            trace_id=getattr(root, "trace_id", None),
            _span=context_manager,
            _langfuse_client=langfuse_client,
        )

    except Exception as e:
        return TraceHandle(enabled=False, trace_id=None)