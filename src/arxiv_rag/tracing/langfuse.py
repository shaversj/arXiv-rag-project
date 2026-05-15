from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any


@dataclass
class TraceHandle:
    enabled: bool
    trace_id: str | None = None
    tool_events: list[dict] = field(default_factory=list)
    answer: str | None = None
    root_observation: Any | None = None

    def record_tool(self, *, name: str, input_payload: dict, output_payload: dict) -> None:
        self.tool_events.append(
            {"name": name, "input": input_payload, "output": output_payload}
        )
        if self.root_observation is not None:
            self.root_observation.update(
                metadata={"tool_events": self.tool_events}
            )

    def record_result(self, *, answer: str) -> None:
        self.answer = answer
        if self.root_observation is not None:
            self.root_observation.update(output=answer)

    def close(self) -> None:
        if self.root_observation is not None:
            self.root_observation.__exit__(None, None, None)
            self.root_observation = None


def build_tracer(*, input_payload: dict | None = None) -> TraceHandle:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")
    if not (public_key and secret_key and host):
        return TraceHandle(enabled=False, trace_id=None)

    try:
        langfuse_module = import_module("langfuse")
        client = langfuse_module.Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=host,
        )
        context_manager = client.start_as_current_observation(
            name="run_agent_turn",
            as_type="span",
            input=input_payload,
        )
        root = context_manager.__enter__()
    except Exception:
        return TraceHandle(enabled=False, trace_id=None)

    return TraceHandle(
        enabled=True,
        trace_id=getattr(root, "trace_id", None),
        root_observation=context_manager,
    )