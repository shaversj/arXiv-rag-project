from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client
from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor

_instrumented = False
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def init_observability() -> None:
    global _instrumented

    if _instrumented:
        return

    load_dotenv(_ENV_PATH)

    langfuse = get_client()
    if not langfuse.auth_check():
        raise RuntimeError("Langfuse authentication failed")

    ClaudeAgentSDKInstrumentor().instrument()
    _instrumented = True
