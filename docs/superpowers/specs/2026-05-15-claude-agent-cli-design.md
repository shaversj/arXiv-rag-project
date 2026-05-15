# Claude Agent CLI Design

## Overview

This design replaces the current FastAPI and browser chat interface with a
library-first Claude-powered agent that answers arXiv questions by calling a
local retrieval tool. The system will expose:

- a reusable Python entrypoint for other applications
- a thin interactive CLI for direct use
- Langfuse tracing for per-turn observability

The v1 agent is retrieval-only. It may search the local arXiv index, synthesize
an answer from retrieved papers, and cite the papers it used. It will not write
data, perform administrative actions, or expose arbitrary tool access.

## Goals

- Replace HTTP and browser chat dependencies with a Python-native interface
- Use the Claude Agent SDK as the orchestration layer for tool calling
- Preserve the existing retrieval engine as the source of truth for paper search
- Provide a clean library API for future embedding into other applications
- Add Langfuse tracing with request-scoped visibility into tool usage and output

## Non-Goals

- No FastAPI service or OpenAI-compatible API in v1
- No OpenWebUI integration
- No multi-tool or admin-capable agent behavior
- No database schema changes as part of this migration
- No requirement for Langfuse to be configured in local development

## Recommended Approach

Build a reusable Python agent core and a thin CLI wrapper on top of it.

This keeps the core behavior independent from any specific interface while still
giving an immediate interactive entrypoint. The Claude Agent SDK will manage the
conversation and tool invocation. A single in-process retrieval tool will bridge
the agent to the current query engine. Langfuse will trace each turn at the
service layer rather than through a web framework.

## Architecture

```text
User / Python caller
        |
        v
  Agent entrypoint
        |
        v
 Claude Agent SDK session
        |
        v
search_arxiv_papers tool
        |
        v
   QueryEngine / PostgresStore
        |
        v
  Retrieved paper context
        |
        v
 Claude final answer with citations
        |
        v
 Langfuse trace + CLI/library result
```

## Components

### 1. Retrieval Module

The existing retrieval code remains the source of truth and should be moved into
the package namespace if needed so the rest of the system imports it cleanly.

Responsibilities:

- initialize the embedding model and database store
- execute semantic search against pgvector
- return normalized paper results in a stable internal shape

The retrieval layer should remain independent from Claude SDK and Langfuse so it
can be tested directly.

### 2. Agent Module

Add a new package area such as `src/arxiv_rag/agent/` containing:

- Claude Agent SDK client/session setup
- system prompt for retrieval-only behavior
- custom tool registration
- response parsing helpers
- citation-aware answer assembly

The agent should expose a library-first function, for example:

```python
async def run_agent_turn(messages: list[dict[str, str]]) -> AgentTurnResult:
    ...
```

`messages` should be simple role/content chat history objects so other Python
applications can call the agent without depending on CLI-specific models.

`AgentTurnResult` should include at least:

- `answer`: final assistant text
- `citations`: structured references to papers used
- `papers`: retrieved paper metadata returned by the tool
- `trace_id`: optional Langfuse trace identifier when tracing is enabled

### 3. Retrieval Tool Adapter

Register one Claude SDK tool, tentatively `search_arxiv_papers`, as an
in-process SDK MCP tool.

Tool contract:

- input: natural-language query and optional limit
- execution: call the local retrieval module
- output: compact structured results with paper ids, titles, authors, abstract
  snippets, categories, and relevance scores

The tool should be the only capability exposed to the agent in v1. Claude SDK
options should explicitly allow only this tool and should avoid file-editing or
shell-oriented permissions.

### 4. Tracing Module

Add `src/arxiv_rag/tracing/` with Langfuse setup and helpers.

Behavior:

- initialize Langfuse only when required environment variables are present
- create one top-level observation/span per user turn
- record input messages, tool arguments, retrieval latency, selected papers, and
  final answer
- return trace metadata to the caller when available

Tracing must fail open. If Langfuse is unavailable or not configured, the agent
and CLI should continue to work without tracing.

### 5. CLI Module

Add a thin chat interface, for example `src/arxiv_rag/chat_cli.py`, that:

- starts an interactive REPL-style session
- collects user messages and maintains local chat history
- calls the library entrypoint for each turn
- prints the final answer and citations
- supports graceful exit commands such as `exit` and `quit`

The CLI should not own retrieval or tracing logic; it should only translate
terminal interaction into library calls.

## Prompt and Tooling Strategy

The system prompt should narrowly define the agent's role:

- answer questions about arXiv papers using the retrieval tool
- use retrieved paper data as the basis for claims
- cite supporting papers in the final answer
- avoid inventing unavailable evidence
- acknowledge uncertainty or missing evidence when retrieval is weak

Because the agent is retrieval-only, the prompt should forbid actions outside
search and synthesis. This keeps behavior aligned with the narrow tool surface.

## Citation Strategy

The final answer should include lightweight citations that reference the papers
actually returned and used by the retrieval tool. The exact format can be simple
and deterministic, such as bracketed numeric references tied to a citation list.

Each citation should include enough metadata for follow-up:

- paper id
- title
- authors
- arXiv URL

The library result should return both the rendered answer and the structured
citation objects so downstream callers can present them differently if needed.

## Error Handling

The design should handle four failure classes explicitly:

1. Retrieval initialization failures
2. Empty or low-confidence retrieval results
3. Claude SDK or model execution failures
4. Langfuse initialization or trace submission failures

Expected behavior:

- retrieval/setup failures return a clear user-facing error
- empty retrieval results produce a graceful answer explaining that no relevant
  papers were found
- Claude failures surface a concise fallback error without exposing internals
- Langfuse failures are logged and ignored for user-facing behavior

## Testing Strategy

Add tests at two levels.

Unit tests:

- retrieval tool adapter output shaping
- citation formatting and deduplication
- Langfuse fail-open behavior
- library entrypoint behavior with mocked Claude responses

Integration-style tests:

- end-to-end agent orchestration with a mocked Claude layer and stub retrieval
- CLI interaction coverage for one or two representative happy/error paths

Live model calls should not be required in CI.

## Migration Plan

The migration should preserve the current retrieval functionality while removing
the obsolete interfaces.

Planned changes:

- remove FastAPI app usage from the primary interaction flow
- remove the static browser chat UI from the primary interaction flow
- move or wrap `query_engine.py` under `src/arxiv_rag/` for package-native use
- add Claude SDK, Langfuse, and CLI entrypoint code
- update project documentation and commands

The HTTP and browser files may either be deleted or left temporarily unused
during implementation, but the end state should make the CLI/library path the
documented and supported interface.

## Configuration

Configuration should continue to center on `config.yaml` and environment
variables.

Additions:

- Claude model selection and related SDK options
- Langfuse host/public key/secret key via environment variables
- optional CLI defaults such as retrieval limit if needed

Sensitive values must remain in environment variables rather than committed
files.

## Deployment and Usage

The primary usage mode becomes local Python execution, for example through `uv`.

Representative workflows:

- run the interactive CLI locally
- import the agent entrypoint from another Python program
- use Langfuse in environments where tracing credentials are configured

This keeps the system simple for local research workflows while preserving a
clean path to future integrations.

## Risks and Mitigations

- Claude SDK integration complexity
  - Mitigation: keep one tool and a narrow prompt in v1
- Trace noise or broken observability
  - Mitigation: centralize Langfuse instrumentation in one module
- Package boundary drift between retrieval and agent code
  - Mitigation: define a stable internal result model for retrieved papers
- Regression during removal of existing interfaces
  - Mitigation: keep retrieval tests focused on unchanged behavior and update
    docs to make the new supported path explicit

## Success Criteria

The migration is successful when:

- users can chat with the system through a local CLI
- Python callers can invoke the same agent through a library API
- the Claude agent can call exactly one retrieval tool
- answers include citations derived from retrieved papers
- Langfuse traces each turn when configured
- local development works even when Langfuse is not configured
