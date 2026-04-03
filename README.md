# gemini-acp

A Python client for Gemini CLI via the Agent Client Protocol (ACP). Sends
a prompt, receives a text response — nothing more.

Designed as a shared dependency for tools that need Gemini summarization
without managing the ACP protocol themselves.

## What it does

- Spawns `gemini --acp` as a subprocess
- Communicates via JSON-RPC (ACP protocol) over stdio
- Sends one prompt, collects the streamed response, returns the text
- Read-only: Gemini can read files in the workspace; it cannot write,
  execute commands, or access files outside the workspace root

## What it does not do

- Session management (each call is stateless)
- Caching (consumers manage their own cache)
- Retry logic (consumers decide whether to retry or fall back)
- File writes or terminal execution (blocked by policy)

## Installation

```bash
pip install -e path/to/gemini-acp        # base (graceful degradation)
pip install -e "path/to/gemini-acp[acp]" # with ACP library (required for actual use)
```

The base install provides `summarize_via_gemini` and `ACP_AVAILABLE`.
Without the `[acp]` extra, `ACP_AVAILABLE` is `False` and all calls
return `None` — consumers fall back to alternative backends.

### Prerequisites

- Python 3.11+
- Gemini CLI 0.34+ on PATH (`gemini --version`)
- Authenticated: `gemini auth login`

## Usage

### Python library

```python
from gemini_acp import summarize_via_gemini, ACP_AVAILABLE

if ACP_AVAILABLE:
    result = summarize_via_gemini(
        text="Long article text here...",
        prompt="Summarize in 2 sentences.",
        model="gemini-3-flash-preview",  # optional; empty = CLI default
        timeout=30,                       # seconds
    )
    if result:
        print(result)
```

`summarize_via_gemini` returns the response text as a string, or `None`
on any failure (timeout, missing binary, ACP error). It never raises.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | The text to include in the prompt |
| `prompt` | `str` | required | Instruction prepended to the text |
| `model` | `str` | `""` | Gemini model name; empty uses CLI default |
| `timeout` | `int` | `30` | Total wall-clock seconds for the entire operation |

### Model selection

| Model | Speed | Cost | Use case |
|-------|-------|------|----------|
| `gemini-3-flash-preview` | Fast | Low | Summarization, triage |
| `gemini-3.1-pro-preview` | Slower | Higher | Complex analysis |
| `""` (empty) | CLI default | Varies | Whatever the CLI is configured with |

## Architecture

```
Consumer (scholarposter, tldr-scholar, etc.)
    │
    ▼
summarize_via_gemini(text, prompt, model, timeout)
    │
    ├─ ACP_AVAILABLE is False? → return None
    ├─ gemini not on PATH? → return None
    │
    ▼
_run_sync( _run_prompt(...) )
    │
    ├─ No event loop running → asyncio.run()
    └─ Event loop running → ThreadPoolExecutor fallback
            │
            ▼
        _run_prompt(prompt_text, model, timeout, cwd)
            │
            ├─ spawn_agent_process("gemini", "--acp", "--model", ...)
            ├─ conn.initialize() ─┐
            ├─ conn.new_session() ┤── shared deadline (single timeout budget)
            └─ conn.prompt()    ──┘
                    │
                    ▼
            _GeminiClient (ACP callbacks)
                ├─ read_text_file: read-only within cwd
                ├─ write_text_file: DENIED
                ├─ create_terminal: DENIED
                ├─ request_permission: allow reads, deny writes
                └─ session_update: accumulate response text
```

### Timeout model

A single deadline spans all three ACP phases (initialize, new_session, prompt).
If initialization takes 10 seconds of a 30-second budget, the prompt phase gets
the remaining 20 seconds. This prevents cumulative timeout overruns where three
independent `wait_for` calls each get the full timeout.

### Security posture

The ACP client enforces a read-only policy:

- **File reads**: allowed within the workspace root (`cwd`). Paths outside the
  root are rejected with an ACP error.
- **File writes**: unconditionally denied.
- **Terminal execution**: unconditionally denied.
- **Permissions**: read operations are approved automatically; write operations
  are cancelled.

This means Gemini can read source files or documents in the workspace to
inform its response, but it cannot modify anything.

## Graceful degradation

The package is designed to be a non-breaking dependency:

| Condition | Behavior |
|-----------|----------|
| `agent-client-protocol` not installed | `ACP_AVAILABLE = False`; all calls return `None` |
| `gemini` binary not on PATH | Returns `None`; logs warning |
| ACP connection timeout | Returns `None`; logs warning |
| Any ACP protocol error | Returns `None`; logs warning |
| Gemini returns empty response | Returns `None` |

Consumers handle `None` by falling back to alternative backends
(e.g., Lemonade, Ollama, extractive summarization).

## Development

```bash
cd gemini-acp
pip install -e ".[acp,dev]"
pytest
```

## Consumers

- **scholarposter** — cross-posts Mastodon toots with Gemini-powered summarization
- **tldr-scholar** — standalone academic text summarizer
