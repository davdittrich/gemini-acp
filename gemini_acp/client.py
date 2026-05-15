"""Gemini CLI client via Agent Client Protocol (ACP).

Provides summarize_via_gemini() for text summarization using Gemini CLI
in ACP mode (--acp). Falls back to None if the Gemini binary is
unavailable, allowing the summarizer fallback chain to continue.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import glob
import json
import os
import shutil
import time as _time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from acp import spawn_agent_process, text_block, PROTOCOL_VERSION, RequestError
from acp.schema import (
    ClientCapabilities,
    FileSystemCapabilities,
    ReadTextFileResponse,
    WriteTextFileResponse,
    RequestPermissionResponse,
    DeniedOutcome,
    AllowedOutcome,
    AgentMessageChunk,
    TextContentBlock,
    UsageUpdate,
)

ACP_AVAILABLE = True


@dataclass
class GeminiUsage:
    tokens_used: int
    cost_usd: float | None
    cost_currency: str | None
    is_estimated: bool = False
    cost_is_estimated: bool = False


class _GeminiClient:
    """Minimal ACP client — read-only, no terminal, no writes."""

    def __init__(self, cwd: str):
        self._cwd = cwd
        self._response_text = ""
        self._usage: GeminiUsage | None = None

    async def read_text_file(self, path: str, session_id: str, limit: int | None = None, line: int | None = None, **kw) -> "ReadTextFileResponse":
        resolved = Path(path).resolve()
        root = Path(self._cwd).resolve()
        if not resolved.is_relative_to(root):
            raise RequestError(-32602, f"Outside workspace: {resolved}")
        return ReadTextFileResponse(content=resolved.read_text("utf-8"))

    async def write_text_file(self, content: str, path: str, session_id: str, **kw) -> "WriteTextFileResponse":
        raise RequestError(-32601, "Write not permitted")

    async def request_permission(self, options, session_id: str, tool_call=None, **kw) -> "RequestPermissionResponse":
        is_write = getattr(tool_call, "kind", None) in ("edit", "write")
        if is_write:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        first = options[0] if options else None
        if first is None:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id=first.option_id, outcome="selected")
        )

    async def session_update(self, session_id: str, update, **kw) -> None:
        if isinstance(update, AgentMessageChunk):
            if isinstance(update.content, TextContentBlock):
                self._response_text += update.content.text
        elif isinstance(update, UsageUpdate):
            self._usage = GeminiUsage(
                tokens_used=update.used,
                cost_usd=update.cost.amount if update.cost else None,
                cost_currency=update.cost.currency if update.cost else None,
            )

    async def create_terminal(self, *a, **kw):
        raise RequestError(-32601, "Terminal not permitted")

    async def terminal_output(self, *a, **kw):
        raise RequestError(-32601, "Terminal not available")

    async def release_terminal(self, *a, **kw):
        return None

    async def wait_for_terminal_exit(self, *a, **kw):
        raise RequestError(-32601, "Terminal not available")

    async def kill_terminal(self, *a, **kw):
        return None

    async def ext_method(self, method: str, params=None):
        raise RequestError(-32601, f"Unknown: {method}")

    async def ext_notification(self, method: str, params=None) -> None:
        pass

    def on_connect(self, conn) -> None:
        pass  # Required by ACP protocol; this client doesn't use the connection directly


_CAPABILITIES = ClientCapabilities(
    fs=FileSystemCapabilities(read_text_file=True, write_text_file=False),
    terminal=False,
)


_MODEL_PRICING: dict[str, dict[str, float]] = {
    # Gemini 2.5 Flash
    "gemini-2.5-flash":         {"input": 0.15,  "cached": 0.0375, "output": 0.60,  "thoughts": 3.50},
    "gemini-2.5-flash-preview": {"input": 0.15,  "cached": 0.0375, "output": 0.60,  "thoughts": 3.50},
    # Gemini 2.5 Pro
    "gemini-2.5-pro":           {"input": 1.25,  "cached": 0.3125, "output": 10.00, "thoughts": 10.00},
    "gemini-2.5-pro-preview":   {"input": 1.25,  "cached": 0.3125, "output": 10.00, "thoughts": 10.00},
    # Gemini 3.1 Pro Preview (assume 2.5 Pro rates until official pricing released)
    "gemini-3.1-pro-preview":   {"input": 1.25,  "cached": 0.3125, "output": 10.00, "thoughts": 10.00},
    "gemini-3.1-flash-preview": {"input": 0.15,  "cached": 0.0375, "output": 0.60,  "thoughts": 3.50},
}


def _calculate_cost(tokens: dict, model: str) -> float | None:
    pricing = _MODEL_PRICING.get(model)
    if not pricing:
        return None
    per_m = 1_000_000
    non_cached_input = max(0, tokens.get("input", 0) - tokens.get("cached", 0))
    return (
        non_cached_input * pricing["input"] / per_m
        + tokens.get("cached", 0) * pricing["cached"] / per_m
        + tokens.get("output", 0) * pricing["output"] / per_m
        + tokens.get("thoughts", 0) * pricing["thoughts"] / per_m
    )


def _read_usage_from_session_file(start_wall: float) -> "GeminiUsage | None":
    """Read real token counts from the Gemini CLI session JSONL written during this call.

    Gemini CLI writes one JSON object per line to
    ~/.gemini/tmp/<name>/chats/session-*.jsonl.  Looks for a file modified at
    or after start_wall (with 2s tolerance), then scans lines in reverse for
    the last entry that has a 'tokens' dict containing 'total'.
    Returns GeminiUsage with is_estimated=False, or None if no matching file
    is found or it cannot be parsed.
    """
    pattern = os.path.expanduser("~/.gemini/tmp/**/chats/session-*.jsonl")
    try:
        files = [
            f for f in glob.glob(pattern, recursive=True)
            if os.path.getmtime(f) >= start_wall - 2.0
        ]
    except OSError:
        return None
    if not files:
        return None
    session_file = max(files, key=os.path.getmtime)
    try:
        with open(session_file) as fh:
            lines = fh.readlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "tokens" in msg:
                tok = msg["tokens"]
                model = msg.get("model", "")
                cost = _calculate_cost(tok, model)
                return GeminiUsage(
                    tokens_used=tok.get("total", 0),
                    cost_usd=cost,
                    cost_currency="USD" if cost is not None else None,
                    is_estimated=False,
                    cost_is_estimated=cost is not None,
                )
    except Exception:
        return None
    return None


async def _run_prompt(prompt_text: str, model: str = "", timeout: float = 30.0,
                      cwd: str = ".") -> tuple[str | None, GeminiUsage | None]:
    """Send a prompt to Gemini via ACP. Returns response text or None on failure.

    Uses a single shared deadline across all phases (connect + session + prompt)
    to prevent cumulative timeout overruns.
    """
    deadline = _time.monotonic() + timeout

    def _remaining() -> float:
        return max(0.1, deadline - _time.monotonic())

    start_wall = _time.time()
    flags = []
    if model:
        flags.extend(["--model", model])

    client = _GeminiClient(cwd=cwd)

    try:
        async with spawn_agent_process(
            client, "gemini", "--acp", *flags,
            cwd=cwd,
        ) as (conn, proc):
            try:
                await asyncio.wait_for(
                    conn.initialize(
                        protocol_version=PROTOCOL_VERSION,
                        client_capabilities=_CAPABILITIES,
                    ),
                    timeout=_remaining(),
                )
                session = await asyncio.wait_for(
                    conn.new_session(cwd=cwd, mcp_servers=[]),
                    timeout=_remaining(),
                )
                await asyncio.wait_for(
                    conn.prompt(
                        session_id=session.session_id,
                        prompt=[text_block(prompt_text)],
                    ),
                    timeout=_remaining(),
                )
            finally:
                # Gemini stays alive after prompt (interactive mode). Kill before
                # __aexit__ calls conn.close() so the stream gets EOF immediately
                # rather than hanging waiting for a live process to close its pipe.
                with suppress(ProcessLookupError):
                    proc.kill()
    except asyncio.TimeoutError:
        logger.warning(f"Gemini ACP timed out after {timeout}s")
        return (None, None)
    except FileNotFoundError:
        logger.warning("Gemini CLI not found on PATH")
        return (None, None)
    except Exception as e:
        logger.warning(f"Gemini ACP error: {e}")
        return (None, None)

    text = client._response_text.strip()
    if client._usage is None and text:
        # Try to read real token counts from Gemini CLI session file
        usage: GeminiUsage | None = _read_usage_from_session_file(start_wall)
        if usage is None:
            # No session file found — fall back to character-count estimate
            estimated_tokens = (len(prompt_text) + len(text)) // 4
            usage = GeminiUsage(
                tokens_used=estimated_tokens,
                cost_usd=None,
                cost_currency=None,
                is_estimated=True,
            )
    else:
        usage = client._usage
    return (text if text else None, usage)


def _run_sync(coro) -> tuple[str | None, GeminiUsage | None] | None:
    """Run async coroutine synchronously, handling existing event loops.

    Uses asyncio.run() when no loop is running (normal cron execution).
    Falls back to ThreadPoolExecutor if called from an async context.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        return asyncio.run(coro)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()


def summarize_via_gemini(text: str, prompt: str, model: str = "",
                         timeout: int = 30) -> tuple[str | None, GeminiUsage | None]:
    """Summarize text using Gemini CLI via ACP.

    Returns (response_text, usage) where usage is a GeminiUsage or None.
    Falls back to (None, None) if Gemini is not installed or any error occurs.
    The summarizer fallback chain handles the None text case.
    """
    if not shutil.which("gemini"):
        logger.warning("Gemini CLI not found on PATH")
        return (None, None)

    full_prompt = f"{prompt}\n\n{text}"
    result = _run_sync(_run_prompt(full_prompt, model=model, timeout=float(timeout)))
    if result is None:
        return (None, None)
    return result
