"""Tests for gemini_acp.client — ACP-based Gemini summarization."""
from __future__ import annotations

from unittest.mock import patch


class TestSummarizeViaGemini:
    def test_returns_none_when_gemini_not_on_path(self):
        with patch("gemini_acp.client.shutil") as mock_shutil:
            mock_shutil.which.return_value = None
            from gemini_acp.client import summarize_via_gemini
            result = summarize_via_gemini("text", "prompt")
        assert result == (None, None)

    def test_returns_none_on_timeout(self):
        """When _run_sync returns None (from timeout inside _run_prompt), result is (None, None)."""
        with patch("gemini_acp.client.shutil") as mock_shutil, \
             patch("gemini_acp.client._run_sync", return_value=None):
            mock_shutil.which.return_value = "/usr/bin/gemini"
            from gemini_acp.client import summarize_via_gemini
            result = summarize_via_gemini("text", "prompt", timeout=1)
        assert result == (None, None)

    def test_model_passed_as_kwarg(self):
        """Verify model parameter is forwarded to _run_prompt."""
        call_args = {}
        async def _capture_prompt(*_, model="", **__):
            call_args["model"] = model
            return ("summary", None)

        with patch("gemini_acp.client.shutil") as mock_shutil, \
             patch("gemini_acp.client._run_prompt", side_effect=_capture_prompt):
            mock_shutil.which.return_value = "/usr/bin/gemini"
            from gemini_acp.client import summarize_via_gemini
            result = summarize_via_gemini("text", "prompt", model="gemini-3-flash-preview")
        assert call_args["model"] == "gemini-3-flash-preview"
        assert result == ("summary", None)

    def test_empty_model_not_passed_as_flag(self):
        """When model is empty, no --model flag should be in the command."""
        call_args = {}
        async def _capture_prompt(*_, model="", **__):
            call_args["model"] = model
            return ("summary", None)

        with patch("gemini_acp.client.shutil") as mock_shutil, \
             patch("gemini_acp.client._run_prompt", side_effect=_capture_prompt):
            mock_shutil.which.return_value = "/usr/bin/gemini"
            from gemini_acp.client import summarize_via_gemini
            result = summarize_via_gemini("text", "prompt", model="")
        assert call_args["model"] == ""
        assert result == ("summary", None)


def test_proc_kill_called_on_timeout():
    """proc.kill() must be called when conn.prompt() times out."""
    import asyncio
    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_proc = MagicMock()
    mock_conn = MagicMock()
    mock_conn.initialize = AsyncMock(return_value=MagicMock())
    mock_conn.new_session = AsyncMock(return_value=MagicMock())
    mock_conn.prompt = AsyncMock(side_effect=asyncio.TimeoutError)

    @asynccontextmanager
    async def fake_spawn(*_, **__):
        yield mock_conn, mock_proc

    with patch("gemini_acp.client.spawn_agent_process", fake_spawn):
        from gemini_acp.client import _run_prompt
        result = asyncio.run(_run_prompt("test prompt", timeout=1.0))

    assert result == (None, None)
    mock_proc.kill.assert_called_once()


class TestRunSync:
    def test_works_without_event_loop(self):
        """Normal cron context — no pre-existing event loop."""
        from gemini_acp.client import _run_sync

        async def _simple():
            return "hello"

        result = _run_sync(_simple())
        assert result == "hello"


# --- New tests for UsageUpdate / GeminiUsage ---

def test_usage_update_captured():
    """When UsageUpdate fires, GeminiUsage is populated in the result tuple."""
    import asyncio
    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock, MagicMock, patch
    from acp.schema import UsageUpdate, Cost
    from gemini_acp.client import GeminiUsage  # type: ignore[reportAttributeAccessIssue]

    captured_client = {}

    @asynccontextmanager
    async def fake_spawn(client_obj, *_, **__):
        captured_client['ref'] = client_obj
        mock_conn = MagicMock()
        mock_conn.initialize = AsyncMock(return_value=MagicMock())
        mock_conn.new_session = AsyncMock(return_value=MagicMock())
        async def fake_prompt(**kw):
            from acp.schema import AgentMessageChunk, TextContentBlock
            await client_obj.session_update('s', UsageUpdate(used=1234, cost=Cost(amount=0.002345, currency='USD'), size=8192, session_update='usage_update'))
            await client_obj.session_update('s', AgentMessageChunk(content=TextContentBlock(type='text', text='summary'), session_update='agent_message_chunk'))
        mock_conn.prompt = AsyncMock(side_effect=fake_prompt)
        yield mock_conn, MagicMock()

    with patch('gemini_acp.client.spawn_agent_process', fake_spawn):
        from gemini_acp.client import _run_prompt
        result = asyncio.run(_run_prompt('test', timeout=10.0))
        assert result is not None
        text, usage = result

    assert text == 'summary'
    assert isinstance(usage, GeminiUsage)
    assert usage.tokens_used == 1234  # type: ignore[reportAttributeAccessIssue]
    assert abs(usage.cost_usd - 0.002345) < 1e-9  # type: ignore[reportAttributeAccessIssue]
    assert usage.cost_currency == 'USD'  # type: ignore[reportAttributeAccessIssue]


def test_no_usage_update_returns_estimated():
    """When no UsageUpdate fires, usage is an estimated GeminiUsage with is_estimated=True."""
    import asyncio
    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock, MagicMock, patch

    @asynccontextmanager
    async def fake_spawn(client_obj, *_, **__):
        mock_conn = MagicMock()
        mock_conn.initialize = AsyncMock(return_value=MagicMock())
        mock_conn.new_session = AsyncMock(return_value=MagicMock())
        async def fake_prompt(**kw):
            from acp.schema import AgentMessageChunk, TextContentBlock
            await client_obj.session_update('s', AgentMessageChunk(content=TextContentBlock(type='text', text='hello'), session_update='agent_message_chunk'))
        mock_conn.prompt = AsyncMock(side_effect=fake_prompt)
        yield mock_conn, MagicMock()

    with patch('gemini_acp.client.spawn_agent_process', fake_spawn):
        from gemini_acp.client import _run_prompt
        result = asyncio.run(_run_prompt('test', timeout=10.0))
        assert result is not None
        text, usage = result

    assert text == 'hello'
    assert usage is not None
    assert usage.is_estimated is True  # type: ignore[reportAttributeAccessIssue]
    assert usage.cost_usd is None  # type: ignore[reportAttributeAccessIssue]
    assert usage.cost_currency is None  # type: ignore[reportAttributeAccessIssue]
    # estimated_tokens = (len("test") + len("hello")) // 4 = 2
    assert usage.tokens_used == 2  # type: ignore[reportAttributeAccessIssue]


def test_usage_update_no_cost():
    """UsageUpdate with cost=None → cost_usd=None, cost_currency=None."""
    import asyncio
    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock, MagicMock, patch
    from acp.schema import UsageUpdate
    from gemini_acp.client import GeminiUsage  # type: ignore[reportAttributeAccessIssue]

    @asynccontextmanager
    async def fake_spawn(client_obj, *_, **__):
        mock_conn = MagicMock()
        mock_conn.initialize = AsyncMock(return_value=MagicMock())
        mock_conn.new_session = AsyncMock(return_value=MagicMock())
        async def fake_prompt(**kw):
            from acp.schema import AgentMessageChunk, TextContentBlock
            await client_obj.session_update('s', UsageUpdate(used=500, cost=None, size=4096, session_update='usage_update'))
            await client_obj.session_update('s', AgentMessageChunk(content=TextContentBlock(type='text', text='ok'), session_update='agent_message_chunk'))
        mock_conn.prompt = AsyncMock(side_effect=fake_prompt)
        yield mock_conn, MagicMock()

    with patch('gemini_acp.client.spawn_agent_process', fake_spawn):
        from gemini_acp.client import _run_prompt
        result = asyncio.run(_run_prompt('test', timeout=10.0))
        assert result is not None
        text, usage = result

    assert isinstance(usage, GeminiUsage)
    assert usage.tokens_used == 500  # type: ignore[reportAttributeAccessIssue]
    assert usage.cost_usd is None  # type: ignore[reportAttributeAccessIssue]
    assert usage.cost_currency is None  # type: ignore[reportAttributeAccessIssue]
