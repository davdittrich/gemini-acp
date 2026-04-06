"""Tests for gemini_acp.client — ACP-based Gemini summarization."""
from __future__ import annotations

from unittest.mock import patch


class TestSummarizeViaGemini:
    def test_returns_none_when_gemini_not_on_path(self):
        with patch("gemini_acp.client.shutil") as mock_shutil:
            mock_shutil.which.return_value = None
            from gemini_acp.client import summarize_via_gemini
            result = summarize_via_gemini("text", "prompt")
        assert result is None

    def test_returns_none_on_timeout(self):
        """When _run_sync returns None (from timeout inside _run_prompt), result is None."""
        with patch("gemini_acp.client.shutil") as mock_shutil, \
             patch("gemini_acp.client._run_sync", return_value=None):
            mock_shutil.which.return_value = "/usr/bin/gemini"
            from gemini_acp.client import summarize_via_gemini
            result = summarize_via_gemini("text", "prompt", timeout=1)
        assert result is None

    def test_model_passed_as_kwarg(self):
        """Verify model parameter is forwarded to _run_prompt."""
        call_args = {}
        async def _capture_prompt(prompt_text, model="", timeout=30.0, cwd="."):
            call_args["model"] = model
            return "summary"

        with patch("gemini_acp.client.shutil") as mock_shutil, \
             patch("gemini_acp.client._run_prompt", side_effect=_capture_prompt):
            mock_shutil.which.return_value = "/usr/bin/gemini"
            from gemini_acp.client import summarize_via_gemini
            result = summarize_via_gemini("text", "prompt", model="gemini-3-flash-preview")
        assert call_args["model"] == "gemini-3-flash-preview"
        assert result == "summary"

    def test_empty_model_not_passed_as_flag(self):
        """When model is empty, no --model flag should be in the command."""
        call_args = {}
        async def _capture_prompt(prompt_text, model="", timeout=30.0, cwd="."):
            call_args["model"] = model
            return "summary"

        with patch("gemini_acp.client.shutil") as mock_shutil, \
             patch("gemini_acp.client._run_prompt", side_effect=_capture_prompt):
            mock_shutil.which.return_value = "/usr/bin/gemini"
            from gemini_acp.client import summarize_via_gemini
            result = summarize_via_gemini("text", "prompt", model="")
        assert call_args["model"] == ""


class TestRunSync:
    def test_works_without_event_loop(self):
        """Normal cron context — no pre-existing event loop."""
        from gemini_acp.client import _run_sync
        import asyncio

        async def _simple():
            return "hello"

        result = _run_sync(_simple())
        assert result == "hello"
