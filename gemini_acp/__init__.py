"""Gemini ACP shared package — provides summarize_via_gemini for any consumer."""
from gemini_acp.client import summarize_via_gemini, ACP_AVAILABLE, GeminiUsage

__all__ = ["summarize_via_gemini", "ACP_AVAILABLE", "GeminiUsage"]
