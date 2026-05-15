"""Gemini ACP shared package — provides summarize_via_gemini for any consumer."""
from gemini_acp.client import summarize_via_gemini as summarize_via_gemini, ACP_AVAILABLE as ACP_AVAILABLE, GeminiUsage as GeminiUsage

__all__ = ["summarize_via_gemini", "ACP_AVAILABLE", "GeminiUsage"]
