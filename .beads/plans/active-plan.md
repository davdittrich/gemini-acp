---
title: "gemini-acp: Clean code pass (6 issues)"
status: draft
created: 2026-04-03
work_units: 1
baseline_tests: 6
---

# gemini-acp Clean Code Pass

## Issues

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 2 | REQUIRED | `from pathlib import Path` inside async method | Move to module top |
| 3 | REQUIRED | `_conn` stored but never read | Remove field + `on_connect` body; keep method signature (ACP requires it) |
| 4 | REQUIRED | Unused `MagicMock` and `pytest` imports in tests | Remove |
| 5 | SUGGESTION | `_get_capabilities` lazy init is redundant | Replace with module-level constant guarded by `ACP_AVAILABLE` |
| 6 | SUGGESTION | `import os` unused | Remove |
| 7 | SUGGESTION | `import concurrent.futures` unconditional | Keep — 1ms cost not worth lazy import complexity for 1 call site |

## WU-1: All fixes in one pass

**Files:** `gemini_acp/client.py`, `tests/test_gemini_acp.py`

### Changes to `gemini_acp/client.py`

1. **Move `from pathlib import Path` to module top** (after `import shutil`)

2. **Remove `_conn` field and `on_connect` body** — the method must exist (ACP
   interface), but the body can be a no-op:
   ```python
   def on_connect(self, conn) -> None:
       pass  # Required by ACP protocol; this client doesn't use the connection directly
   ```
   Remove `self._conn = None` from `__init__`.

3. **Remove `import os`** (unused — `os.environ.copy()` was removed in a prior refactor)

4. **Replace `_get_capabilities` lazy init with guarded constant:**
   ```python
   _CAPABILITIES = None
   if ACP_AVAILABLE:
       _CAPABILITIES = ClientCapabilities(
           fs=FileSystemCapabilities(read_text_file=True, write_text_file=False),
           terminal=False,
       )
   ```
   Delete the `_get_capabilities()` function. In `_run_prompt`, replace
   `client_capabilities=_get_capabilities()` with `client_capabilities=_CAPABILITIES`.
   This is safe because `_run_prompt` is only called when `ACP_AVAILABLE` is True
   (guarded by `summarize_via_gemini`).

5. **Keep `import concurrent.futures`** at module top — the 1ms import cost is
   not worth the readability loss of a lazy import for a 2-line function.

### Changes to `tests/test_gemini_acp.py`

6. **Remove unused imports:**
   ```python
   # Before:
   from unittest.mock import MagicMock, patch
   import pytest
   # After:
   from unittest.mock import patch
   ```

### TDD Tests
- All 6 existing tests pass unchanged (behavioral equivalence)
- `from gemini_acp.client import _CAPABILITIES` → is not None when ACP installed
- No new behavioral tests needed — these are pure refactors

## Success Criteria

1. All 6 existing tests pass
2. No unused imports in source or tests
3. `_conn` field removed
4. `_get_capabilities()` function removed; `_CAPABILITIES` is a module constant
5. `pathlib.Path` imported at module top
