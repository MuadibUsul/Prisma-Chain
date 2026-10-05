"""Log redaction: no key material may reach any log path (B2-01f).

Two layers:

1. **Registered secrets** — every secret representation the process knows
   about (hex/base64 of the protocol seed, the account scalar) is replaced
   by ``«redacted»`` wherever it appears, in the message or its arguments.
2. **Key-shaped patterns** — any 64+ character hex run is replaced by
   ``«redacted-hex»`` even if it was never registered, so a new code path
   that logs a raw key still cannot leak it.

``tests/test_no_secrets_in_logs.py`` asserts both layers.
"""

from __future__ import annotations

import logging
import re

_HEX_KEY = re.compile(r"\b[0-9a-fA-F]{64,}\b")
_REDACTED = "«redacted»"
_REDACTED_HEX = "«redacted-hex»"

_registered: set[str] = set()


def register_secret(value: str) -> None:
    """Register a secret representation for redaction (ignores short values)."""
    if isinstance(value, str) and len(value) >= 16:
        _registered.add(value)


def clear_registered_secrets() -> None:
    _registered.clear()


def scrub(text: str) -> str:
    for secret in _registered:
        text = text.replace(secret, _REDACTED)
    return _HEX_KEY.sub(_REDACTED_HEX, text)


class SecretRedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = scrub(record.msg)
        if record.args:
            record.args = tuple(
                scrub(arg) if isinstance(arg, str) else arg for arg in record.args)
        if record.exc_text:
            record.exc_text = scrub(record.exc_text)
        elif record.exc_info:
            # The handler would format the traceback later, after this
            # filter has run: scrub it now and pin it into exc_text.
            record.exc_text = scrub(logging.Formatter().formatException(record.exc_info))
        return True


def install(logger: logging.Logger | None = None) -> SecretRedactingFilter:
    """Install the filter on one logger (default: root) and return it."""
    filt = SecretRedactingFilter()
    (logger or logging.getLogger()).addFilter(filt)
    return filt
