"""Lightweight structured logging for the CLI.

Every log line is timestamped (UTC wall time + seconds since process start) and
flushed immediately so remote API runs show live progress instead of appearing
to hang. Logs go to stderr so they don't contaminate JSON/JSONL output on
stdout (used by `runner_import.py` and piped tools).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterator


_START = time.monotonic()
_QUIET = False


def set_quiet(quiet: bool) -> None:
    global _QUIET
    _QUIET = quiet


def is_quiet() -> bool:
    return _QUIET or os.environ.get("HINAING_EVAL_QUIET") == "1"


def log(message: str, *, prefix: str = "eval") -> None:
    if is_quiet():
        return
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    elapsed = time.monotonic() - _START
    sys.stderr.write(f"[{now} +{elapsed:7.1f}s] [{prefix}] {message}\n")
    sys.stderr.flush()


@contextmanager
def timed(label: str, *, prefix: str = "eval") -> Iterator[None]:
    log(f"▶ {label}", prefix=prefix)
    t0 = time.monotonic()
    try:
        yield
    finally:
        dt = time.monotonic() - t0
        log(f"✓ {label} ({dt:.2f}s)", prefix=prefix)


@asynccontextmanager
async def heartbeat(label: str, *, every: float = 15.0, prefix: str = "eval") -> AsyncIterator[None]:
    """Emit a 'still waiting' line every ``every`` seconds until the block exits.

    Use this around any single long-running HTTP request that has no native
    progress signal (e.g. the non-streaming /insights/snapshot endpoint) so the
    operator can tell the eval is live rather than wedged.
    """
    stop = asyncio.Event()
    t0 = time.monotonic()

    async def _tick() -> None:
        ticks = 0
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=every)
                return
            except asyncio.TimeoutError:
                ticks += 1
                log(
                    f"… still waiting: {label} (elapsed {time.monotonic() - t0:.1f}s, tick #{ticks})",
                    prefix=prefix,
                )

    task = asyncio.create_task(_tick())
    log(f"▶ {label}", prefix=prefix)
    try:
        yield
    finally:
        stop.set()
        try:
            await task
        except Exception:
            pass
        log(f"✓ {label} ({time.monotonic() - t0:.2f}s)", prefix=prefix)


def kv(**fields: Any) -> str:
    """Render k=v pairs for compact structured log lines."""
    return " ".join(f"{k}={v}" for k, v in fields.items())
